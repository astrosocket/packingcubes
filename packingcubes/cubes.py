from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Collection
from pathlib import Path
from typing import cast

import h5py  # type: ignore
import numpy as np
from numba import (  # type:ignore
    get_num_threads,
    get_thread_id,
    njit,
    objmode,
    prange,
    threading_layer,
    types,
)
from numba.typed import List
from numpy.typing import NDArray

import packingcubes.bounding_box as bbox
from packingcubes.bounding_box import BoundingBox, make_bounding_box
from packingcubes.data_objects import (
    DataContainer,
    Dataset,
    GadgetishHDF5Dataset,
    HDF5Dataset,
    InMemory,
    MultiParticleDataset,
    subview,
)
from packingcubes.octree import _DEFAULT_PARTICLE_THRESHOLD
from packingcubes.packed_tree import (
    PackedTree,
)
from packingcubes.packed_tree.fixed_distance_heap import FixedDistanceHeap
from packingcubes.packed_tree.packed_tree_numba import (
    PackedTreeNumba,
    _construct_tree,
    _index_tuple_type,
    _list_index_tuple,
    _process_slice_against_heap,
    euclidean_distance_particle,
)

LOGGER = logging.getLogger(__name__)

nthreads = get_num_threads()


# need to test parallelism - have issues with using the tbb backend
# so it's useful to print diagnostic info
@njit(parallel=True)
def test_parallel():
    a = np.zeros((10,))
    for i in prange(len(a)):
        a[i] = i
    return a


test_parallel()
layer = threading_layer()
LOGGER.debug(f"Running on the {layer} threading layer with {nthreads} threads")
if layer == "tbb":
    LOGGER.warning(
        "Parallel support for cubes is known to be flaky on the tbb threading "
        "layer. If you are having difficulties, consider switching to the omp "
        "layer by setting the NUMBA_THREADING_LAYER environmental variable or "
        "by setting numba.config.THREADING_LAYER. See "
        "https://numba.readthedocs.io/en/stable/user/threading-layer.html for "
        "more information."
    )


# from https://stackoverflow.com/a/27434050
class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            contents = f.read()

        # parse arguments in the file and store them in a blank namespace
        data = parser.parse_args(contents.split(), namespace=None)
        for k, v in vars(data).items():
            # set arguments in the target namespace if they have not been set yet
            if getattr(namespace, k, None) is None:
                setattr(namespace, k, v)


def _process_args(argv=None):
    if argv is None:
        # need to skip caller or it's picked up as the snapshot file
        argv = sys.argv[1:]

    description = """
    Run the packingcubes program on a snapshot file.
    
    Default is to use the bounding box provided by the simulation,
    so if that's sufficient you do not need to provide x/y/z/dx/dy/dz
    """
    epilog = """
    If particle types are specified (using -t or --particle-types), the 
    snapshot file should be specified with -- SNAPSHOT OUTPUT at the end.
    
    Additional arguments can be read from a file by specifying the file with
    `@filename` anywhere among the argument string. Any arguments found in the
    file will overwrite previously specified arguments and be overwritten by 
    arguments specified later. Note that this is different behavior from the 
    -c/--config argument!
    """
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase output verbosity"
    )
    parser.add_argument(
        "-n",
        "--side-length",
        default=-1,
        help="""
        number of cells per side [3-32], default -1 means use the lowest number
        of cells such that n**3 > # of available threads
        """,
        type=int,
        dest="n",
    )
    parser.add_argument(
        "-p",
        "--particle-threshold",
        help=(
            "the maximum number of particles per octree leaf."
            f"\nDefault: {_DEFAULT_PARTICLE_THRESHOLD}"
        ),
        default=_DEFAULT_PARTICLE_THRESHOLD,
        type=int,
    )
    box_args = parser.add_argument_group(
        "Box parameters", "Arguments to override parts of the default bounding box"
    )
    for ax in "xyz":
        box_args.add_argument(
            f"-{ax}", help=f"minimum bounding box {ax} coordinate", type=float
        )
        box_args.add_argument(
            f"-d{ax}", help=f"bounding box size in {ax} direction", type=float
        )
    conf_arg = parser.add_argument(
        "-c",
        "--config",
        help=(
            "Read in specified config file for arguments (CLI arguments will override)"
        ),
        type=open,
        action=LoadFromFile,
    )
    parser.add_argument(
        "-t",
        "--particle-types",
        help=(
            "Names of particles to include "
            "(Can be integers or strings, 0 <=> PartType0)"
        ),
        nargs="+",
    )
    parser.add_argument("snapshot", help="Path to the snapshot file", type=str)
    parser.add_argument(
        "output",
        help="""
        Name of hdf5 file to save cubes information to. If not specified, cubes
        information will be discarded!
        """,
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--force-overwrite",
        help="Flag to overwrite cubes data contained in OUTPUT",
        action="store_true",
    )
    parser.add_argument(
        "--no-saving-dataset",
        help="""
        Don't save sorted particle positions and shuffle lists
        Normally sorted particle positions/shuffle lists are saved within a
        sidecar file to the snapshot. This flag disables that behavior.
        """,
        action="store_true",
    )

    args = parser.parse_args(argv)
    if args.n != -1 and (args.n < 3 or 32 < args.n):
        raise ValueError(
            f"Cubes per side needs to be within [3, 32]. You provided {args.n}"
        )
    if args.particle_types is not None:
        for i, n in enumerate(args.particle_types):
            try:
                args.particle_types[i] = f"PartType{int(n)}"
            except ValueError:
                continue
    if args.verbose >= 2:
        loglvl = logging.DEBUG
    elif args.verbose >= 1:
        loglvl = logging.INFO
    else:
        loglvl = LOGGER.level
    LOGGER.setLevel(loglvl)
    return args


def _process_box(*, dataset: Dataset, args) -> BoundingBox:
    box = dataset.bounding_box.box

    for i, ax in enumerate("xyz"):
        x: float | None = getattr(args, ax)
        if x is not None:
            box[i] = x
        dx: float | None = getattr(args, f"d{ax}")
        if dx is not None:
            box[3 + i] = dx

    return make_bounding_box(box)


def _get_cube_boxes(
    *, data: DataContainer, box: BoundingBox, cubes_per_side: int
) -> list[BoundingBox]:
    cube_size = box.size / cubes_per_side
    cube_boxes = List.empty_list(bbox.bbn_type)
    for i in range(cubes_per_side):
        for j in range(cubes_per_side):
            for k in range(cubes_per_side):
                cube_pos = box.position + [i, j, k] * cube_size
                cube = bbox.BoundingBox(np.hstack((cube_pos, cube_size)))
                cube_boxes.append(cube)
    cube_boxes.append(data.bounding_box)  # don't forget leftovers cube!
    return cube_boxes


@njit
def _prune_empty(
    num_particles: int,
    cube_indices: NDArray,
    cube_boxes: List[BoundingBox],
) -> tuple[NDArray, List[BoundingBox]]:
    num_retained = 0
    num_cubes = len(cube_indices)
    for i in range(num_cubes):
        cube_start = cube_indices[i]
        cube_stop = cube_indices[i + 1] if i + 1 < num_cubes else num_particles
        num_retained += cube_stop > cube_start

    new_indices = np.empty((num_retained,), dtype=np.int_)
    new_boxes = List.empty_list(bbox.bbn_type)
    ind = 0
    for i in range(num_cubes):
        cube_start = cube_indices[i]
        cube_stop = cube_indices[i + 1] if i + 1 < num_cubes else num_particles
        if cube_stop > cube_start:
            new_indices[ind] = cube_start
            new_boxes.append(cube_boxes[i])
            ind += 1

    return new_indices, new_boxes


@njit(cache=True, inline="always")
def _cube_position(x: float, y: float, z: float, cubes_per_side: int, box: BoundingBox):
    # TODO: add zoom bins
    # note: can't use normalize_to_box because it clips the coordinates
    cube_x = np.floor((x - box.box[0]) / box.box[3] * cubes_per_side)
    cube_x = cubes_per_side - 1 if x == box.box[0] + box.box[3] else cube_x
    cube_y = np.floor((y - box.box[1]) / box.box[4] * cubes_per_side)
    cube_y = cubes_per_side - 1 if y == box.box[1] + box.box[4] else cube_y
    cube_z = np.floor((z - box.box[2]) / box.box[5] * cubes_per_side)
    cube_z = cubes_per_side - 1 if z == box.box[2] + box.box[5] else cube_z
    if (
        (cube_x < 0 or cube_x >= cubes_per_side)
        or (cube_y < 0 or cube_y >= cubes_per_side)
        or (cube_z < 0 or cube_z >= cubes_per_side)
    ):
        # with objmode(string=types.unicode_type):
        #     xstr = f"x: cx={cube_x} x={x} bx={box.x} dx={box.dx}"
        #     ystr = f"y: cy={cube_y} y={y} by={box.y} dy={box.dy}"
        #     zstr = f"z: cz={cube_z} z={z} bz={box.z} dz={box.dz}"
        #     string = xstr+"\n"+ystr+"\n"+zstr
        # print("Special cube point:\n"+string)
        return cubes_per_side**3
    return np.int64((cube_x * cubes_per_side + cube_y) * cubes_per_side + cube_z)


@njit(cache=True)
def _pretty(matrix: NDArray):
    assert len(matrix.shape) == 2
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    max_str_width = int(np.max(np.ceil(np.log10(matrix))))
    lines = ""
    for i in range(n_rows):
        line = ""
        for j in range(n_cols):
            nstr = f"{matrix[i, j]}"
            for _ in range(max_str_width - len(nstr)):
                line = line + " "
            line = line + nstr
            if j < nthreads - 1:
                line = line + ", "
        lines = lines + line + "\n"


@njit(parallel=True)
def _cube(data: DataContainer, cubes_per_side: int, box: BoundingBox):
    """
    Bin the loaded particles into the different cubes
    """
    num_cubes = cubes_per_side**3 + 1
    # print(f"Begin cubing into {num_cubes} cubes")

    chopping_block = np.zeros((num_cubes, nthreads), dtype=np.uint64)

    # print("Begin chopping")
    positions = data.positions
    for i in prange(len(positions)):
        x, y, z = positions[i, 0], positions[i, 1], positions[i, 2]
        cube = _cube_position(x, y, z, cubes_per_side, box)
        tid = get_thread_id()
        chopping_block[cube, tid] += 1

    # print("Chopping complete")
    # numba doesn't support cumsum with the axis=1 argument, so do it
    # manually
    chopped = np.zeros_like(chopping_block)
    for i in range(1, nthreads):
        chopped[:, i] += chopped[:, i - 1] + chopping_block[:, i - 1]
    for i in range(1, num_cubes):
        chopped[i, :] += chopped[i - 1, -1] + chopping_block[i - 1, -1]
    # print("Data chopped")
    # print("Statistics:")
    # print(pretty(chopped))

    shuffle_list = np.empty(
        (
            len(
                positions,
            )
        ),
        dtype=np.uint64,
    )
    new_positions = np.empty(
        (len(positions), 3),
        dtype=np.float64,
    )

    thread_offsets = np.zeros_like(chopped)

    # print("Begin dicing")
    for i in prange(len(positions)):
        x, y, z = positions[i, 0], positions[i, 1], positions[i, 2]
        cube = _cube_position(x, y, z, cubes_per_side, box)
        tid = get_thread_id()

        offset = thread_offsets[cube, tid] + chopped[cube, tid]

        thread_offsets[cube, tid] += 1

        # if offset > len(shuffle_list):
        #     print(
        #         f"Offset too large: {offset}! i={i} cube={cube} tid={tid}"
        #     )
        #     print("thread_offsets:")
        #     print(pretty(thread_offsets))

        shuffle_list[offset] = i
        new_positions[offset, 0] = x
        new_positions[offset, 1] = y
        new_positions[offset, 2] = z

    index = data._index
    for i in prange(len(positions)):
        positions[i, 0] = new_positions[i, 0]
        positions[i, 1] = new_positions[i, 1]
        positions[i, 2] = new_positions[i, 2]
        index[i] = shuffle_list[i]

    # print("Dicing complete\nCubing complete")

    return chopped[:, 0]


@njit(parallel=True)
def _make_trees(
    data: DataContainer,
    cube_indices: NDArray,
    cube_boxes: List[BoundingBox],
    particle_threshold: int,
):
    trees = List.empty_list(types.uint32[:])
    num_cubes = len(cube_boxes)
    # pre-allocate in serial
    for _ in range(num_cubes):
        # empty arrays were giving a numba typing error
        trees.append(np.array([0], dtype=np.uint32))

    particle_overflow = False
    for i in prange(num_cubes):
        cube_inds = (
            cube_indices[i],
            cube_indices[i + 1] if i + 1 < num_cubes else len(data),
        )
        # Note: prange indices are uint64 in parallel mode but current
        # TypedList _get_item implementation casts to intp type, which can
        # be int64. We'll explicitly cast to avoid the warning and because
        # len(cubes) **better** be < 2**63 !
        li = np.int_(i)
        box = cube_boxes[li]

        sub_data = subview(data, cube_inds[0], cube_inds[1])

        if i == num_cubes - 1 and len(sub_data) >= 2**32:
            particle_overflow = True

        # print(f"Making tree for cube {i}. inds=({cube_inds[0]}, {cube_inds[1]})")
        tree = _construct_tree(
            data=sub_data,
            box=box,
            particle_threshold=particle_threshold,
        )
        trees[li] = tree

    if particle_overflow:
        with objmode():
            LOGGER.warn(
                "Requested cubes bounding box is too small. Leftovers box has "
                "more than 2**32 particles and likely will be invalid."
            )
    return trees


def _process_cubes_per_side(cubes_per_side: int):
    if cubes_per_side > 0:
        return cubes_per_side
    cubes_per_side = 3
    flag = cubes_per_side**3 + 1 < nthreads
    while cubes_per_side < 32 and flag:
        cubes_per_side += 1
        flag = cubes_per_side**3 + 1 < nthreads
    return cubes_per_side


def make_cubes(
    *,
    dataset: MultiParticleDataset,
    cubes_per_side: int = -1,
    cube_box: bbox.BoxLike | None = None,
    particle_threshold: int | None = None,
    particle_types: str | Collection[str] | None = None,
    save_dataset: bool = True,
    **kwargs,
) -> dict[str, dict[str, NDArray | list[bbox.BoundingBox] | list[PackedTree]]]:
    """
    Create a cubes_dict from the provided dataset

    Args:
        dataset: MultiParticleDataset
        The dataset containing particle data. Will be sorted in-place, but will
        not save updated positional information unless save_dataset is True

        cubes_per_side: int, optional
        Number of cubes on a side. Dataset will be divided into cubes_per_side**3
        cubes, plus an additional cube to catch any remaining particles (if the
        cube_box is smaller than the actual data extants). Note: due to the
        PackedTree's packed format, cubes must contain fewer than ~4 billion
        particles. If cubes_per_side is too small to support this, a ValueError
        will be raised. The limit is per-particle-type.

        cube_box: BoxLike, optional
        A box-like object (i.e. something that can convert to a (6,) ndarray)
        that delineates the region of data to be cubed. Any particles outside
        this region will fall into an overflow cube. Useful for zoom-in
        simulations or other datasets with sparse outer regions. Default is the
        data bounding box

        particle_threshold: int, optional
        Maximum number of particles in a tree leaf node. Default is 400

        particle_types: Collection[str], optional
        Collection of particle types to include. Default is dataset.particle_types

        save_dataset: bool, optional
        Whether to save the sorted dataset positions out to a file. The data
        will be sorted in memory either way. Default True.

    Returns:
        cubes_dict: dict
        A dictionary with 3 components:
            cube_indices - contains the data offsets for each cube's
            particles (i.e. cube 0 is from cubes_indices[0]:cubes_indices[1]

            cube_boxes - containes the bounding box for each cube

            cube_trees - contains the PackedTree for each cube

    Raises:
        ValueError if requested particle types aren't in the dataset
        ValueError if too few cubes were requested for the number of particles

    """
    cubes = {}

    particle_types = (
        [particle_types] if isinstance(particle_types, str) else particle_types
    )
    if particle_types is None:
        requested_types = set(dataset.particle_types)
        particle_numbers = np.array(list(dataset.particle_numbers.values()))
        LOGGER.info(f"Found particle types: {dataset.particle_types}")
    else:
        requested_types = set(particle_types)
        data_types = set(dataset.particle_types)
        if not requested_types <= data_types:
            raise ValueError(
                f"Requested {requested_types - data_types} "
                "but it is not present in the dataset "
                f"({data_types})."
            )
        LOGGER.info(f"Using {requested_types}. Skipping {data_types - requested_types}")
        particle_numbers = np.array(
            [dataset.particle_numbers[i] for i in requested_types]
        )

    cube_box = dataset.bounding_box if cube_box is None else cube_box
    cube_box = bbox.make_bounding_box(cube_box)

    cubes_per_side = _process_cubes_per_side(cubes_per_side)

    num_cubes = cubes_per_side**3 + 1
    if np.any(particle_numbers / num_cubes > 2**31):
        raise ValueError(
            f"Insufficient number of cubes specified. Requested {cubes_per_side}"
            f" cubes per side, leading to >={particle_numbers / num_cubes}"
            f" particles per cube. Max per cube supported is {2**32}"
        )

    particle_threshold = (
        _DEFAULT_PARTICLE_THRESHOLD
        if particle_threshold is None
        else particle_threshold
    )

    for pt in requested_types:
        LOGGER.info(f"Processing {pt}")
        dataset.particle_type = pt
        data = dataset.data_container

        LOGGER.info("Cubing")
        cube_indices = _cube(data, cubes_per_side, cube_box)

        # check cube sizes
        if np.any(np.diff(cube_indices[:-1]) >= 2**32):
            raise ValueError(
                f"Requested number of cubes is insufficient for {pt}. At least"
                " one cube has more than 2**32 particles, the max allowed for"
                " a packed tree."
            )

        LOGGER.info("Getting boxes")
        cube_boxes = _get_cube_boxes(
            data=data, box=cube_box, cubes_per_side=cubes_per_side
        )

        LOGGER.info("Removing empties")
        cube_indices, cube_boxes = _prune_empty(len(data), cube_indices, cube_boxes)

        LOGGER.info("Making trees")
        trees = _make_trees(data, cube_indices, cube_boxes, particle_threshold)

        LOGGER.info("Converting to PackedTrees")
        ptrees = [
            PackedTree(
                source=t,
                particle_threshold=particle_threshold,
                bounding_box=b,
            )
            for t, b in zip(trees, cube_boxes, strict=True)
        ]

        cubes[pt] = {
            "cube_indices": cube_indices,
            "cube_boxes": cube_boxes,
            "cube_trees": ptrees,
        }
        LOGGER.info(f"Done with {pt}")

        if save_dataset:
            dataset.save()

    return cubes


@njit(parallel=True)
def _get_particle_indices_in_shape(
    cubes: List[BoundingBox],
    trees: List[PackedTreeNumba],
    cube_offsets: NDArray,
    shape: bbox.BoundingVolume,
) -> NDArray[np.int_]:
    """
    Get the particle start-stop tuples in the specified shape
    """
    indices = List.empty_list(_list_index_tuple)
    for _ in range(len(cubes)):
        indices.append(List.empty_list(_index_tuple_type))

    # get particle indices from each tree
    for i in prange(len(cubes)):
        # Note: prange indices are uint64 in parallel mode but current
        # TypedList _get_item implementation casts to intp type, which can
        # be int64. We'll explicitly cast to avoid the warning and because
        # len(cubes) **better** be < 2**63 !
        li = np.int_(i)
        overlap = shape.check_box_overlap(cubes[li])
        if overlap:
            indices[li] = trees[li]._get_particle_indices_in_shape(
                containment_obj=shape
            )

    # add cube offset and flatten list of indices
    num_indices = 0
    for i in prange(len(indices)):
        li = np.int_(i)
        num_indices += len(indices[li])
    flattened_indices = np.empty((num_indices, 3), dtype=np.int_)
    current_index = 0
    # doing this in parallel is probably more effort than worth it
    for i in range(len(indices)):
        cube_indices = indices[i]
        cube_offset = cube_offsets[i]
        for cube_start, cube_end, partial in cube_indices:
            # flattened_indices.append(
            #     (cube_start + cube_offset, cube_end + cube_offset, partial)
            # )
            flattened_indices[current_index, 0] = cube_start + cube_offset
            flattened_indices[current_index, 1] = cube_end + cube_offset
            flattened_indices[current_index, 2] = partial
            current_index += 1

    return flattened_indices


@njit(parallel=True)
def _parallel_expand_all_data_indices(
    slices: NDArray[np.int_],
    shape: bbox.BoundingVolume,
):
    num_particles = 0
    offsets = np.empty((slices.shape[0],), dtype=np.int_)
    for i in range(slices.shape[0]):
        # can't parallelize this because offsets should be the cumsum
        offsets[i] = num_particles
        num_particles += slices[i, 1] - slices[i, 0]

    indices = np.empty((num_particles,), dtype=np.int64)
    #  ignore information about partial/full, just return indices as
    # fast as possible
    for i in prange(slices.shape[0]):
        start = slices[i, 0]
        end = slices[i, 1]
        offset = offsets[i]
        for j, index in enumerate(range(start, end)):
            indices[offset + j] = index
    return indices


@njit(parallel=True)
def _parallel_expand_all_shuffle_indices(
    slices: NDArray[np.int_], shape: bbox.BoundingVolume, data: DataContainer
):
    num_particles = 0
    offsets = np.empty((slices.shape[0],), dtype=np.int_)
    for i in range(slices.shape[0]):
        # can't parallelize this because offsets should be the cumsum
        offsets[i] = num_particles
        num_particles += slices[i, 1] - slices[i, 0]

    indices = np.empty((num_particles,), dtype=np.int64)
    #  ignore information about partial/full, just return indices as
    # fast as possible
    for i in prange(slices.shape[0]):
        start = slices[i, 0]
        end = slices[i, 1]
        offset = offsets[i]
        size = start - end
        shuffle = data._index[start:end]
        for j in range(size):
            indices[offset + j] = shuffle[j]
    return indices


@njit(parallel=True)
def _parallel_expand_data_indices(
    slices: NDArray[np.int_],
    shape: bbox.BoundingVolume,
    data: DataContainer,
) -> NDArray[np.int_]:
    num_particles = 0
    offsets = np.empty((slices.shape[0],), dtype=np.int_)
    for i in range(slices.shape[0]):
        # can't parallelize this because offsets should be the cumsum
        offsets[i] = num_particles
        num_particles += slices[i, 1] - slices[i, 0]

    indices = np.empty((num_particles,), dtype=np.int64)

    num_contained = 0
    for i in prange(slices.shape[0]):
        start = slices[i, 0]
        end = slices[i, 1]
        partial = slices[i, 2]
        offset = offsets[i]
        size = end - start
        if not partial:
            # fully enclosed
            for j, index in enumerate(range(start, end)):
                indices[offset + j] = index
            num_contained += size
            continue
        positions = data._positions[start:end, 0:3]
        j = 0
        ind = offset
        for x, y, z in positions:
            if shape.contains_point(x, y, z):
                indices[ind] = j + start
                ind += 1
            j += 1  # noqa: SIM113
        num_contained += ind - offset
        end_bound = offset + size
        while ind < end_bound:
            indices[ind] = -1
            ind += 1

    # not parallelizable since we're shrinking the array
    out_indices = np.empty((num_contained,), dtype=np.int_)
    ind = 0
    for i in range(len(indices)):
        index = indices[i]
        if index >= 0:
            out_indices[ind] = index
            ind += 1

    return out_indices


@njit(parallel=True)
def _parallel_expand_shuffle_indices(
    slices: NDArray[np.int_],
    shape: bbox.BoundingVolume,
    data: DataContainer,
) -> NDArray[np.int_]:
    num_particles = 0
    offsets = np.empty((slices.shape[0],), dtype=np.int_)
    for i in range(slices.shape[0]):
        # can't parallelize this because offsets should be the cumsum
        offsets[i] = num_particles
        num_particles += slices[i, 1] - slices[i, 0]

    indices = np.empty((num_particles,), dtype=np.int64)

    num_contained = 0
    for i in prange(slices.shape[0]):
        start = slices[i, 0]
        end = slices[i, 1]
        partial = slices[i, 2]
        offset = offsets[i]
        size = end - start
        shuffle = data._index[start:end]
        if not partial:
            # fully enclosed
            for j in range(size):
                indices[offset + j] = shuffle[j]
            num_contained += size
            continue
        positions = data._positions[start:end, 0:3]
        j = 0
        ind = offset
        for x, y, z in positions:
            if shape.contains_point(x, y, z):
                indices[ind] = shuffle[j]
                ind += 1
            j += 1  # noqa: SIM113
        num_contained += ind - offset
        while ind < offset + size:
            indices[ind] = -1
            ind += 1

    # not parallelizable since we're shrinking the array
    out_indices = np.empty((num_contained,), dtype=np.int_)
    ind = 0
    for i in range(len(indices)):
        index = indices[i]
        if index >= 0:
            out_indices[ind] = index
            ind += 1

    return out_indices


@njit
def _get_particle_index_list_in_shape(
    cubes: List[BoundingBox],
    trees: List[PackedTreeNumba],
    cube_offsets: NDArray,
    shape: bbox.BoundingVolume,
    data: DataContainer | None,
    use_data_indices: bool,  # noqa: FBT001, FBT002
) -> NDArray[np.int_]:
    """
    Get the array of particle indices in the specified shape
    """
    slices = _get_particle_indices_in_shape(cubes, trees, cube_offsets, shape)

    if use_data_indices:
        if data is None:
            return _parallel_expand_all_data_indices(slices, shape)
        return _parallel_expand_data_indices(slices, shape, data)
    if data is None:
        return _parallel_expand_all_shuffle_indices(slices, shape, data)
    return _parallel_expand_shuffle_indices(slices, shape, data)


@njit(parallel=False)
def _get_closest_cube(
    cubes: List[BoundingBox],
    xyz: NDArray,
    distance_function: Callable[[float, float, float, float, float, float], float],
) -> np.int_:
    """
    Return the index of the closest cube to a point
    """
    x, y, z = xyz
    cube_dists = np.empty((len(cubes),), dtype=np.float64)
    for i in prange(len(cubes)):
        li = np.int64(i)
        px, py, pz = cubes[li].project_point_on_box(xyz)
        cube_dists[i] = distance_function(x, y, z, px, py, pz)

    return np.argmin(cube_dists)


@njit
def _get_closest_particles(
    cubes: List[BoundingBox],
    trees: List[PackedTreeNumba],
    cube_indices: NDArray,
    data: DataContainer,
    xyz: NDArray,
    k: int,
    distance_function: Callable[[float, float, float, float, float, float], float],
    distance_upper_bound: float,
    use_shuffle: bool,  # noqa: FBT001, FBT002
    return_sorted: bool,  # noqa: FBT001, FBT002
) -> tuple[NDArray, NDArray]:
    """
    Return the k-closest particle distances and their indices

    Args:
        cubes: List[BoundingBox]
        The cube boxes

        trees: List[PackedTreeNumba]
        The cube trees

        cube_indices: NDArray
        The cube index offsets

        data: DataContainer
        The container of the position data

        xyz: NDArray
        The 3 Cartesian coordinates

        k: positive int
        The number of particles to return. No verification of sign is performed

        distance_function: Callable[[float, float, float, float, float, float], float]
        The distance function between two Cartesian points,
        e.g. d = distance_function(x1, y1, z1, x2, y2, z2)

        distance_upper_bound: float
        The maximum distance to consider particles within. May result in fewer
        than k particles being returned if too stringent

        use_shuffle: bool
        Flag to return shuffle indices instead of sorted data indices

    Returns:
        dists: NDArray[float]
        K-length vector of distances

        inds: NDArray[int]
        K-length vector of particle indices
    """
    x, y, z = xyz

    num_cubes = len(cubes)
    containing_cube = _get_closest_cube(cubes, xyz, distance_function)

    cube_start = cube_indices[containing_cube]
    cube_end = (
        cube_indices[containing_cube + 1]
        if containing_cube + 1 < num_cubes
        else len(data)
    )

    # need to pass sub_data because otherwise tree indices will be wrong
    sub_data = subview(data, cube_start, cube_end)
    # we don't need the heap to be sorted because it's still being processed
    dists, inds = trees[containing_cube].get_closest_particles(
        sub_data,
        xyz,
        distance_function,
        distance_upper_bound,
        k,
        use_shuffle,  # noqa: FBT003
        False,  # noqa: FBT003 # return_sorted
    )

    # data inds don't include cube offsets
    if not use_shuffle:
        for i in range(len(inds)):
            inds[i] += cube_start

    # dists[0] is max distance due to heap invariant, *as long as we're not
    # returning the sorted version!*
    if dists[0] == 0:
        # we've found k particles and we're done
        return dists, inds

    # heap can be recreated easily from dists, inds
    heap = FixedDistanceHeap(k, -1)
    heap.distances = dists
    heap.indices = inds
    heap.max_distance = heap.distances[0]

    # separate in case we allow making FDHs from arrays
    max_dist = heap.max_distance
    search_box = BoundingBox(
        np.array(
            [
                x - max_dist,
                y - max_dist,
                z - max_dist,
                2 * max_dist,
                2 * max_dist,
                2 * max_dist,
            ]
        )
    )

    slices = _get_particle_indices_in_shape(cubes, trees, cube_indices, search_box)

    for i in range(slices.shape[0]):
        s, e = slices[i, 0], slices[i, 1]
        # skip slice if it's a subslice of the node we already looked at
        if cube_start <= s < cube_end:
            continue
        _process_slice_against_heap(
            heap, data, xyz, distance_function, s, e, use_shuffle
        )

    distances, indices = heap.distances, heap.indices
    if return_sorted:
        distances, indices = heap.sorted()

    return distances, indices


class ParticleCubes:
    """
    The cubes for a single particle type
    """

    cube_indices: NDArray
    """ Array if cube indices into the dataset """
    cube_boxes: List[BoundingBox]
    """ The bounding boxes for each cube """
    cube_trees: list[PackedTree]
    """ The packed trees for each cube """
    _numba_trees: List[PackedTreeNumba]
    """ The PackedTreeNumba for each cube """

    def __init__(
        self,
        *,
        cube_indices: NDArray,
        cube_boxes: List[BoundingBox],
        cube_trees: list[NDArray] | list[PackedTree] | list[NDArray | PackedTree],
        **kwargs,
    ):
        particle_threshold = getattr(
            kwargs, "particle_threshold", _DEFAULT_PARTICLE_THRESHOLD
        )
        self.cube_indices = cube_indices
        self.cube_boxes = cube_boxes

        self.cube_trees = []
        for t, b in zip(cube_trees, cube_boxes, strict=True):
            if isinstance(t, PackedTree):
                self.cube_trees.append(t)
                continue
            self.cube_trees.append(
                PackedTree(
                    bounding_box=b,
                    source=np.array(t),
                    particle_threshold=particle_threshold,
                )
            )

        self._numba_trees = List([t._tree for t in self.cube_trees])

    def _get_particle_indices_in_shape(
        self,
        shape: bbox.BoundingVolume,
    ) -> NDArray[np.int_]:
        """
        Return all particles contained within the shape

        This is a private version that uses a premade bounding volume

        Args:
            shape: BoundingVolume
            The shape to search in

        Returns:
            indices: NDArray[int]
            Array of particle start-stop indices contained within shape
        """
        return _get_particle_indices_in_shape(
            cubes=self.cube_boxes,
            trees=self._numba_trees,
            cube_offsets=self.cube_indices,
            shape=shape,
        )

    def get_particle_indices_in_box(
        self,
        box: bbox.BoxLike,
    ) -> NDArray[np.int_]:
        """
        Return all particles contained within the box

        Args:
            box: BoxLike
            Box to check

        Returns:
            indices: NDArray[int]
            Array of particle start-stop indices contained within box
        """
        numba_box = bbox.make_bounding_box(box)

        return self._get_particle_indices_in_shape(numba_box)

    def get_particle_indices_in_sphere(
        self,
        center: NDArray,
        radius: float,
    ) -> NDArray[np.int_]:
        """
        Return all particles contained within the sphere defined by center and radius

        Args:
            center: NDArray
            Center point of the sphere

            radius: float
            Radius of the sphere

        Returns:
            indices: NDArray[int]
            Array of particle start-stop indices contained within sphere
        """
        sph = bbox.make_bounding_sphere(center=center, radius=radius, unsafe=True)

        return self._get_particle_indices_in_shape(sph)

    def _get_particle_index_list_in_shape(
        self,
        data: DataContainer | Dataset,
        shape: bbox.BoundingVolume,
        *,
        use_data_indices: bool = True,
        strict: bool = False,
    ) -> NDArray[np.int_]:
        """
        Return all particle indices contained within the shape

        This is a private version that uses a premade bounding volume

        Args:
            data: DataContainer | Dataset
            Dataset containing the particle positions. Pass a DataContainer
            object for a slight performance increase

            box: BoundingBox
            The bounding box of the shape

            use_data_indices: bool, optional
            Flag to return indices into the sorted dataset (True, default) or
            into the shuffle list (False)

            strict: bool, optional
            Flag to specify whether only particles inside the shape will
            be returned. If False (default), additional nearby particles may be
            included for signficantly increased performance

        Returns:
            indices: Array[int]
            Array of particle indices contained within shape
        """
        return _get_particle_index_list_in_shape(
            cubes=self.cube_boxes,
            trees=self._numba_trees,
            cube_offsets=self.cube_indices,
            shape=shape,
            data=data.data_container if isinstance(data, Dataset) else data,
            use_data_indices=use_data_indices,
        )

    def get_particle_index_list_in_box(
        self,
        data: DataContainer | Dataset,
        box: bbox.BoxLike,
        *,
        use_data_indices: bool = True,
        strict: bool = False,
    ) -> NDArray[np.int_]:
        """
        Return all particle indices contained within the box

        Args:
            data: DataContainer | Dataset
            Dataset containing the particle positions. Pass a DataContainer
            object for a slight performance increase

            box: BoxLike
            The box to search in

            use_data_indices: bool, optional
            Flag to return indices into the sorted dataset (True, default) or
            into the shuffle list (False)

            strict: bool, optional
            Flag to specify whether only particles inside the shape will
            be returned. If False (default), additional nearby particles may be
            included for signficantly increased performance

        Returns:
            indices: Array[int]
            Array of particle indices contained within shape
        """
        numba_box = bbox.make_bounding_box(box)
        return self._get_particle_index_list_in_shape(
            data=data,
            shape=numba_box,
            use_data_indices=use_data_indices,
            strict=strict,
        )

    def get_particle_index_list_in_sphere(
        self,
        data: DataContainer | Dataset,
        center: NDArray,
        radius: float,
        *,
        use_data_indices: bool = True,
        strict: bool = False,
    ) -> NDArray[np.int_]:
        """
        Return all particle indices contained within the sphere

        Args:
            data: DataContainer | Dataset
            Dataset containing the particle positions. Pass a DataContainer
            object for a slight performance increase

            center: NDArray
            Center point of the sphere

            radius: float
            Radius of the sphere

            use_data_indices: bool, optional
            Flag to return indices into the sorted dataset (True, default) or
            into the shuffle list (False)

            strict: bool, optional
            Flag to specify whether only particles inside the shape will
            be returned. If False (default), additional nearby particles may be
            included for signficantly increased performance

        Returns:
            indices: NDArray[int]
            Array of particle indices contained within the sphere
        """
        sph = bbox.make_bounding_sphere(radius, center=center, unsafe=True)
        return self._get_particle_index_list_in_shape(
            data=data,
            shape=sph,
            use_data_indices=use_data_indices,
            strict=strict,
        )

    def get_closest_particles(
        self,
        *,
        data: DataContainer | Dataset,
        xyz: NDArray,
        distance_upper_bound: float | None = None,
        p: float | None = None,
        k: int | None = None,
        return_shuffle_indices: bool | None = None,
        return_sorted: bool | None = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Get kth nearest particle distances and indices to point

        Args:
            data: DataContainer | Dataset
            Source of particle position data

            xyz: ArrayLike
            Coordinates of point to check

            distance_upper_bound: nonnegative float, optional
            Return only neighbors from other nodes within this distance. This
            is used for tree pruning, so if you are doing a series of
            nearest-neighbor queries, it may help to supply the distance to the
            nearest neighbor of the most recent point.

            p: float, optional
            Which Minkowski p-norm to use. 1 is the sum of absolute-values
            distance ("Manhattan" distance). 2 is the usual Euclidean distance.
            Infinity is the maximum-coordinate-difference distance. Currently,
            only p=2 is supported.

            k: int, optional
            Number of closest particles to return. Default 1

            return_shuffle_indices: bool, optional
            Flag to return the shuffle indices instead of the data indices.
            Default False.

            return_sorted: bool, optional
            Flag to return the distances and indices in distance-sorted order.
            Set to False for a performance boost. Default True

        Returns:
            distances: NDArray[float]
            Distances to the kth nearest neighbors. Has shape (min(N,k),),
            where N is the number of particles in the sphere bounded by
            distance_upper_bound

            indices: NDArray[int]
            Indices in data of the kth nearest neighbors. Has same shape as
            distances

        Raises:
            NotImplementedError
            If a p value of then 2 is provided
        """

        p = 2 if p is None else p
        if p != 2:
            raise NotImplementedError("Only p=2 is currently supported")
        distance_function = euclidean_distance_particle

        distance_upper_bound = (
            1e100 if distance_upper_bound is None else distance_upper_bound
        )

        k = 1 if k is None else k
        return_shuffle_indices = (
            False if return_shuffle_indices is None else return_shuffle_indices
        )
        return_sorted = True if return_sorted is None else return_sorted

        return _get_closest_particles(
            self.cube_boxes,
            self._numba_trees,
            self.cube_indices,
            data if isinstance(data, DataContainer) else data.data_container,
            xyz,
            k,
            distance_function,
            distance_upper_bound,
            return_shuffle_indices,
            return_sorted,
        )

    def _get_pilis_cubes(
        self,
        *,
        data: DataContainer | Dataset,
        odata: DataContainer | Dataset,
        ocubes: ParticleCubes,
        r: float,
        p: float = 2.0,
        k: int = 1,
        strict: bool = True,
    ):
        raise NotImplementedError(
            """
            Still in progress. Try a PackedTree for this functionatlity
            """
        )

    def save(
        self,
        dataset: str | Path | HDF5Dataset,
        *,
        force_overwrite: bool = False,
    ) -> Path:
        dataset = _check_overwrite(dataset, force_overwrite=force_overwrite)
        save_cube(
            dataset,
            pt="PartType0",
            cube_indices=self.cube_indices,
            cube_boxes=self.cube_boxes,
            cube_trees=self.cube_trees,
        )
        return dataset.filepath if isinstance(dataset, Dataset) else Path(dataset)


def has_cubes(dataset: str | Path | MultiParticleDataset):
    """Return true if the dataset contains a packingcubes structure"""
    # TODO: This whole function probably needs to be refactored somewhere else
    if dataset is None:
        raise ValueError("Need a dataset to check!")
    if isinstance(dataset, HDF5Dataset):
        return "cubes" in dataset._top_level_groups
    if isinstance(dataset, (str, Path)):
        with h5py.File(dataset) as file:
            return "cubes" in file
    return False


def _check_overwrite(
    dataset: str | Path | HDF5Dataset, *, force_overwrite: bool = False
) -> str | Path | HDF5Dataset:
    if not has_cubes(dataset):
        return dataset
    if force_overwrite:
        LOGGER.warning(
            f"Dataset {dataset} already contains cubes structure. Overwriting!"
        )
    else:
        old_filepath = (
            dataset.filepath if isinstance(dataset, HDF5Dataset) else Path(dataset)
        )
        new_filepath = (
            old_filepath.parent / f"{old_filepath.stem}_cubes{old_filepath.suffix}"
        )
        LOGGER.info(
            f"Dataset {old_filepath} already contains cubes structure."
            f"Saving to {new_filepath} instead."
        )
        dataset = new_filepath
    return dataset


def save_cube(
    dataset: str | Path | HDF5Dataset,
    pt: str,
    cube_indices: NDArray,
    cube_boxes: list[bbox.BoundingBox],
    cube_trees: list[PackedTree],
):
    """
    Save an individual cube's data to the dataset
    """
    filepath = dataset.filepath if isinstance(dataset, HDF5Dataset) else dataset
    with h5py.File(filepath, "a") as file:
        cubes = file.create_group(f"cubes/{pt}")
        cubes["indices"] = cube_indices
        cubes["number"] = len(cube_indices)
        for i, (box, tree) in enumerate(zip(cube_boxes, cube_trees, strict=True)):
            cubes[f"box_{i}"] = box.box
            cubes[f"tree_{i}"] = tree.packed_form


def load_cubes(
    dataset: str | MultiParticleDataset,
) -> dict[str, dict]:
    """
    Load cubes data from a dataset. See make_cubes for a description of the output
    """
    cubes_dict = {}
    if isinstance(dataset, Dataset) and not isinstance(dataset, HDF5Dataset):
        raise NotImplementedError("We can only load Cubes from HDF5 datasets")
    if not has_cubes(dataset):
        raise ValueError("No cubes in provided dataset")
    filepath = dataset.filepath if isinstance(dataset, HDF5Dataset) else dataset
    with h5py.File(
        filepath,
    ) as file:
        cubes_group = file["cubes"]
        pts = list(cubes_group.keys())
        for pt in pts:
            cubes = cubes_group[pt]
            cube_indices = cubes["indices"]
            number = cubes["number"]
            cube_boxes = []
            cube_trees = []
            for i in range(number):
                cube_boxes.append(bbox.make_bounding_box(cubes[f"box_{i}"]))
                cube_trees.append(PackedTree(source=cubes[f"tree_{i}"]))
            cubes_dict[pt] = {
                "cube_indices": cube_indices,
                "cube_boxes": cube_boxes,
                "cube_trees": cube_trees,
            }
    return cubes_dict


class CubesError(Exception):
    pass


def _has_trees(cubes_dict: dict[str, dict]) -> bool:
    """
    Check if cubes_dict has trees in it
    """
    return all("cube_trees" in cubes for _, cubes in cubes_dict.items())


def _add_trees_to_cubes_dict(
    *, cubes_dict: dict[str, dict], dataset: MultiParticleDataset, **kwargs
):
    """
    Generate missing PackedTrees from dataset on per-particle-type basis
    """
    particle_threshold = getattr(
        kwargs, "particle_threshold", _DEFAULT_PARTICLE_THRESHOLD
    )
    for pt, cubes in cubes_dict.items():
        dataset.particle_type = pt
        cube_indices = cast(NDArray, cubes["cube_indices"])
        cube_boxes = cast(list[bbox.BoundingBox], cubes["cube_boxes"])
        cube_trees = _make_trees(
            data=dataset.data_container,
            cube_indices=cube_indices,
            cube_boxes=cube_boxes,
            particle_threshold=particle_threshold,
        )


class MultiCubes:
    cubes_dict: dict[str, ParticleCubes]
    """ Mapping from particle type to ParticleCubes for this dataset """

    def __init__(
        self,
        *,
        cubes_dict: dict[str, dict],
        **kwargs,
    ):
        self.cubes_dict = {}
        for pt, cubes in cubes_dict.items():
            cube_indices = cast(NDArray, cubes["cube_indices"])
            cube_boxes = cast(list[bbox.BoundingBox], cubes["cube_boxes"])
            cube_trees = cast(
                list[PackedTree] | list[NDArray] | list[PackedTree | NDArray],
                cubes["cube_trees"],
            )
            self.cubes_dict[pt] = ParticleCubes(
                cube_indices=cube_indices,
                cube_boxes=cube_boxes,
                cube_trees=cube_trees,
                **kwargs,
            )

    @property
    def particle_types(self):
        return self.cubes_dict.keys()

    def get_single_cubes(self, particle_type: str) -> ParticleCubes:
        """
        Return the ParticleCubes instance corresponding to the specified type.
        """
        return self.cubes_dict[particle_type]

    def _get_particle_indices_in_shape(
        self,
        shape: bbox.BoundingVolume,
        *,
        particle_types: str | Collection[str] | None = None,
    ) -> dict[str, NDArray[np.int_]]:
        """
        Return all particles contained within the shape

        Args:
            particle_types: str | Collection[str]
            Particle type(s) to include

            shape: BoundingVolume
            The shape to check

            particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to self.particle_types
        Returns:
            indices: dict[str, NDArray[int]]
            Dictionary of arrays of particle start-stop indices plus partiality
            flag contained within shape, organized by particle type
        """
        if particle_types is None:
            particle_types = self.particle_types
        if isinstance(particle_types, str):
            particle_types = [particle_types]
        inds = {}
        for pt in particle_types:
            inds[pt] = self.cubes_dict[pt]._get_particle_indices_in_shape(
                shape,
            )
        return inds

    def get_particle_indices_in_box(
        self,
        box: bbox.BoxLike,
        *,
        particle_types: str | Collection[str] | None = None,
    ) -> dict[str, NDArray[np.int_]]:
        """
        Return all particles contained within the box

        Args:
            box: BoxLike
            Box to check

            particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to self.particle_types
        Returns:
            indices: dict[str, NDArray[int]]
            Dictionary of arrays of particle start-stop indices plus partiality
            flag contained within box, organized by particle type
        """
        numba_box = bbox.make_bounding_box(box)
        return self._get_particle_indices_in_shape(
            numba_box, particle_types=particle_types
        )

    def get_particle_indices_in_sphere(
        self,
        center: NDArray,
        radius: float,
        *,
        particle_types: str | Collection[str] | None = None,
    ) -> dict[str, NDArray[np.int_]]:
        """
        Return all particles contained within the sphere defined by center and radius

        Args:
            particle_types: str | Collection[str]
            Particle type(s) to include

            center: NDArray
            Center point of the sphere

            radius: float
            Radius of the sphere

            particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to self.particle_types
        Returns:
            indices: dict[str, NDArray[int]]
            Dictionary of arrays of particle start-stop indices plus partiality
            flag contained within sphere, organized by particle type
        """
        sph = bbox.make_bounding_sphere(radius, center=center, unsafe=True)
        return self._get_particle_indices_in_shape(sph, particle_types=particle_types)

    def _get_particle_index_list_in_shape(
        self,
        *,
        data: DataContainer | Dataset,
        shape: bbox.BoundingVolume,
        particle_types: str | Collection[str] | None = None,
        strict: bool = True,
        use_data_indices: bool = True,
    ) -> dict[str, NDArray[np.int_]]:
        """
        Return all particles contained within the sphere defined by center and radius

        Args:
            data: DataContainer | Dataset
            Dataset containing the particle positions. Pass a DataContainer
            object for a slight performance increase

            center: NDArray
            Center point of the sphere

            radius: float
            Radius of the sphere

            particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to self.particle_types

            strict: bool, optional
            Flag to specify whether only particles inside the sphere will
            be returned. If False (default), additional nearby particles may be
            included for signficantly increased performance

            use_data_indices: bool, optional
            Flag to return indices into the sorted dataset (True, default) or
            into the shuffle list (False)

        Returns:
            indices: NDArray[int]
            List of original particle indices contained within sphere
        """
        if particle_types is None:
            particle_types = self.particle_types
        if isinstance(particle_types, str):
            particle_types = [particle_types]
        data = data.data_container if isinstance(data, Dataset) else data
        inds = {}
        for pt in particle_types:
            inds[pt] = self.cubes_dict[pt]._get_particle_index_list_in_shape(
                data,
                shape,
                use_data_indices=use_data_indices,
            )
        return inds

    def get_particle_index_list_in_box(
        self,
        *,
        data: DataContainer | Dataset,
        box: bbox.BoxLike,
        particle_types: str | Collection[str] | None = None,
        strict: bool = True,
        use_data_indices: bool = True,
    ) -> dict[str, NDArray[np.int_]]:
        """
        Return all particles contained within the sphere defined by center and radius

        Args:
            data: DataContainer | Dataset
            Dataset containing the particle positions. Pass a DataContainer
            object for a slight performance increase

            box: BoxLike
            Box to check

            particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to self.particle_types

            strict: bool, optional
            Flag to specify whether only particles inside the sphere will
            be returned. If False (default), additional nearby particles may be
            included for signficantly increased performance

            use_data_indices: bool, optional
            Flag to return indices into the sorted dataset (True, default) or
            into the shuffle list (False)

        Returns:
            indices: NDArray[int]
            List of original particle indices contained within sphere
        """
        bbn = bbox.make_bounding_box(box)
        return self._get_particle_index_list_in_shape(
            data=data,
            shape=bbn,
            particle_types=particle_types,
            strict=strict,
            use_data_indices=use_data_indices,
        )

    def get_particle_index_list_in_sphere(
        self,
        *,
        data: DataContainer | Dataset,
        center: NDArray,
        radius: float,
        particle_types: str | Collection[str] | None = None,
        strict: bool = True,
        use_data_indices: bool = True,
    ) -> dict[str, NDArray[np.int_]]:
        """
        Return all particles contained within the sphere defined by center and radius

        Args:
            data: DataContainer | Dataset
            Dataset containing the particle positions. Pass a DataContainer
            object for a slight performance increase

            center: NDArray
            Center point of the sphere

            radius: float
            Radius of the sphere

            particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to self.particle_types

            strict: bool, optional
            Flag to specify whether only particles inside the sphere will
            be returned. If False (default), additional nearby particles may be
            included for signficantly increased performance

            use_data_indices: bool, optional
            Flag to return indices into the sorted dataset (True, default) or
            into the shuffle list (False)

        Returns:
            indices: NDArray[int]
            List of original particle indices contained within sphere
        """
        sph = bbox.make_bounding_sphere(radius, center=center, unsafe=True)
        return self._get_particle_index_list_in_shape(
            data=data,
            shape=sph,
            particle_types=particle_types,
            strict=strict,
            use_data_indices=use_data_indices,
        )

    def get_closest_particles(
        self,
        *,
        data: DataContainer | Dataset,
        xyz: NDArray,
        particle_types: str | Collection[str] | None = None,
        distance_upper_bound: float | None = None,
        p: float | None = None,
        k: int | None = None,
        return_shuffle_indices: bool | None = None,
        return_sorted: bool | None = None,
    ):
        """
        Get kth nearest particle distances and indices to point

        Args:
            data: DataContainer | Dataset
            Source of particle position data

            xyz: ArrayLike
            Coordinates of point to check

            particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to self.particle_types

            distance_upper_bound: nonnegative float, optional
            Return only neighbors from other nodes within this distance. This
            is used for tree pruning, so if you are doing a series of
            nearest-neighbor queries, it may help to supply the distance to the
            nearest neighbor of the most recent point.

            p: float, optional
            Which Minkowski p-norm to use. 1 is the sum of absolute-values
            distance ("Manhattan" distance). 2 is the usual Euclidean distance.
            Infinity is the maximum-coordinate-difference distance. Currently,
            only p=2 is supported.

            k: int, optional
            Number of closest particles to return. Default 1

            return_shuffle_indices: bool, optional
            Flag to return the shuffle indices instead of the data indices.
            Default False.

            return_sorted: bool, optional
            Flag to return the distances and indices in distance-sorted order.
            Set to False for a performance boost. Default True

        Returns:
            distances: NDArray[float]
            Distances to the kth nearest neighbors. Has shape (min(N,k),),
            where N is the number of particles in the sphere bounded by
            distance_upper_bound

            indices: NDArray[int]
            Indices in data of the kth nearest neighbors. Has same shape as
            distances

        Raises:
            NotImplementedError
            If a p value of then 2 is provided
        """
        if particle_types is None:
            particle_types = self.particle_types
        if isinstance(particle_types, str):
            particle_types = [particle_types]
        data = data.data_container if isinstance(data, Dataset) else data
        inds = {}
        for pt in particle_types:
            inds[pt] = self.cubes_dict[pt].get_closest_particles(
                data=data,
                xyz=xyz,
                distance_upper_bound=distance_upper_bound,
                p=p,
                k=k,
                return_shuffle_indices=return_shuffle_indices,
                return_sorted=return_sorted,
            )
        return inds

    def save(
        self,
        dataset: str | Path | HDF5Dataset,
        *,
        force_overwrite: bool = False,
    ) -> Path:
        """
        Save cubes information to specified file

        Args:
            dataset: str | HDF5Dataset
            Location to store cubes data.

            force_overwrite: bool, optional
            If dataset already contains cubes data, overwrite if True.
            Default False
        """
        dataset = _check_overwrite(dataset, force_overwrite=force_overwrite)

        for pt, cubes in self.cubes_dict.items():
            save_cube(
                dataset,
                pt=pt,
                cube_indices=cubes.cube_indices,
                cube_boxes=cubes.cube_boxes,
                cube_trees=cubes.cube_trees,
            )
        return dataset.filepath if isinstance(dataset, Dataset) else Path(dataset)


def Cubes(
    *,
    dataset: str | NDArray | MultiParticleDataset | None = None,
    cubes_dict: dict[str, dict] | None = None,
    **kwargs,
) -> ParticleCubes | MultiCubes:
    if cubes_dict is None and dataset is None:
        raise CubesError("Must provide either a cubes_dict or dataset!")
    dataset = (
        InMemory(positions=dataset) if isinstance(dataset, np.ndarray) else dataset
    )
    # we only want to load the dataset if we need to
    if cubes_dict is None:
        assert dataset is not None
        try:
            cubes_dict = load_cubes(dataset)
        except (NotImplementedError, ValueError):
            dataset = (
                GadgetishHDF5Dataset(filepath=dataset)
                if isinstance(dataset, str)
                else dataset
            )
            cubes_dict = make_cubes(dataset=dataset, **kwargs)
    else:
        if not _has_trees(cubes_dict):
            if dataset is None:
                raise CubesError("cubes_dict has no trees and dataset not provided")
            dataset = (
                GadgetishHDF5Dataset(filepath=dataset)
                if isinstance(dataset, str)
                else dataset
            )
            _add_trees_to_cubes_dict(cubes_dict=cubes_dict, dataset=dataset, **kwargs)
    if len(cubes_dict) == 1:
        cubes = next(iter(cubes_dict.values()))
        return ParticleCubes(**cubes, **kwargs)
    return MultiCubes(cubes_dict=cubes_dict, **kwargs)


def make_ParticleCubes(**kwargs) -> ParticleCubes:
    """
    Wrapper for Cubes that explicitly returns ParticleCubes or raises an error
    """
    cubes = Cubes(**kwargs)
    if not isinstance(cubes, ParticleCubes):
        raise CubesError(
            f"""
            Multiple particle types present. Please specify one of 
            {cubes.particle_types} as particle_types=PARTICLE_TYPE.
            """
        )
    return cubes


if __name__ == "__main__":
    logging.basicConfig()
    args = _process_args()
    if args.output and has_cubes(args.output) and not args.force_overwrite:
        sys.exit(
            "Provided output file already contains cubes data and"
            " you did not specify --force-overwrite"
        )
    dataset = GadgetishHDF5Dataset(filepath=args.snapshot, sorted_filepath=args.output)
    box = _process_box(dataset=dataset, args=args)
    cubes_dict = make_cubes(
        dataset=dataset,
        cubes_per_side=args.n,
        cube_box=box,
        particle_threshold=args.particle_threshold,
        particle_types=args.particle_types,
        save_dataset=not args.no_save_dataset,
    )
    LOGGER.info(cubes_dict.keys())
    cubes = Cubes(dataset=dataset, cubes_dict=cubes_dict)
    if args.output:
        cubes.save(args.output)
