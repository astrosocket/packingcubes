from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from collections.abc import Collection
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
    MultiParticleDataset,
    subview,
)
from packingcubes.octree import _DEFAULT_PARTICLE_THRESHOLD
from packingcubes.packed_tree import (
    PackedTree,
)
from packingcubes.packed_tree.packed_tree_numba import (
    PackedTreeNumba,
    _construct_tree,
    _index_tuple_type,
    _list_index_tuple,
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
        default=3,
        help="number of cells per side [3-32], default 3",
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
        help="Name of hdf5 file to save cubes information to",
        type=str,
    )
    parser.add_argument(
        "--force-overwrite",
        help="Flag to overwrite cubes data contained in OUTPUT",
        action="store_true",
    )

    args = parser.parse_args(argv)
    if args.n < 3 or 32 < args.n:
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
                cube = make_bounding_box(np.hstack((cube_pos, cube_size)))
                cube_boxes.append(cube)
    cube_boxes.append(data.bounding_box)  # don't forget leftovers cube!
    return cube_boxes


@njit(cache=True)
def _cube_position(x: float, y: float, z: float, cubes_per_side: int, box: BoundingBox):
    # TODO: add zoom bins
    num_cubes = cubes_per_side**3 + 1
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
        return num_cubes - 1
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

    shuffle_list = np.zeros(
        (
            len(
                positions,
            )
        ),
        dtype=np.uint64,
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

    # print("Dicing complete\nCubing complete")

    return shuffle_list, chopped[:, 0]


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

    for i in prange(num_cubes):
        cube_inds = (
            cube_indices[i],
            cube_indices[i + 1] - 1 if i + 1 < num_cubes else len(data) - 1,
        )
        box = cube_boxes[i]

        sub_data = subview(data, cube_inds[0], cube_inds[1])

        if i == num_cubes - 1 and len(sub_data) >= 2**32:
            with objmode():
                LOGGER.warn(
                    "Requested cubes bounding box is too small. Leftovers box has "
                    "more than 2**32 particles and likely will be invalid."
                )

        # print(f"Making tree for cube {i}. inds=({cube_inds[0]}, {cube_inds[1]})")
        tree = _construct_tree(
            data=sub_data,
            particle_threshold=particle_threshold,
        )
        trees[i] = tree
    return trees


def make_cubes(
    *,
    dataset: MultiParticleDataset,
    cubes_per_side: int = 3,
    cube_box: bbox.BoxLike | None = None,
    particle_threshold: int = _DEFAULT_PARTICLE_THRESHOLD,
    particle_types: Collection[str] | None = None,
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

    num_cubes = cubes_per_side**3 + 1
    if np.any(particle_numbers / num_cubes > 2**31):
        raise ValueError(
            f"Insufficient number of cubes specified. Requested {cubes_per_side}"
            f" cubes per side, leading to >={particle_numbers / num_cubes}"
            f" particles per cube. Max per cube supported is {2**32}"
        )

    for pt in requested_types:
        LOGGER.info(f"Processing {pt}")
        dataset.particle_type = pt
        data = dataset.data_container

        LOGGER.info("Cubing")
        shuffle_list, cube_indices = _cube(data, cubes_per_side, cube_box)

        # check cube sizes
        if np.any(np.diff(cube_indices[:-1]) >= 2**32):
            raise ValueError(
                f"Requested number of cubes is insufficient for {pt}. At least"
                " one cube has more than 2**32 particles, the max allowed for"
                " a packed tree."
            )

        dataset.reorder(shuffle_list)

        LOGGER.info("Getting boxes")
        cube_boxes = _get_cube_boxes(
            data=data, box=cube_box, cubes_per_side=cubes_per_side
        )

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


_big_index_tuple = types.UniTuple(types.uint64, 2)


@njit(parallel=True)
def _get_particle_indices_in_shape(
    cubes: List[BoundingBox],
    trees: List[PackedTreeNumba],
    cube_offsets: NDArray,
    shape: bbox.BoundingVolume,
    shape_box: bbox.BoundingBox,
) -> List[tuple[int, int]]:
    """
    Get the particle start-stop tuples in the specified shape
    """
    indices = List.empty_list(_list_index_tuple)
    for _ in range(len(cubes)):
        indices.append(List.empty_list(_index_tuple_type))

    # get particle indices from each tree
    shape_midpoint = np.array(shape_box.midplane())
    for i in prange(len(cubes)):
        # Note: prange indices are uint64 in parallel mode but current
        # TypedList _get_item implementation casts to intp type, which can
        # be int64. We'll explicitly cast to avoid the warning and because
        # len(cubes) **better** be < 2**63 !
        li = np.int_(i)
        px, py, pz = cubes[li].project_point_on_box(shape_midpoint)
        overlap = shape.contains_point(px, py, pz)
        if overlap:
            indices[li] = trees[li]._get_particle_indices_in_shape(
                bounding_box=shape_box, containment_obj=shape
            )

    # add cube offset and flatten list of indices
    flattened_indices = List.empty_list(_big_index_tuple)
    for cube_indices, cube_offset in zip(indices, cube_offsets):  # noqa: B905
        for cube_start, cube_end, _ in cube_indices:
            flattened_indices.append((cube_start + cube_offset, cube_end + cube_offset))

    return flattened_indices


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

    def get_particle_indices_in_box(
        self,
        box: bbox.BoxLike,
    ) -> list[tuple[int, int]]:
        """
        Return all particles contained within the box

        Args:
            box: BoxLike
            Box to check

        Returns:
            indices: list[tuple[int, int]]
            List of particle start-stop indices contained within box
        """
        with objmode(numba_box=bbox.bbn_type):
            numba_box = bbox.make_bounding_box(box)
        return _get_particle_indices_in_shape(
            cubes=self.cube_boxes,
            trees=self._numba_trees,
            cube_offsets=self.cube_indices,
            shape_box=numba_box.copy(),
            shape=numba_box,
        )

    def get_particle_indices_in_sphere(
        self,
        center: NDArray,
        radius: float,
    ) -> list[tuple[int, int]]:
        """
        Return all particles contained within the sphere defined by center and radius

        Args:
            center: NDArray
            Center point of the sphere

            radius: float
            Radius of the sphere

        Returns:
            indices: list[tuple[int, int]]
            List of particle start-stop indices contained within sphere
        """
        with objmode(sph=bbox.bs_type):
            sph = bbox.make_bounding_sphere(center=center, radius=radius)

        return _get_particle_indices_in_shape(
            cubes=self.cube_boxes,
            trees=self._numba_trees,
            cube_offsets=self.cube_indices,
            shape_box=sph.bounding_box,
            shape=sph,
        )


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


class Cubes:
    cubes_dict: dict[str, ParticleCubes]
    """ Mapping from particle type to ParticleCubes for this dataset """

    def __init__(
        self,
        *,
        dataset: MultiParticleDataset,
        cubes_dict: dict[str, dict] | None = None,
        **kwargs,
    ):
        if cubes_dict is None:
            with contextlib.suppress(NotImplementedError, ValueError):
                cubes_dict = load_cubes(dataset)
        self._make_cubes(dataset=dataset, cubes_dict=cubes_dict, **kwargs)

    def _make_cubes(
        self,
        *,
        dataset: MultiParticleDataset,
        cubes_dict: dict[str, dict] | None,
        **kwargs,
    ):
        if cubes_dict is None:
            cubes_dict = make_cubes(dataset=dataset, **kwargs)
        self.cubes_dict = {}
        for pt, cubes in cubes_dict.items():
            cube_indices = cast(NDArray, cubes["cube_indices"])
            cube_boxes = cast(list[bbox.BoundingBox], cubes["cube_boxes"])
            if not hasattr(cubes, "cube_trees"):
                particle_threshold = getattr(
                    kwargs, "particle_threshold", _DEFAULT_PARTICLE_THRESHOLD
                )
                cube_trees = _make_trees(
                    data=dataset.data_container,
                    cube_indices=cube_indices,
                    cube_boxes=cube_boxes,
                    particle_threshold=particle_threshold,
                )
            else:
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

    def get_particle_indices_in_box(
        self,
        box: bbox.BoxLike,
        *,
        particle_types: str | Collection[str] | None = None,
    ) -> dict[str, list[tuple[int, int]]]:
        """
        Return all particles contained within the box

        Args:
            box: BoxLike
            Box to check

            particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to self.particle_types
        Returns:
            indices: dict[str, list[tuple[int, int]][
            Dictionary of lists of particle start-stop indices contained
            within box, organized by particle type
        """
        if particle_types is None:
            particle_types = self.particle_types
        if isinstance(particle_types, str):
            particle_types = [particle_types]
        inds = {}
        for pt in particle_types:
            inds[pt] = self.cubes_dict[pt].get_particle_indices_in_box(box)
        return inds

    def get_particle_indices_in_sphere(
        self,
        center: NDArray,
        radius: float,
        *,
        particle_types: str | Collection[str] | None = None,
    ) -> dict[str, list[tuple[int, int]]]:
        """
        Return all particles contained within the sphere defined by center and radius

        Args:
            particle_types: str | Collection[str]
            Particle type(s) to include

            center: NDArray
            Center point of the sphere

            radius: float
            Radius of the sphere

        Returns:
            indices: dict[str, list[tuple[int, int]][
            Dictionary of lists of particle start-stop indices contained
            within sphere, organized by particle type
        """
        if particle_types is None:
            particle_types = self.particle_types
        if isinstance(particle_types, str):
            particle_types = [particle_types]
        inds = {}
        for pt in particle_types:
            inds[pt] = self.cubes_dict[pt].get_particle_indices_in_sphere(
                center, radius
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
        if has_cubes(dataset):
            if force_overwrite:
                LOGGER.warning(
                    f"Dataset {dataset} already contains cubes structure. Overwriting!"
                )
            else:
                old_filepath = (
                    dataset.filepath
                    if isinstance(dataset, HDF5Dataset)
                    else Path(dataset)
                )
                new_filepath = (
                    old_filepath.parent
                    / f"{old_filepath.stem}_cubes{old_filepath.suffix}"
                )
                LOGGER.info(
                    f"Dataset {old_filepath} already contains cubes structure."
                    f"Saving to {new_filepath} instead."
                )
                dataset = new_filepath

        for pt, cubes in self.cubes_dict.items():
            save_cube(
                dataset,
                pt=pt,
                cube_indices=cubes.cube_indices,
                cube_boxes=cubes.cube_boxes,
                cube_trees=cubes.cube_trees,
            )
        return dataset.filepath if isinstance(dataset, Dataset) else Path(dataset)


if __name__ == "__main__":
    logging.basicConfig()
    args = _process_args()
    if has_cubes(args.output) and not args.force_overwrite:
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
    )
    LOGGER.info(cubes_dict.keys())
    cubes = Cubes(dataset=dataset, cubes_dict=cubes_dict)
    cubes.save(args.output)
