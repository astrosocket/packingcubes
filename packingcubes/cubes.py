from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Collection

import numpy as np
from numba import (  # type:ignore
    get_num_threads,
    get_thread_id,
    njit,
    objmode,
    prange,
    types,
)
from numba.typed import List
from numpy.typing import NDArray

import packingcubes.bounding_box as bbox
from packingcubes.bounding_box import BoundingBox, make_bounding_box
from packingcubes.data_objects import (
    DataContainer,
    GadgetishHDF5Dataset,
    HDF5Dataset,
    subview,
)
from packingcubes.octree import _DEFAULT_PARTICLE_THRESHOLD
from packingcubes.packed_tree import PackedTree, PackedTreeNumba, _construct_tree

LOGGER = logging.getLogger(__name__)

nthreads = get_num_threads()


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
    snapshot file should be specified with -- SNAPSHOT_NAME at the end.
    
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
    parser.add_argument("snapshot", help="path to the snapshot file", type=str)

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


def _process_box(*, dataset: HDF5Dataset, args) -> BoundingBox:
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
    cube_boxes = []
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
    cube_x = np.floor((x - box.x) / box.dx * cubes_per_side)
    cube_x = cubes_per_side - 1 if x == box.x + box.dx else cube_x
    cube_y = np.floor((y - box.y) / box.dy * cubes_per_side)
    cube_y = cubes_per_side - 1 if y == box.y + box.dy else cube_y
    cube_z = np.floor((z - box.z) / box.dz * cubes_per_side)
    cube_z = cubes_per_side - 1 if z == box.z + box.dz else cube_z
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
    dataset: HDF5Dataset,
    cubes_per_side: int,
    cube_box: BoundingBox,
    particle_threshold: int,
    particle_types: Collection | None = None,
):
    cubes = {}

    if particle_types is None:
        requested_types = dataset.particle_types
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

        cubes[pt] = {
            "cube_indices": cube_indices,
            "cube_boxes": cube_boxes,
            "cube_trees": trees,
        }
        LOGGER.info(f"Done with {pt}")

    return cubes


@njit(parallel=True)
def _get_particle_indices_in_shape(
    cubes: List[BoundingBox],
    trees: List[PackedTreeNumba],
    cube_offsets: NDArray,
    shape: bbox.BoundingVolume,
    shape_box: bbox.BoundingBox,
) -> List[tuple[int, int]]:
    indices = List.empty_list(List[tuple[int, int]])
    for _ in range(len(cubes)):
        indices.append(List.empty_list(tuple[int, int]))

    # get particle indices from each tree
    shape_midpoint = np.array(shape_box.midplane())
    for i in prange(len(cubes)):
        overlap = shape.contains(cubes[i].project_point_on_box(shape_midpoint))
        if overlap:
            indices[i] = trees[i]._get_particle_indices_in_shape(
                bounding_box=shape_box, containing_obj=shape
            )

    # add cube offset and flatten list of indices
    flattened_indices = List.empty_list(tuple[int, int])
    for cube_indices, cube_offset in zip(indices, cube_offsets, strict=True):
        for cube_start, cube_end in cube_indices:
            flattened_indices.append(cube_start + cube_offset, cube_end + cube_offset)

    return flattened_indices


class Cubes:
    dataset: HDF5Dataset
    cube_indices: NDArray
    cube_boxes: List[BoundingBox]
    cube_trees: list[PackedTree]
    _numba_trees: List[PackedTreeNumba]

    def __init__(
        self,
        dataset: HDF5Dataset,
        cube_indices: NDArray,
        cube_boxes: List[BoundingBox],
        cube_trees: list[NDArray] | list[PackedTree],
        **kwargs,
    ):
        self.dataset = dataset
        self.cube_indices = cube_indices
        self.cube_boxes = cube_boxes

        if isinstance(cube_trees[0], PackedTree):
            self.cube_trees = []
            for t in cube_trees:
                if not isinstance(t, PackedTree):
                    raise TypeError("Cannot mix PackedTrees and non-PackedTrees")
                self.cube_trees.append(t)
        else:
            self.cube_trees = [
                PackedTree(dataset=self.dataset, source=np.array(t)) for t in cube_trees
            ]

        self._numba_trees = List([t._tree for t in self.cube_trees])

    def get_particle_indices_in_box(
        self,
        box: bbox.BoxLike,
    ) -> list[tuple[int, int]]:
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
        with objmode(sph=bbox.bs_type):
            sph = bbox.make_bounding_sphere(center=center, radius=radius)

        return _get_particle_indices_in_shape(
            cubes=self.cube_boxes,
            trees=self._numba_trees,
            cube_offsets=self.cube_indices,
            shape_box=sph.bounding_box,
            shape=sph,
        )


if __name__ == "__main__":
    logging.basicConfig()
    args = _process_args()
    dataset = GadgetishHDF5Dataset(filepath=args.snapshot)
    box = _process_box(dataset=dataset, args=args)
    cubes_dict = make_cubes(
        dataset=dataset,
        cubes_per_side=args.n,
        cube_box=box,
        particle_threshold=args.particle_threshold,
        particle_types=args.particle_types,
    )
    LOGGER.info(cubes_dict.keys())
    cubes = Cubes(dataset=dataset, **cubes_dict)
