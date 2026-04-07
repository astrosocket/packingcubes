from __future__ import annotations

import logging
from collections.abc import Callable

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
from packingcubes.bounding_box import BoundingBox
from packingcubes.data_objects import (
    DataContainer,
    subview,
)
from packingcubes.packed_tree.fixed_distance_heap import FixedDistanceHeap
from packingcubes.packed_tree.packed_tree_numba import (
    PackedTreeNumba,
    _construct_tree,
    _index_tuple_type,
    _list_index_tuple,
    _process_slice_against_heap,
)

LOGGER = logging.getLogger(__name__)


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
LOGGER.debug(f"Running on the {layer} threading layer with {get_num_threads()} threads")
if layer == "tbb":
    LOGGER.warning(
        "Parallel support for cubes is known to be flaky on the tbb threading "
        "layer. If you are having difficulties, consider switching to the omp "
        "layer by setting the NUMBA_THREADING_LAYER environmental variable or "
        "by setting numba.config.THREADING_LAYER. See "
        "https://numba.readthedocs.io/en/stable/user/threading-layer.html for "
        "more information."
    )


@njit
def prune_empty(
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
    nthreads = get_num_threads()
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
def cube(data: DataContainer, cubes_per_side: int, box: BoundingBox):
    """
    Bin the loaded particles into the different cubes
    """
    num_cubes = cubes_per_side**3 + 1
    # print(f"Begin cubing into {num_cubes} cubes")

    nthreads = get_num_threads()

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
def make_trees(
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


@njit(parallel=True)
def get_particle_indices_in_shape(
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
def get_particle_index_list_in_shape(
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
    slices = get_particle_indices_in_shape(cubes, trees, cube_offsets, shape)

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
def get_closest_particles(
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

    slices = get_particle_indices_in_shape(cubes, trees, cube_indices, search_box)

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
