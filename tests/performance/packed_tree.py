from numba import njit
from numba.typed import List
from numpy.typing import NDArray

import packingcubes.bounding_box as bbox
import packingcubes.data_objects as data_objects
import packingcubes.packed_tree as optree
from packingcubes.packed_tree import PackedTree

_NUM_REPS = 10_000


def packed_octree_creation(ds):
    return PackedTree(dataset=ds)


def packed_octree_query_ball_point(
    tree: PackedTree, *, centers: list[NDArray], radii: list[float], **kwargs
):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = tree.get_particle_indices_in_sphere(
            center=c,
            radius=r,
        )


def packed_octree_query_ball_point_indices(
    tree: PackedTree,
    *,
    data: data_objects.DataContainer,
    centers: list[NDArray],
    radii: list[float],
    **kwargs,
):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = tree.get_particle_index_list_in_sphere(
            data=data,
            center=c,
            radius=r,
            strict=True,
        )


@njit
def _packed_octree_qbp_jitted(
    tree: optree.packed_tree_numba.PackedTreeNumba,
    spheres: List[bbox.BoundingSphere],
    num_reps: int,
):
    for i in range(len(spheres)):
        sph = spheres[i]
        for _ in range(num_reps):
            sph_inds = tree._get_particle_indices_in_shape(sph)


def packed_octree_qbp_jitted(
    tree: PackedTree, *, centers: list[NDArray], radii: list[float], **kwargs
):
    spheres = List.empty_list(bbox.bs_type)
    for c, r in zip(centers, radii, strict=True):
        sph = bbox.make_bounding_sphere(r, center=c, unsafe=True)
        spheres.append(sph)
    num_reps = _NUM_REPS
    _packed_octree_qbp_jitted(tree._tree, spheres, num_reps)
