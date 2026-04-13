from numpy.typing import NDArray

import packingcubes.octree as octree
from packingcubes.octree import PythonOctree


def python_octree_creation(ds):
    return PythonOctree(
        dataset=ds,
    )


def python_octree_query_ball_point(
    tree: octree.Octree, *, centers: list[NDArray], radii: list[float], **kwargs
):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = tree.get_particle_indices_in_sphere(
            center=c,
            radius=r,
        )
