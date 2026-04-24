# ruff: noqa: D103
"""Creation and search methods for SciPy's KDTree"""

from numpy.typing import NDArray
from scipy.spatial import KDTree

import packingcubes.octree as octree


def kdtree_creation(ds):
    return KDTree(
        data=ds.positions, copy_data=True, leafsize=octree._DEFAULT_PARTICLE_THRESHOLD
    )


def kdtree_query_ball_point(
    tree: KDTree,
    *,
    centers: list[NDArray],
    radii: list[float],
    particle_numbers: list[int],
    **kwargs,
):
    for i, (c, r) in enumerate(zip(centers, radii, strict=True)):
        sph_inds = tree.query_ball_point(x=c, r=r)
        if particle_numbers and len(sph_inds) != particle_numbers[i]:
            raise ValueError(
                f"""
                Particle number mismatch: expected {particle_numbers[i]} particles
                for ball {i} and got {len(sph_inds)}.
                """
            )


def kdtree_query(tree: KDTree, *, centers: list[NDArray], k: int, **kwargs):
    for c in centers:
        dd, ii = tree.query(c, k=k)
