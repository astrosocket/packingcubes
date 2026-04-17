# ruff: noqa: D103
from numpy.typing import NDArray

import packingcubes.octree as octree
from packingcubes.packed_tree import KDTree


def packed_kdtree_creation(ds):
    return KDTree(data=ds, leafsize=octree._DEFAULT_PARTICLE_THRESHOLD)


def packed_kdtree_query_ball_point(
    tree: KDTree,
    *,
    centers: list[NDArray],
    radii: list[float],
    particle_numbers: list[int],
    **kwargs,
):
    for i, (c, r) in enumerate(zip(centers, radii, strict=True)):
        # Ensure kdtree output matches scipy's regardless of defaults
        sph_inds = tree.query_ball_point(
            x=c,
            r=r,
            strict=True,
            return_sorted=False,
            return_lists=False,
            return_data_indices=True,
        )
        if particle_numbers and len(sph_inds) != particle_numbers[i]:
            raise ValueError(
                f"""
                Particle number mismatch: expected {particle_numbers[i]} particles
                for ball {i} and got {len(sph_inds)}.
                """
            )


def packed_kdtree_query(tree: KDTree, *, centers: list[NDArray], k: int, **kwargs):
    for c in centers:
        dd, ii = tree.query(c, k=k, return_data_indices=True, return_sorted=True)
