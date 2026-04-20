# ruff: noqa: D103
"""Creation and search methods for ParticleCubes objects"""

from numpy.typing import NDArray

import packingcubes.cubes as cubes
from packingcubes.cubes import ParticleCubes as ParticleCubes


def cubes_creation(ds):
    return cubes.Cubes(
        dataset=ds,
        save_dataset=False,
    )


def cubes_query_ball_points(
    cubes: ParticleCubes, *, centers: list[NDArray], radii: list[float], **kwargs
):
    for c, r in zip(centers, radii, strict=True):
        sph_inds = cubes.get_particle_indices_in_sphere(
            center=c,
            radius=r,
        )
