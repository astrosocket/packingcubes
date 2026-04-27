"""Implementation of multiple-particle-type packingcubes"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

import packingcubes.bounding_box as bbox
from packingcubes.cubes.particle_cubes import ParticleCubes, check_overwrite, save_cube
from packingcubes.data_objects import (
    DataContainer,
    Dataset,
    HDF5Dataset,
)
from packingcubes.packed_tree import (
    PackedTree,
)

LOGGER = logging.getLogger(__name__)


class MultiCubes:
    """The cubes for multiple particle types"""

    _cubes_dict: dict[str, ParticleCubes]
    """ Mapping from particle type to ParticleCubes for this dataset """

    def __init__(
        self,
        *,
        cubes_dict: dict[str, dict] | Mapping[str, ParticleCubes],
        **kwargs,
    ):
        self._cubes_dict = {}
        for pt, cubes in cubes_dict.items():
            if isinstance(cubes, ParticleCubes):
                self._cubes_dict[pt] = cubes
                continue
            cube_indices = cast(NDArray, cubes["cube_indices"])
            cube_boxes = cast(list[bbox.BoundingBox], cubes["cube_boxes"])
            cube_trees = cast(
                list[PackedTree] | list[NDArray] | list[PackedTree | NDArray],
                cubes["cube_trees"],
            )
            self._cubes_dict[pt] = ParticleCubes(
                cube_indices=cube_indices,
                cube_boxes=cube_boxes,
                cube_trees=cube_trees,
                **kwargs,
            )

    @property
    def particle_types(self):
        """Return the list of particle types with cubes"""
        return self._cubes_dict.keys()

    def get_single_cubes(self, particle_type: str) -> ParticleCubes:
        """Return the ParticleCubes instance corresponding to the specified type."""
        return self._cubes_dict[particle_type]

    def _get_particle_indices_in_shape(
        self,
        shape: bbox.BoundingVolume,
        *,
        particle_types: str | Collection[str] | None = None,
    ) -> dict[str, NDArray[np.int_]]:
        """Return all particles contained within the shape

        Parameters
        ----------
        particle_types: str | Collection[str]
            Particle type(s) to include

        shape: BoundingVolume
            The shape to check

        particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to self.particle_types

        Returns
        -------
        indices: dict[str, NDArray[int]]
            Dictionary of arrays of particle start-stop indices plus partiality
            flag contained within shape, organized by particle type

        See Also
        --------
        [ParticleCubes._get_particle_indices_in_shape][ParticleCubes._get_particle_indices_in_shape]
        """
        if particle_types is None:
            particle_types = self.particle_types
        if isinstance(particle_types, str):
            particle_types = [particle_types]
        inds = {}
        for pt in particle_types:
            inds[pt] = self._cubes_dict[pt]._get_particle_indices_in_shape(
                shape,
            )
        return inds

    def get_particle_indices_in_box(
        self,
        box: bbox.BoxLike,
        *,
        particle_types: str | Collection[str] | None = None,
    ) -> dict[str, NDArray[np.int_]]:
        """Return all particles contained within the box

        Parameters
        ----------
        box: BoxLike
            Box to check

        particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to self.particle_types

        Returns
        -------
        indices: dict[str, NDArray[int]]
            Dictionary of arrays of particle start-stop indices plus partiality
            flag contained within box, organized by particle type

        See Also
        --------
        [ParticleCubes.get_particle_indices_in_box][ParticleCubes.get_particle_indices_in_box]
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
        """Return all particles contained within the sphere defined by center and radius

        Parameters
        ----------
        particle_types: str | Collection[str]
            Particle type(s) to include

        center: NDArray
            Center point of the sphere

        radius: float
            Radius of the sphere

        particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to self.particle_types

        Returns
        -------
        indices: dict[str, NDArray[int]]
            Dictionary of arrays of particle start-stop indices plus partiality
            flag contained within sphere, organized by particle type

        See Also
        --------
        [ParticleCubes.get_particle_indices_in_sphere][ParticleCubes.get_particle_indices_in_sphere]
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
        """Return all particles contained within the sphere defined by center and radius

        Parameters
        ----------
        data: DataContainer | Dataset
            Dataset containing the particle positions. Pass a DataContainer
            object for a slight performance increase

        center: NDArray
            Center point of the sphere

        radius: float
            Radius of the sphere

        particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to `self.particle_types`

        strict: bool, optional
            Flag to specify whether only particles inside the sphere will
            be returned. If `False` (default), additional nearby particles may be
            included for signficantly increased performance

        use_data_indices: bool, optional
            Flag to return indices into the sorted dataset (`True`, default) or
            into the shuffle list (`False`)

        Returns
        -------
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
            inds[pt] = self._cubes_dict[pt]._get_particle_index_list_in_shape(
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
        """Return all particles contained within the sphere defined by center and radius

        Parameters
        ----------
        data: DataContainer | Dataset
            Dataset containing the particle positions. Pass a `DataContainer`
            object for a slight performance increase

        box: BoxLike
            Box to check

        particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to `self.particle_types`

        strict: bool, optional
            Flag to specify whether only particles inside the sphere will
            be returned. If `False` (default), additional nearby particles may be
            included for signficantly increased performance

        use_data_indices: bool, optional
            Flag to return indices into the sorted dataset (`True`, default) or
            into the shuffle list (`False`)

        Returns
        -------
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
        """Return all particles contained within the sphere defined by center and radius

        Parameters
        ----------
        data: DataContainer | Dataset
            Dataset containing the particle positions. Pass a `DataContainer`
            object for a slight performance increase

        center: NDArray
            Center point of the sphere

        radius: float
            Radius of the sphere

        particle_types: str | Collection[str], optional
            Particle type(s) to include. Defaults to `self.particle_types`

        strict: bool, optional
            Flag to specify whether only particles inside the sphere will
            be returned. If `False` (default), additional nearby particles may be
            included for signficantly increased performance

        use_data_indices: bool, optional
            Flag to return indices into the sorted dataset (`True`, default) or
            into the shuffle list (`False`)

        Returns
        -------
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
        """Get kth nearest particle distances and indices to point

        Parameters
        ----------
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

        Returns
        -------
        distances: NDArray[float]
            Distances to the kth nearest neighbors. Has shape (min(N,k),),
            where N is the number of particles in the sphere bounded by
            distance_upper_bound

        indices: NDArray[int]
            Indices in data of the kth nearest neighbors. Has same shape as
            distances

        Raises
        ------
        NotImplementedError
            If a p value of other then 2 is provided
        """
        if particle_types is None:
            particle_types = self.particle_types
        if isinstance(particle_types, str):
            particle_types = [particle_types]
        data = data.data_container if isinstance(data, Dataset) else data
        inds = {}
        for pt in particle_types:
            inds[pt] = self._cubes_dict[pt].get_closest_particles(
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
        """Save cubes information to specified file

        Parameters
        ----------
        dataset: str | HDF5Dataset
            Location to store cubes data.

        force_overwrite: bool, optional
            If dataset already contains cubes data, overwrite if True.
            Default False

        Returns
        -------
        :
            Path to the saved cubes information
        """
        dataset = check_overwrite(dataset, force_overwrite=force_overwrite)

        for pt, cubes in self._cubes_dict.items():
            save_cube(
                dataset,
                pt=pt,
                cube_indices=cubes.cube_indices,
                cube_boxes=cubes.cube_boxes,
                cube_trees=cubes.cube_trees,
            )
        return dataset.filepath if isinstance(dataset, Dataset) else Path(dataset)
