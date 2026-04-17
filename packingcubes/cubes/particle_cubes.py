"""Implmentation of single-particle-type packing cubes"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py  # type: ignore
import numpy as np
from numba.typed import List
from numpy.typing import NDArray

import packingcubes.bounding_box as bbox
from packingcubes.bounding_box import BoundingBox
from packingcubes.cubes.cubes_numba import (
    get_closest_particles,
    get_particle_index_list_in_shape,
    get_particle_indices_in_shape,
)
from packingcubes.data_objects import (
    DataContainer,
    Dataset,
    HDF5Dataset,
    MultiParticleDataset,
)
from packingcubes.octree import _DEFAULT_PARTICLE_THRESHOLD
from packingcubes.packed_tree import (
    PackedTree,
)
from packingcubes.packed_tree.packed_tree_numba import (
    PackedTreeNumba,
    euclidean_distance_particle,
)

LOGGER = logging.getLogger(__name__)


class ParticleCubes:
    """The cubes for a single particle type"""

    cube_indices: NDArray
    """ Array of cube indices into the dataset """
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
        """Return all particles contained within the shape

        This is a private version that uses a premade bounding volume

        Parameters
        ----------
        shape: BoundingVolume
            The shape to search in

        Returns
        -------
        indices: Xx3 NDArray[np.int_]
            Array of index information. Each row describes a chunk/slice of data
            in the form `[start, stop, partial]`, where partial is a flag - (1)
            if the data chunk is entirely contained within `shape`, (0) otherwise.
        """
        return get_particle_indices_in_shape(
            cubes=self.cube_boxes,
            trees=self._numba_trees,
            cube_offsets=self.cube_indices,
            shape=shape,
        )

    def get_particle_indices_in_box(
        self,
        box: bbox.BoxLike,
    ) -> NDArray[np.int_]:
        """Return all particles contained within the box

        Parameters
        ----------
        box: BoxLike
            Box to check

        Returns
        -------
        indices: Xx3 NDArray[np.int_]
            Array of index information. Each row describes a chunk/slice of data
            in the form `[start, stop, partial]`, where partial is a flag - (1)
            if the data chunk is entirely contained within `box`, (0) otherwise.
        """
        numba_box = bbox.make_bounding_box(box)

        return self._get_particle_indices_in_shape(numba_box)

    def get_particle_indices_in_sphere(
        self,
        center: NDArray,
        radius: float,
    ) -> NDArray[np.int_]:
        """Return all particles contained within the sphere defined by center and radius

        Parameters
        ----------
        center: NDArray
            Center point of the sphere

        radius: float
            Radius of the sphere

        Returns
        -------
        indices: Xx3 NDArray[np.int_]
            Array of index information. Each row describes a chunk/slice of data
            in the form `[start, stop, partial]`, where partial is a flag - (1)
            if the data chunk is entirely contained within the sphere, (0)
            otherwise.
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
        """Return all particle indices contained within the shape

         This is a private version that uses a premade bounding volume

        Parameters
        ----------
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

        Returns
        -------
        indices: Array[int]
            Array of particle indices contained within shape
        """
        return get_particle_index_list_in_shape(
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
        """Return all particle indices contained within the box

        Parameters
        ----------
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

        Returns
        -------
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
        """Return all particle indices contained within the sphere

        Parameters
        ----------
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

        Returns
        -------
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
        """Get kth nearest particle distances and indices to point.

        Parameters
        ----------
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
            If a p value of greater than 2 is provided
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

        return get_closest_particles(
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
            Still in progress. Try a PackedTree for this functionality
            """
        )

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


def check_overwrite(
    dataset: str | Path | HDF5Dataset, *, force_overwrite: bool = False
) -> str | Path | HDF5Dataset:
    """
     Check if it is safe to overwrite cubes structure, returning new file if not

     It's safe to overwrite cubes structure in two cases:

      1. It doesn't exist
      2. It does exist and force_overwrite is True

     If it's not safe to overwrite the cubes structure, return a path to a new
     file, specified as `dataset.filepath.stem+"_cubes.hdf5"`. Note that we do
     not check if the new filepath already exists, so this could clobber the
     information in that file!

    Parameters
    ----------
    dataset: str | Path | HDF5Dataset
        The location to check
    force_overwrite: bool, optional
        Force writing the cubes structure in the provided file, even if one
        already exists. Default False

    Returns
    -------
    :
        The location to write the cubes structure to

    """
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
    """Save an individual cube's data to the dataset"""
    filepath = dataset.filepath if isinstance(dataset, HDF5Dataset) else dataset
    with h5py.File(filepath, "a") as file:
        cubes = file.create_group(f"cubes/{pt}")
        cubes["indices"] = cube_indices
        cubes["number"] = len(cube_indices)
        for i, (box, tree) in enumerate(zip(cube_boxes, cube_trees, strict=True)):
            cubes[f"box_{i}"] = box.box
            cubes[f"tree_{i}"] = tree.packed_form
