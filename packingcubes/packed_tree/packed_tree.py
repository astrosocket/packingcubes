from __future__ import annotations

import logging
from collections.abc import Buffer, Iterable, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree
from packingcubes.data_objects import Dataset, InMemory
from packingcubes.packed_tree.packed_node import PackedNode
from packingcubes.packed_tree.packed_tree_meta import (
    TreeMeta,
    TreeMetaError,
    create_metadata,
    extract_metadata,
    pack_metadata,
)
from packingcubes.packed_tree.packed_tree_numba import PackedTreeNumba, _construct_tree

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(capture=True)


class PackedTree(octree.Octree):
    """
    Public packed octree interface

    This interface defines the methods for creating, manipulating, and
    traversing a packingcubes packed octree.

    Attributes:
        data: Dataset
        The backing dataset

        particle_threshold: int
        The maximum leaf size before splitting, used in tree construction

    """

    _tree: PackedTreeNumba
    """
    The actual in-memory representation of the tree as a numba-fied class
    object
    """
    metadata: TreeMeta
    """
    The metadata for this packed tree
    """
    particle_threshold: int
    """
    The maximum leaf size before splitting, used in tree construction
    """

    def __init__(
        self,
        *,
        dataset: NDArray | Dataset | None = None,
        source: Buffer | None = None,
        particle_threshold: int | None = None,
        bounding_box: bbox.BoxLike | None = None,
        copy_data: bool = False,
    ):
        """
        Note: must provide either dataset or source. If provided source does
        not include metadata, must additionally provide either dataset or
        bounding_box.

        Args:
            dataset: NDArray | Dataset, optional
            An (N,3) array or Dataset containing particle data

            source: Buffer | None, optional
            Pre-computed packed buffer containing this tree. Leave out to
            compute the tree from scratch.

            particle_threshold: int, optional
            Number of particles allowed in a leaf before splitting. Defaults to
            octree._DEFAULT_PARTICLE_THRESHOLD

            bounding_box: BoxLike, optional
            Bounding box of the tree. Required if metadata needs to be created
            and dataset is not provided.
            Will override the dataset bounding box.

            copy_data: bool, optional
            If dataset is just an array, flag to copy data prior to construction.
            Defaults to False
        """
        if particle_threshold is None:
            particle_threshold = octree._DEFAULT_PARTICLE_THRESHOLD

        self.particle_threshold = particle_threshold

        if dataset is not None and not isinstance(dataset, Dataset):
            dataset = InMemory(positions=dataset.copy() if copy_data else dataset)

        # from some empirical testing. doesn't need to be exact anyway
        # estimated_size_in_bytes = (
        #     len(data) / 10 ** (np.floor(np.log10(self.particle_threshold))) * 1.2
        # ).astype(int) * 20
        # self.tree = np.array((estimated_size_in_bytes,), dtype=np.bytes_)

        metadata = None
        if source is not None:
            try:
                metadata, packed = extract_metadata(source)
            except TreeMetaError as ve:
                packed = np.frombuffer(source, dtype=np.uint32)
        elif dataset is not None:
            data = dataset.data_container
            bounding_box = (
                data.bounding_box
                if bounding_box is None
                else bbox.make_bounding_box(bounding_box)
            )
            packed = _construct_tree(
                data, box=bounding_box, particle_threshold=particle_threshold
            )
        else:
            raise ValueError("Must provide one of dataset or source")

        if metadata is None:
            if bounding_box is not None:
                box = bbox.make_bounding_box(bounding_box)
            elif dataset is not None:
                box = dataset.bounding_box
            else:
                raise ValueError("Metadata creation requires dataset or bounding_box")
            metadata = create_metadata(
                box=box,
                packed=packed,
                particle_threshold=particle_threshold,
            )
        self.metadata = metadata
        self._tree = PackedTreeNumba(
            self.metadata.bounding_box, packed, self.metadata.particle_threshold
        )

    @property
    def packed_tree(self) -> memoryview:
        """
        Return a memoryview of the tree's backing byte array
        """
        return memoryview(self._tree.tree)

    @property
    def packed_meta(self) -> memoryview:
        """
        Return a memoryview of the tree's packed metadata
        """
        packed_meta = pack_metadata(self.metadata, self._tree.tree)
        return memoryview(packed_meta)

    @property
    def packed_form(self) -> NDArray[np.uint32]:
        """
        Return this tree in full packed form
        """
        packed_meta = np.frombuffer(self.packed_meta, dtype=np.uint32)
        packed_tree = np.frombuffer(self.packed_tree, dtype=np.uint32)
        return np.hstack((packed_meta, packed_tree))

    def __iter__(self) -> Iterator[octree.OctreeNode]:
        """
        Iterate through all nodes of the octree. Note that no guarantee is made
        of what order the nodes are traversed in
        """
        return map(PackedNode, self._tree._get_nodes_numba())

    def get_leaves(self) -> Iterable[octree.OctreeNode]:
        """
        Return a list of all leaf octree nodes in depth-first order
        """
        return map(PackedNode, self._tree.get_leaves())

    def get_node(self, tag: str) -> octree.OctreeNode | None:
        """
        Return the node corresponding to the provided tag or None if not found

        Args:
            tag: str
            The tag to search for

        Returns:
            node
            Node in octree with specified tag or None if it does not exist
        """
        pnn = self._tree.get_node(tag)
        return PackedNode(pnn) if pnn else None

    def get_particle_indices_in_box(
        self,
        *,
        box: bbox.BoxLike,
    ) -> list[tuple[int, int]]:
        """
        Return all particles contained within the box

        Args:
            box: BoxLike
            Box to check

        Returns:
            indices: list[tuple[int, int]]
            List of particle start-stop indices contained within sphere
        """
        bounding_box = bbox.make_bounding_box(box)
        return self._tree._get_particle_indices_in_shape(bounding_box, bounding_box)

    def get_particle_indices_in_sphere(
        self,
        *,
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
        sph = bbox.make_bounding_sphere(center=center, radius=radius)
        return self._tree._get_particle_indices_in_shape(sph.bounding_box, sph)

    def get_particle_index_list_in_box(
        self,
        *,
        dataset: Dataset,
        box: bbox.BoxLike,
        strict: bool = False,
    ) -> NDArray[np.int64]:
        """
        Return all particles contained within the box

        Args:
            box: BoxLike
            Box to check

            strict: bool, optional
            Flag to specify whether only particles inside containment_obj will
            be returned. If False (default), additional nearby particles may be
            included for signficantly increased performance

        Returns:
            indices: NDArray[int]]
            List of original particle indices contained within sphere
        """
        bounding_box = bbox.make_bounding_box(box)
        if strict:
            return self._tree._get_particle_index_list_in_shape_strict(
                dataset.data_container, bounding_box, bounding_box
            )
        return self._tree._get_particle_index_list_in_shape(
            dataset.data_container, bounding_box, bounding_box
        )

    def get_particle_index_list_in_sphere(
        self,
        *,
        dataset: Dataset,
        center: NDArray,
        radius: float,
        strict: bool = False,
    ) -> NDArray[np.int64]:
        """
        Return all particles contained within the sphere defined by center and radius

        Args:
            center: NDArray
            Center point of the sphere

            radius: float
            Radius of the sphere

            strict: bool, optional
            Flag to specify whether only particles inside containment_obj will
            be returned. If False (default), additional nearby particles may be
            included for signficantly increased performance

        Returns:
            indices: NDArray[int]
            List of original particle indices contained within sphere
        """
        sph = bbox.make_bounding_sphere(radius, center=center)
        bounding_box = sph.bounding_box
        if strict:
            return self._tree._get_particle_index_list_in_shape_strict(
                dataset.data_container, bounding_box, sph
            )
        return self._tree._get_particle_index_list_in_shape(
            dataset.data_container, bounding_box, sph
        )

    def get_closest_particle(
        self, xyz: ArrayLike, *, check_neighbors: bool = True
    ) -> tuple[np.int_, float]:
        """
        Get nearest particle index (and distance) to point

        Steps:
          1. Find smallest node contaning point
          2. Find closest particle in this node
          3. Check if neighboring nodes have closer particles
              1. Check if neighboring node is closer than closest particle
              2. Compare particles in neighbor node to closest

        Args:
            xyz: ArrayLike
            Coordinates of point to check

            check_neighbors: bool, optional
            Flag to check whether we should look at neighbors of the smallest
            containing node. Default True

        Returns:
            closest_ind: int
            Absolute index of closest particle

            closest_dist: float
            Distance to closest particle
        """
        return self._tree.get_closest_particle(xyz, check_neighbors)
