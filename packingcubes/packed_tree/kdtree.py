from __future__ import annotations

import logging
import warnings

import numpy as np
from numpy.typing import NDArray

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree
from packingcubes.data_objects import Dataset, InMemory
from packingcubes.packed_tree.packed_tree import PackedTree

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(capture=True)


class KDTreeWarning(octree.OctreeWarning):
    pass


class KDTreeError(octree.OctreeError):
    pass


class KDTreeAPI:
    """
    Class to mimic the SciPy KDTree API using PackedTrees

    Will provide identical API to SciPy's KDTree to the extent possible given
    that PackedTrees are fundamentally different. Where 1-1 matches for a
    requested method, argument, or functionality are not possible, raise an
    KDTreeError if there is nothing similar and emit a KDTreeWarning explaining
    the replacement otherwise.

    Warning! PackedTrees are not robust against large amounts of degenerate
    input data! Please sanitize data prior to usage if expecting data
    degeneracy levels above ~100 (i.e. 100 data points with the same values).
    Note that multiple degenerate regions are acceptable, assuming they are
    sufficiently separated.
    """

    _tree: PackedTree
    _dataset: Dataset
    data: NDArray
    """
    The n data points of dimension m to be indexed. The data is only copied if
    the "kd-tree" is built with copy_data=True.
    """
    n: int
    """
    The number of data points.
    """
    leafsize: int
    """
    The number of points at which the algorithm switches over to brute-force
    """
    maxs: NDArray
    """
    The maximum value in each dimension of the n data points
    """
    mins: NDArray
    """
    The minimum value in each dimension of the n data points
    """
    size: int
    """
    The number of nodes in the tree.
    """

    def __init__(
        self,
        data: NDArray,
        leafsize: int | None = None,
        compact_nodes: bool | None = None,  # noqa: FBT001
        copy_data: bool = False,  # noqa: FBT001, FBT002
        balanced_tree: bool | None = None,  # noqa: FBT001
        boxsize=None,
    ):
        if compact_nodes is not None:
            if boxsize is None:
                extra = (
                    "which is set from the outermost datapoints."
                    " So setting compact_nodes does nothing."
                )
            else:
                extra = f"which has been manually set to {boxsize}."
            warnings.warn(
                "Node sizes in PackedTrees are set by the geometry of the"
                "total bounding box, " + extra,
                KDTreeWarning,
                stacklevel=1,
            )
        if leafsize is None:
            LOGGER.info(
                "Using the default PackedTree leaf size "
                f"({octree._DEFAULT_PARTICLE_THRESHOLD}) instead of the KDTree's (10)"
            )
        if balanced_tree is not None and balanced_tree:
            warnings.warn(
                "PackedTree nodes are split at the middle of the bounding box, "
                "independent of the data contained. Setting balanced_tree does "
                "nothing.",
                KDTreeWarning,
                stacklevel=1,
            )
        data_shape = data.shape
        if len(data_shape) != 2 or data_shape[1] != 3:
            raise KDTreeError(
                "PackedTrees only support 3-dimensional data. Provided data "
                + (
                    f"was {data_shape[1]}-dimensional"
                    if len(data_shape) >= 2
                    else "was 1-dimensional."
                )
            )
        self._dataset = InMemory(positions=data.copy() if copy_data else data)
        self.data = self._dataset.positions
        self.n = len(self.data)
        data_box = self._dataset.bounding_box
        self.mins = data_box.box[:3]
        self.maxs = data_box.box[:3] + data_box.box[3:]

        boxsize = np.atleast_1d(boxsize)
        box_warning = """
        PackedTrees do not need or expect data points to be normalized to the
        [0, 1) interval. We will assume that the data is within [0, L_i] with
        no wrapping. If you have negative or overly-large data values, simply
        use the full 6 terms and provide the full bounding box or pass None to
        generate it from the data extents. If you truly need the toroidal
        geometry, please impose that before calling the constructor. \n\n
        You can suppress this message by passing the full 6 terms (or None).
        """
        match len(boxsize):
            case 1:
                warnings.warn(box_warning, KDTreeWarning, stacklevel=1)
                box = np.array([0, 0, 0, boxsize, boxsize, boxsize])
            case 3:
                warnings.warn(box_warning, KDTreeWarning, stacklevel=1)
                box = np.hstack(([0, 0, 0], boxsize.flatten()))
            case 6:
                box = boxsize
            case _:
                raise KDTreeError(
                    f"Cannot handle boxsize argument with length {len(boxsize)}. "
                    "Supported options are 1, 3, and 6."
                )
        bounding_box = bbox.make_bounding_box(box)
        self.leafsize = (
            octree._DEFAULT_PARTICLE_THRESHOLD if leafsize is None else leafsize
        )
        self._tree = PackedTree(
            dataset=self._dataset,
            particle_threshold=leafsize,
            bounding_box=bounding_box,
        )
        # Each node is 5 fields, so number of nodes = length/5
        self.size = int(len(self._tree._tree.tree) / 5)
