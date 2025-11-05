from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import ArrayLike

from packingcubes.data_objects import Dataset

LOGGER = logging.getLogger(__name__)


# Convenience functions
def _partition(data: Dataset, lo: int, hi: int, ax: int, pivot: float) -> int:
    """
    Partition a portion of 3D data between lo and hi along the axis ax

    Returns the (effective) pivot location. Note that the pivot is not required
    to be in the data, in which case the returned location will be lo (if all
    elements are greater than pivot) or hi+1 (if all elements are less than
    the pivot)
    """
    assert 0 <= lo, f"Low index out of bounds {lo=}"
    assert hi < len(data), f"High index out of bounds {hi=}"
    if lo > hi:
        # nothing to do
        return lo
    r = hi  # right edge (won't change)
    while lo < hi:
        # pivot is not likely to be in data
        # need additional check to guard against out-of-bounds
        # access
        while lo < r and data.positions[lo, ax] < pivot:
            lo += 1
        # Don't want hi<=lo anyway, so don't need equivalent left edge constant
        while hi > lo and data.positions[hi, ax] >= pivot:
            hi -= 1
        data._swap(lo, hi)
    if lo >= hi:
        # undo extraneous last swap if lo>=hi
        data._swap(lo, hi)
    # we may be in a position where all elements are less then the pivot
    if lo == r and data.positions[lo, ax] < pivot:
        return r + 1
    return lo


def _partition_data(
    data: Dataset,
    box: ArrayLike,
    lo: int,
    hi: int,
) -> List[int]:
    """
    Partition 3D data from lo to hi contained in bounding box into 8 octants.

    Returns child_list of the octants starting indices. First octant assumed
    to start at lo.
    Note that if child_list[i] octant contains no data, then
    child_list[i]==child_list[i+1]. Note also that child_list[i] can be greater
    than hi. This means all data was contained in octants
    [lo,*child_list[0:i-1]]
    """
    pass


def _get_child_box(box: ArrayLike, ind: int) -> ArrayLike:
    """
    Get indth new child box of current box

    New child box is defined as the suboctant described by position ind
    with size (box[3]/2, box[4]/2, box[5]/2)
    """
    # Use z-index order for now, but other possibilities
    # like Hilbert curves exist - and see
    # https://math.stackexchange.com/questions/2411867/3d-hilbert-curve-without-double-length-edges
    # for a possible "Hilbert" curve that may be better?
    # Raising errors for invalid boxes coming from data
    if np.any(np.isinf(box) | np.isnan(box)):
        raise ValueError("Box values must be finite numbers")
    if np.any(box[3:] <= 0):
        raise ValueError("box dimensions must be >0")
    # using assert because octree members know better
    assert isinstance(ind, int) and 0 <= ind and ind < 8, (
        f"Octree code passed an invalid index: {ind}!"
    )
    child_box = box.copy()
    child_box[3:] /= 2.0
    x, y, z = ((ind & 2) / 2, ind & 1, (ind & 4) / 4)
    child_box[0] = child_box[0] + child_box[3] * x
    child_box[1] = child_box[1] + child_box[4] * y
    child_box[2] = child_box[2] + child_box[5] * z
    return child_box


def _in_box(box: ArrayLike, x: float, y: float, z: float) -> bool:
    """
    Check if point inside box
    """
    return (
        ((box[0] <= x) & (x <= box[0] + box[3]))
        & ((box[1] <= y) & (y <= box[1] + box[4]))
        & ((box[2] <= z) & (z <= box[2] + box[5]))
    )


def _box_neighbors_in_node(
    box_ind: int,
) -> ArrayLike[int]:
    """
    Return the neighbor boxes that are in the same node. Assuming 0-based indexing
    """
    pass


def project_point_on_box(box: ArrayLike, x: float, y: float, z: float) -> tuple[float]:
    """
    Return coordinates of projection of (x, y, z) on box face.

    This is the closest point on the box to (x, y, z). Possibly will be
    jittered *into* the box for determining sub-boxes.
    """
    pass


def morton(positions: ArrayLike, box: ArrayLike) -> ArrayLike:
    """
    Given array of positions and box to look at, compute morton encoding
    Note that there are no checks on whether positions are actually inside
    the box
    Note that returned morton encoding is 1-indexed
    """
    # Map positions to box s.t. [0,0,0] and [1,1,1] are left-front-bottom and
    # right-back-top
    boxed_pos = (positions - box[:3]) / box[3:]
    # note that np.round rounds 0.5 to 0 (due to round-to-even choice)
    morton = (
        1
        + np.round(boxed_pos[:, 0]) * 2
        + np.round(boxed_pos[:, 1]) * 1
        + np.round(boxed_pos[:, 2]) * 4
    )
    return morton


# This will be an in-place octree, so each node has a
# view of the backing array, indices to the start of
# each subnode/leaf, box size, and node start/end
# Note that node_ind+part_num=next_sibling_ind
@dataclass
class OctreeNode:
    """
    Internal representation of a node in the octree

    Class holding the octree information in a recursive substructure. Note that
    the octree is currently fully initialized on creation of the root (node
    creation contains a recursive node creation call). Each node forms the root
    of it's own octree and can have children that are either a

    Inputs:
        data: Dataset
        The backing Dataset for the octree

        node_start, node_end: int, optional
        The start and end data indices for this node. Defaults to 0 and
        len(data)

        box: ArrayLike, optional
        Bounding box of the form [x, y, z, dx, dy, dz] where (x, y, z) is the
        left-front-bottom corner. All particles are assumed to lie inside.
        Defaults to [0, 0, 0, 1, 1, 1]

        tag: List[int], optional
        List of 1-based z-order indices describing the current box.
        E.g. if assuming the default bounding box, the box
        [0.25, 0.25, 0.75, 0.25, 0.25, 0.25] would be [5,1]

        parent: None | OctreeNode, optional
        Parent node of this node. None (default) if root of the entire tree.

        _particle_threshold: int, optional
        Configuration parameter for how many particles will be contained in a
        leaf before splitting into subtrees. Note that currently particle
        sorting stops at this level, i.e. leaves are unsorted. Default 1.
    """

    data: Dataset
    node_start: int
    node_end: int
    box: ArrayLike  # [x,y,z,dx,dy,dz]
    # empty for root node, otherwise list of morton indices s.t. [7,2,1] is the
    # tag for the left-front-bottom subnode of the left-back-bottom subnode of
    # right-front-top subnode of the root
    tag: List[int]
    children: List[OctreeNode]
    parent: None | OctreeNode

    def __init__(
        self,
        *,
        data: Dataset,
        node_start: int = 0,
        node_end: int = None,
        box: ArrayLike = np.array([0, 0, 0, 1, 1, 1], dtype=float),
        tag: List[int] = None,
        parent: None | OctreeNode = None,
        _particle_threshold=1,
    ) -> None:
        self.data = data
        self.node_start = node_start
        self.node_end = len(data) - 1 if node_end is None else node_end
        self.children = []
        self.box = box.astype(float)
        self.tag = [] if tag is None else tag
        self.parent = parent
        self._particle_threshold = _particle_threshold

        # begin recursion
        self._construct()

    def _construct(self) -> None:
        """
        Main recursive call of octree construction. This is where the particles
        get partitioned and child nodes created
        """
        pass

    def __repr__(self):
        return (
            f"OctreeNode (id:{self.tag}) from {self.node_start}-{self.node_end} ({len(self)} particles) with"
            + f" box {self.box} and {len(self.children)} child node"
            + f"{'' if len(self.children) == 1 else 's'}"
        )

    def __len__(self):
        """
        Return number of particles held by this node
        """
        return self.node_end - self.node_start + 1

    @property
    def slice(self):
        """
        Return slice of data corresponding to this node
        """
        return slice(self.node_start, self.node_end + 1)

    def distance(self, x: float, y: float, z: float) -> ArrayLike:
        """
        Return array of particle distances from given point
        """
        pass


class Octree:
    """
    Public octree class

    Use this class for creating, manipulating, and traversing an octree based
    on a Dataset object. The octree will be composed of OctreeNodes and is
    currently computed in a top-down approach

    Inputs:
        data: Dataset
        A Dataset containing particle data

        _particle_threshold: int
        Number of particles allowed in a leaf before splitting

    """

    root: OctreeNode

    def __init__(self, data: Dataset, _particle_threshold: int) -> None:
        self.root = OctreeNode(
            data=data, box=data.bounding_box, _particle_threshold=_particle_threshold
        )

    def get_closest_particle(self, x: float, y: float, z: float) -> None:
        """
        Get nearest particle index to point
        """
        pass
