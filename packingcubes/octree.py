from __future__ import annotations

import logging
from typing import List

import numpy as np
from numpy.typing import ArrayLike

import packingcubes.bounding_box as bbox
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
    if lo < 0:
        raise ValueError(f"Low index out of bounds {lo=}")
    if len(data) <= hi:
        raise ValueError(f"High index out of bounds {hi=}")
    if lo > hi:
        # nothing to do
        return lo
    if lo == hi:
        # return early to avoid unnecessary swap
        # ternary to avoid data cast
        return lo + (1 if data.positions[lo, ax] < pivot else 0)
    r = hi  # right edge (constant)
    while lo < hi:
        # pivot is not guaranteed to be in data
        # need additional check to guard against out-of-bounds
        # access
        while lo < r and data.positions[lo, ax] < pivot:
            lo += 1
        # Don't want hi<=lo anyway, so don't need equivalent left edge constant
        while hi > lo and data.positions[hi, ax] >= pivot:
            hi -= 1
        data._swap(lo, hi)
    if lo >= hi:
        # undo extraneous last swap if lo>hi. If lo==hi, swapping does nothing
        # but is expensive, so avoid if possible
        data._swap(lo, hi)
    # we may be in a position where all elements are less then the pivot
    # this is equivalent to the single element case
    # if lo == r:
    #     return r + (1 if data.positions[lo, ax] < pivot else 0)
    return lo + (1 if data.positions[lo, ax] < pivot else 0)


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
    midplane = bbox.midplane(box)
    child_list = [-1, -1, -1, -1, -1, -1, -1]
    # child_list is defined such that data in c0 (child node 1) is prior to
    # child_list[0], c1 data is between child_list[0] and child_list[1], etc.
    # i.e. c0 < 0 < c1 < 1 < c2 < 2 < c3 < 3 < c4 < 4 < c5 < 5 < c6 < 6 < c7
    # z-index ordering, so box is
    #    /6-----/8
    #  5/-+---7/ |
    #  | /2---+-/4
    #  1/-----3/
    # z-partition
    child_list[3] = _partition(data, lo, hi, 2, midplane[2])
    # x-partition and y-partition - note that these combine to be only two
    # passes through the list. Also, only perform partition if we actually
    # have data available
    if lo < child_list[3]:
        child_list[1] = _partition(data, lo, child_list[3] - 1, 0, midplane[0])
        if lo < child_list[1]:
            child_list[0] = _partition(data, lo, child_list[1] - 1, 1, midplane[1])
        if child_list[1] < hi:
            child_list[2] = _partition(
                data, child_list[1], child_list[3] - 1, 1, midplane[1]
            )
    if child_list[3] < hi:
        child_list[5] = _partition(data, child_list[3], hi, 0, midplane[0])
        if 0 < child_list[5]:
            child_list[4] = _partition(
                data, child_list[3], child_list[5] - 1, 1, midplane[1]
            )
        if child_list[5] < hi:
            child_list[6] = _partition(data, child_list[5], hi, 1, midplane[1])
    # fix any empty values
    # There's probably a cleverer way to do this...
    fill_value = lo
    for i in range(7):
        if child_list[i] < 0:
            child_list[i] = fill_value
        else:
            fill_value = hi + 1
    return child_list


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
    if np.any(~np.isfinite(box)):
        raise ValueError("Box values must be finite numbers")
    if np.any(box[3:] <= 0):
        raise ValueError("box dimensions must be >0")
    # using assert because octree members know better
    assert isinstance(ind, int) and 0 <= ind and ind < 8, (
        f"Octree code passed an invalid index: {ind}!"
    )
    child_box = box.copy()
    child_box[3:] /= 2.0
    x, y, z = ((ind & 1) / 1, (ind & 2) / 2, (ind & 4) / 4)
    child_box[0] = child_box[0] + child_box[3] * x
    child_box[1] = child_box[1] + child_box[4] * y
    child_box[2] = child_box[2] + child_box[5] * z
    return child_box


def _box_neighbors_in_node(
    box_ind: int,
) -> ArrayLike[int]:
    """
    Return the neighbor boxes that are in the same node. Assuming 0-based indexing
    """
    # TODO: look into a case/control-flow free formula
    match box_ind:
        case 0:  # 1
            return [1, 2, 4]  # [2,3,5]
        case 1:  # 2
            return [0, 3, 5]  # [1,4,6]
        case 2:  # 3
            return [0, 3, 6]  # [1,4,7]
        case 3:  # 4
            return [1, 2, 7]  # [2,3,8]
        case 4:  # 5
            return [0, 5, 6]  # [1,6,7]
        case 5:  # 6
            return [1, 4, 7]  # [2,5,8]
        case 6:  # 7
            return [2, 5, 7]  # [3,6,8]
        case _:  # 8
            return [3, 5, 6]  # [4,6,7]


def morton(positions: ArrayLike, box: ArrayLike) -> ArrayLike[int]:
    """
    Given array of positions and box to look at, compute morton encoding
    Note that there are no checks on whether positions are actually inside
    the box
    Note that returned morton encoding is 1-indexed
    """
    # old approach:
    # Map positions to box s.t. [0,0,0] and [1,1,1] are left-front-bottom and
    # right-back-top
    # Note that np.round/rint rounds 0.5 to 0 (due to round-to-even choice)
    # to match _partition method, we need to round 0.5 to 1. The following is
    # from https://stackoverflow.com/a/34219827
    # boxed_pos = np.trunc(bbox.normalize_to_box(positions, box) + 0.5).astype(int)
    # morton = 1 + boxed_pos[:, 0] * 1 + boxed_pos[:, 1] * 2 + boxed_pos[:, 2] * 4
    # This and similar approaches don't handle subnormal values and similar edge
    # cases well (due to the normalization step), so we've changed the approach
    # new approach
    # This could technically still fail, since e.g. a box with x=1e10, dx=1e-8,
    # would have a midplane of 1.000000000000000001e10=1e10 to within floating
    # point, putting anything in subbox 1 in subbox 2. But it'll match the
    # behavior of _partition, and if your data looks like that, there's not
    # much we can do...
    midplane = bbox.midplane(box)
    morton = (
        1
        + (positions[:, 0] >= midplane[0])
        + 2 * (positions[:, 1] >= midplane[1])
        + 4 * (positions[:, 2] >= midplane[2])
    ).astype(int)
    return morton


# This will be an in-place octree, so each node has a
# view of the backing array, indices to the start of
# each subnode/leaf, box size, and node start/end
# Note that node_ind+part_num=next_sibling_ind
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
        # sort everything, even if we have fewer than 8 leaves
        # this step also generates child_list
        self.child_list = _partition_data(
            self.data, self.box, self.node_start, self.node_end
        )
        # Loop through children sections. If any section has more than
        # _particle_threshold child, recurse.
        # Note this step could be easily converted into a while loop with a
        # stack of subsections instead of recursion
        # Note also child_list[0] is the offset to child[1]
        # child[0] starts at node_start and ends at
        # node_start+child_list[0] -> child_list only
        # Note also that there are currently no checks to
        # ensure that everything is not just being dumped
        # into one child, causing infinite recursion
        for i in range(8):
            child1 = self.child_list[i - 1] if i > 0 else self.node_start
            child2 = self.child_list[i] if i < 7 else self.node_end
            if child2 - child1 > self._particle_threshold:
                # recursing on child i not i-1!
                child_box = _get_child_box(self.box, i)
                LOGGER.debug(
                    f"Making child box{i + 1} for {child2}-{child1}={child2 - child1} particles in box {child_box}"
                )
                node = OctreeNode(
                    data=self.data,
                    node_start=child1,
                    node_end=child2 - 1,
                    box=child_box,
                    tag=self.tag + [i + 1],  # e.g. i=1 corresponds to subtree 3
                    parent=self,
                )
                self.children.append(node)

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
    def slice(self, *, leaf_ind: None | int = None):
        """
        Return slice of data corresponding to this node
        """
        if leaf_ind is None:
            return slice(self.node_start, self.node_end + 1)
        if not isinstance(leaf_ind, int):
            raise TypeError("leaf_ind must be an int")
        child_start = self.child_list[leaf_ind] if leaf_ind > 0 else self.node_start
        child_end = self.child_list[leaf_ind + 1] if leaf_ind < 6 else self.node_end
        return slice(child_start, child_end)

    def distance(
        self, x: float, y: float, z: float, *, leaf_ind: None | int = None
    ) -> ArrayLike:
        """
        Return array of particle distances from given point
        """
        slice_ = self.slice(leaf_ind=leaf_ind)
        pos = self.data.positions
        point = np.array(x, y, z)
        return np.sqrt(np.sum((pos[slice_, 0] - point) ** 2, axis=1))


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
