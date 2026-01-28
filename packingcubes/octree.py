from __future__ import annotations

import abc
import logging
import warnings
from collections.abc import Iterable, Iterator, Sized
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numba import njit  # type: ignore
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

import packingcubes.bounding_box as bbox
from packingcubes.configuration import FIELD_FORMAT
from packingcubes.data_objects import DataContainer, Dataset

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(capture=True)

_DEFAULT_PARTICLE_THRESHOLD = 400


# Octants
class Octants(IntEnum):
    LEFTFRONTBOTTOM = 1
    RIGHTFRONTBOTTOM = 2
    LEFTBACKBOTTOM = 3
    RIGHTBACKBOTTOM = 4
    LEFTFRONTTOP = 5
    RIGHTFRONTTOP = 6
    LEFTBACKTOP = 7
    RIGHTBACKTOP = 8


class OctreeError(Exception):
    pass


class OctreeWarning(UserWarning):
    pass


class ContainmentFunc(Protocol):
    def __call__(self, xyz: NDArray) -> NDArray[np.bool_]:
        pass


# Convenience functions
@njit
def _partition(data: DataContainer, lo: int, hi: int, ax: int, pivot: float) -> int:
    """
    Partition a portion of 3D data between lo and hi along the axis ax

    Returns the (effective) pivot location. Note that the pivot is not required
    to be in the data, in which case the returned location will be lo (if all
    elements are greater than pivot) or hi+1 (if all elements are less than
    the pivot)
    """
    if lo < 0:
        raise ValueError(f"Low index out of bounds, lo={lo}")
    if lo > hi:
        # nothing to do
        return lo
    if lo == hi:
        # return early to avoid unnecessary swap
        # ternary to avoid data cast
        return lo + (1 if data.positions[lo, ax] < pivot else 0)
    if len(data) <= hi:
        # OOB upper index only an issue if we're actually going to use it
        raise ValueError(f"High index out of bounds, hi={hi}")
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


@njit
def _partition_data(
    data: DataContainer,
    box: bbox.BoundingBox,
    lo: int,
    hi: int,
) -> list[int]:
    """
    Partition 3D data from lo to hi contained in bounding box into 8 octants.

    Returns child_list of the octants starting indices. First octant assumed
    to start at lo.
    Note that if child_list[i] octant contains no data, then
    child_list[i]==child_list[i+1]. Note also that child_list[i] can be greater
    than hi. This means all data was contained in octants
    [lo,*child_list[0:i-1]]
    """
    midplane = box.midplane()
    child_list = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    # child_list is defined such that data in c0 (child node 1) is prior to
    # child_list[0], c1 data is between child_list[0] and child_list[1], etc.
    # i.e.
    # lo <= c0 < 0 <= c1 < 1 <= c2 < 2 <= c3 < 3 and
    #  3 <= c4 < 4 <= c5 < 5 <= c6 < 6 <= c7 <= hi
    # HOWEVER not all child nodes will have data.
    # That means child_list[i] is the _minimum possible_ index that could
    # contain data for that child. Thus, e.g. if data has morton coding [1,6,8]
    # child_list should be (note c0 starts at 0) [1,1,1,1,1,1,2,2]
    # z-index ordering, so box is
    #    /7-----/8
    #  5/-+---6/ |
    #  | /3---+-/4
    #  1/-----2/

    # z-partition
    # LOGGER.debug(f"Partitioning 1-8: s:{lo} e:{hi}")
    # LOGGER.debug(f"{morton(data.positions[lo : hi + 1], box)}")
    zsplit = _partition(data, lo, hi, 2, midplane[2])

    # x-partition and y-partition - note that these combine to be only two
    # passes through the list.
    # LOGGER.debug(f"Partitioning 1-4: s:{lo} e:{zsplit - 1}")
    # LOGGER.debug(f"{morton(data.positions[lo:zsplit], box)}")
    ysplit1 = _partition(data, lo, zsplit - 1, 1, midplane[1])

    # LOGGER.debug(f"Partitioning 1-2: s:{lo} e:{ysplit1 - 1}")
    # LOGGER.debug(f"{morton(data.positions[lo:ysplit1], box)}")
    xsplit1 = _partition(data, lo, ysplit1 - 1, 0, midplane[0])

    # LOGGER.debug(f"Partitioning 3-4: s:{ysplit1} e:{zsplit - 1}")
    # LOGGER.debug(f"{morton(data.positions[ysplit1:zsplit], box)}")
    xsplit2 = _partition(data, ysplit1, zsplit - 1, 0, midplane[0])

    # LOGGER.debug(f"Partitioning 5-8: s:{zsplit} e:{hi}")
    # LOGGER.debug(f"{morton(data.positions[zsplit : hi + 1], box)}")
    ysplit2 = _partition(data, zsplit, hi, 1, midplane[1])

    # LOGGER.debug(f"Partitioning 5-6: s:{zsplit} e:{ysplit2 - 1}")
    # LOGGER.debug(f"{morton(data.positions[zsplit:ysplit2], box)}")
    xsplit3 = _partition(data, zsplit, ysplit2 - 1, 0, midplane[0])

    # LOGGER.debug(f"Partitioning 7-8: s:{ysplit2} e:{hi}")
    # LOGGER.debug(f"{morton(data.positions[ysplit2 : hi + 1], box)}")
    xsplit4 = _partition(data, ysplit2, hi, 0, midplane[0])

    # LOGGER.debug(
    #     f"y1:{xsplit1} x1:{ysplit1} y2:{xsplit2} "
    #     f"z:{zsplit} y3:{xsplit3} x2:{ysplit2} y4:{xsplit4}"
    # )

    child_list[0] = lo  #       indices of child_0 (morton=1): lo, xsplit1-1
    child_list[1] = xsplit1  # indices of child_1 (morton=2): xsplit1, ysplit1-1
    child_list[2] = ysplit1  # indices of child_2 (morton=3): ysplit1, xsplit2-1
    child_list[3] = xsplit2  # indices of child_3 (morton=4): xsplit2, zsplit-1
    child_list[4] = zsplit  #  indices of child_4 (morton=5): zsplit, ysplit3-1
    child_list[5] = xsplit3  # indices of child_5 (morton=6): xsplit3, ysplit2-1
    child_list[6] = ysplit2  # indices of child_6 (morton=7): ysplit2, xsplit4-1
    child_list[7] = xsplit4  # indices of child_7 (morton=8): xsplit4, hi
    child_list[8] = hi + 1

    # LOGGER.debug(f"{lo=} {hi=} {child_list}")
    return child_list


def _box_neighbors_in_node(
    box_ind: int,
) -> list[Octants]:
    """
    Return the neighbor Octants that are in the same node.
    """
    # TODO: look into a case/control-flow free formula
    match box_ind:
        case Octants.LEFTFRONTBOTTOM:  # 1
            return [
                Octants.RIGHTFRONTBOTTOM,
                Octants.LEFTBACKBOTTOM,
                Octants.LEFTFRONTTOP,
            ]  # [2,3,5]
        case Octants.RIGHTFRONTBOTTOM:  # 2
            return [
                Octants.LEFTFRONTBOTTOM,
                Octants.RIGHTBACKBOTTOM,
                Octants.RIGHTFRONTTOP,
            ]  # [1,4,6]
        case Octants.LEFTBACKBOTTOM:  # 3
            return [
                Octants.LEFTFRONTBOTTOM,
                Octants.RIGHTBACKBOTTOM,
                Octants.LEFTBACKTOP,
            ]  # [1,4,7]
        case Octants.RIGHTBACKBOTTOM:  # 4
            return [
                Octants.RIGHTFRONTBOTTOM,
                Octants.LEFTBACKBOTTOM,
                Octants.RIGHTBACKTOP,
            ]  # [2,3,8]
        case Octants.LEFTFRONTTOP:  # 5
            return [
                Octants.LEFTFRONTBOTTOM,
                Octants.LEFTBACKTOP,
                Octants.RIGHTFRONTTOP,
            ]  # [1,6,7]
        case Octants.RIGHTFRONTTOP:  # 6
            return [
                Octants.RIGHTFRONTBOTTOM,
                Octants.LEFTFRONTTOP,
                Octants.RIGHTBACKTOP,
            ]  # [2,5,8]
        case Octants.LEFTBACKTOP:  # 7
            return [
                Octants.LEFTBACKBOTTOM,
                Octants.LEFTFRONTTOP,
                Octants.RIGHTBACKTOP,
            ]  # [3,6,8]
        case Octants.RIGHTBACKTOP:  # 8
            return [
                Octants.RIGHTBACKBOTTOM,
                Octants.RIGHTFRONTTOP,
                Octants.LEFTBACKTOP,
            ]  # [4,6,7]
        case _:
            raise ValueError(
                f"Invalid {box_ind=} specified! Valid options are in octree.Octants",
            )


def morton(positions: NDArray, box: bbox.BoxLike) -> NDArray[np.int_]:
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
    box = bbox.make_bounding_box(box)
    midplane = box.midplane()
    # We do not attempt to convert to Octants here because that would create a
    # numpy *object* array. Just leave them as ints
    return (
        1
        + (positions[:, 0] >= midplane[0])
        + 2 * (positions[:, 1] >= midplane[1])
        + 4 * (positions[:, 2] >= midplane[2])
    ).astype(int)


def full_morton(positions: NDArray, box: bbox.BoxLike) -> NDArray[np.uint64]:
    """
    Return the full morton indices of positions in box

    Computes and returns the full morton encoding of positions in the provided
    box using bit interleaving. Essentially:
    ```python
    xyz = (0b1111, 0b0101, 0b1001)
    # morton = 0bz4y4x4_z3y3x3_z2y2x2_z1y1x1
    morton = 0b101_101_001_111
    ```
    Note that the final morton code needs to fit into a uint64 (the largest
    width uint numpy/numba supports), which means we only have **21** bits to
    work with per coordinate (instead of 64). This corresponds to a dynamic
    range of ~2e6, well below what most larger simulations work with, so use
    these indices with **caution**!

    Args:
        positions: NDArray
        Array of particle positions. Note: any precision beyond 21 bits will be
        lost!

        box: BoundingBox
        Bounding box of all positions

    Returns:
        mortons: NDArray[np.uint64]
        Full morton indices of the provided positions.
    """
    box = bbox.make_bounding_box(box)

    # Normalize the coordinates to [0,1)
    normalized = box.normalize_to_box(positions).astype(np.float32)
    # Convert to [0, 2^32]
    int_pos = (normalized * 0x0000_0001_0000_0000).astype(np.uint64)
    # only allowed 21 bits, so bit shift everything 11 spots
    int_pos = int_pos >> np.uint64(11)
    # initialize morton codes
    mortons = np.zeros((len(int_pos),), dtype=np.uint64)
    # Compute morton indices using the magic bits implementation from
    # https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
    # mortons[0] = mortons[0] << 2
    for i in range(3):
        x = int_pos[:, i].astype(np.uint64)
        x = x & np.uint64(0x1FFFFF)  # this step is probably unnecessary
        # shift left 32 bits, OR with self, and with
        # 00011111000000000000000000000000000000001111111111111111
        x = (x | x << np.uint64(32)) & np.uint64(0x1F00000000FFFF)
        # shift left 16 bits, OR with self, and with
        # 00011111000000000000000011111111000000000000000011111111
        x = (x | x << np.uint64(16)) & np.uint64(0x1F0000FF0000FF)
        # shift left 8 bits, OR with self, and with
        # 0001000000001111000000001111000000001111000000001111000000000000
        x = (x | x << np.uint64(8)) & np.uint64(0x100F00F00F00F00F)
        # shift left 4 bits, OR with self, and with
        # 0001000011000011000011000011000011000011000011000011000100000000
        x = (x | x << np.uint64(4)) & np.uint64(0x10C30C30C30C30C3)
        # shift left 2 bits, OR with self, and with
        # 1001001001001001001001001001001001001001001001001001001001001
        x = (x | x << np.uint64(2)) & np.uint64(0x1249249249249249)
        mortons = mortons | (x << np.uint64(i))
    return x


@njit
def pre_sort(
    data: DataContainer, particle_threshold=_DEFAULT_PARTICLE_THRESHOLD
) -> dict:
    root_box = data.bounding_box
    partition_list = [("0", root_box, 0, len(data) - 1)]
    node_bounds = {"0": (0, len(data) - 1)}
    max_depth = root_box.max_depth()
    while partition_list:
        partition = partition_list.pop()
        tag = partition[0]
        box = partition[1]
        child_list = _partition_data(data, box, partition[2], partition[3])

        for i in range(1, 9):
            child1 = child_list[i - 1]
            child2 = child_list[i]

            if child1 >= child2:
                continue

            child_tag = tag + str(i)
            child_box = box.get_child_box(i - 1)
            node_bounds[child_tag] = (child1, child2 - 1)

            if child2 - child1 >= particle_threshold and len(child_tag) < max_depth:
                partition_list.append((child_tag, child_box, child1, child2 - 1))
    return node_bounds


class OctreeNode(Sized):
    """
    Interface for a node in an octree

    Attributes:
        node_start, node_end: int
        The start and end data indices for this node. OctreeNodes have
        node_end+1 - node_start particles

        box: bounding_box.BoundingBox
        Bounding box of the form [x, y, z, dx, dy, dz] where (x, y, z) is the
        left-front-bottom corner. All particles are assumed to lie inside.

        tag: str
        Str of 1-based z-order indices describing the current box.
        E.g. if assuming the unit bounding box, the box
        [0.25, 0.25, 0.75, 0.25, 0.25, 0.25] would be [5,1]. Root node is "0"

        children: Iterable[OctreeNode]
        Iterable of this node's children. Empty if leaf node

        parent: OctreeNode
        Parent node of this node. None if root of the entire tree.

    """

    @property
    @abc.abstractmethod
    def node_start(self) -> int:
        """
        The start data index for this node. OctreeNodes have
        node_end+1 - node_start particles
        """
        raise NotImplementedError("Defined by implemented class")

    @property
    @abc.abstractmethod
    def node_end(self) -> int:
        """
        The end data index for this node. OctreeNodes have
        node_end+1 - node_start particles
        """
        raise NotImplementedError("Defined by implemented class")

    @property
    @abc.abstractmethod
    def box(self) -> bbox.BoundingBox:
        """
        Bounding box of the form [x, y, z, dx, dy, dz] where (x, y, z) is the
        left-front-bottom corner. All particles are assumed to lie inside.
        """
        raise NotImplementedError("Defined by implemented class")

    @property
    @abc.abstractmethod
    def tag(self) -> str:
        """
        String  of 1-based z-order indices describing the current box.
        E.g. if assuming the unit bounding box, the box
        [0.25, 0.25, 0.75, 0.25, 0.25, 0.25] would be 51. Root node is "0"
        """
        raise NotImplementedError("Defined by implemented class")

    @property
    def slice(self) -> slice:
        """
        Return slice of data corresponding to this node
        """
        return slice(self.node_start, self.node_end + 1)

    def __repr__(self):
        return (
            f"{type(self).__name__} (id:{self.tag})"
            + f" from {self.node_start}-{self.node_end} ({len(self)} particles) with"
            + f" box {self.box}"
        )

    def __len__(self):
        """
        Return number of particles held by this node
        """
        return np.maximum(self.node_end - self.node_start + 1, 0)


# This will be an in-place octree, so each node has a
# view of the backing array, indices to the start of
# each subnode/leaf, box size, and node start/end
# Note that node_ind+part_num=next_sibling_ind
class PythonOctreeNode(OctreeNode):
    """
    Internal representation of a node in the octree

    Class holding the octree information in a recursive substructure. Note that
    the octree is currently fully initialized on creation of the root (node
    creation contains a recursive node creation call). Each node forms the root
    of its own octree and has either 8 children or is a leaf node.

    Attributes:
        data: DataContainer
        The backing DataContainer for the octree

        node_start, node_end: int
        The start and end data indices for this node. OctreeNodes have
        node_end+1 - node_start particles

        box: numpy.array
        Bounding box of the form [x, y, z, dx, dy, dz] where (x, y, z) is the
        left-front-bottom corner. All particles are assumed to lie inside.

        tag: str
        String of 1-based z-order indices describing the current box.
        E.g. if assuming the unit bounding box, the box
        [0.25, 0.25, 0.75, 0.25, 0.25, 0.25] would be "051". Root node is "0"

        children: List[OctreeNode]
        List of this node's children. Empty if leaf node

        parent: OctreeNode
        Parent node of this node. None if root of the entire tree.

    """

    data: DataContainer
    """ Reference to backing DataContainer of this OctreeNode """
    _node_start: int
    """ Starting index in data of this OctreeNode """
    _node_end: int
    """ 
    Ending index in data of this OctreeNode. OctreeNodes have 
    node_end+1 - node_start particles 
    """
    _box: bbox.BoundingBox  # [x,y,z,dx,dy,dz]
    """ 
    BoundingBox with field box that is a (6, ) array defined as 
    [x, y, z, dx, dy, dz] where (x,y,z) is the left-front-bottom corner 
    """
    # empty for root node, otherwise list of morton indices s.t. [7,2,1] is the
    # tag for the left-front-bottom subnode of the left-back-bottom subnode of
    # right-front-top subnode of the root
    _tag: str
    """ 
    String of morton indices describing the location of this OctreeNode in the
    overall tree. Root node is "0"
    """
    _children: list[PythonOctreeNode | None]
    """ 
    List of OctreeNode children with None as placeholders for empty children.
    Empty if leaf node.
    """
    _parent: PythonOctreeNode | None
    """ Reference to parent node or None if root """

    # The following are currently only used when packing
    if TYPE_CHECKING:
        _index: int
        """
        Index of node in packed array
        """
        _last_child: bool
        """
        Whether this node is the last child node when listed in order
        """

    def __init__(
        self,
        *,
        data: DataContainer,
        node_start: int | None = None,
        node_end: int | None = None,
        box: bbox.BoundingBox | None = None,
        tag: str | None = None,
        parent: PythonOctreeNode | None = None,
        particle_threshold: int = 1,
        pbar: tqdm | None = None,
    ) -> None:
        """
        Args:
            data: DataContainer
            The backing DataContainer for the octree

            node_start, node_end: int, optional
            The start and end data indices for this node. Defaults to 0 and
            len(data)-1

            box: BoundingBox, optional
            Bounding box of the form [x, y, z, dx, dy, dz] where (x, y, z) is the
            left-front-bottom corner. All particles are assumed to lie inside.
            Defaults to data.bounding_box

            tag: str, optional
            String of 1-based z-order indices describing the current box.
            E.g. if assuming the unit bounding box, the box
            [0.25, 0.25, 0.75, 0.25, 0.25, 0.25] would be "051". Defaults to "0"
            for the root node

            parent: OctreeNode, optional
            Parent node of this node. None (default) if root of the entire tree.

            particle_threshold: int, optional
            Configuration parameter for how many particles will be contained in a
            leaf before splitting into subtrees. Note that currently particle
            sorting stops at this level. Default 1.

            pbar: tqdm, optional
            A tqdm() instance for optional progress updates
        """
        particle_threshold = int(particle_threshold)
        if particle_threshold <= 0:
            raise OctreeError("particle_threshold must be positive!")
        self._particle_threshold = particle_threshold

        if not len(data):
            raise OctreeError("Empty DataContainer provided")
        self.data = data

        if node_start is None:
            self._node_start = 0
        elif node_start < 0:
            raise OctreeError(f"Invalid start index: {node_start}")
        else:
            self._node_start = node_start
        if node_end is None:
            self._node_end = len(data) - 1
        elif node_end > len(data) - 1:
            raise OctreeError(f"Invalid end index: {node_end}")
        else:
            self._node_end = node_end
        self._children = []
        self._box = box if box is not None else data.bounding_box
        self._tag = tag if tag is not None else "0"
        self._parent = parent

        if self._parent is None:
            # we are at the root, set max depth based on floating point
            # precision
            self._max_depth = self.box.max_depth() - 1
            if len(self) == 1:
                self._max_depth = 0
        else:
            self._max_depth = self._parent._max_depth - 1
        if self._max_depth < 0:
            raise OctreeError(
                "Attempted to create an OctreeNode with negative max depth!",
            )

        self._pbar = pbar

        # begin recursion
        self._construct()

    def _construct(self) -> None:
        """
        Main recursive call of octree construction. This is where the particles
        get partitioned and child nodes created
        """
        # sort everything, even if we have fewer than 8 leaves
        # this step also generates child_list
        child_list = _partition_data(
            self.data,
            self._box,
            self._node_start,
            self._node_end,
        )

        # Only look at children if current node has at least
        # _particle_threshold particles and we haven't hit recursion limit.
        # Ideally, we'd continue sorting until we hit 1 particle
        # but we likely can/should switch to a different sorting method, like
        # insertion sort on Morton codes or similar
        # Recursion limit is so we don't hit floating point precision errors.
        if len(self) <= self._particle_threshold or self._max_depth < 1:
            LOGGER.debug(
                "Exiting on particle threshold"
                if len(self) <= self._particle_threshold
                else f"Exiting on recursion limit {len(self.tag)}",
            )
            if len(self) > self._particle_threshold:
                warnings.warn(
                    f"Bad data detected at a depth of {len(self.tag)}.\n"
                    f"Box ({self.box}) does not support further splitting.\n"
                    "We will stop tree construction here to avoid loss of "
                    "precision",
                    OctreeWarning,
                    stacklevel=1,
                )
            if self._pbar is not None:
                self._pbar.update(len(self))
            return

        # Loop through children sections and recurse.
        # Note this step could be easily converted into a while loop with a
        # stack of subsections instead of recursion
        # Note also child_list[0] is the offset to child[1]
        # child[0] starts at node_start and ends at
        # node_start+child_list[0] -> child_list only
        children: list[PythonOctreeNode | None] = []
        for i in range(1, 9):
            child1 = child_list[i - 1]
            child2 = child_list[i]

            # append empty child as None
            if child1 >= child2:
                children.append(None)
                continue

            # recursing on child i not i-1!
            child_box = self.box.get_child_box(i - 1)
            # LOGGER.debug(
            #     f"Making child box{i + 1} for {child2}-{child1}"
            #     f"={child2 - child1} particles in box {child_box}"
            # )
            node = PythonOctreeNode(
                data=self.data,
                node_start=child1,
                node_end=child2 - 1,
                box=child_box,
                tag=self._tag + str(i),
                parent=self,
                particle_threshold=self._particle_threshold,
                pbar=self._pbar,
            )
            children.append(node)
        # Only _keep_ children if above _particle_threshold
        if len(self) > self._particle_threshold:
            self._children.extend(children)
        if self._pbar is not None and self.is_leaf:
            self._pbar.update(len(self))

    @property
    def node_start(self) -> int:
        """
        The start data index for this node. OctreeNodes have
        node_end+1 - node_start particles
        """
        return self._node_start

    @property
    def node_end(self) -> int:
        """
        The end data index for this node. OctreeNodes have
        node_end+1 - node_start particles
        """
        return self._node_end

    @property
    def box(self) -> bbox.BoundingBox:
        """
        Bounding box of the form [x, y, z, dx, dy, dz] where (x, y, z) is the
        left-front-bottom corner. All particles are assumed to lie inside.
        """
        return self._box

    @property
    def tag(self) -> str:
        """
        String of 1-based z-order indices describing the current box.
        E.g. if assuming the unit bounding box, the box
        [0.25, 0.25, 0.75, 0.25, 0.25, 0.25] would be [5,1]
        """
        return self._tag

    @property
    def children(self) -> list[PythonOctreeNode | None]:
        return self._children

    @property
    def parent(self) -> PythonOctreeNode | None:
        return self._parent

    @property
    def is_leaf(self) -> bool:
        """
        Return whether this node is a leaf
        """
        if not hasattr(self, "_is_leaf"):
            self._is_leaf = not bool(self.children)
        return self._is_leaf

    def __repr__(self):
        return (
            f"{type(self).__name__} (id:{self.tag})"
            + f" from {self.node_start}-{self.node_end} ({len(self)} particles) with"
            + f" box {self.box}"
            + f" and {len([c for c in self.children if c is not None])} child node"
            + f"{'' if len(self.children) == 1 else 's'}"
        )

    def distance(
        self,
        xyz: NDArray,
    ) -> NDArray[np.float64]:
        """
        Return array of particle distances from given point
        """
        pos = self.data.positions
        return np.sqrt(np.sum((pos[self.slice, :] - xyz) ** 2, axis=1))

    def closest_particle_index(self, xyz: NDArray) -> tuple[np.int_, float]:
        """
        Return index (and distance) of closest particle
        """
        distances = self.distance(xyz)
        ind = np.argmin(distances)
        return ind + self.node_start, distances[ind]


def _top_down_containing_node(
    node: PythonOctreeNode,
    xyz: NDArray,
) -> PythonOctreeNode | None:
    """
    For a given point, return the smallest child node that contains point or None
    """
    while len(node) > 1:
        # find leaf or smallest subtree containing point
        for child in node.children:
            if child is not None and child.box.contains(xyz):
                node = child
                break  # for child loop
        else:
            break  # while loop
    return node if node.box.contains(xyz) else None


def _bottom_up_containing_node(node: PythonOctreeNode, xyz: NDArray):
    """
    For a given point, return the smallest parent node that contains point or None
    """
    while node.parent is not None:
        if node.box.contains(xyz):
            return node
        node = node.parent
    return node if node.box.contains(xyz) else None


@njit
def _point_in_sphere(
    point: NDArray, center: NDArray, radius: float
) -> NDArray[np.bool_]:
    """
    Return true if point is closer than (<=) radius to center. Vectorizable
    """
    # We don't need to calculate the actual distance, we only need to compare dist^2
    dist2 = np.sum(np.atleast_2d((point - center) ** 2), axis=1)
    return dist2 <= radius**2


class Octree(Iterable[OctreeNode], Protocol):
    """
    Public octree interface

    This protocol defines the methods for manipulating and traversing a
    packingcubes octree.

    """

    def __iter__(self) -> Iterator[OctreeNode]:
        """
        Iterate through all nodes of the octree. Note that no guarantee is made
        of what order the nodes are traversed in
        """
        ...

    def get_leaves(self) -> Iterable[OctreeNode]: ...
    def get_node(self, tag: str) -> OctreeNode | None: ...

    def get_particle_indices_in_box(
        self,
        *,
        box: bbox.BoxLike,
    ) -> list[tuple[int, int]]:
        pass

    def get_particle_indices_in_sphere(
        self,
        *,
        center: NDArray,
        radius: float,
    ) -> list[tuple[int, int]]:
        pass

    def get_closest_particle(
        self, xyz: ArrayLike, *, check_neighbors: bool = True
    ) -> tuple[np.int_, float]:
        pass


class PythonOctree(Octree):
    """
    Public octree class

    Use this class for creating, manipulating, and traversing an octree based
    on a Dataset object. The octree will be composed of OctreeNodes and is
    currently computed in a top-down approach

    Attributes:
        root: OctreeNode
        The root node of the octree

    """

    root: PythonOctreeNode
    """The root node of the octree"""

    def __init__(
        self,
        dataset: Dataset,
        *,
        particle_threshold: int | None = None,
        show_pbar: bool = False,
    ) -> None:
        """
        Args:
            dataset: Dataset
            A Dataset containing particle data

            particle_threshold: int, optional
            Number of particles allowed in a leaf before splitting. Defaults to
            _DEFAULT_PARTICLE_THRESHOLD

            show_pbar: bool, optional
            Show a progress bar during tree construction
        """
        if particle_threshold is None:
            particle_threshold = _DEFAULT_PARTICLE_THRESHOLD

        pbar = None
        if show_pbar:
            pbar = tqdm(total=len(dataset), miniters=1000)

        data = dataset.data_container

        self.root = PythonOctreeNode(
            data=data,
            box=data.bounding_box,
            particle_threshold=particle_threshold,
            pbar=pbar,
        )

        if pbar is not None:
            pbar.close()

    def get_leaves(self) -> list[PythonOctreeNode]:
        """
        Return a list of all leaf octree nodes in depth-first order
        """
        if not hasattr(self, "_leaves"):
            leaves: list[PythonOctreeNode] = []
            nodes: list[PythonOctreeNode | None] = [self.root]
            while nodes:
                node = nodes.pop()
                if node is None:
                    continue
                if node.is_leaf:
                    leaves.append(node)
                else:
                    nodes.extend(reversed(node.children))
            self._leaves = leaves
        return self._leaves

    def get_node(self, tag: str) -> PythonOctreeNode | None:
        """
        Return the node corresponding to the provided tag or None if not found

        Args:
            tag: str
            The tag to search for

        Returns:
            node
            Node in octree with specified tag or None if it does not exist
        """
        if not tag or tag == "0":
            return self.root

        node = self.root
        for t in tag[1:]:
            node = node.children[int(t - 1)]  # type: ignore
            if node is None:
                return None

        return node

    def __iter__(self) -> Iterator[PythonOctreeNode]:
        """
        Return all nodes as pre-order tree traversal
        """
        nodes = [self.root]
        while nodes:
            node = nodes.pop()
            yield node
            if not node.is_leaf:
                nodes.extend(filter(None, reversed(node.children)))

    def _get_containing_node_of_point(
        self,
        xyz: NDArray,
        *,
        start_node: PythonOctreeNode | None = None,
        top_down: bool = True,
    ) -> PythonOctreeNode | None:
        """
        Return smallest node containing point

        Defaults to a top-down approach from root. Can provide a start_node to
        short-cut search. Can also go bottom-up; requires start_node.
        """
        if not top_down and start_node is None:
            raise ValueError("start_node **must** be provided for bottom-up traversal!")
        node = self.root if start_node is None else start_node
        if top_down:
            return _top_down_containing_node(node, xyz)

        # find first parent that contains point, then see if parent
        # can be refined
        node = _bottom_up_containing_node(node, xyz)
        if node is not None:
            return _top_down_containing_node(node, xyz)
        return None

    def _get_containing_node_of_pointlist(
        self,
        points: NDArray,
        *,
        start_node: PythonOctreeNode | None = None,
        top_down: bool = True,
    ) -> PythonOctreeNode:
        """
        Return smallest node containing all points in array or root

        Defaults to a top-down approach from root. Can provide a start_node to
        short-cut search. Can also go bottom-up; requires start_node.
        """
        # Basic approach: find containing node for first point on list, then
        # for each remaining point, traverse up parents until point is
        # contained
        # This is equivalent to computing bounding-OctreeNode (like a
        # bounding-box but only octree-aligned boxes are allowed)
        node = None
        for point in points:
            if node is None:
                node = self._get_containing_node_of_point(
                    point,
                    start_node=start_node,
                    top_down=top_down,
                )
            else:
                node = _bottom_up_containing_node(node, point)
        return node if node is not None else self.root

    def _get_nodes_in_shape(
        self,
        *,
        bounding_box: bbox.BoundingBox,
        containment_obj: bbox.BoundingVolume | None = None,
    ) -> tuple[list[PythonOctreeNode], list[PythonOctreeNode]]:
        """
        Return lists of all nodes entirely inside and partially inside shape

        Both node lists will be in z-index order, but are likely to be
        interleaved, i.e.
        entirely_in[i].z_order
            < partial_leaves[j].z_order
                < entirely_in[i + 1].z_order
                    < partial_leaves[j + 1].z_order

        Args:
            bounding_box: BoundingBox
            Shape bounding box

            containment_test: callable, optional
            Function to test if point(s) are inside shape. Should have the
            signature
            containment_test(point: ArrayLike) -> NDArray[bool]
            Defaults to testing if point(s) are inside the provided bounding
            box

        Returns:
            entirely_in: List[OctreeNode]
            List of nodes that are entirely within shape. Nodes may be internal nodes

            partial_leaves: List[OctreeNode]
            List of leaf nodes that are only partially within shape.

        """
        if containment_obj is None:
            containment_obj = bounding_box

        # node containing bounding box
        bbox_vertices = bounding_box.get_box_vertices()
        bbox_center = bounding_box.get_box_center()
        start_node = self._get_containing_node_of_pointlist(bbox_vertices)

        # depth-first traversal
        # shouldn't be any difference in performance between breadth-first
        # and depth-first (assuming serial), but this way the nodes will be
        # returned in z-order
        entire_nodes = []
        partial_leaves = []
        child_queue = list(reversed(start_node.children))
        while len(child_queue):
            node = child_queue.pop()

            if node is None:
                continue

            # Test if node entirely contained in shape
            node_vertices = node.box.get_box_vertices()

            vertices_enclosed = sum(containment_obj.contains(node_vertices))

            if vertices_enclosed:
                if vertices_enclosed == len(node_vertices):
                    entire_nodes.append(node)
                elif node.is_leaf:
                    partial_leaves.append(node)
                else:
                    # need to reverse input for depth-first search
                    child_queue.extend(reversed(node.children))
                continue

            # Also need to check closest point. Should take care of overlapping
            # edges
            closest_point = node.box.project_point_on_box(bbox_center)
            if containment_obj.contains(closest_point):
                if node.is_leaf:
                    partial_leaves.append(node)
                else:
                    child_queue.extend(reversed(node.children))

        return entire_nodes, partial_leaves

    def _get_nodes_in_sphere(
        self,
        *,
        center: NDArray,
        radius: float,
    ) -> tuple[list[PythonOctreeNode], list[PythonOctreeNode]]:
        """
        Return lists of all nodes entirely inside and nodes partially inside sphere

        Calls _get_nodes_in_shape with sphere's bounding box and
        _point_in_sphere as the containment_test

        Args:
            center: NDArray
            Coordinates of sphere center

            radius: float
            Sphere radius

        Returns:
            entirely_in: List[OctreeNode]
            List of nodes that are entirely within shape. Nodes may be internal nodes

            partial_leaves: List[OctreeNode]
            List of leaf nodes that are only partially within shape.

        Raises:
            IndexError
            When there are unexpected issues with the queue system.
        """

        if len(center) != 3:
            raise ValueError("Center should be a 3 element array")

        sph = bbox.BoundingSphere(center, radius)

        return self._get_nodes_in_shape(
            bounding_box=sph.bounding_box,
            containment_obj=sph,
        )

    def _get_particle_indices_in_shape(
        self,
        *,
        bounding_box: bbox.BoundingBox,
        containment_obj: bbox.BoundingVolume | None = None,
    ) -> list[tuple[int, int]]:
        """
        Return all particles contained within a shape that fits inside bounding box

        Args:
            bounding_box: BoundingBox
            Shape bounding box

            containment_test: Callable[NDArray], optional
            Function to test if point(s) are inside shape. Should have the
            (vectorized) signature
            containment_test(point: ArrayLike) -> NDArray[bool]
            Defaults to testing if point(s) are inside the provided bounding
            box

        Returns:
            indices: list[tuple[int, int]]
            List of particle start-stop indices contained within shape
        """

        entirely_in, partial_leaves = self._get_nodes_in_shape(
            bounding_box=bounding_box,
            containment_obj=containment_obj,
        )
        # reversed stack is a queue if you're not adding anything new and node
        # lists should be much shorter than final particle index list
        entirely_in.reverse()
        partial_leaves.reverse()

        # initialize with empty index list so hstack doesn't complain
        indices = []
        while entirely_in or partial_leaves:
            if entirely_in and partial_leaves:
                node, is_full = (
                    (entirely_in.pop(), True)
                    if entirely_in[0].node_start < partial_leaves[0].node_start
                    else (partial_leaves.pop(), False)
                )
            elif entirely_in:
                node, is_full = entirely_in.pop(), True
            else:
                node, is_full = partial_leaves.pop(), False

            indices.append((node.node_start, node.node_end + 1))

        return indices

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

        return self._get_particle_indices_in_shape(
            bounding_box=bounding_box,
        )

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

        if len(center) != 3:
            raise ValueError("Center should be a 3 element array")

        sph = bbox.make_bounding_sphere(center=center, radius=radius)

        return self._get_particle_indices_in_shape(
            bounding_box=sph.bounding_box,
            containment_obj=sph,
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
        xyz = np.atleast_1d(xyz)

        # ensure point is in octree, project if not
        if not self.root.box.contains(xyz):
            # Project point onto root
            xyz = self.root.box.project_point_on_box(xyz)

        node = self._get_containing_node_of_point(xyz)
        node = node if node is not None else self.root

        # get closest particle in that box
        closest_ind, in_box_dist = node.closest_particle_index(xyz)

        def _distance(xyz: NDArray, pxyz: NDArray) -> NDArray:
            return np.sqrt(np.sum(np.atleast_2d((xyz - pxyz) ** 2), axis=1))

        closest_dist = in_box_dist

        # Need to check all nodes in neighborhood, which is all nodes
        # overlapping with the sphere with same radius as the closest distance
        entirely_in, partial_leaves = self._get_nodes_in_sphere(
            center=xyz,
            radius=closest_dist,
        )
        neighbors = entirely_in + partial_leaves
        for neighbor in neighbors:
            pxyz = neighbor.box.project_point_on_box(xyz)
            neigh_proj_dist = _distance(xyz, pxyz)
            # Only care about neighbor if neighbor box could contain a closer
            # point than what we've found so far
            if neigh_proj_dist < closest_dist:
                # check all particles in neighbor box
                # TODO: convert non-leaf boxes to leaves
                neighbor_closest_ind, neighbor_box_dist = (
                    neighbor.closest_particle_index(xyz)
                )
                # neighbor particle is closer than current best, switch
                if neighbor_box_dist < closest_dist:
                    closest_ind, closest_dist = neighbor_closest_ind, neighbor_box_dist

        return closest_ind, closest_dist

    def to_packed(self) -> bytes:
        """
        Convert to a packed bytestream as described in [Packed Format](packed_format)

        Returns:
            packed: bytes
            Array of FIELD_FORMATs describing this tree
        """
        num_nodes = sum(1 for n in self)
        # print(f"Packing {num_nodes} into bytes")
        packed = memoryview(bytearray(num_nodes * 5 * 4)).cast(FIELD_FORMAT)
        nodes: list[PythonOctreeNode] = [self.root]
        current = 0
        while nodes:
            node = nodes.pop()
            node._index = current

            # we know each node is at least 20 bytes = 5 fields long
            packed[current] = 5
            packed[current + 1] = node.node_start
            packed[current + 2] = node.node_end

            # pack metadata
            children = node.children
            child_flag = 0
            last_child = 0
            for i, child in enumerate(children):
                if child:
                    child_flag += 1 << i
                    last_child = i
            my_index = int(node.tag[-1]) if node.tag else 0
            level = len(node.tag)
            metadata = pack_node_metadata(child_flag, my_index, level, 0)
            packed[current + 3] = metadata

            # increment current position
            current += 4

            # add parent_offset
            if node.is_leaf:
                packed[current] = node._index - (
                    node.parent._index if node.parent else 0
                )
                current += 1
                # if last leaf among siblings, track up until no longer last
                # sibling, appending parent_offset and updating length for
                # each node, since we now know how long the node is
                while node.parent and getattr(node, "_last_child", False):
                    node = node.parent
                    # set parent_offset
                    packed[current] = node._index - (
                        node.parent._index if node.parent else 0
                    )
                    current += 1
                    # update node length
                    packed[node._index] += current - node._index - 5
            else:
                # Set flag on last child to update tree and look at children
                children[last_child]._last_child = True  # type: ignore[union-attr]
                nodes.extend(filter(None, reversed(children)))
        return packed.tobytes()


@njit
def unpack_node_metadata(
    metadata: int,
) -> tuple[int, int, int, int]:
    """
    Unpack a node metadata field into child_flag, my_index, level, and empty
    """
    return (
        metadata >> 24,
        (metadata >> 16) & 255,
        (metadata >> 8) & 255,
        metadata & 255,
    )


@njit
def pack_node_metadata(child_flag: int, my_index: int, level: int, empty: int) -> int:
    """
    Pack child_flag, my_index, level, and empty ints into a metadata int
    """
    return (child_flag << 24) + (my_index << 16) + (level << 8) + empty
