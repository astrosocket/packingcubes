from __future__ import annotations

import logging
import warnings
from enum import IntEnum
from functools import partial
from typing import List

import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

import packingcubes.bounding_box as bbox
from packingcubes.data_objects import Dataset

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(True)

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
    if lo > hi:
        # nothing to do
        return lo
    if lo == hi:
        # return early to avoid unnecessary swap
        # ternary to avoid data cast
        return lo + (1 if data.positions[lo, ax] < pivot else 0)
    if len(data) <= hi:
        # OOB upper index only an issue if we're actually going to use it
        raise ValueError(f"High index out of bounds {hi=}")
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
    #     f"y1:{xsplit1} x1:{ysplit1} y2:{xsplit2} z:{zsplit} y3:{xsplit3} x2:{ysplit2} y4:{xsplit4}"
    # )

    # child_list[-1] = lo #    indices of child_0 (morton=1): lo, xsplit1-1
    child_list[0] = xsplit1  # indices of child_1 (morton=2): xsplit1, ysplit1-1
    child_list[1] = ysplit1  # indices of child_2 (morton=3): ysplit1, xsplit2-1
    child_list[2] = xsplit2  # indices of child_3 (morton=4): xsplit2, zsplit-1
    child_list[3] = zsplit  #  indices of child_4 (morton=5): zsplit, ysplit3-1
    child_list[4] = xsplit3  # indices of child_5 (morton=6): xsplit3, ysplit2-1
    child_list[5] = ysplit2  # indices of child_6 (morton=7): ysplit2, xsplit4-1
    child_list[6] = xsplit4  # indices of child_7 (morton=8): xsplit4, hi

    # LOGGER.debug(f"{lo=} {hi=} {child_list}")
    return child_list


def _box_neighbors_in_node(
    box_ind: int,
) -> ArrayLike[Octants]:
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
                f"Invalid {box_ind=} specified! Valid options are in octree.Octants"
            )


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
    # We do not attempt to convert to Octants here because that would create a
    # numpy *object* array. Just leave them as ints
    return morton


def _convert_list_to_tag_str(tag: List[Octants]) -> str:
    """
    Convert list of Octants to a str
    """
    return "".join([f"{t.value}" for t in tag])


def _convert_tag_str_to_list(tag_str: str) -> List[Octants]:
    """
    Convert string that looks like a list of Octants into one
    """
    return [Octants(int(t)) for t in tag_str]


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
    of its own octree and has either 8 children or is a leaf node.

    Attributes:
        data: Dataset
        The backing Dataset for the octree

        node_start, node_end: int
        The start and end data indices for this node. OctreeNodes have
        node_end+1 - node_start particles

        box: numpy.array
        Bounding box of the form [x, y, z, dx, dy, dz] where (x, y, z) is the
        left-front-bottom corner. All particles are assumed to lie inside.

        tag: List[int]
        List of 1-based z-order indices describing the current box.
        E.g. if assuming the unit bounding box, the box
        [0.25, 0.25, 0.75, 0.25, 0.25, 0.25] would be [5,1]

        children: List[OctreeNode]
        List of this node's children. Empty if leaf node

        parent: None | OctreeNode
        Parent node of this node. None if root of the entire tree.

    """

    data: Dataset
    """ Reference to backing dataset of this OctreeNode """
    node_start: int
    """ Starting index in data of this OctreeNode """
    node_end: int
    """ Ending index in data of this OctreeNode. OctreeNodes have node_end+1 - node_start particles """
    box: bbox.BoundingBox  # [x,y,z,dx,dy,dz]
    """ BoundingBox with field box that is a (6, ) array defined as [x, y, z, dx, dy, dz] where (x,y,z) is the left-front-bottom corner """
    # empty for root node, otherwise list of morton indices s.t. [7,2,1] is the
    # tag for the left-front-bottom subnode of the left-back-bottom subnode of
    # right-front-top subnode of the root
    tag: List[Octants]
    """ List of morton indices describing the location of this OctreeNode in the overall tree """
    children: List[None | OctreeNode]
    """ List of OctreeNode children with None as placeholders for empty children. Empty if leaf node """
    parent: None | OctreeNode
    """ Reference to parent node or None if root """

    def __init__(
        self,
        *,
        data: Dataset,
        node_start: int = None,
        node_end: int = None,
        box: ArrayLike = None,
        tag: List[Octants] = None,
        parent: None | OctreeNode = None,
        particle_threshold: int = 1,
        pbar: tqdm.tqdm = None,
    ) -> None:
        """
        Args:
            data: Dataset
            The backing Dataset for the octree

            node_start, node_end: int, optional
            The start and end data indices for this node. Defaults to 0 and
            len(data)-1

            box: ArrayLike, optional
            Bounding box of the form [x, y, z, dx, dy, dz] where (x, y, z) is the
            left-front-bottom corner. All particles are assumed to lie inside.
            Defaults to data.bounding_box

            tag: List[int], optional
            List of 1-based z-order indices describing the current box.
            E.g. if assuming the unit bounding box, the box
            [0.25, 0.25, 0.75, 0.25, 0.25, 0.25] would be [5,1]. Defaults to
            the empty list

            parent: None | OctreeNode, optional
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
            raise OctreeError("Empty dataset provided")
        self.data = data

        if node_start is None:
            self.node_start = 0
        elif node_start < 0:
            raise OctreeError(f"Invalid start index: {node_start}")
        else:
            self.node_start = node_start
        if node_end is None:
            self.node_end = len(data) - 1
        elif node_end > len(data) - 1:
            raise OctreeError(f"Invalid end index: {node_end}")
        else:
            self.node_end = node_end
        self.children = []
        self.box = box if box is not None else data.bounding_box
        self.box = bbox.make_valid(self.box)
        self.tag = tag if tag is not None else []
        self.parent = parent

        if self.parent is None:
            # we are at the root, set max depth based on floating point
            # precision
            self._max_depth = bbox.max_depth(self.box) - 1
            if len(self) == 1:
                self._max_depth = 0
        else:
            self._max_depth = self.parent._max_depth - 1
        if self._max_depth < 0:
            raise OctreeError(
                "Attempted to create an OctreeNode with negative max depth!"
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
            self.box,
            self.node_start,
            self.node_end,
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
                else f"Exiting on recursion limit {len(self.tag)}"
            )
            if len(self) > self._particle_threshold:
                warnings.warn(
                    f"Bad data detected at a depth of {len(self.tag)}.\n"
                    f"Box ({self.box}) does not support further splitting.\n"
                    "We will stop tree construction here to avoid loss of "
                    "precision",
                    OctreeWarning,
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
        children = []
        for i in range(8):
            child1 = child_list[i - 1] if i > 0 else self.node_start
            # need node_end + 1 because we have child2 - 1 below
            child2 = child_list[i] if i < 7 else self.node_end + 1

            # append empty child as None
            if child1 >= child2:
                children.append(None)
                continue

            # recursing on child i not i-1!
            child_box = bbox.get_child_box(self.box, i)
            # LOGGER.debug(
            #     f"Making child box{i + 1} for {child2}-{child1}"
            #     f"={child2 - child1} particles in box {child_box}"
            # )
            node = OctreeNode(
                data=self.data,
                node_start=child1,
                node_end=child2 - 1,
                box=child_box,
                tag=self.tag + [Octants(i + 1)],  # e.g. i=1 corresponds to subtree 3
                parent=self,
                particle_threshold=self._particle_threshold,
                pbar=self._pbar,
            )
            children.append(node)
        # Only _keep_ children if above _particle_threshold
        if len(self) > self._particle_threshold:
            self.children.extend(children)
        if self._pbar is not None and self.is_leaf:
            self._pbar.update(len(self))

    def __repr__(self):
        return (
            f"OctreeNode (id:{_convert_list_to_tag_str(self.tag)}) from {self.node_start}-{self.node_end} ({len(self)} particles) with"
            + f" box {self.box} and {len([c for c in self.children if c is not None])} child node"
            + f"{'' if len(self.children) == 1 else 's'}"
        )

    def __len__(self):
        """
        Return number of particles held by this node
        """
        return np.maximum(self.node_end - self.node_start + 1, 0)

    @property
    def is_leaf(self) -> bool:
        """
        Return whether this node is a leaf
        """
        return not bool(len(self.children))

    @property
    def slice(self):
        """
        Return slice of data corresponding to this node
        """
        return slice(self.node_start, self.node_end + 1)

    def distance(
        self,
        x: float,
        y: float,
        z: float,
    ) -> ArrayLike:
        """
        Return array of particle distances from given point
        """
        slice_ = self.slice()
        pos = self.data.positions
        point = np.array(x, y, z)
        return np.sqrt(np.sum((pos[slice_, 0] - point) ** 2, axis=1))

    def closest_particle_index(self, x: float, y: float, z: float) -> tuple[int, float]:
        """
        Return index (and distance) of closest particle
        """
        distances = self.distance(x, y, z)
        ind = np.argmin(distances)
        return ind + self.node_start, distances[ind]


def _top_down_containing_node(
    node: OctreeNode,
    xyz: ArrayLike,
):
    """
    For a given point, return the smallest child node that contains point or None if none do
    """
    while len(node) > 1:
        # find leaf or smallest subtree containing point
        for child in node.children:
            if child is not None and bbox.in_box(child.box, xyz):
                node = child
                break  # for child loop
        else:
            break  # while loop
    return node if bbox.in_box(node.box, xyz) else None


def _bottom_up_containing_node(node: OctreeNode, xyz: ArrayLike):
    """
    For a given point, return the smallest parent node that contains point or None if none do
    """
    while node.parent is not None:
        if bbox.in_box(node.box, xyz):
            return node
        node = node.parent
    return node if bbox.in_box(node.box, xyz) else None


def _point_in_sphere(point: ArrayLike, center: ArrayLike, radius: float) -> bool:
    """
    Return true if point is closer than (<=) radius to center. Vectorizable
    """
    # We don't need to calculate the actual distance, we only need to compare dist^2
    dist2 = np.sum(np.atleast_2d((point - center) ** 2), axis=1)
    return dist2 <= radius**2


class Octree:
    """
    Public octree class

    Use this class for creating, manipulating, and traversing an octree based
    on a Dataset object. The octree will be composed of OctreeNodes and is
    currently computed in a top-down approach

    Attributes:
        root: OctreeNode
        The root node of the octree

    """

    root: OctreeNode
    """The root node of the octree"""

    def __init__(
        self, data: Dataset, *, particle_threshold: int = None, show_pbar: bool = False
    ) -> None:
        """
        Args:
            data: Dataset
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
            pbar = tqdm(total=len(data), miniters=1000)

        self.root = OctreeNode(
            data=data,
            box=data.bounding_box,
            particle_threshold=particle_threshold,
            pbar=pbar,
        )

        if show_pbar:
            pbar.close()

    def get_leaves(self):
        """
        Return a list of all leaf octree nodes in depth-first order
        """
        if hasattr(self, "_leaves"):
            return self._leaves
        leaves = []
        nodes = [self.root]
        while nodes:
            node = nodes.pop()
            if node is None:
                continue
            if node.is_leaf:
                leaves.append(node)
            else:
                nodes.extend(reversed(node.children))
        self._leaves = leaves
        return leaves

    def get_node(self, tag: str | List[Octants]) -> OctreeNode:
        """
        Return the node corresponding to the provided tag or None if not found

        Args:
            tag: str | List[Octants]
            The tag to search for

        Returns:
            node | None
            Node in octree with specified tag or None if it does not exist
        """
        if isinstance(tag, str):
            tag = _convert_tag_str_to_list(tag)

        if not tag:
            return self.root

        node = self.root
        for t in tag:
            node = node.children[t]
            if node is None:
                return None

        return node

    def __iter__(self):
        """
        Return all nodes as pre-order tree traversal
        """
        nodes = [self.root]
        while nodes:
            node = nodes.pop()
            if node is None:
                continue
            yield node
            if not node.is_leaf:
                nodes.extend(reversed(node.children))

    def get_containing_node_of_point(
        self,
        xyz: ArrayLike,
        *,
        start_node: None | OctreeNode = None,
        top_down: bool = True,
    ) -> OctreeNode:
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
        else:
            # find first parent that contains point, then see if parent
            # can be refined
            node = _bottom_up_containing_node(node, xyz)
            if node is not None:
                return _top_down_containing_node(node, xyz)
            return None

    def get_containing_node_of_pointlist(
        self,
        points: ArrayLike,
        *,
        start_node: None | OctreeNode = None,
        top_down: bool = True,
    ) -> OctreeNode:
        """
        Return smallest node containing all points in array

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
                node = self.get_containing_node_of_point(
                    point, start_node=start_node, top_down=top_down
                )
            else:
                node = _bottom_up_containing_node(node, point)
        return node

    def _get_nodes_in_shape(
        self, *, bounding_box: bbox.BoxLike, containment_test: callable = None
    ) -> tuple[List[OctreeNode], List[OctreeNode]]:
        """
        Return list of all nodes entirely inside shape and all nodes partially inside shape

        Both node lists will be in z-index order, but are likely to be
        interleaved, i.e.
        entirely_in[i].z_order
            < partial_leaves[j].z_order
                < entirely_in[i + 1].z_order
                    < partial_leaves[j + 1].z_order

        Args:
            bounding_box: BoxLike
            Shape bounding box

            containment_test: callable, optional
            Function to test if point(s) are inside shape. Should have the
            signature
            containment_test(point: ArrayLike[float]) -> ArrayLike[bool]
            Defaults to testing if point(s) are inside the provided bounding
            box

        Returns:
            entirely_in: List[OctreeNode]
            List of nodes that are entirely within shape. Nodes may be internal nodes

            partial_leaves: List[OctreeNode]
            List of leaf nodes that are only partially within shape.

        Raises:
            IndexError:
            When there are unexpected issues with the queue system.
        """
        if containment_test is None:
            containment_test = partial(bbox.in_box, box=bounding_box)

        # node containing bounding box
        bbox_vertices = bbox.get_box_vertices(bounding_box)
        bbox_center = bbox.get_box_center(bounding_box)
        start_node = self.get_containing_node_of_pointlist(bbox_vertices)

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
            node_vertices = bbox.get_box_vertices(node.box)
            # raise NotImplementedError("Need to exclude nodes that are entirely outside")

            vertices_enclosed = sum(containment_test(node_vertices))

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
            closest_point = bbox.project_point_on_box(node.box, bbox_center)
            if containment_test(closest_point):
                if node.is_leaf:
                    partial_leaves.append(node)
                else:
                    child_queue.extend(reversed(node.children))

        return entire_nodes, partial_leaves

    def _get_nodes_in_sphere(
        self, *, center: ArrayLike[float], radius: float
    ) -> tuple[List[OctreeNode], List[OctreeNode]]:
        """
        Return list of all nodes entirely inside sphere and all nodes partially inside sphere

        Calls _get_nodes_in_shape with sphere's bounding box and
        _point_in_sphere as the containment_test

        Args:
            center: ArrayLike
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

        # sphere bounding box
        bounding_box = bbox.BoundingBox(
            np.array(
                [
                    center[0] - radius,
                    center[1] - radius,
                    center[2] - radius,
                    2 * radius,
                    2 * radius,
                    2 * radius,
                ]
            )
        )

        containment_test = partial(_point_in_sphere, center=center, radius=radius)

        return self._get_nodes_in_shape(
            bounding_box=bounding_box, containment_test=containment_test
        )

    def _get_particle_indices_in_shape(
        self,
        *,
        bounding_box: bbox.BoxLike,
        containment_test: callable = None,
        strict: bool = False,
    ) -> ArrayLike:
        """
        Return all particles contained within a shape that fits inside bounding box

        Args:
            bounding_box: BoxLike
            Shape bounding box

            containment_test: callable, optional
            Function to test if point(s) are inside shape. Should have the
            (vectorized) signature
            containment_test(point: ArrayLike[float]) -> ArrayLike[bool]
            Defaults to testing if point(s) are inside the provided bounding
            box

            strict: bool, optional
            Flag describing whether each particle in a partially overlapping
            node should undergo containment_test (strict=True). Setting to
            False allows indices to include particles outside (but "nearby")
            shape. Defaults to False

        Returns:
            indices: ArrayLike
            Array of particle indices contained within shape
        """
        entirely_in, partial_leaves = self._get_nodes_in_shape(
            bounding_box=bounding_box, containment_test=containment_test
        )
        # reversed stack is a queue if you're not adding anything new and node
        # lists should be much shorter than final particle index list
        entirely_in.reverse()
        partial_leaves.reverse()

        # initialize with empty index list so hstack doesn't complain
        indices = [np.array([], dtype=int)]
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

            node_indices = np.arange(node.node_start, node.node_end + 1, dtype=int)

            if is_full or not strict:
                indices.append(node_indices)
            else:
                positions = node.data.positions[node_indices]
                mask = containment_test(positions)
                indices.append(node_indices[mask])

        # indices is now a list of numpy index arrays. Stack'em
        # Note that all elements should be unique due to octree
        # construction
        return np.hstack(indices)

    def get_particle_indices_in_box(
        self, *, box: ArrayLike, strict: bool = False
    ) -> ArrayLike:
        """
        Return all particles contained within the box

        Args:
            box: ArrayLike
            Box to check

            strict: bool, optional
            Flag describing whether each particle in a partially overlapping
            node should be tested for being inside the box. Setting to
            False allows indices to include particles outside (but "nearby")
            box. Defaults to False

        Returns:
            indices: ArrayLike
            Array of particle indices contained within sphere
        """
        bounding_box = bbox.BoundingBox(box.copy())

        return self._get_particle_indices_in_shape(
            bounding_box=bounding_box, strict=strict
        )

    def get_particle_indices_in_sphere(
        self, *, center: ArrayLike, radius: float, strict: bool = False
    ) -> ArrayLike:
        """
        Return all particles contained within the sphere defined by center and radius

        Args:
            center: ArrayLike
            Center point of the sphere

            radius: float
            Radius of the sphere

            strict: bool, optional
            Flag describing whether each particle in a partially overlapping
            node should be tested for sphere containment. Setting to
            False allows indices to include particles outside (but "nearby")
            sphere. Defaults to False

        Returns:
            indices: ArrayLike
            Array of particle indices contained within sphere
        """

        if len(center) != 3:
            raise ValueError("Center should be a 3 element array")

        # sphere bounding box
        bounding_box = bbox.BoundingBox(
            np.array(
                [
                    center[0] - radius,
                    center[1] - radius,
                    center[2] - radius,
                    2 * radius,
                    2 * radius,
                    2 * radius,
                ]
            )
        )

        containment_test = partial(_point_in_sphere, center=center, radius=radius)

        return self._get_particle_indices_in_shape(
            bounding_box=bounding_box, containment_test=containment_test, strict=strict
        )

    def get_closest_particle(self, xyz: ArrayLike, *, check_neighbors=True) -> int:
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
        # ensure point is in octree, project if not
        if not bbox.in_box(self.root.box, xyz):
            # Project point onto root
            xyz = bbox.project_point_on_box(self.root.box, xyz)

        node = self.get_containing_node_of_point(xyz)

        # get closest particle in that box
        closest_ind, in_box_dist = node.closest_particle_index(xyz)

        def _distance(xyz: ArrayLike, pxyz: ArrayLike):
            return np.sqrt(np.sum((xyz - pxyz) ** 2, axis=1))

        closest_dist = in_box_dist

        # Need to check all nodes in neighborhood, which is all nodes
        # overlapping with the sphere with same radius as the closest distance
        entirely_in, partial_leaves = self._get_nodes_in_sphere(
            xyz, radius=closest_dist
        )
        neighbors = entirely_in + partial_leaves
        for neighbor in neighbors:
            pxyz = bbox.project_point_on_box(neighbor.box, xyz)
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
