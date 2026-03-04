from __future__ import annotations

import logging

import numpy as np
from numba import (  # type: ignore
    TypingError,
    njit,
    types,
    uint8,
    uint32,
)
from numba.experimental import jitclass
from numba.extending import as_numba_type
from numba.typed import List
from numba.types import ListType, string

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(capture=True)


@jitclass(
    [
        ("node_start", uint32),
        ("node_end", uint32),
        ("box", bbox.bbn_type),
        ("tag", string),
        ("index", uint32),
        ("is_leaf", types.boolean),
    ]
)
class PackedNodeNumba:
    """
    Private representation of a node in a packed tree

    This class is what should be emitted by PackedTreeNumba methods that return
    nodes, rather than CurrentNode instances.

    Note that this only represents the node at time of creation. Changes do
    not propagate in either direction.
    """

    def __init__(
        self,
        node_start: int,
        node_end: int,
        box: bbox.BoundingBox,
        tag: str | None = None,
    ):
        """
        Initialize a packed root node

        Args:
            node_end: int
            Number of particles in dataset

            box: BoxLike
            Bounding box of dataset
        """
        self.node_start = node_start
        self.node_end = node_end
        self.box = box
        self.tag = "0" if tag is None else tag
        self.index = np.uint(0)
        self.is_leaf = True

    def copy(self) -> PackedNodeNumba:
        node = PackedNodeNumba(
            self.node_start,
            self.node_end,
            self.box.copy(),
            self.tag,
        )
        node.index = self.index
        return node


# Effectively, this class is just a python wrapper for a PackedNodeNumba
class PackedNode(octree.OctreeNode):
    """
    Public representation of a node in the packed tree

    Intended to be the equivalent of PythonOctreeNode for packed trees,
    primarily for typing checks. Note that this only represents the node
    at time of creation. Changes do not propagate in either direction.

    """

    _node: PackedNodeNumba

    def __init__(self, node: PackedNodeNumba):
        self._node = node

    def __eq__(self, obj):
        if not isinstance(self, PackedNodeNumba):
            return False
        return (
            self.node_start == obj.node_start
            and self.node_end == obj.node_end
            and self.tag == obj.tag
            and np.all(self.box.box == obj.box.box)
        )

    @property
    def node_start(self) -> int:
        """
        The start data index for this node. OctreeNodes have
        node_end+1 - node_start particles
        """
        return self._node.node_start

    @property
    def node_end(self) -> int:
        """
        The end data index for this node. OctreeNodes have
        node_end+1 - node_start particles
        """
        return self._node.node_end

    @property
    def box(self) -> bbox.BoundingBox:
        """
        Bounding box of the form [x, y, z, dx, dy, dz] where (x, y, z) is the
        left-front-bottom corner. All particles are assumed to lie inside.
        """
        return self._node.box

    @property
    def tag(self) -> str:
        """
        List of 1-based z-order indices describing the current box.
        E.g. if assuming the unit bounding box, the box
        [0.25, 0.25, 0.75, 0.25, 0.25, 0.25] would be [5,1]
        """
        return self._node.tag

    def copy(self) -> PackedNode:
        """Return a deep copy of this node"""
        return PackedNode(self._node.copy())


try:
    pack_node_type = as_numba_type(PackedNodeNumba)
except TypingError:
    pack_node_type = type(PackedNodeNumba)


@njit
def _convert_list_to_tag_str(tag: list[int]) -> str:
    """
    Convert list of ints to a str
    """
    return "0" + "".join([str(t) for t in tag])


@njit
def _create_from_current_node(node: CurrentNode) -> PackedNodeNumba:
    """Copy a CurrentNode instance into a PackedNodeNumba"""
    packed = PackedNodeNumba(
        int(node.node_start),
        int(node.node_end),
        node.box.copy(),
        _convert_list_to_tag_str(node.tag),
    )
    packed.index = np.uint(node.index)
    packed.is_leaf = is_leaf(node)
    return packed


tagtype = ListType(uint8)


# Numba version
@jitclass(
    [
        ("index", uint32),
        ("node_start", uint32),
        ("node_end", uint32),
        ("tag", tagtype),
        ("child_flag", uint8),
        ("my_index", uint8),
        ("level", uint8),
        ("empty", uint8),
        ("box", bbox.bbn_type),
    ]
)
class CurrentNode:
    index: int
    node_start: int
    node_end: int
    tag: List[int]
    child_flag: int
    my_index: int
    level: int
    empty: int
    box: bbox.BoundingBox

    def __init__(
        self,
        box: bbox.BoundingBox,
        index: int = 0,
        node_start: int = 0,
        node_end: int = 0,
        tag: List[int] | None = None,
        child_flag: int = 0,
        my_index: int = 0,
        level: int = 0,
        empty: int = 0,
    ):
        self.box = box
        self.index = index
        self.node_start = node_start
        self.node_end = node_end
        self.tag = List.empty_list(uint8) if tag is None else None
        self.child_flag = child_flag
        self.my_index = my_index
        self.level = level
        self.empty = empty


try:
    curr_node_type = as_numba_type(CurrentNode)
except TypingError:
    curr_node_type = type(CurrentNode)


@njit
def _update_node_state(node: CurrentNode, child_index: int, old_my_index: int):
    """
    Update the internal state of this node depending on tree taversal direction

    This has been separated out of _update_current_node so that it can be used
    by _construct_node_recursive, since these parts are not directly encoded in
    the packed tree
    """
    # Since some information, like the box and tag, are not stored in the
    # tree, we need to update the node's state. This is dependent on which
    # direction we're traveling: up the tree requires shortening the tag and
    # expanding the box; down the tree requires lengthening the tag and
    # shrinking the box.
    if not child_index:
        # moving to parent - need index of current node to remove offsets
        # need zero-based for positions
        node.tag.pop()
        curr_child_index = old_my_index - 1  # need 0-based index
        node.box.box[0] -= node.box.box[3] * (curr_child_index & 1)
        node.box.box[1] -= node.box.box[4] * ((curr_child_index & 2) >> 1)
        node.box.box[2] -= node.box.box[5] * ((curr_child_index & 4) >> 2)
        # need to grow box _after_ moving
        node.box.box[3] *= 2
        node.box.box[4] *= 2
        node.box.box[5] *= 2
    else:
        # 1-based index is stored
        if node.my_index != child_index:
            raise octree.OctreeError(
                f"Child index ({child_index}) does not "
                + f"match expected ({node.my_index})"
            )
        node.tag.append(np.uint8(child_index))
        # need to shrink box _before_ moving
        node.box.box[3] /= 2
        node.box.box[4] /= 2
        node.box.box[5] /= 2
        # need zero-based for positions
        child_index -= 1
        node.box.box[0] += node.box.box[3] * (child_index & 1)
        node.box.box[1] += node.box.box[4] * ((child_index & 2) >> 1)
        node.box.box[2] += node.box.box[5] * ((child_index & 4) >> 2)


@njit
def get_name(node: CurrentNode) -> str:
    """
    Get the name (tag) of this CurrentNode
    """
    return _convert_list_to_tag_str(node.tag)


@njit
def get_children(node: CurrentNode) -> list[int]:
    """
    Return a list of 0-based children indices for this CurrentNode
    """
    return List([i for i in range(8) if node.child_flag & (1 << i)])


@njit
def is_leaf(node: CurrentNode) -> bool:
    """
    Return True if node is a leaf node
    """
    return not bool(node.child_flag)


@njit
def is_root(node: CurrentNode) -> bool:
    """
    Return True if node is the root node
    """
    return not node.index
