from __future__ import annotations

import datetime
import logging
import warnings
from array import array
from collections.abc import Buffer, Iterable, Iterator, Sequence
from typing import NamedTuple

import numpy as np
from numba import (  # type: ignore
    TypingError,
    int64,
    njit,
    objmode,
    types,
    uint8,
    uint32,
)
from numba.experimental import jitclass
from numba.extending import as_numba_type
from numba.typed import List
from numba.types import ListType, string
from numpy.typing import ArrayLike, NDArray
from xxhash import xxh3_64_intdigest

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree
from packingcubes.configuration import FIELD_FORMAT
from packingcubes.data_objects import DataContainer, Dataset, InMemory, dc_type

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


@njit
def _my_pack_bits(bool_array: NDArray) -> int:
    """
    Private numba-compatible implementation of numpy's packbits
    """
    if bool_array.shape != (8,):
        raise ValueError("Only (8,) arrays are supported.")
    out = 0
    for i in range(8):
        out = out | (bool_array[i] << i)
    return out


@njit
def _update_current_node(
    tree: Sequence, index: int, node: CurrentNode, child_index: int
):
    """
    Update the parameters of the node based on the new position

    Note we do no checking for invalid arguments here. This method should
    **not** be called externally.

    Args:
        index: int
        Position in the packed tree

        node: CurrentNode
        Node to update

        child_index: int
        Which node to transition to: 0 for parent, 1-8 for the requested
        child.
    """
    # currently at a node boundary
    # print(
    #     "Moving from {index} ({bytes}->{unpacked}) to ".format(
    #         index=node.index,
    #         bytes=self.tree[node.index : (node.index + 4)],
    #         unpacked=[
    #             node.node_start,
    #             node.node_end,
    #             node.child_flag,
    #             node.my_index,
    #             node.level,
    #         ],
    #     ),
    #     end="",
    # )
    # store current my_index in case we're going up tree
    old_my_index = node.my_index
    node.index = index
    node.node_start, node.node_end, metadata = tree[(index + 1) : (index + 4)]
    node.child_flag, node.my_index, node.level, node.empty = (
        octree.unpack_node_metadata(metadata)
    )
    # print(
    #     "{index} ({bytes}->{unpacked}) to ".format(
    #         index=node.index,
    #         bytes=self.tree[node.index : (node.index + 4)],
    #         unpacked=[
    #             node.node_start,
    #             node.node_end,
    #             node.child_flag,
    #             node.my_index,
    #             node.level,
    #         ],
    #     )
    # )

    _update_node_state(node, child_index, old_my_index)


@njit
def _move_to_child(tree: Sequence, node: CurrentNode, child_ind: int) -> int:
    """
    Move pointer to specified child node and return offset

    Args:
        node: CurrentNode
        Node to update

        child_index: int
        Which (0-based) child node to move to

    Returns:
        offset: int
        The number of fields moved or zero if the child DNE
    """
    # currently at a node boundary
    # flag for children is at boundary + 3 fields
    child_flag = tree[node.index + 3] >> 24
    # could also use current_node.child_flag
    if not child_flag & (1 << child_ind):
        return 0

    temp = int((255 >> 8 - child_ind) & child_flag)
    num_skip = 0
    while temp > 0:
        temp &= temp - 1
        num_skip += 1

    # children start at boundary + 4 fields
    current = old = node.index
    current += 4
    for _ in range(num_skip):
        if current > len(tree):
            raise IndexError(f"Invalid index: {current}")
        current += tree[current]

    _update_current_node(tree, current, node, child_ind + 1)
    return current - old


@njit
def _move_to_parent(tree: Sequence, node: CurrentNode):
    """
    Move pointer to parent node and return offset (0 if at root)
    """
    # currently at node boundary
    # amount to move back is at end of node, or skip_length-1 fields from
    # self.current
    node_len = tree[node.index]
    pl = tree[node.index + node_len - 1]
    if pl:
        # only move up if we're not already at root
        _update_current_node(tree, node.index - pl, node, 0)

    return pl


@njit
def _construct_tree(
    data: DataContainer,
    box: bbox.BoundingBox | None = None,
    particle_threshold: int = octree._DEFAULT_PARTICLE_THRESHOLD,
) -> NDArray:
    """
    Construct a packed tree, generating a node at a time recursively

    Functions as the initial entry point into _construct_node_recursive.

    Tip: if you know that all the particles are in a subregion of the data's
    bounding box, specifying a smaller box can significantly speed up tree
    construction and especially search.

    Args:
        data: DataContainer
        The backing data of this tree

        box: BoundingBox | None, optional
        Bounding box of the tree. Defaults to the data's bounding box

        particle_threshold: int, optional
        The maximum number of particles a leaf node can hold before splitting
        Defaults to octree._DEFAULT_PARTICLE_THRESHOLD

    Returns:
        tree: NDArray[uint32]
        Array of uint32s representing the packed tree.
    """
    # TODO: Ideally this would be kwargs-only, but the mix of kwargs and
    # default arguments is unsupported still per
    # https://github.com/numba/numba/issues/9251
    node = CurrentNode(
        # node_start=0,
        # node_end=len(data) - 1,
        # index=0,
        # child_flag=0,
        # my_index=0,
        # level=1,
        # empty=0,
        # box=data.bounding_box if box is None else box,
        data.bounding_box if box is None else box,
        0,
        0,
        len(data) - 1 if len(data) > 0 else 0,
        None,
        0,
        0,
        1,
        0,
    )

    max_depth = node.box.max_depth()  # BoundingBox

    tree = List.empty_list(uint32)

    _construct_node_recursive(
        data=data,
        tree=tree,
        node=node,
        parent_index=np.uint32(0),
        max_depth=max_depth,
        particle_threshold=particle_threshold,
    )

    return np.array([b for b in tree], dtype=np.uint32)  # noqa: C416


@njit
def _construct_node_recursive(
    data: DataContainer,
    tree: list[np.uint32],
    node: CurrentNode,
    parent_index: int,
    max_depth: int,
    particle_threshold: int = octree._DEFAULT_PARTICLE_THRESHOLD,
) -> int:
    """
    Construct a packed tree, generating a node at a time recursively

    Args:
        data: DataContainer
        The backing data of this tree

        tree: list[uint32]
        List of uint32s representing the in-progress packed tree. Initial
        call should pass an empty list.

        node: CurrentNode
        Pointer to the node this invocation is constructing. Initial call
        should pass a node corresponding to the root

        parent_index: int
        Index of the current node's parent in the in-progress tree. Initial
        call should pass 0

        max_depth: int
        The maximum depth supported by the bounding box of the data before
        floating point errors will occur. A node with level >=max_depth is
        required/guaranteed to be a leaf node

        particle_threshold: int, optional
        The maximum number of particles a leaf node can hold before splitting
        Defaults to octree._DEFAULT_PARTICLE_THRESHOLD

    Returns:
        The index of the next node in the tree at the same level or higher
    """
    # we need to cache various properties for when we recurse
    index = node.index
    node_start = node.node_start
    node_end = node.node_end
    num_particles = node_end - node_start + 1
    my_index = node.my_index
    level = node.level
    empty = node.empty
    # All nodes have the following 4 fields, plus parent_offset
    # but we don't know the skip_length, child_flag, or parent_offset field
    # values for internal nodes yet. Put placeholders for now, they'll be
    # updated later
    tree.append(np.uint32(5))
    tree.append(np.uint32(node_start))
    tree.append(np.uint32(node_end))
    tree.append(octree.pack_node_metadata(0, my_index, level, empty))

    # base case: fewer than particle threshold or reached depth limit
    if num_particles <= particle_threshold or node.level >= max_depth:
        tree.append(np.uint32(index - parent_index))
        # print(_convert_list_to_tag_str(node.tag), index, parent_index)
        _move_to_parent(tree, node)
        return index + 5

    # Need to partition and do each child one-by-one
    # This partitioning is why we need to keep track of the box
    # need node_end + 1 because we have child2 - 1 below
    child_list = np.array(octree._partition_data(data, node.box, node_start, node_end))

    # update metadata since we can compute the child flag now
    # numpy packbits is not yet supported by numba
    # child_flag = int(np.packbits(child_list[:8] < child_list[1:], bitorder="little"))
    # we'll use our own implementation for now
    # note: this actually is faster by a factor of ~2 (1.18 us vs 550ns)
    # since we don't need special cases!
    child_flag = _my_pack_bits(child_list[:8] < child_list[1:])
    tree[index + 3] = octree.pack_node_metadata(child_flag, my_index, level, empty)

    child_index = index + 4
    for i in range(1, 9):
        child1 = child_list[i - 1]
        child2 = child_list[i]

        if child1 >= child2:
            continue
        # can't use _move_to_child, since the data doesn't exist yet (we
        # don't even know how many children there are to skip over!)
        # need to manually change
        node.index = child_index
        node.node_start = child1
        node.node_end = child2 - 1
        node.my_index = i
        node.level += 1
        _update_node_state(node, i, 0)  # old_my_index is unused
        child_index = _construct_node_recursive(
            data, tree, node, index, max_depth, particle_threshold
        )

    tree[index] = np.uint32(child_index + 1 - index)
    tree.append(np.uint32(index - parent_index))
    _move_to_parent(tree, node)

    return child_index + 1


# treetype = MemoryView(uint32, 1, "C")
_index_tuple_type = types.UniTuple(uint32, 2)
_list_index_tuple = types.ListType(_index_tuple_type)


@jitclass([("data", dc_type), ("tree", uint32[:]), ("particle_threshold", int64)])
class PackedTreeNumba:
    """
    Private jitted octree interface

    This interface defines the methods for manipulating and traversing a
    packingcubes octree.

    Attributes:
        data: Dataset
        The backing dataset

        tree: array
        The actual tree in memory

    """

    box: bbox.BoundingBox
    """ Backing dataset """
    tree: NDArray
    """ 
    Tree representation in memory
    """
    particle_threshold: int
    """
    Maximum number of particles in a leaf
    """

    def __init__(
        self,
        box: bbox.BoundingBox,
        tree: NDArray,
        particle_threshold: int = octree._DEFAULT_PARTICLE_THRESHOLD,
    ):
        self.box = box
        self.tree = tree
        self.particle_threshold = particle_threshold

    def _make_root_node(self) -> CurrentNode:
        """
        Return a CurrentNode pointer at the tree root
        """
        # child flag located at field 3
        child_flag, my_index, level, empty = octree.unpack_node_metadata(self.tree[3])
        # return CurrentNode(
        #     node_start=self.tree[1],
        #     node_end=self.tree[2],
        #     index=0,
        #     child_flag=child_flag,
        #     my_index=my_index,
        #     level=level,
        #     empty=empty,
        #     box=self.box,
        # )
        # kwargs aren't supported
        return CurrentNode(
            self.box,
            0,
            self.tree[1],
            self.tree[2],
            None,
            child_flag,
            my_index,
            level,
            empty,
        )

    # The __iter__ dunder method is currently unsupported in Numba
    # def __iter__(self) -> Iterator[PackedNode]:
    #     """
    #     Return all nodes as pre-order tree traversal
    #     """
    #     return map(_create_from_current_node, self._iterate_nodes())

    # Numba appears to have trouble with the following in a number of ways
    # including yields from closures and the yield statement in general
    # (Essentially, even a version that should have been supported produced
    # errors with malloc and would break immediately when the yields were
    # included, but worked fine and successfully traverses the tree when they
    # were left out
    def _iterate_nodes(self) -> Iterator[CurrentNode]:
        raise NotImplementedError(
            "Not currently Numba supported. Try _get_nodes_numba instead."
        )
        node = self._make_root_node()
        yield node
        children_to_visit = [get_children(node)]
        while children_to_visit:
            children_generator = children_to_visit[-1]
            try:
                child = next(children_generator)
            except StopIteration:
                children_to_visit.pop()
                self._move_to_parent(node)
                continue
            self._move_to_child(node, child)
            yield node
            children_to_visit.append(get_children(node))

    # must return the list because we can't return iter() (I think)
    # Note this is not well documented - I think it's the same reasoning as
    # why we can't return a range object, but it's not specified as such
    def _get_nodes_numba(self) -> list[PackedNodeNumba]:
        """
        Return a list containing a "copy" of all nodes in the tree
        """
        node = self._make_root_node()
        nodes = List([_create_from_current_node(node)])

        next_child = List([0])
        while next_child:
            child = next_child[-1]
            while child < 8 and not _move_to_child(self.tree, node, child):
                child = child + 1

            if child >= 8:
                # no more children to visit
                next_child.pop()
                _move_to_parent(self.tree, node)
                continue

            next_child[-1] = child + 1

            nodes.append(_create_from_current_node(node))

            next_child.append(0)

        return nodes

    # see comments on __iter__, _iterate_nodes, _get_nodes_numba
    # def get_leaves(self) -> Iterable[PackedNode]:
    # return map(_create_from_current_node, filter(is_leaf, self._iterate_nodes()))

    def get_leaves(self) -> list[PackedNodeNumba]:
        """
        Return the list of PackedNodeNumba leaves

        Note that the node objects are generated on the fly and are not backed
        by the tree; any modifications will not propagate.
        """
        return [n for n in self._get_nodes_numba() if n.is_leaf]

    def _get_current_node(self, tag: str) -> CurrentNode | None:
        """
        Get the CurrentNode object represented by tag or None if non-existent
        """
        with objmode(int_tag=tagtype):
            if tag[0] == "0":
                tag = tag[1:]
            if tag:
                int_tag = List([np.uint8(int(child) - 1) for child in tag])
            else:
                int_tag = List.empty_list(uint8)
        node = self._make_root_node()
        for child in int_tag:
            offset = _move_to_child(self.tree, node, child)
            if not offset and child:
                # check child to ensure tags starting with 0 are allowed
                return None
        return node

    def get_node(self, tag: str) -> PackedNodeNumba | None:
        """
        Get a PackedNodeNumba object represented by tag or None if non-existent

        Note that this object is effectively a copy; any modifications will not
        propagate.
        """
        node = self._get_current_node(tag)
        return _create_from_current_node(node)

    def _top_down_containing_node(
        self, node: CurrentNode | None, xyz: NDArray
    ) -> CurrentNode | None:
        """
        For a given point, return the smallest child node that contains point or None
        """
        if node is None:
            node = self._make_root_node()
        if not node.box.contains(xyz):
            return None
        while not is_leaf(node):
            children = get_children(node)
            for child in children:
                _move_to_child(self.tree, node, child)
                if node.box.contains(xyz):
                    break
                _move_to_parent(self.tree, node)
            else:
                break  # while loop
        return node if node.box.contains(xyz) else None

    def _bottom_up_containing_node(
        self, node: CurrentNode, xyz: NDArray
    ) -> CurrentNode | None:
        """
        For a given point, return the smallest parent node that contains point or None
        """
        while not is_root(node):
            if node.box.contains(xyz):
                return node
            _move_to_parent(self.tree, node)
        return node if node.box.contains(xyz) else None

    def _get_containing_node_of_point(
        self,
        xyz: NDArray,
        start_node: CurrentNode | None = None,
        top_down: bool = True,  # noqa: FBT001, FBT002
    ) -> CurrentNode | None:
        """
        Return smallest node containing point

        Defaults to a top-down approach from root. Can provide a start_node to
        short-cut search. Can also go bottom-up; requires start_node.
        """
        if not top_down and start_node is None:
            raise ValueError("start_node **must** be provided for bottom-up traversal!")
        node = self._make_root_node() if start_node is None else start_node
        if top_down:
            return self._top_down_containing_node(node, xyz)

        # find first parent that contains point, then see if parent
        # can be refined
        containing_node = self._bottom_up_containing_node(node, xyz)
        if containing_node is not None:
            return self._top_down_containing_node(containing_node, xyz)
        return None

    def _get_containing_node_of_pointlist(
        self,
        points: NDArray,
        start_node: CurrentNode | None = None,
        top_down: bool = True,  # noqa: FBT001, FBT002
    ) -> CurrentNode:
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
                    start_node,
                    top_down,
                )
            else:
                node = self._bottom_up_containing_node(node, point)
        return node if node is not None else self._make_root_node()

    def _get_nodes_in_shape(
        self,
        bounding_box: bbox.BoundingBox,
        containment_obj: bbox.BoundingVolume,
    ) -> tuple[list[PackedNodeNumba], list[PackedNodeNumba]]:
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

            containment_obj: BoundingVolume
            Object with bounding box specified by bounding_box. Provides
            a more exact containment test (e.g. contained within a sphere).

        Returns:
            entirely_in: Iterable[OctreeNode]
            Nodes that are entirely within shape. Nodes may be internal nodes

            partial_leaves: Iterable[OctreeNode]
            Leaf nodes that are only partially within shape.

        Raises:
            IndexError:
            When there are unexpected issues with the queue system.
        """

        bbox_center = bounding_box.get_box_center()
        node = self._make_root_node()

        entire_nodes = List.empty_list(pack_node_type)
        partial_leaves = List.empty_list(pack_node_type)

        # check root
        node_vertices = node.box.get_box_vertices()
        if sum(containment_obj.contains(node_vertices)) == len(node_vertices):
            # if all root vertices enclosed, shape is bigger than tree...
            entire_nodes.append(_create_from_current_node(node))
            return entire_nodes, partial_leaves

        next_child = List([0])
        while next_child:
            child = next_child[-1]
            while child < 8 and not _move_to_child(self.tree, node, child):
                child = child + 1

            if child >= 8:
                # no more children to visit
                next_child.pop()
                _move_to_parent(self.tree, node)
                continue
            next_child[-1] = child + 1

            # Test if node entirely contained in shape
            node_vertices = node.box.get_box_vertices()
            vertices_enclosed = sum(containment_obj.contains(node_vertices))

            if not vertices_enclosed:
                # check for edge overlap
                closest_point = node.box.project_point_on_box(bbox_center)
                # need bool to convert NDArray[bool] to bool
                # if containment_test returns something besides a length-1
                # array, we should error (that's why we don't just e.g. take
                # the first element)
                partial = bool(containment_obj.contains(closest_point))
            else:
                # check for degree of containment
                partial = 0 < vertices_enclosed < len(node_vertices)

            if partial and not is_leaf(node):
                # visit children
                next_child.append(0)
                continue

            # for remaining cases we will move to parent regardless
            if vertices_enclosed and not partial:
                # all vertices enclosed
                entire_nodes.append(_create_from_current_node(node))
            elif partial:
                # (less than all vertices enclosed or edge overlap) and leaf
                partial_leaves.append(_create_from_current_node(node))
            _move_to_parent(self.tree, node)

        return entire_nodes, partial_leaves

    def _get_nodes_in_sphere(
        self,
        center: NDArray,
        radius: float,
    ) -> tuple[list[PackedNodeNumba], list[PackedNodeNumba]]:
        """
        Return lists of all nodes entirely inside and nodes partially inside sphere

        Calls _get_nodes_in_shape using a BoundingSphere

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
        # sphere bounding box
        with objmode(sph=bbox.bs_type):
            sph = bbox.make_bounding_sphere(center=center, radius=radius)

        return self._get_nodes_in_shape(
            sph.bounding_box,
            sph,
        )

    def _get_particle_indices_in_shape(
        self,
        bounding_box: bbox.BoundingBox,
        containment_obj: bbox.BoundingVolume,
    ) -> list[tuple[int, int]]:
        """
        Return all particles contained within a shape that fits inside bounding box

        Args:
            bounding_box: BoundingBox
            Shape bounding box

            containment_obj: BoundingVolume
            Object with bounding box specified by bounding_box. Provides
            a more exact containment test (e.g. contained within a sphere).

        Returns:
            indices: list[tuple[int, int]]
            List of particle start-stop indices contained within shape
        """

        # node containing bounding box
        bbox_center = bounding_box.get_box_center()
        node = self._make_root_node()

        indices = List.empty_list(_index_tuple_type)

        # check root
        node_vertices = node.box.get_box_vertices()
        if sum(containment_obj.contains(node_vertices)) == len(node_vertices):
            indices.append((node.node_start, np.uint32(node.node_end + 1)))
            return indices

        next_child = List([0])
        while next_child:
            child = next_child[-1]
            while child < 8 and not _move_to_child(self.tree, node, child):
                child = child + 1

            if child >= 8:
                # no more children to visit
                next_child.pop()
                _move_to_parent(self.tree, node)
                continue
            next_child[-1] = child + 1

            # Test if node entirely contained in shape
            node_vertices = node.box.get_box_vertices()
            vertices_enclosed = sum(containment_obj.contains(node_vertices))

            if not vertices_enclosed:
                # Check for edge overlap
                closest_point = node.box.project_point_on_box(bbox_center)
                partial = bool(containment_obj.contains(closest_point))
            else:
                # check for degree of containment
                partial = 0 < vertices_enclosed < len(node_vertices)

            if partial and not is_leaf(node):
                # visit children
                next_child.append(0)
                continue

            # for remaining cases we will move to parent regardless
            if vertices_enclosed or partial:
                # at least some overlap
                indices.append((node.node_start, np.uint32(node.node_end + 1)))

            _move_to_parent(self.tree, node)

        # Note that all elements should be unique and pre-sorted due to octree
        # construction
        return indices

    def get_particle_indices_in_box(
        self,
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
        with objmode(numba_box=bbox.bbn_type):
            numba_box = bbox.make_bounding_box(box)
        return self._get_particle_indices_in_shape(numba_box.copy(), numba_box)

    def get_particle_indices_in_sphere(
        self,
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
        with objmode(sph=bbox.bs_type):
            sph = bbox.make_bounding_sphere(center=center, radius=radius)

        return self._get_particle_indices_in_shape(
            sph.bounding_box,
            sph,
        )

    def _node_node_query_ball_tree(
        self, other: PackedTreeNumba, r: float
    ) -> List[List[tuple[int, int]]]:
        node = self._make_root_node()
        query = List.empty_list(_list_index_tuple)
        with objmode(sph=bbox.bs_type):
            sph = bbox.make_bounding_sphere(center=[0, 0, 0], radius=r)

        next_child = List([0])
        while next_child:
            child = next_child[-1]
            while child < 8 and not _move_to_child(self.tree, node, child):
                child = child + 1

            if child >= 8:
                # no more children to visit
                next_child.pop()
                _move_to_parent(self.tree, node)
                continue

            next_child[-1] = child + 1

            next_child.append(0)
            if is_leaf(node):
                sph.center[0], sph.center[1], sph.center[2] = node.box.midplane()
                query.append(
                    other._get_particle_indices_in_shape(sph.bounding_box, sph)
                )

        return query

    def count_neighbors(self, other: PackedTreeNumba, r: float) -> int:
        node = self._make_root_node()
        num_pairs = 0
        with objmode(sph=bbox.bs_type):
            sph = bbox.make_bounding_sphere(center=[0, 0, 0], radius=r)

        next_child = List([0])
        while next_child:
            child = next_child[-1]
            while child < 8 and not _move_to_child(self.tree, node, child):
                child = child + 1

            if child >= 8:
                # no more children to visit
                next_child.pop()
                _move_to_parent(self.tree, node)
                continue

            next_child[-1] = child + 1

            next_child.append(0)
            if is_leaf(node):
                sph.center[0], sph.center[1], sph.center[2] = node.box.midplane()
                pair_nodes = other._get_particle_indices_in_shape(sph.bounding_box, sph)
                for other_start, other_end in pair_nodes:
                    num_pairs += (node.node_end - node.node_start + 1) * (
                        other_end - other_start + 1
                    )

        return num_pairs

    def _get_particle_index_list_in_shape(
        self,
        data: DataContainer,
        bounding_box: bbox.BoundingBox,
        containment_obj: bbox.BoundingVolume,
    ) -> NDArray[np.int64]:
        """
        Return all particles contained within a shape that fits inside bounding box

        Args:
            bounding_box: BoundingBox
            Shape bounding box

            containment_obj: BoundingVolume
            Object with bounding box specified by bounding_box. Provides
            a more exact containment test (e.g. contained within a sphere).

        Returns:
            indices: list[tuple[int]]
            List of particle start-stop indices contained within shape
        """
        slices = self._get_particle_indices_in_shape(bounding_box, containment_obj)

        # the following is an attempt to mimic what I _think_ the expand_ranges
        # function from swiftsimio (which is GPL 3) does.
        num_particles = 0
        for s in slices:
            num_particles += s[1] - s[0] + 1

        indices = np.empty((num_particles,), dtype=np.int64)
        ind = 0
        for s in slices:
            for i, index in enumerate(range(s[0], s[1] + 1)):
                indices[ind + i] = data._index[index]
            ind += s[1] - s[0] + 1

        return indices

    def get_closest_particle(
        self,
        xyz: ArrayLike,
        check_neighbors: bool = True,  # noqa: FBT001, FBT002
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

        Raises:
            NotImplementedError
        """
        raise NotImplementedError


try:
    packed_tree_type = as_numba_type(PackedTreeNumba)
except TypingError:
    packed_tree_type = type(PackedTreeNumba)


def _print_packed(packed: memoryview | array, *, expected_node_start=0):
    """
    Attempt to print a packed tree for debugging
    """
    if isinstance(packed, memoryview) and packed.format != FIELD_FORMAT:
        packed.cast("b").cast(FIELD_FORMAT)
    position = 0
    child_ind = [0]
    while position + 3 < len(packed):
        skip_length, node_start, node_end, metadata = packed[position : position + 4]
        child_flag, my_index, level, empty = octree.unpack_node_metadata(metadata)
        print(  # noqa
            f"{'.' * (level - 1)} sl:{skip_length:4} "
            f"ns:{node_start:4} ne:{node_end:6} "
            f"cf: {child_flag:08b}={
                ''.join(
                    str(n + 1)
                    for n in np.where(
                        np.unpackbits(np.uint8(child_flag), bitorder='little')
                    )
                )
            } "
            f"my:{my_index} ll:{level} em:{empty} (m:{metadata})",
            end="",
        )
        if position + skip_length < len(packed):
            if skip_length >= 5 + bool(child_flag):
                print(f" po: {packed[position + skip_length - 1]}", end="")  # noqa
            else:
                print(" po: DNE", end="")  # noqa
        else:
            print(" po: NA", end="")  # noqa
        print(f" in: {position}")  # noqa

        if node_start != expected_node_start:
            print(  # noqa
                f"Error at index {position}. "
                f"Expected to start at {expected_node_start} "
                f"but started at {node_start}"
            )
            return

        position += 4
        child_ind[-1] -= 1
        if child_flag:
            # internal node: continue with children
            child_ind.append(int(np.bitwise_count(child_flag)))
            continue
        position += 1
        expected_node_start = node_end + 1
        while child_ind and not child_ind[-1]:
            child_ind.pop()
            position += 1

    print(f"remaining: {packed[position:]}")  # noqa


class TreeMeta(NamedTuple):
    """The metadata for a PackedTree"""

    creation_timestamp: np.float64
    """ Creation time as a UTC timestamp"""
    checksum: int
    """ Actual checksum """
    bounding_box: bbox.BoundingBox
    """ Bounding box used when creating the tree """
    particle_threshold: int = octree._DEFAULT_PARTICLE_THRESHOLD
    """ Particle threshold used when creating the tree """
    meta_size: int = 144
    """ Header size in bytes """
    field_type: np.dtype = np.dtype(np.uint32)
    """ Field type (e.g. uint32)"""
    packed_spec_version: str = "1.0.0"
    """ Current version of the packed metadata standard in semver format"""
    checksum_method: str = "xxh3_64_intdigest"
    """ Method for computing the checksum """


class TreeMetaError(Exception):
    pass


def create_metadata(
    box: bbox.BoundingBox,
    packed: NDArray[np.uint32],
    particle_threshold: int = octree._DEFAULT_PARTICLE_THRESHOLD,
):
    """
    Create a tree metadata object from the provided info

    Args:
        box: BoundingBox
        The bounding box used to create the tree

        packed: NDArray[uint32]
        The packed tree data

        particle_threshold: int, optional
        The particle threshold to split leaves. Defaults to
        optree._DEFAULT_PARTICLE_THRESHOLD

    Returns:
        A TreeMeta object containing the tree's metadata
    """
    creation = np.float64(datetime.datetime.now(tz=datetime.UTC).timestamp())
    checksum = xxh3_64_intdigest(bytes(packed))
    return TreeMeta(
        field_type=packed.dtype,
        creation_timestamp=creation,
        checksum=checksum,
        bounding_box=box,
        particle_threshold=particle_threshold,
    )


def pack_metadata(
    metadata: TreeMeta, packed_tree: NDArray[np.uint32]
) -> NDArray[np.uint32]:
    """
    Pack tree metadata

    Args:
        metadata: TreeMeta
        The metadata of the tree

        packed_tree: NDArray
        The packed tree data

    Returns:
        Packed metadata
    """
    packed_meta: NDArray = np.zeros(
        (int(metadata.meta_size / np.dtype(metadata.field_type).itemsize),),
        dtype=metadata.field_type,
    )
    packed_meta[0] = metadata.meta_size
    version = np.array(
        [np.uint8(v) for v in metadata.packed_spec_version.split(".")], dtype=np.uint32
    )
    packed_meta[1] = np.uint32(
        (np.uint32(np.int8(4)) << 24)
        | (version[0] << 16)
        | (version[1] << 8)
        | (version[2])
    )

    def to_field(
        x: np.float32 | np.float64 | np.int64 | np.uint64 | NDArray,
    ) -> NDArray:
        return np.frombuffer(x.tobytes(), dtype=metadata.field_type.type)

    ft = metadata.field_type.type
    packed_meta[2:4] = to_field(
        np.float64(datetime.datetime.now(tz=datetime.UTC).timestamp())
    )
    packed_meta[4:21] = to_field(np.array(metadata.checksum_method))
    packed_meta[21:23] = to_field(np.uint64(xxh3_64_intdigest(packed_tree)))
    packed_meta[23:35] = to_field(metadata.bounding_box.box)
    packed_meta[35] = np.uint32(metadata.particle_threshold)
    return packed_meta


def extract_metadata(source: Buffer) -> tuple[TreeMeta, NDArray[np.uint32]]:
    """
    Extract the metadata and packed tree information from a buffer

    Args:
        source: Buffer
        A Buffer containing the packed data

    Returns:
        metadata
        The tree's metadata

        packed_tree
        The actual packed tree data as a numpy array

    Raises:
        ValueError
        If the metadata does not match an expected format

        NotImplementedError
        If an unimplemented data format is specified (currently only uint32)

        OctreeError
        If the metadata checksum does not match the computed checksum
    """
    combined = np.frombuffer(source, dtype=np.uint32)
    meta_size = combined[0]
    if meta_size != 144:
        # need to check endianness
        dts = np.dtype(np.uint32).newbyteorder("S")
        meta_size = np.asarray(meta_size, dtype=dts)
        if meta_size != 144:
            raise TreeMetaError(
                "Unknown metadata format! Expected first "
                f"field to be equal to 144, got {meta_size}"
            )
        combined = np.asarray(combined, dtype=dts)
    # extract field type and packed specification version
    ft_version = combined[1]
    ft = np.int8(ft_version >> 24)
    if ft != 4:
        raise NotImplementedError(
            f"Only uint32 (ft=4) is currently available (provided {ft})"
        )
    version_list = [0xFF & (ft_version >> i) for i in range(16, -1, -8)]
    version = f"{version_list[0]}.{version_list[1]}.{version_list[2]}"

    def from_field(x: NDArray, dt: np.dtype) -> NDArray:
        return np.frombuffer(x.tobytes(), dtype=dt)

    creation_time = np.float64(from_field(combined[2:4], np.dtype(np.float64)))
    checksum_method = str(from_field(combined[4:21], np.dtype("U17"))[0])
    if checksum_method != "xxh3_64_intdigest":
        LOGGER.warning(
            f"Unknown checksum method: {checksum_method}. Expected xxh3_64_intdigest"
        )
    checksum = int(from_field(combined[21:23], np.dtype(np.uint64)))
    box_array = from_field(combined[23:35], np.dtype(np.float64))
    box = bbox.make_bounding_box(box_array)
    particle_threshold = combined[35]
    packed_tree = combined[36:]
    computed_checksum = xxh3_64_intdigest(packed_tree)
    if checksum != computed_checksum:
        raise octree.OctreeError(
            f"Packed tree checksum {computed_checksum}"
            f" does not match header ({checksum})!"
        )
    metadata = TreeMeta(
        meta_size=meta_size,
        field_type=np.dtype(np.uint32),
        creation_timestamp=creation_time,
        checksum=checksum,
        bounding_box=box,
        particle_threshold=particle_threshold,
        checksum_method=checksum_method,
        packed_spec_version=version,
    )
    return metadata, packed_tree


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
        return self._tree.get_particle_indices_in_box(box)

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
        return self._tree.get_particle_indices_in_sphere(center, radius)

    def get_particle_index_list_in_box(
        self,
        *,
        dataset: Dataset,
        box: bbox.BoxLike,
    ) -> NDArray[np.int64]:
        """
        Return all particles contained within the box

        Args:
            box: BoxLike
            Box to check

        Returns:
            indices: NDArray[int]]
            List of original particle indices contained within sphere
        """
        bounding_box = bbox.make_bounding_box(box)
        return self._tree._get_particle_index_list_in_shape(
            dataset.data_container, bounding_box, bounding_box
        )

    def get_particle_index_list_in_sphere(
        self,
        *,
        dataset: Dataset,
        center: NDArray,
        radius: float,
    ) -> NDArray[np.int64]:
        """
        Return all particles contained within the sphere defined by center and radius

        Args:
            center: NDArray
            Center point of the sphere

            radius: float
            Radius of the sphere

        Returns:
            indices: NDArray[int]
            List of original particle indices contained within sphere
        """
        sph = bbox.make_bounding_sphere(radius, center=center)
        bounding_box = sph.bounding_box
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
                f"was {data_shape[1]}-dimensional"
                if len(data_shape) >= 2
                else "was 1-dimensional."
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
