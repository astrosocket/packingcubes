from __future__ import annotations

import logging
from array import array
from collections.abc import Buffer, Iterable, Iterator, Sequence

import numpy as np
from numba import int64, njit, objmode, types, uint8, uint32  # type: ignore
from numba.core.errors import TypingError
from numba.experimental import jitclass
from numba.extending import as_numba_type, overload
from numba.typed import List
from numba.types import ListType, string
from numpy.typing import ArrayLike, NDArray

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree
from packingcubes.configuration import FIELD_FORMAT
from packingcubes.data_objects import DataContainer, Dataset, dc_type

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


class PackedNode(octree.OctreeNode):
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
        return PackedNode(self._node.copy())


pack_node_type = as_numba_type(PackedNodeNumba)


@njit
def _convert_list_to_tag_str(tag: list[int]) -> str:
    """
    Convert list of ints to a str
    """
    return "0" + "".join([str(t) for t in tag])


@njit
def _create_from_current_node(node: CurrentNode) -> PackedNodeNumba:
    packed = PackedNodeNumba(
        int(node.node_start),
        int(node.node_end),
        node.box.copy(),
        _convert_list_to_tag_str(node.tag),
    )
    packed.index = np.uint(node.index)
    packed.is_leaf = is_leaf(node)
    return packed


# Dataclass version
# @dataclass
# class CurrentNode:
#     node_end: int
#     tag: list[int]
#     box: bbox.BoundingBox
#     index: int = 0
#     node_start: int = 0
#     child_flag: int = 0
#     my_index: int = 0
#     level: int = 0
#     empty: int = 0

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


curr_node_type = as_numba_type(CurrentNode)


@njit
def _update_node_state(node: CurrentNode, child_index: int, old_my_index: int):
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
        node.box.box[3:] *= 2
    else:
        # 1-based index is stored
        if node.my_index != child_index:
            raise octree.OctreeError(
                f"Child index ({child_index}) does not "
                + f"match expected ({node.my_index})"
            )
        node.tag.append(child_index)
        # need to shrink box _before_ moving
        node.box.box[3:] /= 2
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
    return not bool(node.child_flag)


@njit
def is_root(node: CurrentNode) -> bool:
    return not node.index


@njit
def _my_pack_bits(bool_array: NDArray) -> int:
    if bool_array.shape != (8,):
        raise ValueError("Only (8,) arrays are supported.")
    out = 0
    for i in range(8):
        out = out | (bool_array[i] << i)
    return out


@overload(np.bitwise_count)
def _my_bitwise_count(n: int):
    """
    Return the number of 1 bits in a number
    """

    def bitwise_count64(n: int) -> int:
        return sum([1 for i in range(64) if np.uint64(n) & (np.uint64(1) << i)])

    if isinstance(n, types.Integer):
        return bitwise_count64
    raise TypingError(f"Unsupported type: {n}")


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

    num_skip = np.bitwise_count((255 >> 8 - child_ind) & child_flag)

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


def _construct_tree(
    data: DataContainer, particle_threshold: int = octree._DEFAULT_PARTICLE_THRESHOLD
) -> NDArray:
    node = CurrentNode(
        node_start=0,
        node_end=len(data) - 1,
        index=0,
        child_flag=0,
        my_index=0,
        level=1,
        empty=0,
        box=data.bounding_box,
    )

    max_depth = node.box.max_depth()  # BoundingBox

    tree = List.empty_list(uint32)

    _construct_node_recursive(
        data=data,
        tree=tree,
        node=node,
        parent_index=0,
        max_depth=max_depth,
        particle_threshold=particle_threshold,
    )

    return np.array(tree, dtype=np.uint32)


@njit
def _construct_node_recursive(
    data: DataContainer,
    tree: list[int],
    node: CurrentNode,
    parent_index: int,
    max_depth: int,
    particle_threshold: int = octree._DEFAULT_PARTICLE_THRESHOLD,
    # pbar: tqdm | None = None
) -> int:
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
    tree.append(5)
    tree.append(node_start)
    tree.append(node_end)
    tree.append(octree.pack_node_metadata(0, my_index, level, empty))

    # base case: fewer than particle threshold or reached depth limit
    if num_particles <= particle_threshold or node.level >= max_depth:
        tree.append(index - parent_index)
        # print(_convert_list_to_tag_str(node.tag), index, parent_index)
        _move_to_parent(tree, node)
        # if pbar is not None:
        #     pbar.update(num_particles)
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
        # child_index = self._construct_node(node, index, pbar)
        child_index = _construct_node_recursive(
            data, tree, node, index, max_depth, particle_threshold
        )

    tree[index] = child_index + 1 - index
    tree.append(index - parent_index)
    _move_to_parent(tree, node)

    return child_index + 1


# treetype = MemoryView(uint32, 1, "C")
_index_tuple_type = types.UniTuple(uint32, 2)


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

    data: DataContainer
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
        data: DataContainer,
        tree: NDArray,
        particle_threshold: int = octree._DEFAULT_PARTICLE_THRESHOLD,
    ):
        self.data = data
        self.tree = tree
        self.particle_threshold = particle_threshold

    def _make_root_node(self) -> CurrentNode:
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
        #     box=self.data.bounding_box,
        # )
        # kwargs aren't supported
        return CurrentNode(
            self.data.bounding_box,
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
        return [n for n in self._get_nodes_numba() if n.is_leaf]

    def _get_current_node(self, tag: str) -> CurrentNode | None:
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

            containment_test: callable, optional
            Function to test if point(s) are inside shape. Should have the
            signature
            containment_test(point: NDArray) -> NDArray[bool]
            Defaults to testing if point(s) are inside the provided bounding
            box

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

            containment_test: Callable[NDArray], optional
            Function to test if point(s) are inside shape. Should have the
            (vectorized) signature
            containment_test(point: NDArray) -> NDArray[bool]
            Defaults to testing if point(s) are inside the provided bounding
            box

        Returns:
            indices: list[tuple[int]]
            List of particle start-stop indices contained within shape
        """

        # node containing bounding box
        bbox_center = bounding_box.get_box_center()
        node = self._make_root_node()

        indices = List.empty_list(_index_tuple_type)

        # check root
        node_vertices = node.box.get_box_vertices()
        if sum(containment_obj.contains(node_vertices)) == len(node_vertices):
            indices.append((node.node_start, node.node_end + 1))
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
                indices.append((node.node_start, node.node_end + 1))

            _move_to_parent(self.tree, node)

        # Note that all elements should be unique and pre-sorted due to octree
        # construction
        return indices

    def get_particle_indices_in_box(
        self,
        box: bbox.BoxLike,
    ) -> list[tuple[int, int]]:
        with objmode(numba_box=bbox.bbn_type):
            numba_box = bbox.make_bounding_box(box)
        return self._get_particle_indices_in_shape(numba_box.copy(), numba_box)

    def get_particle_indices_in_sphere(
        self,
        center: NDArray,
        radius: float,
    ) -> list[tuple[int, int]]:
        with objmode(sph=bbox.bs_type):
            sph = bbox.make_bounding_sphere(center=center, radius=radius)

        return self._get_particle_indices_in_shape(
            sph.bounding_box,
            sph,
        )

    def get_closest_particle(
        self,
        xyz: ArrayLike,
        check_neighbors: bool = True,  # noqa: FBT001, FBT002
    ) -> tuple[np.int_, float]:
        raise NotImplementedError


packed_tree_type = as_numba_type(PackedTreeNumba)


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


# TODO: add tree metadata when saving/loading
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

    dataset: Dataset
    """
    The dataset backing this octree
    """
    _tree: PackedTreeNumba
    """
    The actual in-memory representation of the tree as a numba-fied class
    object
    """
    particle_threshold: int
    """
    The maximum leaf size before splitting, used in tree construction
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        source: Buffer | None = None,
        particle_threshold: int | None = None,
    ):
        """
        Args:
            dataset: Dataset
            A Dataset containing particle data

            source: Buffer | None
            Pre-computed packed buffer containing this tree. Will be computed
            if present

            particle_threshold: int, optional
            Number of particles allowed in a leaf before splitting. Defaults to
            octree._DEFAULT_PARTICLE_THRESHOLD
        """
        if particle_threshold is None:
            particle_threshold = octree._DEFAULT_PARTICLE_THRESHOLD

        self.particle_threshold = particle_threshold

        self.dataset = dataset
        data = dataset.data_container

        # from some empirical testing. doesn't need to be exact anyway
        # estimated_size_in_bytes = (
        #     len(self.data) / 10 ** (np.floor(np.log10(self.particle_threshold))) * 1.2
        # ).astype(int) * 20
        # self.tree = np.array((estimated_size_in_bytes,), dtype=np.bytes_)

        if source is None:
            packed = _construct_tree(data, particle_threshold=particle_threshold)
        else:
            packed = np.array(source, dtype=np.uint32)

        self._tree = PackedTreeNumba(data, packed, particle_threshold)

    @property
    def packed(self) -> memoryview:
        """
        Return a memoryview of the tree's backing byte array
        """
        return memoryview(self._tree.tree)

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
