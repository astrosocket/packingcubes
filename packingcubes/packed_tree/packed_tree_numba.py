"""Jitclass version of PackedTree and supporting functions

Underlying jitclass of a PackedTree that contains the actual implementation
of most of the functions along with several supporting functions. While not
strictly a private module, the functions contained here are not intended for
use generally, but for extensions.
"""

from __future__ import annotations

import logging
from array import array
from collections.abc import Callable, Iterator, Sequence

import numpy as np
from numba import (  # type: ignore
    TypingError,
    int64,
    njit,
    objmode,
    types,
    uint32,
)
from numba.experimental import jitclass
from numba.extending import as_numba_type
from numba.typed import List
from numpy.typing import NDArray

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree
from packingcubes.configuration import FIELD_FORMAT
from packingcubes.data_objects import DataContainer, dc_type
from packingcubes.packed_tree.fixed_distance_heap import FixedDistanceHeap
from packingcubes.packed_tree.packed_node import (
    CurrentNode,
    PackedNodeNumba,
    _create_from_current_node,
    _update_node_state,
    get_children,
    is_leaf,
    is_root,
    pack_node_type,
)

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(capture=True)


@njit
def euclidean_distance(xyz: NDArray, pxyz: NDArray) -> NDArray:
    """Compute the Euclidean distance between two arrays"""
    return np.sqrt(np.sum(np.atleast_2d((xyz - pxyz) ** 2), axis=1))


@njit(fastmath=True)
def euclidean_distance_particle(
    x: float, y: float, z: float, px: float, py: float, pz: float
) -> float:
    """Compute the Euclidean distance between two points"""
    return np.sqrt(euclidean_d2(x, y, z, px, py, pz))


@njit(fastmath=True)
def euclidean_d2(
    x: float, y: float, z: float, px: float, py: float, pz: float
) -> float:
    """Compute the squared Euclidean distance between two points"""
    return (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2


@njit
def _process_slice_against_heap(
    heap: FixedDistanceHeap,
    data: DataContainer,
    xyz: NDArray,
    distance_function: Callable[[float, float, float, float, float, float], float],
    start: int,
    end: int,
    use_shuffle: bool,  # noqa: FBT001, FBT002
):
    """Try every particle in slice against distance heap iteratively"""
    x, y, z = xyz
    positions = data._positions[start:end, 0:3]
    if use_shuffle:
        indices = data._index[start:end]

    num_particles = end - start
    for i in range(num_particles):
        px, py, pz = positions[i, :]
        dist = np.float64(distance_function(x, y, z, px, py, pz))
        index = np.int_(start + i)
        if use_shuffle:
            index = np.int_(indices[i])
        heap.try_replace(dist, index)


@njit
def _my_pack_bits(bool_array: NDArray) -> int:
    """Private numba-compatible implementation of numpy's packbits"""
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
    """Update the parameters of the node based on the new position

    Note we do no checking for invalid arguments here. This method should
    **not** be called externally.

    Parameters
    ----------
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
    """Move pointer to specified child node and return offset

    Parameters
    ----------
    node: CurrentNode
        Node to update

    child_index: int
        Which (0-based) child node to move to

    Returns
    -------
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
    """Move pointer to parent node and return offset (0 if at root)"""
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
def _process_leaf_for_pilis_tree(
    node: CurrentNode,
    data: DataContainer,
    box: bbox.BoundingBox,
    r: float,
    r2: float,
    rsq: float,
    otree: PackedTreeNumba,
    odata: DataContainer,
    d2_function: Callable[[float, float, float, float, float, float], float],
    strict: bool,  #  noqa: FBT001, FBT002
    query: list[NDArray[np.int64]],
):
    # unwrap loops to reduce complexity
    box.box[0] = node.box.box[0] - r
    box.box[1] = node.box.box[1] - r
    box.box[2] = node.box.box[2] - r
    box.box[3] = node.box.box[3] + r2
    box.box[4] = node.box.box[4] + r2
    box.box[5] = node.box.box[5] + r2
    pil = otree._get_particle_index_list_in_shape(odata, box)
    for index in range(node.node_start, node.node_end + 1):
        if strict:
            reduced_pil = List.empty_list(np.int64)
            x, y, z = data._positions[index, :]
            for oindex in pil:
                ox, oy, oz = odata._positions[oindex, :]
                d = d2_function(x, y, z, ox, oy, oz)
                if d < rsq:
                    reduced_pil.append(oindex)
            query.append(np.asarray(reduced_pil))
        else:
            query.append(pil)


@njit
def _construct_tree(
    data: DataContainer,
    box: bbox.BoundingBox | None = None,
    particle_threshold: int = octree._DEFAULT_PARTICLE_THRESHOLD,
) -> NDArray:
    """Construct a packed tree, generating a node at a time recursively

    Functions as the initial entry point into
    [_construct_node_recursive][_construct_node_recursive].

    Tip
    ---
    If you know that all the particles are in a subregion of the data's
    bounding box, specifying a smaller box can significantly speed up tree
    construction and especially search.

    Parameters
    ----------
    data: DataContainer
        The backing data of this tree

    box: BoundingBox | None, optional
        Bounding box of the tree. Defaults to the data's bounding box

    particle_threshold: int, optional
        The maximum number of particles a leaf node can hold before splitting
        Defaults to `octree._DEFAULT_PARTICLE_THRESHOLD`

    Returns
    -------
    tree: NDArray[uint32]
        Array of `uint32`s representing the packed tree.
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
    """Construct a packed tree, generating a node at a time recursively

    Parameters
    ----------
    data: DataContainer
        The backing data of this tree

    tree: list[uint32]
        List of `uint32`s representing the in-progress packed tree. Initial
        call should pass an empty list.

    node: CurrentNode
        Pointer to the node this invocation is constructing. Initial call
        should pass a node corresponding to the root

    parent_index: int
        Index of the current node's parent in the in-progress tree. Initial
        call should pass `0`

    max_depth: int
        The maximum depth supported by the bounding box of the data before
        floating point errors will occur. A node with `level >=max_depth` is
        required/guaranteed to be a leaf node

    particle_threshold: int, optional
        The maximum number of particles a leaf node can hold before splitting.
        Defaults to `octree._DEFAULT_PARTICLE_THRESHOLD`

    Returns
    -------
    :
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
_index_tuple_type = types.UniTuple(uint32, 3)
_list_index_tuple = types.ListType(_index_tuple_type)


@jitclass([("data", dc_type), ("tree", uint32[:]), ("particle_threshold", int64)])
class PackedTreeNumba:
    """Private jitted octree interface

    This interface defines the methods for manipulating and traversing a
    packingcubes octree.
    """

    box: bbox.BoundingBox
    """ 
    Bounding box for the tree

    Effectively the root level bounding box
    """
    tree: NDArray
    """ 
    Tree representation in memory

    Note that this does not include any of the metadata. `tree[0]` should the
    start of the root node data in packed tree format.
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

    def __len__(self):
        """Return the number of particles in the tree"""
        return self.tree[2] - self.tree[1] + 1

    def _make_root_node(self) -> CurrentNode:
        """Return a CurrentNode pointer at the tree root"""
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
            self.box.copy(),
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
        """Return a list containing a "copy" of all nodes in the tree"""
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
        """Return the list of PackedNodeNumba leaves

        Note that the node objects are generated on the fly and are not backed
        by the tree; any modifications will not propagate.
        """
        return [n for n in self._get_nodes_numba() if n.is_leaf]

    def _get_current_node(self, tag: str) -> CurrentNode | None:  # noqa: C901
        """Get the CurrentNode object represented by tag or None if non-existent"""
        node = self._make_root_node()
        for child_str in tag:
            # ideally we'd just do int(child_str), but that's not supported
            if child_str == "1":
                child = 1
            elif child_str == "2":
                child = 2
            elif child_str == "3":
                child = 3
            elif child_str == "4":
                child = 4
            elif child_str == "5":
                child = 5
            elif child_str == "6":
                child = 6
            elif child_str == "7":
                child = 7
            elif child_str == "8":
                child = 8
            else:
                return None
            offset = _move_to_child(self.tree, node, child)
            if not offset and child:
                # check child to ensure tags starting with 0 are allowed
                return None
        return node

    def get_node(self, tag: str) -> PackedNodeNumba | None:
        """Get a PackedNodeNumba object represented by tag or None if non-existent

        Note that this object is effectively a copy; any modifications will not
        propagate.

        Parameters
        ----------
        tag:
            The name of the node to search for. Nodes are named by child order,
            in z-order so e.g. "012" is the front-mid-left-bottom grandchild.

        Returns
        -------
        :
            A copy of the node in the tree or `None` if it DNE
        """
        node = self._get_current_node(tag)
        return _create_from_current_node(node)

    def _top_down_containing_node(
        self, node: CurrentNode | None, xyz: NDArray
    ) -> CurrentNode | None:
        """Given point, return the smallest child node that contains point or None"""
        if node is None:
            node = self._make_root_node()
        if not node.box.contains_point(xyz[0], xyz[1], xyz[2]):
            return None
        while not is_leaf(node):
            children = get_children(node)
            for child in children:
                _move_to_child(self.tree, node, child)
                if node.box.contains_point(xyz[0], xyz[1], xyz[2]):
                    break
                _move_to_parent(self.tree, node)
            else:
                break  # while loop
        return node if node.box.contains_point(xyz[0], xyz[1], xyz[2]) else None

    def _bottom_up_containing_node(
        self, node: CurrentNode, xyz: NDArray
    ) -> CurrentNode | None:
        """Given point, return the smallest parent node that contains point or None"""
        while not is_root(node):
            if node.box.contains_point(xyz[0], xyz[1], xyz[2]):
                return node
            _move_to_parent(self.tree, node)
        return node if node.box.contains_point(xyz[0], xyz[1], xyz[2]) else None

    def _get_containing_node_of_point(
        self,
        xyz: NDArray,
        start_node: CurrentNode | None = None,
        top_down: bool = True,  # noqa: FBT001, FBT002
    ) -> CurrentNode | None:
        """Return smallest node containing point

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
        """Return smallest node containing all points in array or root

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
        """Return lists of all nodes entirely inside and partially inside shape

        Both node lists will be in z-index order, but are likely to be
        interleaved, i.e.
        entirely_in[i].z_order
            < partial_leaves[j].z_order
                < entirely_in[i + 1].z_order
                    < partial_leaves[j + 1].z_order

        Parameters
        ----------
        bounding_box: BoundingBox
            Shape bounding box

        containment_obj: BoundingVolume
            Object with bounding box specified by bounding_box. Provides
            a more exact containment test (e.g. contained within a sphere).

        Returns
        -------
        entirely_in: Iterable[OctreeNode]
            Nodes that are entirely within shape. Nodes may be internal nodes

        partial_leaves: Iterable[OctreeNode]
            Leaf nodes that are only partially within shape.

        Raises
        ------
        IndexError
            When there are unexpected issues with the queue system.
        """
        bbox_center = bounding_box.get_box_center()
        node = self._make_root_node()

        entire_nodes = List.empty_list(pack_node_type)
        partial_leaves = List.empty_list(pack_node_type)

        # check root
        overlap = containment_obj.check_box_overlap(node.box)
        if overlap == 8:
            # if all 8 root vertices enclosed, shape is bigger than tree...
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
            overlap = containment_obj.check_box_overlap(node.box)
            partial = 0 < overlap < 8  # 8 box vertices

            if partial and not is_leaf(node):
                # visit children
                next_child.append(0)
                continue

            # for remaining cases we will move to parent regardless
            if overlap == 8:
                # all 8 vertices enclosed
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
        """Return lists of all nodes entirely inside and nodes partially inside sphere

        Calls _get_nodes_in_shape using a BoundingSphere

        Parameters
        ----------
        center: NDArray
            Coordinates of sphere center

        radius: float
            Sphere radius

        Returns
        -------
        entirely_in: List[OctreeNode]
            List of nodes that are entirely within shape. Nodes may be internal nodes

        partial_leaves: List[OctreeNode]
            List of leaf nodes that are only partially within shape.

        Raises
        ------
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
        containment_obj: bbox.BoundingVolume,
    ) -> list[tuple[int, int, int]]:
        """Return all particles contained within a shape

        Parameters
        ----------
        containment_obj: BoundingVolume
            Object to test containment in

        Returns
        -------
        indices: list[tuple[int, int, int]]
            List of particle start-stop indices contained within shape.
            Each tuple describes a chunk/slice of data in the form
            `(start, stop, partial)`, where partial is a flag:

             - `1` if the data chunk is entirely contained within shape,
             - `0` otherwise.
        """
        node = self._make_root_node()

        indices = List.empty_list(_index_tuple_type)

        # check root - need to check if either all corners are contained or if
        # root is leaf and there exists *any* overlap
        overlap = containment_obj.check_box_overlap(node.box)
        # 8 box vertices
        if overlap == 8 or (overlap and is_leaf(node)):
            indices.append(
                (node.node_start, np.uint32(node.node_end + 1), np.uint32(overlap < 8))
            )
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

            # Compute node overlap
            overlap = containment_obj.check_box_overlap(node.box)
            partial = np.uint32(0 < overlap < 8)  # 8 box vertices

            if partial and not is_leaf(node):
                # visit children
                next_child.append(0)
                continue

            # for remaining cases we will move to parent regardless
            if overlap:
                # at least some overlap
                indices.append((node.node_start, np.uint32(node.node_end + 1), partial))

            _move_to_parent(self.tree, node)

        # Note that all elements should be unique and pre-sorted due to octree
        # construction
        return indices

    def get_particle_indices_in_box(
        self,
        box: bbox.BoxLike,
    ) -> list[tuple[int, int, int]]:
        """Return all particles contained within the box

        Parameters
        ----------
        box: BoxLike
            Box to check

        Returns
        -------
        indices: list[tuple[int, int, int]]
            List of particle start-stop indices contained within `box`.
            Each tuple describes a chunk/slice of data in the form
            `(start, stop, partial)`, where partial is a flag:

             - `1` if the data chunk is entirely contained within `box`,
             - `0` otherwise.
        """
        with objmode(numba_box=bbox.bbn_type):
            numba_box = bbox.make_bounding_box(box)
        return self._get_particle_indices_in_shape(numba_box)

    def get_particle_indices_in_sphere(
        self,
        center: NDArray,
        radius: float,
    ) -> list[tuple[int, int, int]]:
        """Return all particles contained within the sphere defined by center and radius

        Parameters
        ----------
        center: NDArray
            Center point of the sphere

        radius: float
            Radius of the sphere

        Returns
        -------
        indices: list[tuple[int, int, int]]
            List of particle start-stop indices contained within sphere.
            Each tuple describes a chunk/slice of data in the form
            `(start, stop, partial)`, where partial is a flag:

             - `1` if the data chunk is entirely contained within the sphere,
             - `0` otherwise.
        """
        with objmode(sph=bbox.bs_type):
            sph = bbox.make_bounding_sphere(center=center, radius=radius)

        return self._get_particle_indices_in_shape(
            sph,
        )

    def _node_node_query_ball_tree(
        self, other: PackedTreeNumba, r: float
    ) -> List[List[tuple[int, int, int]]]:
        node = self._make_root_node()
        query = List.empty_list(_list_index_tuple)
        with objmode(sph=bbox.bs_type):
            sph = bbox.make_bounding_sphere(center=[0, 0, 0], radius=r)

        def process_leaf(
            node: CurrentNode,
            sph: bbox.BoundingSphere,
            other: PackedTreeNumba,
            query: List[List[tuple[int, int]]],
        ):
            sph.center[0], sph.center[1], sph.center[2] = node.box.midplane()
            query.append(other._get_particle_indices_in_shape(sph))

        # check root node
        if is_leaf(node):
            process_leaf(node, sph, other, query)
            return query
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
                process_leaf(node, sph, other, query)

        return query

    def count_neighbors(self, other: PackedTreeNumba, r: float) -> int:
        """Compute the number of particles within r of particles in other"""
        node = self._make_root_node()
        num_pairs = 0
        with objmode(sph=bbox.bs_type):
            sph = bbox.make_bounding_sphere(center=[0, 0, 0], radius=r)

        def process_leaf(
            node: CurrentNode,
            sph: bbox.BoundingSphere,
            other: PackedTreeNumba,
        ) -> int:
            num_pairs = 0
            sph.center[0], sph.center[1], sph.center[2] = node.box.midplane()
            pair_nodes = other._get_particle_indices_in_shape(sph)
            for other_start, other_end, _ in pair_nodes:
                num_pairs += (node.node_end - node.node_start + 1) * (
                    other_end - other_start + 1
                )
            return num_pairs

        # check root node
        if is_leaf(node):
            return process_leaf(node, sph, other)

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
                num_pairs += process_leaf(node, sph, other)

        return num_pairs

    def _get_particle_index_list_in_shape(
        self,
        data: DataContainer | None,
        containment_obj: bbox.BoundingVolume,
    ) -> NDArray[np.int64]:
        """Return all particles contained within a shape

        If the data argument is specified, will do additional
        containment-checks at the particle level

        Parameters
        ----------
        data: DataContainer | None
            Particle positions information. If `None`, only check whether node
            is within shape, which is equivalent to expanding the node
            start/stops from `_get_particle_indices_in_shape`

        containment_obj: BoundingVolume
            Provides an exact containment test (e.g. contained within a sphere).

        Returns
        -------
        indices: NDArray[int]]
            List of particle indices contained within shape. Will contain
            any additional particles that can be found in the same nodes if
            data is not provided
        """
        slices = self._get_particle_indices_in_shape(containment_obj)

        # the following is an attempt to mimic what I _think_ the expand_ranges
        # function from swiftsimio (which is GPL 3) does.
        num_particles = 0
        for s in slices:
            num_particles += s[1] - s[0]

        indices = np.empty((num_particles,), dtype=np.int64)
        ind = 0
        if data is None:
            #  ignore information about partial/full, just return indices as
            # fast as possible
            for s in slices:
                for i, index in enumerate(range(s[0], s[1])):
                    indices[ind + i] = index
                ind += s[1] - s[0]
            return indices

        for s in slices:
            if not s[2]:
                # fully enclosed
                for i, index in enumerate(range(s[0], s[1])):
                    indices[ind + i] = index
                ind += s[1] - s[0]
                continue
            positions = data._positions[s[0] : s[1], 0:3]
            i = 0
            for x, y, z in positions:
                if containment_obj.contains_point(x, y, z):
                    indices[ind] = i + s[0]
                    ind += 1
                i += 1  # noqa: SIM113

        return indices[0:ind]

    def _get_pilis_tree(
        self,
        data: DataContainer,
        odata: DataContainer,
        otree: PackedTreeNumba,
        r: float,
        d2_function: Callable[[float, float, float, float, float, float], float],
        strict: bool,  # noqa: FBT001, FBT002
    ) -> List[NDArray[np.int64]]:
        """Compute list of all pairs of points in data/odata whose distance < r

        Parameters
        ----------
        data, odata: DataContainer
            The actual particle data for self and other

        r: float
            The maximum distance

        d2_function: JITted Callable
            Function to compute the squared distance between 2 points. Expected
            signature [[float, float, float, float, float, float], float]

        strict: bool
            If False, compare only the approximate node distance. Should be
            significantly faster, but may include substantial amounts of false
            positives

        Returns
        -------
        results: list of arrayss
            For every point data[i], results[i] is the array of indices of points
            within r in odata

        """
        node = self._make_root_node()
        query = List.empty_list(np.int64[:])  # type: ignore
        box = self.box.copy()
        r2 = 2 * r
        rsq = r * r

        if is_leaf(node):
            _process_leaf_for_pilis_tree(
                node, data, box, r, r2, rsq, otree, odata, d2_function, strict, query
            )
            return query

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
                _process_leaf_for_pilis_tree(
                    node,
                    data,
                    box,
                    r,
                    r2,
                    rsq,
                    otree,
                    odata,
                    d2_function,
                    strict,
                    query,
                )

        return query

    def get_closest_particles(
        self,
        data: DataContainer,
        xyz: NDArray,
        distance_function: Callable[[float, float, float, float, float, float], float],
        distance_upper_bound: float,
        k: int = 1,
        use_shuffle: bool = False,  # noqa: FBT001, FBT002
        return_sorted: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        """Get kth nearest particle distances and indices to point

        Parameters
        ----------
        data: DataContainer
            Source of particle position data

        xyz: ArrayLike
            Coordinates of point to check

        distance_function: Callable[[float, float, float, float, float, float], float]
            Function to compute distance. **Must be njit compatible!**
            Needs to accept 6 coordinates: `x, y, z` and `px, py, pz` and return
            the distance as float64

        distance_upper_bound: nonnegative float, optional
            Return only neighbors from other nodes within this distance. This
            is used for tree pruning, so if you are doing a series of
            nearest-neighbor queries, it may help to supply the distance to the
            nearest neighbor of the most recent point.

        k: int, optional
            Number of closest particles to return. Default 1

        use_shuffle: bool, optional
            Flag to return shuffle indices instead of sorted data indices.
            Default False.

        return_sorted: bool, optional
            Return output in order by distance. Default False

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
        OctreeError
            if something goes wrong with the search process

        """
        # ensure point is in octree, project if not
        x, y, z = xyz
        px, py, pz = self.box.project_point_on_box(xyz)
        node = self._get_containing_node_of_point(np.array([px, py, pz]))
        node = node if node is not None else self._make_root_node()

        # because we need the kth nearest particles, make sure the node we're
        # looking at has at least that many particles. This means it might not
        # be a leaf, not that that should matter
        while len(node) < k and not is_root(node):
            _move_to_parent(self.tree, node)

        heap = FixedDistanceHeap(k, len(data))

        _process_slice_against_heap(
            heap,
            data,
            xyz,
            distance_function,
            node.node_start,
            node.node_end + 1,
            use_shuffle,
        )

        max_distance = heap.max_distance

        if max_distance == 0 or is_root(node):
            # We've found k particles that are as close as possible or we
            # already looked at all particles. We're done
            if return_sorted:
                # heap sorting will return early if max_distance==0, so this
                # won't do much extra work
                heap.sorted()
            return heap.distances, heap.indices

        dist = min(max_distance, distance_upper_bound)
        search_box = bbox.BoundingBox(
            np.array([x - dist, y - dist, z - dist, 2 * dist, 2 * dist, 2 * dist])
        )

        # clip search box to tree bounding box
        clip = search_box.clip_to_box(self.box)
        if not clip:
            raise octree.OctreeError("Search box entirely outside of tree")

        box_inds = self._get_particle_indices_in_shape(search_box)

        for s, e, _ in box_inds:
            # skip slice if it's a subslice of the node we already looked at
            if node.node_start <= s < node.node_end:
                continue
            _process_slice_against_heap(
                heap, data, xyz, distance_function, s, e, use_shuffle
            )

        distances, indices = heap.distances, heap.indices
        if return_sorted:
            distances, indices = heap.sorted()

        return distances, indices


try:
    packed_tree_type = as_numba_type(PackedTreeNumba)
except TypingError:
    packed_tree_type = type(PackedTreeNumba)


def _print_packed(packed: memoryview | array, *, expected_node_start=0):
    """Attempt to print a packed tree for debugging"""
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
