from __future__ import annotations

import logging
from array import array
from collections.abc import Buffer, Generator, Iterable, Iterator
from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree
from packingcubes.configuration import FIELD_FORMAT
from packingcubes.data_objects import Dataset

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(capture=True)


class PackedNode(octree.OctreeNode):
    def __init__(
        self,
        *,
        node_start: int | None = None,
        node_end: int,
        box: bbox.BoxLike,
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
        self._node_start = 0 if node_start is None else node_start
        self._node_end = node_end
        self._box = bbox.make_valid(box)
        self._tag = "0" if tag is None else tag
        self._index = np.uint(0)

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
        List of 1-based z-order indices describing the current box.
        E.g. if assuming the unit bounding box, the box
        [0.25, 0.25, 0.75, 0.25, 0.25, 0.25] would be [5,1]
        """
        return self._tag

    def copy(self) -> PackedNode:
        node = PackedNode(
            node_start=self.node_start,
            node_end=self.node_end,
            box=self.box.copy(),
            tag=self.tag,
        )
        node._index = self._index
        return node


def _convert_list_to_tag_str(tag: list[int]) -> str:
    """
    Convert list of ints to a str
    """
    return "0" + "".join(str(t) for t in tag)


def _create_from_current_node(node: CurrentNode) -> PackedNode:
    packed = PackedNode(
        node_start=int(node.node_start),
        node_end=int(node.node_end),
        box=node.box.copy(),
        tag=_convert_list_to_tag_str(node.tag),
    )
    packed._index = np.uint(node.index)
    return packed


@dataclass
class CurrentNode:
    node_end: int
    tag: list[int]
    box: bbox.BoundingBox
    index: int = 0
    node_start: int = 0
    child_flag: int = 0
    my_index: int = 0
    level: int = 0
    empty: int = 0


def get_name(current: CurrentNode) -> str:
    """
    Get the name (tag) of this CurrentNode
    """
    return _convert_list_to_tag_str(current.tag)


def get_children(current: CurrentNode) -> Generator[int]:
    """
    Return a generator of 0-based children indices for this CurrentNode
    """
    return (i for i in range(8) if current.child_flag & (1 << i))


def is_leaf(current: CurrentNode) -> bool:
    return not bool(current.child_flag)


def is_root(current: CurrentNode) -> bool:
    return not current.index


class PackedTree(octree.Octree):
    """
    Public octree interface

    This interface defines the methods for creating, manipulating, and traversing a
    packingcubes octree.

    Attributes:
        node: numpy.uint
        The address of the current node we're on

        data: Dataset
        The backing dataset

        tree: array
        The actual tree in memory

    """

    data: Dataset
    """ Backing dataset """
    tree: memoryview
    """ Tree representation in memory """

    def __init__(
        self,
        data: Dataset,
        *,
        source: Buffer | None,
        particle_threshold: int | None = None,
        show_pbar: bool = False,
    ) -> None:
        if particle_threshold is None:
            self.particle_threshold = octree._DEFAULT_PARTICLE_THRESHOLD

        pbar = None
        if not source and show_pbar:
            pbar = tqdm(total=len(data), miniters=1000)

        self.data = data

        # from some empirical testing. doesn't need to be exact anyway
        # estimated_size_in_bytes = (
        #     len(self.data) / 10 ** (np.floor(np.log10(self.particle_threshold))) * 1.2
        # ).astype(int) * 20
        # self.tree = np.array((estimated_size_in_bytes,), dtype=np.bytes_)

        if not source:
            # initialize to unsigned longs
            self.tree = memoryview(array(FIELD_FORMAT))
            self._construct_tree(pbar)
        else:
            self.tree = memoryview(source)
            if self.tree.format in "bBc":
                self.tree = self.tree.cast(FIELD_FORMAT)

        if pbar is not None:
            pbar.close()
        pass

    def _construct_tree(self, pbar: tqdm | None):
        # self.current = PackedNode(
        #     node_end=len(self.data) - 1, box=self.data.bounding_box
        # )
        # self.root = self.current.copy()
        self.current_node = CurrentNode(
            index=0,
            node_start=0,
            node_end=len(self.data),
            box=self.data.bounding_box,
            tag=[],
        )

        raise NotImplementedError

    def _update_current_node(self, index: int, node: CurrentNode, child_index: int):
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
        node.node_start, node.node_end, metadata = self.tree[(index + 1) : (index + 4)]
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

    def _move_to_child(self, node: CurrentNode, child_ind: int) -> int:
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
        child_flag = self.tree[node.index + 3] >> 24
        # could also use current_node.child_flag
        if not child_flag & (1 << child_ind):
            return 0

        num_skip = np.bitwise_count((255 >> 8 - child_ind) & child_flag)

        # children start at boundary + 4 fields
        current = old = node.index
        current += 4
        for _ in range(num_skip):
            current += self.tree[current]

        self._update_current_node(current, node, child_ind + 1)
        return current - old

    def _move_to_parent(self, node: CurrentNode):
        """
        Move pointer to parent node and return offset (0 if at root)
        """
        # currently at node boundary
        # amount to move back is at end of node, or skip_length-1 fields from
        # self.current
        node_len = self.tree[node.index]
        pl = self.tree[node.index + node_len - 1]
        if pl:
            # only move up if we're not already at root
            self._update_current_node(node.index - pl, node, 0)

        return pl

    def _make_root_node(self) -> CurrentNode:
        # child flag located at field 3
        child_flag, my_index, level, empty = octree.unpack_node_metadata(self.tree[3])
        return CurrentNode(
            node_start=self.tree[1],
            node_end=self.tree[2],
            index=0,
            child_flag=child_flag,
            my_index=my_index,
            level=level,
            empty=empty,
            tag=[],
            box=self.data.bounding_box,
        )

    def __iter__(self) -> Iterator[PackedNode]:
        """
        Return all nodes as pre-order tree traversal
        """
        return map(_create_from_current_node, self._iterate_nodes())

    def _iterate_nodes(self) -> Iterator[CurrentNode]:
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

    def get_leaves(self) -> Iterable[PackedNode]:
        return map(_create_from_current_node, filter(is_leaf, self._iterate_nodes()))

    def get_node(self, tag: str) -> PackedNode | None:
        node = self._make_root_node()
        for child in tag:
            offset = self._move_to_child(node, int(child))
            if not offset:
                return None
        return _create_from_current_node(node)

    def _top_down_containing_node(self, node: CurrentNode, xyz: ArrayLike):
        """
        For a given point, return the smallest child node that contains point or None
        """
        if not bbox.in_box(node.box, xyz):
            return None
        while not is_leaf(node):
            children = get_children(node)
            for child in children:
                self._move_to_child(node, child)
                if bbox.in_box(node.box, xyz):
                    break
                self._move_to_parent(node)
            else:
                break  # while loop
        return node if bbox.in_box(node.box, xyz) else None

    def _bottom_up_containing_node(self, node: CurrentNode, xyz: ArrayLike):
        """
        For a given point, return the smallest parent node that contains point or None
        """
        while not is_root(node):
            if bbox.in_box(node.box, xyz):
                return node
            self._move_to_parent(node)
        return node if bbox.in_box(node.box, xyz) else None

    def _get_containing_node_of_point(
        self,
        xyz: ArrayLike,
        *,
        start_node: CurrentNode | None = None,
        top_down: bool = True,
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
        node = self._bottom_up_containing_node(node, xyz)
        if node is not None:
            return self._top_down_containing_node(node, xyz)
        return None

    def _get_containing_node_of_pointlist(
        self,
        points: NDArray,
        *,
        start_node: CurrentNode | None = None,
        top_down: bool = True,
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
                    start_node=start_node,
                    top_down=top_down,
                )
            else:
                node = self._bottom_up_containing_node(node, point)
        return node if node is not None else self._make_root_node()

    def _get_nodes_in_shape(
        self,
        *,
        bounding_box: bbox.BoxLike,
        containment_test: octree.ContainmentFunc | None = None,
    ) -> tuple[list[PackedNode], list[PackedNode]]:
        """
        Return lists of all nodes entirely inside and partially inside shape

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
            containment_test(point: ArrayLike) -> NDArray[bool]
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
        if containment_test is None:

            def in_box(xyz: ArrayLike) -> NDArray[np.bool_]:
                return bbox.in_box(bounding_box, xyz)

            containment_test = in_box

        # node containing bounding box
        bbox_center = bbox.get_box_center(bounding_box)
        node = self._make_root_node()

        entire_nodes = []
        partial_leaves = []
        child_queue = [get_children(node)]
        while child_queue:
            children = child_queue[-1]
            try:
                child = next(children)
            except StopIteration:
                child_queue.pop()
                self._move_to_parent(node)
                continue
            self._move_to_child(node, child)

            # Test if node entirely contained in shape
            node_vertices = bbox.get_box_vertices(node.box)

            vertices_enclosed = sum(containment_test(node_vertices))

            if vertices_enclosed:
                if vertices_enclosed == len(node_vertices):
                    entire_nodes.append(_create_from_current_node(node))
                    self._move_to_parent(node)
                else:
                    if is_leaf(node):
                        partial_leaves.append(_create_from_current_node(node))
                    # don't need to reverse input since using generator
                    child_queue.append(get_children(node))
                continue

            # Also need to check closest point. Should take care of overlapping
            # edges
            closest_point = bbox.project_point_on_box(node.box, bbox_center)
            if containment_test(closest_point):
                if is_leaf(node):
                    partial_leaves.append(_create_from_current_node(node))
                child_queue.append(get_children(node))
                continue

            # no overlap, discard
            self._move_to_parent(node)

        return entire_nodes, partial_leaves

    def _get_nodes_in_sphere(
        self,
        *,
        center: NDArray,
        radius: float,
    ) -> tuple[list[PackedNode], list[PackedNode]]:
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
                ],
            ),
        )

        containment_test = partial(
            octree._point_in_sphere, center=center, radius=radius
        )

        return self._get_nodes_in_shape(
            bounding_box=bounding_box,
            containment_test=containment_test,
        )

    def _get_particle_indices_in_shape(
        self,
        *,
        bounding_box: bbox.BoxLike,
        containment_test: octree.ContainmentFunc | None = None,
    ) -> list[list[int]]:
        """
        Return all particles contained within a shape that fits inside bounding box

        Args:
            bounding_box: BoxLike
            Shape bounding box

            containment_test: Callable[NDArray], optional
            Function to test if point(s) are inside shape. Should have the
            (vectorized) signature
            containment_test(point: ArrayLike) -> NDArray[bool]
            Defaults to testing if point(s) are inside the provided bounding
            box

        Returns:
            indices: list[tuple[int]]
            List of particle start-stop indices contained within shape
        """

        if containment_test is None:

            def in_box(xyz: ArrayLike) -> NDArray[np.bool_]:
                return bbox.in_box(bounding_box, xyz)

            containment_test = in_box

        # node containing bounding box
        bbox_center = bbox.get_box_center(bounding_box)
        node = self._make_root_node()

        indices = []
        child_queue = [get_children(node)]
        while child_queue:
            children = child_queue[-1]
            try:
                child = next(children)
            except StopIteration:
                child_queue.pop()
                self._move_to_parent(node)
                continue
            self._move_to_child(node, child)

            # Check closest point.
            closest_point = bbox.project_point_on_box(node.box, bbox_center)
            if not containment_test(closest_point):
                # no overlap, discard
                self._move_to_parent(node)
                continue

            # Test if node entirely contained in shape
            node_vertices = bbox.get_box_vertices(node.box)

            vertices_enclosed = sum(containment_test(node_vertices))

            if is_leaf(node) or vertices_enclosed == len(node_vertices):
                indices.append([node.node_start, node.node_end + 1])
                if vertices_enclosed == len(node_vertices):
                    # we're done with this subtree, go back
                    self._move_to_parent(node)
                    continue
            # look at children - leaves return an empty generator
            child_queue.append(get_children(node))

        # indices is now a list of numpy index arrays. Stack'em
        # Note that all elements should be unique and pre-sorted due to octree
        # construction
        return indices

    def _get_particle_indices_in_shape_verts(
        self,
        *,
        bounding_box: bbox.BoxLike,
        containment_test: octree.ContainmentFunc | None = None,
    ) -> list[list[int]]:
        """
        Return all particles contained within a shape that fits inside bounding box

        Args:
            bounding_box: BoxLike
            Shape bounding box

            containment_test: Callable[NDArray], optional
            Function to test if point(s) are inside shape. Should have the
            (vectorized) signature
            containment_test(point: ArrayLike) -> NDArray[bool]
            Defaults to testing if point(s) are inside the provided bounding
            box

        Returns:
            indices: list[tuple[int,int]]
            List of particle start-stop indices contained within shape
        """

        if containment_test is None:

            def in_box(xyz: ArrayLike):
                return bbox.in_box(bounding_box, xyz)

            containment_test = in_box

        # node containing bounding box
        bbox_center = bbox.get_box_center(bounding_box)
        node = self._make_root_node()

        indices = []
        child_queue = [get_children(node)]
        while child_queue:
            children = child_queue[-1]
            try:
                child = next(children)
            except StopIteration:
                child_queue.pop()
                self._move_to_parent(node)
                continue
            self._move_to_child(node, child)

            # Test if node entirely contained in shape
            node_vertices = bbox.get_box_vertices(node.box)

            vertices_enclosed = sum(containment_test(node_vertices))

            if vertices_enclosed:
                if is_leaf(node) or vertices_enclosed == len(node_vertices):
                    indices.append([node.node_start, node.node_end + 1])
                    if vertices_enclosed == len(node_vertices):
                        # we're done with this subtree, go back
                        self._move_to_parent(node)
                        continue
                # look at children - leaves return an empty generator
                child_queue.append(get_children(node))
                continue

            # No vertex overlap, check closest point.
            closest_point = bbox.project_point_on_box(node.box, bbox_center)
            if not containment_test(closest_point):
                # no overlap, discard
                self._move_to_parent(node)
                continue
            if is_leaf(node):
                indices.append([node.node_start, node.node_end + 1])
            child_queue.append(get_children(node))

        # indices is now a list of numpy index arrays. Stack'em
        # Note that all elements should be unique and pre-sorted due to octree
        # construction
        return indices

    def get_particle_indices_in_box(
        self,
        *,
        box: bbox.BoxLike,
    ) -> list[list[int]]:
        return self._get_particle_indices_in_shape_verts(
            bounding_box=bbox.make_valid(box).copy(),
        )

    def get_particle_indices_in_sphere(
        self,
        *,
        center: NDArray,
        radius: float,
    ) -> list[list[int]]:
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
                ],
            ),
        )

        containment_test = partial(
            octree._point_in_sphere, center=center, radius=radius
        )

        return self._get_particle_indices_in_shape_verts(
            bounding_box=bounding_box,
            containment_test=containment_test,
        )

    def get_closest_particle(
        self, xyz: ArrayLike, *, check_neighbors: bool = True
    ) -> tuple[np.int_, float]:
        raise NotImplementedError
