# ruff: noqa: D100, D101, D102, D103, D107

import logging

import conftest as ct
import numpy as np
import pytest
from hypothesis import assume, example, given, note, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hypnp
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    precondition,
    rule,
)
from numpy.typing import ArrayLike

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree
from packingcubes.data_objects import DataContainer, Dataset

LOGGER = logging.getLogger(__name__)


#############################
# Test _partition
#############################
@st.composite
def basic_data_container_with_invalid_bounds(draw):
    basic_data = draw(ct.basic_data_container_strategy())
    lower = draw(st.integers(min_value=-10, max_value=0))
    upper = draw(
        st.integers(min_value=len(basic_data) - 1, max_value=len(basic_data) + 10),
    )
    assume(lower < 0 or upper >= len(basic_data))
    return basic_data, lower, upper


@st.composite
def basic_data_container_with_valid_bounds(draw):
    basic_data = draw(ct.basic_data_container_strategy())
    lower = draw(st.integers(min_value=0, max_value=len(basic_data) - 1))
    upper = draw(st.integers(min_value=0, max_value=len(basic_data) - 1))
    return basic_data, lower, upper


@given(basic_data_container_with_invalid_bounds())
def test_partition_invalid_bounds(basic_data_with_bounds: tuple[Dataset, int, int]):
    basic_data, lower, upper = basic_data_with_bounds
    midplane = basic_data.bounding_box.midplane()

    with pytest.raises(ValueError, match="out of bounds"):
        partition = octree._partition(basic_data, lower, upper, 0, midplane[0])


@given(
    ct.basic_data_container_strategy(), st.integers().filter(lambda i: i < -3 or i >= 3)
)
def test_partition_axis_invalid(basic_data: Dataset, ax: int):
    positions = basic_data.positions
    midplane = basic_data.bounding_box.midplane()

    with pytest.raises(IndexError):
        partition = octree._partition(
            basic_data,
            0,
            len(basic_data) - 1,
            ax,
            midplane[ax],
        )


@given(
    basic_data_container_with_valid_bounds(),
    st.integers(min_value=0, max_value=2),
    st.floats(-1, 2),
)
def test_partition_valid(
    basic_data_with_bounds: tuple[DataContainer, int, int],
    ax: int,
    midplane_scale: float,
):
    basic_data, lower, upper = basic_data_with_bounds
    midplane = basic_data.bounding_box.midplane()

    positions = basic_data.positions
    note(f"x-values: {positions[lower : (upper + 1), ax]}")

    midplane = midplane[ax] * midplane_scale

    num_below = np.sum(positions[lower : (upper + 1), ax] < midplane)
    num_above = np.sum(positions[lower : (upper + 1), ax] >= midplane)
    note(f"{num_below=:} {num_above=:}")

    partition = octree._partition(basic_data, lower, upper, ax, midplane)

    note(f"part x-v: {positions[lower : (upper + 1), ax]}")
    note(f"{partition=:}")

    assert partition == num_below + lower
    assert np.all(positions[lower:partition, ax] < midplane)
    assert np.all(positions[partition : (upper + 1), ax] >= midplane)


#############################
# Test _partition_data
#############################
def is_sorted(a: ArrayLike) -> bool:
    # Note that this is fastest for most cases according to
    # https://stackoverflow.com/a/59589142
    return np.all(a[:-1] <= a[1:])


@given(ct.basic_data_container_strategy())
@settings(print_blob=True, deadline=None)
def test_partition_data_full_box(
    basic_data,
):
    note("Before partition")
    note(f"BoundingBox: {basic_data.bounding_box.box}")
    note(f"Positions :{basic_data.positions[:2, :]}")
    note(
        f"Normalized: {
            basic_data.bounding_box.normalize_to_box(basic_data.positions[:2, :])
        }",
    )
    morton_inds = octree.morton(basic_data.positions, basic_data.bounding_box)
    note(
        f"Morton: {morton_inds[:2]}",
    )

    child_list = octree._partition_data(
        basic_data,
        basic_data.bounding_box,
        0,
        len(basic_data) - 1,
    )

    note("After partition")
    note(
        f"Normalized: {
            basic_data.bounding_box.normalize_to_box(basic_data.positions[:2, :])
        }",
    )
    morton_inds = octree.morton(basic_data.positions, basic_data.bounding_box)
    note(
        f"Morton: {morton_inds[:2]}",
    )

    bin_count = np.bincount(morton_inds, minlength=9)
    # morton inds are 1-8, so bin 0 is always 0.
    expected_child_list = np.cumsum(bin_count)
    note(f"{bin_count=} e:{expected_child_list} g:{child_list}")

    assert is_sorted(morton_inds)
    assert np.all(expected_child_list == child_list)


@given(st.integers(min_value=1, max_value=8))
@settings(deadline=None)
def test_partition_data_sub_box(make_basic_data, child_ind: int):
    basic_data = make_basic_data(num_particles=10).data_container
    # Only want to look at particles in sub-box of full
    # do original partitioning first
    lvl1_child_list = octree._partition_data(
        basic_data,
        basic_data.bounding_box,
        0,
        9,
    )
    root_morton = octree.morton(basic_data.positions, basic_data.bounding_box)
    # Ensure we only test cases where we actually would have a sub_box, ie
    # the number of particles with (morton code == child_ind) > 1
    assume(np.count_nonzero(root_morton == child_ind) > 1)
    note(f"{basic_data.bounding_box=}")
    note(f"{root_morton=}")
    # e.g. for the DEADBEEF data we're interested in the child_ind=3 particles,
    # i.e. child_list[1]:child_list[2]
    note(f"For child box {child_ind}")
    child_ind = child_ind - 1  # convert to 0-index
    lvl1_child_box = basic_data.bounding_box.get_child_box(child_ind)
    note(f"{lvl1_child_box=}")
    lvl1_child_start = lvl1_child_list[child_ind - 1] if child_ind else 0
    lvl1_child_end = (
        max(lvl1_child_list[child_ind] - 1, 0) if child_ind < 7 else len(basic_data) - 1
    )
    note(f"lvl1_child_indices={lvl1_child_start}-{lvl1_child_end}")
    lvl1_child_pos = basic_data.positions[lvl1_child_start : lvl1_child_end + 1].copy()
    normalized_pos = (lvl1_child_pos - lvl1_child_box.box[:3]) / lvl1_child_box.box[3:]
    lvl1_morton = octree.morton(lvl1_child_pos, lvl1_child_box)
    note("Prior to partition:")
    for i, (pos, norm, mort) in enumerate(
        zip(lvl1_child_pos, normalized_pos, lvl1_morton, strict=True),
    ):
        note(f"{i=} {pos}->{norm} = {mort}")
    expected_order = np.sort(lvl1_morton)
    note(f"{expected_order=}")
    child_list = octree._partition_data(
        basic_data,
        lvl1_child_box,
        lvl1_child_start,
        lvl1_child_end,
    )
    lvl1_child_pos = basic_data.positions[lvl1_child_start : lvl1_child_end + 1].copy()
    normalized_pos = (lvl1_child_pos - lvl1_child_box.box[:3]) / lvl1_child_box.box[3:]
    lvl1_morton = octree.morton(lvl1_child_pos, lvl1_child_box)
    note("After partition:")
    for i, (pos, norm, mort) in enumerate(
        zip(lvl1_child_pos, normalized_pos, lvl1_morton, strict=True),
    ):
        note(f"{i=} {pos}->{norm} = {mort}")
    note(
        f"{child_list=} -> {[c - lvl1_child_list[1] for c in child_list]} (normalized)",
    )
    note(f"actual order{lvl1_morton}")
    note("")
    assert np.all(expected_order == lvl1_morton)


#############################
# Test morton
#############################


@example(np.array([[np.nan, 0, 0]]), np.array([0, 0, 0, 1, 1, 1])).xfail(
    reason="NaNs/Infs propagate into indices",
)
@example(np.array([[0, 0, 0]]), np.array([0, 0, 0, 1, -1, 1])).xfail(
    reason="Invalid boxes fail",
    raises=bbox.BoundingBoxError,
)
# The following examples no longer "fail". It's unclear whether the morton
# indices make _sense_, but morton() will happily compute them...
@example(np.array([[0, 0, np.inf]]), np.array([0, 0, 0, 1, 1, 1]))
@example(np.array([[-1, -2e-60, -1]]), np.array([-1, -1, -1, 2, 2, 2]))
@given(
    ct.valid_positions(),
    ct.valid_boxes() | ct.valid_bounding_boxes(),
)
def test_morton(positions: ArrayLike, box: ArrayLike):
    box = bbox.make_bounding_box(box)
    midplane = box.midplane()
    note(f"Midplane for this test is {midplane}")

    mortons = octree.morton(positions, box)

    for morton, pos in zip(mortons, positions, strict=True):
        assert 1 <= morton <= 8
        match morton:
            case 1:
                assert np.all(pos < midplane)
            case 2:
                assert pos[0] >= midplane[0]
                assert pos[1] < midplane[1]
                assert pos[2] < midplane[2]
            case 3:
                assert pos[0] < midplane[0]
                assert pos[1] >= midplane[1]
                assert pos[2] < midplane[2]
            case 4:
                assert pos[0] >= midplane[0]
                assert pos[1] >= midplane[1]
                assert pos[2] < midplane[2]
            case 5:
                assert pos[0] < midplane[0]
                assert pos[1] < midplane[1]
                assert pos[2] >= midplane[2]
            case 6:
                assert pos[0] >= midplane[0]
                assert pos[1] < midplane[1]
                assert pos[2] >= midplane[2]
            case 7:
                assert pos[0] < midplane[0]
                assert pos[1] >= midplane[1]
                assert pos[2] >= midplane[2]
            case 8:
                assert np.all(pos >= midplane)


#############################
#############################
# Test PythonOctreeNode
#############################
#############################
@given(
    data=ct.basic_data_container_strategy(),
    node_start=st.integers(),
    node_end=st.integers(),
    box=ct.valid_bounding_boxes(),
    tag=st.text(alphabet="012345678", min_size=1, max_size=20),
    parent=st.none(),
    particle_threshold=st.integers(),
)
def test_PythonOctreeNode_invalid_ints(
    data: DataContainer,
    node_start: int,
    node_end: int,
    box: bbox.BoundingBox,
    tag: str,
    parent: None | octree.PythonOctreeNode,
    particle_threshold: int,
):
    assume(particle_threshold < 1 or node_start < 0 or node_end > len(data) - 1)
    with pytest.raises(octree.OctreeError) as oerrinfo:
        octree.PythonOctreeNode(
            data=data,
            node_start=node_start,
            node_end=node_end,
            box=box,
            tag=tag,
            parent=parent,
            particle_threshold=particle_threshold,
        )
    if particle_threshold < 1:
        assert "particle_threshold" in str(oerrinfo.value)
    elif node_start < 0:
        assert "Invalid start" in str(oerrinfo.value)
    elif node_end > len(data) - 1:
        assert "Invalid end" in str(oerrinfo.value)
    else:
        pytest.fail(str(oerrinfo.value))


def test_PythonOctreeNode_empty_data(make_basic_data):
    data = make_basic_data(0).data_container

    with pytest.raises(octree.OctreeError) as oerrinfo:
        octree.PythonOctreeNode(
            data=data,
            node_start=0,
            node_end=len(data) - 1,
            box=data.bounding_box,
        )
    assert "Empty DataContainer" in str(oerrinfo.value)


def make_worst_case_duplicate() -> Dataset:
    """
    Worst case duplicate would be some sort of 3D Cantor dust-like fractal
    structure but we don't have time to test that, so we'll just do the lop
    layer -> 8 vertices of the unit cube that are duplicated
    """
    data = Dataset(filepath="")

    positions = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for _ in range(2):
                    positions.append([i, j, k])
    positions = np.array(positions, dtype=float)

    data._box = bbox.make_bounding_box([0, 0, 0, 1, 1, 1])

    data._positions = positions
    data._setup_index()

    return data


@example(make_worst_case_duplicate())
@given(data=ct.data_with_duplicates(max_particles=3))
# This will cause the test to take a long time but is
# required for make_worst_case_duplicate
@settings(deadline=None, print_blob=True)
@pytest.mark.filterwarnings("::packingcubes.octree.OctreeWarning::")
@pytest.mark.filterwarnings("error")
def test_PythonOctreeNode_duplicate_data(data: Dataset):
    note(f"{data.bounding_box=}")
    note(f"{data.positions=}")
    note(f"{data.bounding_box.max_depth()}")

    # if all particles are identical, then the box shouldn't support splitting
    # but we still want to test the recursion limit
    if data.bounding_box.max_depth() < 1:
        with pytest.raises(octree.OctreeError) as oeinfo:
            node = octree.PythonOctreeNode(
                data=data.data_container, particle_threshold=1
            )
        assert "negative max depth" in str(oeinfo.value)
    else:
        with pytest.warns(octree.OctreeWarning, match=r"Bad data detected"):  # noqa PT031
            # need to explicitly set the particle threshold since
            # data_with_duplicates is only guaranteed to produce 1 duplicate
            node = octree.PythonOctreeNode(
                data=data.data_container,
                particle_threshold=1,
            )
            # if no warning was raised we want to know why
            note(f"{node=}")
            for child in node.children:
                note(f"{child}")


#############################
#############################
# Test PythonOctree
#############################
#############################
class PythonOctreeComparison(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.tree = None
        self.current_node = None

    @initialize(
        dataset=ct.basic_data_strategy(), pt=st.none() | st.integers(min_value=1)
    )
    @pytest.mark.filterwarnings("ignore:Bad data")
    def make_octree(self, dataset: Dataset, pt: int):
        assume(dataset.bounding_box.max_depth() > 0)
        try:
            self.tree = octree.PythonOctree(dataset=dataset, particle_threshold=pt)
        except octree.OctreeError as oerr:
            if len(dataset) <= 10:
                note(dataset.positions)
            raise oerr
        self.pt = pt if pt is not None else octree._DEFAULT_PARTICLE_THRESHOLD
        self.current_node = self.tree.root

    @rule()
    def validate_root_attributes(self):
        root = self.tree.root
        data = root.data
        assert root.node_start == 0
        assert root.node_end == len(data) - 1
        assert root.box.box == pytest.approx(data.bounding_box.box.astype(float))
        assert root.tag == "0"
        assert root.parent is None
        assert root._particle_threshold == self.pt

    @rule(child_index=st.integers(min_value=0, max_value=7))
    @precondition(
        lambda oc: oc.current_node is not None and not oc.current_node.is_leaf
    )
    def go_to_child(self, child_index: int):
        note(self.current_node)
        note(f"{self.current_node.is_leaf}")
        self.current_node = self.current_node.children[child_index]

    @rule()
    @precondition(
        lambda oc: oc.current_node is not None and oc.current_node.parent is not None
    )
    def go_to_parent(self):
        self.current_node = self.current_node.parent

    @rule()
    @precondition(lambda oc: oc.tree is not None)
    @precondition(lambda oc: oc.current_node is not None)
    def node_data_sorted(self):
        node = self.current_node
        positions = node.data.positions[node.node_start : node.node_end + 1]
        box = node.box
        mortons = octree.morton(positions=positions, box=box)
        note(node)
        note(box)
        note(positions)
        note(mortons)
        assert is_sorted(mortons)

    @rule()
    @precondition(lambda oc: oc.tree is not None)
    def get_leaves(self):
        leaves = self.tree.get_leaves()
        assert len(leaves)

    @rule()
    @precondition(lambda oc: oc.tree is not None)
    def iter(self):
        for _ in self.tree:
            assert True

    @rule(
        normal_point=hypnp.arrays(
            float, 3, elements=st.floats(min_value=0, max_value=1)
        )
    )
    @precondition(lambda oc: oc.tree is not None)
    def get_containing_node(self, normal_point):
        box = self.tree.root.box.box
        point = box[3:] * normal_point + box[:3]

        with pytest.raises(ValueError, match="bottom-up"):
            self.tree._get_containing_node_of_point(point, top_down=False)

        node = self.tree._get_containing_node_of_point(point)

        assert node.box.contains(point)

        node1 = self.tree._get_containing_node_of_point(
            point, top_down=False, start_node=node
        )

        assert node == node1

        nan_point = np.array([np.nan, np.nan, np.nan])
        node_nan = self.tree._get_containing_node_of_point(nan_point)
        assert node_nan is None
        node_nan = self.tree._get_containing_node_of_point(
            nan_point, top_down=False, start_node=node
        )
        assert node_nan is None

    @rule(
        normal_points=hypnp.arrays(
            float,
            st.tuples(st.integers(min_value=1, max_value=10), st.just(3)),
            elements=st.floats(min_value=0, max_value=1),
        )
    )
    @precondition(lambda oc: oc.tree is not None)
    def get_containing_node_pointlist(self, normal_points):
        box = self.tree.root.box.box
        points = box[3:] * normal_points + box[:3]

        node = self.tree._get_containing_node_of_pointlist(points)

        assert np.all(node.box.contains(points))

        node1 = self.tree._get_containing_node_of_pointlist(
            points, top_down=False, start_node=node
        )

        assert node == node1

    @rule(search_box=ct.valid_bounding_boxes())
    @precondition(lambda oc: oc.tree is not None)
    def get_particle_indices_in_box(self, search_box):
        positions = self.tree.root.data.positions

        ind_tuples = self.tree.get_particle_indices_in_box(box=search_box)
        note("Returned tuples:")
        note(ind_tuples)
        # this returns a list of particle index tuples
        # We don't actually need to know if any of the returned particles are
        # in the sphere, we only need to check that we're not missing any
        actual = np.where(search_box.contains(positions))[0]
        note("Actual contained indices:")
        note(actual)
        if len(actual):
            assert len(ind_tuples)
            tuple_arr = np.atleast_2d(np.array(ind_tuples))
            for pt in actual:
                assert np.any((tuple_arr[:, 0] <= pt) & (pt <= tuple_arr[:, 1]))

    @rule(
        normal_center=hypnp.arrays(
            float, 3, elements=st.floats(min_value=0, max_value=1)
        ),
        normal_radius=st.floats(min_value=4 * np.finfo(np.float32).eps, max_value=2),
    )
    @precondition(lambda oc: oc.tree is not None)
    def get_particle_indices_in_sphere(self, normal_center, normal_radius):
        box = self.tree.root.box.box
        center = box[3:] * normal_center + box[:3]
        dist = np.maximum(
            np.sqrt(np.sum((center - box[:3]) ** 2)),
            np.finfo(np.float32).eps * np.max(np.abs(box[:3])),
        )
        radius = np.maximum(normal_radius * dist, np.finfo(np.float32).eps)
        note(f"Center: {center}")
        note(f"Radius: {radius}")
        note(f"dist: {dist}")

        sph = bbox.make_bounding_sphere(radius, center=center)

        positions = self.tree.root.data.positions

        ind_tuples = self.tree.get_particle_indices_in_sphere(
            center=center, radius=radius
        )
        note("Returned tuples:")
        note(ind_tuples)
        # this returns a list of particle index tuples
        # We don't actually need to know if any of the returned particles are
        # in the sphere, we only need to check that we're not missing any
        actual = np.where(sph.contains(positions))[0]
        note("Actual contained indices:")
        note(actual)
        if len(actual):
            assert len(ind_tuples)
            tuple_arr = np.atleast_2d(np.array(ind_tuples))
            for pt in actual:
                assert np.any((tuple_arr[:, 0] <= pt) & (pt <= tuple_arr[:, 1]))


TestPythonOctreeComparison = PythonOctreeComparison.TestCase
