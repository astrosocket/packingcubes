import logging

import conftest as ct
import numpy as np
import pytest
from hypothesis import assume, example, given, note, settings
from hypothesis import strategies as st
from numpy.typing import ArrayLike

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree
from packingcubes.data_objects import Dataset

LOGGER = logging.getLogger(__name__)


#############################
# Test _partition
#############################
@st.composite
def basic_data_with_invalid_bounds(draw):
    basic_data = draw(ct.basic_data_strategy())
    lower = draw(st.integers(min_value=-10, max_value=0))
    upper = draw(
        st.integers(min_value=len(basic_data) - 1, max_value=len(basic_data) + 10)
    )
    assume(lower < 0 or upper >= len(basic_data))
    return basic_data, lower, upper


@st.composite
def basic_data_with_valid_bounds(draw):
    basic_data = draw(ct.basic_data_strategy())
    lower = draw(st.integers(min_value=0, max_value=len(basic_data) - 1))
    upper = draw(st.integers(min_value=0, max_value=len(basic_data) - 1))
    return basic_data, lower, upper


@given(basic_data_with_invalid_bounds())
def test_partition_invalid_bounds(basic_data_with_bounds: tuple[Dataset, int, int]):
    basic_data, lower, upper = basic_data_with_bounds
    midplane = bbox.midplane(basic_data.bounding_box)

    with pytest.raises(ValueError):
        partition = octree._partition(basic_data, lower, upper, 0, midplane[0])


@given(ct.basic_data_strategy(), st.integers().filter(lambda i: i < -3 or i >= 3))
def test_partition_axis_invalid(basic_data: Dataset, ax: int):
    positions = basic_data.positions
    midplane = bbox.midplane(basic_data.bounding_box)

    with pytest.raises(IndexError):
        partition = octree._partition(
            basic_data, 0, len(basic_data) - 1, ax, midplane[ax]
        )


@given(
    basic_data_with_valid_bounds(),
    st.integers(min_value=0, max_value=2),
    st.floats(-1, 2),
)
def test_partition_valid(
    basic_data_with_bounds: tuple[Dataset, int, int], ax: int, midplane_scale: float
):
    basic_data, lower, upper = basic_data_with_bounds
    midplane = bbox.midplane(basic_data.bounding_box)

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


@given(ct.basic_data_strategy())
@settings(print_blob=True)
def test_partition_data_full_box(
    basic_data,
):
    note("Before partition")
    note(basic_data.bounding_box)
    # note(bbox.normalize_to_box(basic_data.positions, basic_data.bounding_box))
    note(basic_data.positions)
    morton_inds = octree.morton(basic_data.positions, basic_data.bounding_box)
    note(morton_inds)
    note(f"Positions :{basic_data.positions[1:2, :]}")
    note(
        f"Normalized: {bbox.normalize_to_box(basic_data.positions[1:2, :], basic_data.bounding_box)}"
    )
    note(
        f"Morton: {octree.morton(basic_data.positions[1:2, :], basic_data.bounding_box)}"
    )

    child_list = octree._partition_data(
        basic_data,
        basic_data.bounding_box,
        0,
        len(basic_data) - 1,
    )

    note("After partition")
    note(basic_data.bounding_box)
    note(bbox.normalize_to_box(basic_data.positions, basic_data.bounding_box))
    morton_inds = octree.morton(basic_data.positions, basic_data.bounding_box)
    note(morton_inds)

    bin_count = np.bincount(morton_inds, minlength=8)
    # morton inds are 1-8, so bin 0 is always 0. Even though anything
    # in bin 1 should _start_ at index=0 (since we're doing the top-level
    # partition), child_list doesn't contain an explicit index for bin 1
    expected_child_list = np.cumsum(bin_count)[1:8]
    note(f"{bin_count=} e:{expected_child_list} g:{child_list}")

    assert is_sorted(morton_inds)
    assert np.all(expected_child_list == child_list)


@given(st.integers(min_value=1, max_value=8))
def test_partition_data_sub_box(make_basic_data, child_ind: int):
    basic_data = make_basic_data(num_particles=10)
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
    lvl1_child_box = octree._get_child_box(basic_data.bounding_box, child_ind)
    note(f"{lvl1_child_box=}")
    lvl1_child_start = lvl1_child_list[child_ind - 1] if child_ind else 0
    lvl1_child_end = (
        max(lvl1_child_list[child_ind] - 1, 0) if child_ind < 7 else len(basic_data) - 1
    )
    note(f"lvl1_child_indices={lvl1_child_start}-{lvl1_child_end}")
    lvl1_child_pos = basic_data.positions[lvl1_child_start : lvl1_child_end + 1].copy()
    normalized_pos = (lvl1_child_pos - lvl1_child_box[:3]) / lvl1_child_box[3:]
    lvl1_morton = octree.morton(lvl1_child_pos, lvl1_child_box)
    note("Prior to partition:")
    for i, (pos, norm, mort) in enumerate(
        zip(lvl1_child_pos, normalized_pos, lvl1_morton)
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
    normalized_pos = (lvl1_child_pos - lvl1_child_box[:3]) / lvl1_child_box[3:]
    lvl1_morton = octree.morton(lvl1_child_pos, lvl1_child_box)
    note("After partition:")
    for i, (pos, norm, mort) in enumerate(
        zip(lvl1_child_pos, normalized_pos, lvl1_morton)
    ):
        note(f"{i=} {pos}->{norm} = {mort}")
    note(
        f"{child_list=} -> {[c - lvl1_child_list[1] for c in child_list]} (normalized)"
    )
    note(f"actual order{lvl1_morton}")
    note("")
    assert np.all(expected_order == lvl1_morton)


#############################
# test _get_child_box
#############################
@given(ct.invalid_boxes())
def test_get_child_box_invalid_box(box: ArrayLike):
    with pytest.raises(bbox.BoundingBoxError):
        octree._get_child_box(box, 0)


@given(ct.valid_boxes())
def test_get_child_box_valid(box: ArrayLike):
    # box should be of the form [x, y, z, dx, dy, dz]
    x, y, z, dx, dy, dz = box
    dx2, dy2, dz2 = dx / 2.0, dy / 2.0, dz / 2.0
    child_boxes = [
        [x, y, z],  # 1
        [x + dx2, y, z],  # 2
        [x, y + dy2, z],  # 3
        [x + dx2, y + dy2, z],  # 4
        [x, y, z + dz2],  # 5
        [x + dx2, y, z + dz2],  # 6
        [x, y + dy2, z + dz2],  # 7
        [x + dx2, y + dy2, z + dz2],  # 8
    ]
    dxbox = [dx2, dy2, dz2]
    for i in range(8):
        child_box = octree._get_child_box(box, i)
        assert child_box[:3] == pytest.approx(child_boxes[i])
        assert child_box[3:] == pytest.approx(dxbox)


#############################
# Test _box_neighbors_in_node
#############################
@given(st.integers().filter(lambda i: i < 1 or i > 8))
def test_box_neighbors_in_node_invalid(ind: int):
    with pytest.raises(ValueError):
        octree._box_neighbors_in_node(ind)


@pytest.mark.skip(reason="Not implemented yet")
def test_box_neighbors_in_node():
    pass


#############################
# Test morton
#############################


@example(np.array([[np.nan, 0, 0]]), np.array([0, 0, 0, 1, 1, 1])).xfail(
    reason="NaNs/Infs propagate into indices"
)
@example(np.array([[0, 0, 0]]), np.array([0, 0, 0, 1, -1, 1])).xfail(
    reason="Invalid boxes fail", raises=bbox.BoundingBoxError
)
# The following examples no longer "fail". It's unclear whether the morton
# indices make _sense_, but morton() will happily compute them...
@example(np.array([[0, 0, np.inf]]), np.array([0, 0, 0, 1, 1, 1]))
@example(np.array([[-1, -2e-60, -1]]), np.array([-1, -1, -1, 2, 2, 2]))
@given(
    ct.valid_positions(),
    ct.valid_boxes(),
)
def test_morton(positions: ArrayLike, box: ArrayLike):
    midplane = bbox.midplane(box)
    note(f"Midplane for this test is {midplane}")

    mortons = octree.morton(positions, box)

    for morton, pos in zip(mortons, positions):
        assert 1 <= morton <= 8
        match morton:
            case 1:
                assert np.all(pos < midplane)
            case 2:
                assert (
                    pos[0] >= midplane[0]
                    and pos[1] < midplane[1]
                    and pos[2] < midplane[2]
                )
            case 3:
                assert (
                    pos[0] < midplane[0]
                    and pos[1] >= midplane[1]
                    and pos[2] < midplane[2]
                )
            case 4:
                assert (
                    pos[0] >= midplane[0]
                    and pos[1] >= midplane[1]
                    and pos[2] < midplane[2]
                )
            case 5:
                assert (
                    pos[0] < midplane[0]
                    and pos[1] < midplane[1]
                    and pos[2] >= midplane[2]
                )
            case 6:
                assert (
                    pos[0] >= midplane[0]
                    and pos[1] < midplane[1]
                    and pos[2] >= midplane[2]
                )
            case 7:
                assert (
                    pos[0] < midplane[0]
                    and pos[1] >= midplane[1]
                    and pos[2] >= midplane[2]
                )
            case 8:
                assert np.all(pos >= midplane)


#############################
#############################
# Test OctreeNode
#############################
#############################
# Since _construct is used in the constructor, we won't test it separately
@pytest.mark.skip(reason="Not implemented yet")
def test_OctreeNode(make_basic_data):
    pass


#############################
#############################
# Test Octree
#############################
#############################
@pytest.mark.skip("Not implemented yet")
def test_Octree(make_basic_data):
    basic_data = make_basic_data()
    root = octree.Octree(basic_data)
