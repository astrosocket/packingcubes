import logging

import conftest as ct
import numpy as np
import pytest
from hypothesis import assume, example, given, note
from hypothesis import strategies as st
from numpy.typing import ArrayLike

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree

LOGGER = logging.getLogger(__name__)


#############################
# Test _partition
#############################
@example(-3, 9).xfail(reason="Out-of-bounds below")
@example(0, 11).xfail(reason="Out-of-bounds above")
@given(st.integers(min_value=0, max_value=9), st.integers(min_value=0, max_value=9))
def test_partition_bounds(make_basic_data, start: int, end: int):
    basic_data = make_basic_data(num_particles=10)
    positions = basic_data.positions
    # only test x-axis
    note(f"x-values: {positions[start : (end + 1), 0]}")
    num_below = np.sum(positions[start : (end + 1), 0] < 0.5)
    num_above = np.sum(positions[start : (end + 1), 0] >= 0.5)
    note(f"{num_below=:} {num_above=:}")
    partition = octree._partition(basic_data, start, end, 0, 0.5)
    note(f"part x-v: {positions[start : (end + 1), 0]}")
    note(f"{partition=:}")
    assert partition == num_below + start
    assert np.all(positions[start:partition, 0] < 0.5)
    assert np.all(positions[partition : (end + 1), 0] >= 0.5)


@example(4).xfail(reason="invalid axis")
@example(-5).xfail(reason="invalid axis")
@given(st.integers(min_value=0, max_value=2))
def test_partition_axis(make_basic_data, ax: int):
    basic_data = make_basic_data()
    positions = basic_data.positions
    axl = "xyz"[ax]
    note(f"{axl}-values: {positions[:, ax]}")
    num_below = np.sum(positions[:, ax] < 0.5)
    num_above = np.sum(positions[:, ax] >= 0.5)
    note(f"{num_below=:} {num_above=:}")
    partition = octree._partition(basic_data, 0, len(basic_data) - 1, ax, 0.5)
    note(f"part {axl}-v: {positions[:, ax]}")
    note(f"{partition=:}")
    assert partition == num_below
    assert np.all(positions[:partition, ax] < 0.5)
    assert np.all(positions[partition:, ax] >= 0.5)


@given(midplane=st.floats(-1, 2))
def test_partition_midplane(make_basic_data, midplane: float):
    basic_data = make_basic_data()
    positions = basic_data.positions
    # only test x-axis
    note(f"Testing midplane: {midplane}")
    note(f"x-values: {positions[:, 0]}")
    num_below = np.sum(positions[:, 0] < midplane)
    num_above = np.sum(positions[:, 0] >= midplane)
    note(f"{num_below=:} {num_above=:}")
    partition = octree._partition(basic_data, 0, len(basic_data) - 1, 0, midplane)
    note(f"part x-v: {positions[:, 0]}")
    note(f"{partition=:}")
    assert partition == num_below
    assert np.all(positions[:partition, 0] < midplane)
    assert np.all(positions[partition:, 0] >= midplane)


#############################
# Test _partition_data
#############################
def test_partition_data_full_box(make_basic_data):
    basic_data = make_basic_data(num_particles=10, seed=0xDEADBEEF)
    positions_copy = basic_data.positions.copy()
    # With the current random seed (DEADBEEF), basic_data looks like the
    # following: [6,6,6,3,3,6,7,3,4,3], where the numbers are the 1-based
    # top-level z-order indices. So we'd expect the final positions list to
    # look like [3,3,3,3,4,6,6,6,6,7] corresponding to original indices
    #           [9,4,7,3,8,5,0,2,1,6] and the list of children to be
    # [0,0,4,5,5,9,10] #TODO: double check manual indices
    LOGGER.debug("Original order")
    morton_copy = octree.morton(positions_copy, basic_data.bounding_box)
    for pos, mort in zip(positions_copy, morton_copy):
        LOGGER.debug(pos, mort)
    child_list = octree._partition_data(
        basic_data,
        basic_data.bounding_box,
        0,
        9,
    )
    expected_child_list = [0, 0, 4, 5, 5, 9, 10]
    LOGGER.debug(f"{expected_child_list=}")
    LOGGER.debug(f"         {child_list=}")
    assert np.all(expected_child_list == child_list)
    expected_positions_order = np.array([9, 4, 7, 3, 8, 5, 0, 2, 1, 6])
    morton = octree.morton(basic_data.positions, basic_data.bounding_box)
    for i in range(len(positions_copy)):
        LOGGER.debug(f"{i}")
        LOGGER.debug(
            positions_copy[expected_positions_order[i], :],
            morton_copy[expected_positions_order[i]],
        )
        LOGGER.debug(" =?= ")
        LOGGER.debug(basic_data.positions[i, :], morton[i])
    assert np.all(positions_copy[expected_positions_order, :] == basic_data.positions)


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
@pytest.mark.skip(reason="Not implemented yet")
def test_get_child_box_shape():
    pass


@given(ct.invalid_boxes())
def test_get_child_box_invalid(box: ArrayLike):
    with pytest.raises(ValueError) as excinfo:
        child_box = octree._get_child_box(box, 0)
    assert "finite numbers" in str(excinfo.value) or "box dimensions" in str(
        excinfo.value
    )


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
@pytest.mark.skip(reason="Not implemented yet")
def test_box_neighbors_in_node():
    pass


#############################
# Test morton
#############################


@example(np.array([[np.nan, 0, 0]]), np.array([0, 0, 0, 1, 1, 1])).xfail(
    reason="NaNs/Infs propagate into indices"
)
# The following examples no longer "fail". It's unclear whether the morton
# indices make _sense_, but morton() will happily compute them...
@example(np.array([[0, 0, np.inf]]), np.array([0, 0, 0, 1, 1, 1]))
@example(np.array([[0, 0, 0]]), np.array([0, 0, 0, 1, -1, 1]))
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
