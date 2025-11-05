import logging

import numpy as np
import pytest
from hypothesis import example, given, note
from hypothesis import strategies as st
from numpy.typing import ArrayLike

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
@pytest.mark.skip(reason="Not implemented yet")
def test_partition_data_full_box(make_basic_data):
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_partition_data_sub_box(make_basic_data, child_ind: int):
    pass


#############################
# test _get_child_box
#############################
@pytest.mark.skip(reason="Not implemented yet")
def test_get_child_box_shape():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_get_child_box_invalid(box: ArrayLike):
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_get_child_box_valid(
    x: float, y: float, z: float, dx: float, dy: float, dz: float
):
    pass


#############################
# Test morton
#############################
@pytest.mark.skip(reason="Not implemented yet")
def test_morton():
    pass


#############################
# Test OctreeNode
#############################
# Since _construct is used in the constructor, we won't test it separately
@pytest.mark.skip(reason="Not implemented yet")
def test_OctreeNode(make_basic_data):
    pass


#############################
# Test Octree
#############################
@pytest.mark.skip("Not implemented yet")
def test_Octree(make_basic_data):
    basic_data = make_basic_data()
    root = octree.Octree(basic_data)
