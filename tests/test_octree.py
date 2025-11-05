import logging

import pytest
from numpy.typing import ArrayLike

import packingcubes.octree as octree

LOGGER = logging.getLogger(__name__)


#############################
# Test _partition
#############################
@pytest.mark.skip(reason="Not implemented yet")
def test_partition_bounds(make_basic_data, start: int, end: int):
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_partition_axis(make_basic_data, ax: int):
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_partition_midplane(make_basic_data, midplane: float):
    pass


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
