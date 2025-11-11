import logging

import conftest as ct
import numpy as np
import pytest
from hypothesis import given
from numpy.typing import ArrayLike

import packingcubes.bounding_box as bbox

LOGGER = logging.getLogger(__name__)


#############################
# Test in_box
#############################
# filtering out invalid boxes with dx=inf, since they're not a useful test
@given(
    ct.invalid_boxes().filter(lambda b: not np.any(np.isinf(b[3:]))), ct.valid_points()
)
@pytest.mark.filterwarnings("ignore: overflow encountered")
def test_in_box_invalid_box(box: ArrayLike, xyz: tuple[float]):
    # need to account for the cases where e.g. dx=x=0 or dx=inf s.t. xyz may be
    # *technically* inside the box...
    assert not bbox.in_box(box, *xyz) or np.any(
        [(box[3 + i] == 0) & (xyz[i] - box[i] == 0) for i in range(3)]
    )


@given(ct.valid_boxes(), ct.invalid_points())
@pytest.mark.filterwarnings("ignore: overflow encountered")
def test_in_box_invalid_point(box: ArrayLike, xyz: tuple[float]):
    assert not bbox.in_box(box, *xyz)


@given(ct.valid_boxes(), ct.valid_points())
@pytest.mark.filterwarnings("ignore: overflow encountered")
def test_in_box_valid(box: ArrayLike, xyz: tuple[float]):
    inside_box = np.all([box[i] <= xyz[i] <= box[i] + box[i + 3] for i in range(3)])
    assert inside_box == bbox.in_box(box, *xyz)


#############################
# Test midplane
#############################
@given(ct.valid_boxes())
def test_midplane(box: ArrayLike):
    midplane = bbox.midplane(box)

    for i, m in enumerate(midplane):
        assert midplane[i] == box[i] + box[i + 3] / 2


#############################
# Test normalize_to_box
#############################
@given(ct.valid_positions(), ct.valid_boxes())
@pytest.mark.filterwarnings("ignore: overflow encountered")
def test_normalize_to_box(coordinates, box):
    normal_coords = bbox.normalize_to_box(coordinates, box)

    assert np.all((0 <= normal_coords) & (normal_coords <= 1))
    assert normal_coords == pytest.approx(
        np.clip((coordinates - box[:3]) / box[3:], a_min=0, a_max=1)
    )


#############################
# Test _get_neighbor_boxes
#############################
@given(ct.valid_boxes())
@pytest.mark.filterwarnings("ignore: overflow encountered")
def test_get_neighbor_boxes(box: ArrayLike):
    expected_neighbors = np.zeros((6, 6))
    # initialize to 6 copies of box
    for i in range(6):
        expected_neighbors[i, :] = box
    expected_neighbors[0, 0] -= box[3]  # box 0 is shifted by -dx
    expected_neighbors[1, 0] += box[3]  # box 1 is shifted by +dx
    expected_neighbors[2, 1] -= box[4]  # box 2 is shifted by -dy
    expected_neighbors[3, 1] += box[4]  # box 3 is shifted by +dy
    expected_neighbors[4, 2] -= box[5]  # box 4 is shifted by -dz
    expected_neighbors[5, 2] += box[5]  # box 5 is shifted by +dz

    neighbors = bbox.get_neighbor_boxes(box)

    assert expected_neighbors == pytest.approx(neighbors)


#############################
# Test project_point_on_box
#############################
@pytest.mark.skip(reason="Not implemented yet")
def test_project_point_on_box():
    pass
