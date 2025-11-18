import logging

import conftest as ct
import numpy as np
import pytest
from hypothesis import example, given, note
from hypothesis import strategies as st
from numpy.typing import ArrayLike

import packingcubes.bounding_box as bbox

LOGGER = logging.getLogger(__name__)


#############################
# Test _make_valid
#############################
@given(ct.invalid_boxes())
def test_make_valid_invalid_boxes(box):
    with pytest.raises(bbox.BoundingBoxError) as bberr:
        bbox._make_valid(box)
    box = np.atleast_1d(np.squeeze(np.asanyarray(box)))
    if len(box) != 6 or box.shape != (6,):
        assert "wrong dimensions" in str(bberr.value)
    elif np.any(box[3:] <= 0):
        assert "invalid size" in str(bberr.value)
    elif not np.all(np.isfinite(box)):
        assert "not finite" in str(bberr.value)
    else:
        raise Exception(f"Unknown exception: {bberr}!")


@given(ct.valid_boxes())
def test_make_valid_valid_boxes(box):
    valid_box = bbox._make_valid(box)

    assert np.all(valid_box == box)


#############################
# Test in_box
#############################
@given(
    ct.invalid_boxes(),
    ct.valid_positions(),
)
def test_in_box_invalid_box(box: ArrayLike, xyz: ArrayLike):
    with pytest.raises(bbox.BoundingBoxError):
        bbox.in_box(box, xyz)


@given(ct.valid_boxes(), ct.invalid_positions())
def test_in_box_invalid_point(box: ArrayLike, xyz: ArrayLike):
    assert not np.any(bbox.in_box(box, xyz))


@given(ct.valid_boxes(), ct.valid_positions())
def test_in_box_valid(box: ArrayLike, xyz: ArrayLike):
    inside_box = np.ones((1, len(xyz)), dtype=bool)
    for i in range(3):
        inside_box &= (box[i] <= xyz[:, i]) & (xyz[:, i] <= box[i] + box[i + 3])
    assert np.all(inside_box == bbox.in_box(box, xyz))


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
@given(ct.invalid_boxes())
def test_get_neighbor_boxes_invalid(box: ArrayLike):
    with pytest.raises(bbox.BoundingBoxError):
        bbox.get_box_vertices(box, jitter=0)


@given(ct.valid_boxes())
def test_get_neighbor_boxes(box: ArrayLike):
    expected_neighbors = np.zeros((26, 6))
    # initialize to 26 copies of box
    for i in range(26):
        expected_neighbors[i, :] = box

    index = 0
    for dz in [-box[5], 0, box[5]]:
        for dy in [-box[4], 0, box[4]]:
            for dx in [-box[3], 0, box[3]]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                expected_neighbors[index, :3] += [dx, dy, dz]
                index += 1

    neighbors = bbox.get_neighbor_boxes(box)

    assert expected_neighbors == pytest.approx(neighbors)


#############################
# Test get_box_vertices
#############################
@given(ct.invalid_boxes())
def test_get_box_vertices(box: ArrayLike):
    with pytest.raises(bbox.BoundingBoxError):
        bbox.get_box_vertices(box, jitter=0)


@example(np.array([0, 0, 0, 1, 1, 1]), np.nan).xfail(
    reason="NaNs/Infs not valid jitters", raises=ValueError
)
@example(np.array([0, 0, 0, 1, 1, 1]), np.inf).xfail(
    reason="NaNs/Infs not valid jitters", raises=ValueError
)
@example(np.array([0, 0, 0, 1, 1, 1]), -np.inf).xfail(
    reason="NaNs/Infs not valid jitters", raises=ValueError
)
@given(
    ct.valid_boxes(),
    st.floats(allow_infinity=False, allow_nan=False),
)
def test_get_box_vertices_valid(box: ArrayLike, jitter: float):
    expected_vertices = np.zeros((8, 3))
    x, y, z, dx, dy, dz = box

    expected_vertices[0, :] = [x, y, z]
    expected_vertices[1, :] = [x + dx, y, z]
    expected_vertices[2, :] = [x, y + dy, z]
    expected_vertices[3, :] = [x + dx, y + dy, z]
    expected_vertices[4, :] = [x, y, z + dz]
    expected_vertices[5, :] = [x + dx, y, z + dz]
    expected_vertices[6, :] = [x, y + dy, z + dz]
    expected_vertices[7, :] = [x + dx, y + dy, z + dz]

    vertices = bbox.get_box_vertices(box=box, jitter=0)
    vertices_w_jitter = bbox.get_box_vertices(box=box, jitter=jitter)

    # check without jitter
    assert expected_vertices == pytest.approx(vertices)

    # check with jitter
    if jitter != 0:
        for i, (v, vj) in enumerate(zip(vertices, vertices_w_jitter)):
            # need a better test here
            note(
                f"{i=} w/o jitter: {v} w/ jitter:{vj} {'inside' if jitter > 0 else 'outside'}?:{bbox.in_box(box, vj)}"
            )
            assert (jitter > 0 and bbox.in_box(box, vj)) or (
                jitter < 0 and not bbox.in_box(box, vj)
            )


#############################
# Test project_point_on_box
#############################
@pytest.mark.skip(reason="Not implemented yet")
def test_project_point_on_box():
    pass
