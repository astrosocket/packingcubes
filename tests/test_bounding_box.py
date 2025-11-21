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
# Test check_valid
#############################
@given(st.none() | st.text() | ct.invalid_boxes())
def test_check_valid_invalid_boxes(box):
    validation_code = bbox.check_valid(box, raise_error=False)
    with pytest.raises(bbox.BoundingBoxError) as bberrinfo:
        bbox.check_valid(box)

    note(f"{validation_code=}")
    lower_info = str(bberrinfo.value).lower()

    note(f"{lower_info=}")

    if not isinstance(box, np.ndarray):
        assert "not a box" in str(bberrinfo)
        assert bbox.BoundingBoxValidFlag.IS_BOX not in validation_code
        return  # don't check anything else

    box = np.atleast_1d(np.squeeze(np.asanyarray(box)))
    if len(box) != 6 or box.shape != (6,):
        assert bbox.BoundingBoxValidFlag.CORRECT_SHAPE not in validation_code
        assert "wrong shape" in lower_info
    else:
        if not np.all(np.isfinite(box)):
            assert bbox.BoundingBoxValidFlag.FINITE not in validation_code
            assert "inf/nan" in lower_info
        if np.any(box[3:] <= 0):
            assert bbox.BoundingBoxValidFlag.POSITIVE not in validation_code
            assert "invalid size" in lower_info
        if np.all(np.isfinite(box)) & np.any(
            (box[3:] > 0) & (np.abs(box[3:]) <= np.abs(box[:3] * np.finfo(float).eps))
        ):
            assert bbox.BoundingBoxValidFlag.PRECISION not in validation_code
            assert "precision" in lower_info
    # The following is a regression test...
    if "Something is very wrong" in lower_info:
        pytest.fail(f"Unknown exception: {str(bberrinfo.value)}!")


@given(ct.valid_boxes())
def test_check_valid_valid_box(box: ArrayLike):
    # Note this also checks our valid box generator...

    # should not raise an error
    validation_code_w_error = bbox.check_valid(box, raise_error=True)
    validation_code2_no_error = bbox.check_valid(box, raise_error=False)

    # check box is actually valid
    assert isinstance(box, np.ndarray)
    assert box.shape == (6,)
    assert np.all(np.isfinite(box))
    # this handles both negative sizes and precision
    assert np.all(box[3:] > np.abs(box[:3] * np.finfo(float).eps))

    # check output
    assert (
        validation_code_w_error == validation_code2_no_error
        and validation_code_w_error == bbox.BoundingBoxValidFlag.VALID
    )


#############################
# Test make_valid
#############################
@given(ct.invalid_boxes())
def test_make_valid_invalid_boxes(box):
    with pytest.raises(bbox.BoundingBoxError):
        bbox.make_valid(box)
    # error information already checked in test_check_valid


@given(ct.valid_boxes())
def test_make_valid_valid_boxes(box):
    valid_box = bbox.make_valid(box)

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
# Test max_depth
#############################
@given(ct.valid_boxes())
def test_max_depth_valid_box(box):
    max_depth = bbox.max_depth(box)
    note(f"{max_depth=}")

    smallest_dx = 2.0 ** (-max_depth) * box[3:]
    note(f"{smallest_dx=}")

    # test at farthest corner from 0
    farthest_corner = np.abs(box[:3]) + box[3:]
    note(f"{farthest_corner=}")

    np.testing.assert_array_less(farthest_corner, farthest_corner + smallest_dx)

    eps = np.finfo(float).eps
    assert np.any(
        np.isclose(
            farthest_corner, farthest_corner + smallest_dx / 2, atol=eps, rtol=eps
        )
    )


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
# Test get_box_center
#############################
@given(ct.invalid_boxes())
def test_get_box_center_invalid(box: ArrayLike):
    with pytest.raises(bbox.BoundingBoxError):
        bbox.get_box_center(box)


@given(ct.valid_boxes())
def test_get_box_center_valid(box: ArrayLike):
    center = bbox.get_box_center(box)

    assert center == pytest.approx(box[:3] + box[3:] / 2)


#############################
# Test get_box_vertex
#############################
@given(ct.invalid_boxes())
def test_get_box_vertex_invalid_boxes(box: ArrayLike):
    for index in range(8):
        with pytest.raises(bbox.BoundingBoxError):
            bbox.get_box_vertex(box, index=index)


@given(
    ct.valid_boxes(),
    st.floats() | st.integers().filter(lambda i: i < 1 or 8 < i),
    st.floats(allow_infinity=False, allow_nan=False),
)
def test_get_box_vertex_invalid_indices(
    box: ArrayLike, index: int | float, jitter: float
):
    with pytest.raises(ValueError):
        bbox.get_box_vertex(box, index=index, jitter=jitter)


@example(np.array([0, 0, 0, 1, 1, 1]), 1, np.nan).xfail(
    reason="Jitter must be a number", raises=ValueError
)
@example(np.array([0, 0, 0, 1, 1, 1]), 1, np.inf).xfail(
    reason="Jitter must be finite", raises=ValueError
)
@example(np.array([0, 0, 0, 1, 1, 1]), 1, -np.inf).xfail(
    reason="Jitter must be finite", raises=ValueError
)
@given(
    ct.valid_boxes(),
    st.integers(min_value=1, max_value=8),
    st.floats(allow_infinity=False, allow_nan=False),
)
def test_get_box_vertex_valid(box: ArrayLike, index: int, jitter: float):
    vertex = bbox.get_box_vertex(box, index, jitter=0)
    vertex_w_jitter = bbox.get_box_vertex(box, index, jitter=jitter)

    x, y, z, dx, dy, dz = box

    match index:
        case 1:
            assert vertex == pytest.approx(np.array([x, y, z]))
        case 2:
            assert vertex == pytest.approx(np.array([x + dx, y, z]))
        case 3:
            assert vertex == pytest.approx(np.array([x, y + dy, z]))
        case 4:
            assert vertex == pytest.approx(np.array([x + dx, y + dy, z]))
        case 5:
            assert vertex == pytest.approx(np.array([x, y, z + dz]))
        case 6:
            assert vertex == pytest.approx(np.array([x + dx, y, z + dz]))
        case 7:
            assert vertex == pytest.approx(np.array([x, y + dy, z + dz]))
        case 8:
            assert vertex == pytest.approx(np.array([x + dx, y + dy, z + dz]))

    if jitter:
        assert np.all(vertex_w_jitter != vertex)
        assert bbox.in_box(box, vertex_w_jitter) != (jitter < 0)


#############################
# Test get_box_vertices
#############################
@given(ct.invalid_boxes())
def test_get_box_vertices_invalid_boxes(box: ArrayLike):
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
            assert np.all(v != vj)
            assert bbox.in_box(box, vj) != (jitter < 0)


#############################
# Test project_point_on_box
#############################
@given(ct.invalid_boxes(), ct.valid_positions(), st.floats())
def test_project_point_on_box_invalid_box(
    box: ArrayLike, xyz: ArrayLike, jitter: float
):
    with pytest.raises(bbox.BoundingBoxError):
        bbox.project_point_on_box(box, xyz, jitter=jitter)


@given(
    ct.valid_boxes(),
    ct.invalid_positions().filter(lambda a: np.any(np.isnan(a))),
    st.floats(),
)
def test_project_point_on_box_invalid_point_nan(
    box: ArrayLike, xyz: ArrayLike, jitter: float
):
    with pytest.raises(ValueError):
        bbox.project_point_on_box(box, xyz, jitter=jitter)


@example(np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 0]), np.nan).xfail(
    reason="Jitter must be a number", raises=ValueError
)
@example(np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 0]), -1).xfail(
    reason="Negative jitter not supported yet", raises=NotImplementedError
)
# infinite points are allowed
@example(np.array([0, 0, 0, 1, 1, 1]), np.array([np.inf, 0, 0]), 1)
@given(ct.valid_boxes(), ct.valid_positions(), st.floats(min_value=0, allow_nan=False))
def test_project_point_on_box_valid(box: ArrayLike, xyz: ArrayLike, jitter: float):
    pxyz = bbox.project_point_on_box(box, xyz, jitter=0)
    pxyz_w_jitter = bbox.project_point_on_box(box, xyz, jitter=jitter)

    assert np.all(bbox.in_box(box, pxyz))

    xyzs = np.atleast_2d(xyz)
    pxyzs = np.atleast_2d(pxyz)
    pxyz_w_jitters = np.atleast_2d(pxyz_w_jitter)
    for txyz, tpxyz, tpxyz_w_jitter in zip(xyzs, pxyzs, pxyz_w_jitters):
        # test we didn't mess up already contained points
        if bbox.in_box(box, txyz):
            assert txyz == pytest.approx(tpxyz)
        elif jitter:  # if xyz in box then jitter is ignored
            assert np.any((tpxyz != tpxyz_w_jitter) | np.isinf(txyz))
            assert np.all(bbox.in_box(box, tpxyz_w_jitter)) != (jitter < 0)
