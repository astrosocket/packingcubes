import logging

import conftest as ct
import numpy as np
import pytest
from hypothesis import assume, example, given, note, settings
from hypothesis import strategies as st
from numpy.typing import ArrayLike

import packingcubes.bounding_box as bbox

LOGGER = logging.getLogger(__name__)


#############################
# Test check_valid
#############################
@example(
    np.array(
        [
            -6.68159592e307,
            -6.68159592e307,
            -6.68159592e307,
            1.33631918e308,
            1.33631918e308,
            1.33631918e308,
        ]
    )
).via("discovered failure")
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
            (box[3:] > 0) & (np.abs(box[3:]) <= np.abs(box[:3] * np.finfo(float).eps)),
        ):
            assert bbox.BoundingBoxValidFlag.PRECISION not in validation_code
            assert "precision" in lower_info
        with np.errstate(all="raise"):
            try:
                np.abs(box[:3]) + box[3:]
            except FloatingPointError:
                assert bbox.BoundingBoxValidFlag.NOFPERROR not in validation_code
                assert "floating point" in lower_info
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
    # handle floating point
    with np.errstate(all="raise"):
        np.abs(box[:3]) + box[3:]

    # check output
    assert validation_code_w_error == validation_code2_no_error
    assert validation_code_w_error == bbox.BoundingBoxValidFlag.VALID


#############################
# Test BoundingVolume
#############################
@given(ct.valid_positions())
def test_BoundingVolume(xyz: ArrayLike):
    bv = bbox.BoundingVolume()
    with pytest.raises(NotImplementedError):
        bv.contains(xyz)


#############################
# Test make_bounding_box
#############################
@given(ct.invalid_boxes())
def test_make_bounding_box_invalid_boxes(box):
    with pytest.raises(bbox.BoundingBoxError):
        bbox.make_bounding_box(box)
    # error information already checked in test_check_valid


@given(ct.valid_boxes() | ct.valid_bounding_boxes())
def test_make_bounding_box_valid_boxes(box):
    valid_box = bbox.make_bounding_box(box)
    if isinstance(box, bbox.BoundingBox):
        assert np.all(valid_box.box == box.box)
    else:
        assert np.all(valid_box.box == box)


#############################
# Test properties
#############################
@given(ct.valid_bounding_boxes())
def test_properties(bounding_box: bbox.BoundingBox):
    assert bounding_box.box[0] == bounding_box.x
    assert bounding_box.box[1] == bounding_box.y
    assert bounding_box.box[2] == bounding_box.z
    assert bounding_box.box[3] == bounding_box.dx
    assert bounding_box.box[4] == bounding_box.dy
    assert bounding_box.box[5] == bounding_box.dz
    assert np.all(bounding_box.box[:3] == bounding_box.position)
    assert np.all(bounding_box.box[3:] == bounding_box.size)


#############################
# Test copy
#############################
@given(ct.valid_bounding_boxes())
def test_copy(box: bbox.BoundingBox):
    copy = box.copy()

    assert np.all(copy.box == box.box)
    for i in range(6):
        copy.box[i] = 1 if box.box[i] == 0 else 0
        assert copy.box[i] != box.box[i]


#############################
# Test BoundingBox contains
#############################
@given(ct.valid_bounding_boxes(), ct.invalid_positions())
def test_bbox_contains_invalid_point(box: bbox.BoundingBox, xyz: ArrayLike):
    assert not np.any(box.contains(xyz))


@given(ct.valid_bounding_boxes(), ct.valid_positions())
def test_bbox_contains_valid(bounding_box: bbox.BoundingBox, xyz: ArrayLike):
    inside_box = np.ones((1, len(xyz)), dtype=bool)
    box = bounding_box.box
    for i in range(3):
        inside_box &= (box[i] <= xyz[:, i]) & (xyz[:, i] <= box[i] + box[i + 3])
    assert np.all(inside_box == bounding_box.contains(xyz))


#############################
# Test midplane
#############################
@given(ct.valid_bounding_boxes())
def test_midplane(box: bbox.BoundingBox):
    midplane = box.midplane()

    for i, m in enumerate(midplane):
        assert m == box.box[i] + box.box[i + 3] / 2


#############################
# Test max_depth
#############################
@given(ct.valid_bounding_boxes())
@settings(deadline=None)
def test_max_depth_valid_box(bounding_box: bbox.BoundingBox):
    max_depth = bounding_box.max_depth()
    note(f"{max_depth=}")

    smallest_dx = 2.0 ** (-max_depth) * bounding_box.box[3:]
    note(f"{smallest_dx=}")

    # test at farthest corner from 0
    farthest_corner = np.abs(bounding_box.box[:3]) + bounding_box.box[3:]
    note(f"{farthest_corner=}")

    np.testing.assert_array_less(farthest_corner, farthest_corner + smallest_dx)

    eps = np.finfo(float).eps
    assert np.any(
        np.isclose(
            farthest_corner,
            farthest_corner + smallest_dx / 2,
            atol=eps,
            rtol=eps,
        ),
    )


#############################
# Test normalize_to_box
#############################
@given(ct.valid_positions(), ct.valid_bounding_boxes())
@pytest.mark.filterwarnings("ignore: overflow encountered")
@settings(deadline=None)
def test_normalize_to_box(coordinates: ArrayLike, bounding_box: bbox.BoundingBox):
    normal_coords = bounding_box.normalize_to_box(coordinates)

    assert np.all((0 <= normal_coords) & (normal_coords <= 1))
    assert normal_coords == pytest.approx(
        np.clip(
            (coordinates - bounding_box.box[:3]) / bounding_box.box[3:],
            a_min=0,
            a_max=1,
        ),
    )


#############################
# Test _get_neighbor_boxes
#############################
@given(ct.valid_bounding_boxes())
@settings(deadline=None)
def test_get_neighbor_boxes(bounding_box: bbox.BoundingBox):
    expected_neighbors = np.zeros((26, 6))
    # initialize to 26 copies of box
    for i in range(26):
        expected_neighbors[i, :] = bounding_box.box

    index = 0
    for dz in [-bounding_box.box[5], 0, bounding_box.box[5]]:
        for dy in [-bounding_box.box[4], 0, bounding_box.box[4]]:
            for dx in [-bounding_box.box[3], 0, bounding_box.box[3]]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                expected_neighbors[index, :3] += [dx, dy, dz]
                index += 1

    neighbors = bounding_box.get_neighbor_boxes()

    assert expected_neighbors == pytest.approx(neighbors)


#############################
# Test get_child_box
#############################
@given(
    ct.valid_bounding_boxes(),
    st.integers().filter(lambda i: i < 0 or i > 7) | st.floats(),
)
@settings(deadline=None)
def test_get_child_box_invalid_index(
    bounding_box: bbox.BoundingBox,
    index: int | float,
):
    with pytest.raises(
        (OverflowError, ValueError),
        match="invalid index|int too big|int value is too large",
    ):
        bounding_box.get_child_box(index)


@given(ct.valid_bounding_boxes())
def test_get_child_box_valid(bounding_box: bbox.BoundingBox):
    # box should be of the form [x, y, z, dx, dy, dz]
    x, y, z, dx, dy, dz = bounding_box.box
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
        child_box = bounding_box.get_child_box(i)
        assert child_box.box[:3] == pytest.approx(child_boxes[i])
        assert child_box.box[3:] == pytest.approx(dxbox)


#############################
# Test get_box_center
#############################
@given(ct.valid_bounding_boxes())
def test_get_box_center_valid(bounding_box: bbox.BoundingBox):
    center = bounding_box.get_box_center()

    assert center == pytest.approx(bounding_box.box[:3] + bounding_box.box[3:] / 2)


#############################
# Test get_box_vertex
#############################
@given(
    ct.valid_bounding_boxes(),
    st.floats() | st.integers().filter(lambda i: i < 0 or 7 < i),
    st.floats(allow_infinity=False, allow_nan=False),
)
@settings(deadline=None)
def test_get_box_vertex_invalid_indices(
    bounding_box: bbox.BoundingBox,
    index: int | float,
    jitter: float,
):
    with pytest.raises(
        (OverflowError, ValueError),
        match=("int too big|too large|out of bounds|must be an int|must be finite"),
    ):
        bounding_box.get_box_vertex(index, jitter)


@example(bbox.make_bounding_box([0, 0, 0, 1, 1, 1]), 1, np.nan).xfail(
    reason="Jitter must be a number",
    raises=ValueError,
)
@example(bbox.make_bounding_box([0, 0, 0, 1, 1, 1]), 1, np.inf).xfail(
    reason="Jitter must be finite",
    raises=ValueError,
)
@example(bbox.make_bounding_box([0, 0, 0, 1, 1, 1]), 1, -np.inf).xfail(
    reason="Jitter must be finite",
    raises=ValueError,
)
@given(
    ct.valid_bounding_boxes(),
    st.integers(min_value=1, max_value=8),
    st.floats(allow_infinity=False, allow_nan=False),
)
def test_get_box_vertex_valid(
    bounding_box: bbox.BoundingBox, index: int, jitter: float
):
    vertex = bounding_box.get_box_vertex(index, 0)
    vertex_w_jitter = bounding_box.get_box_vertex(index, jitter)

    x, y, z, dx, dy, dz = bounding_box.box

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
        assert bounding_box.contains(vertex_w_jitter) != (jitter < 0)


#############################
# Test get_box_vertices
#############################
@example(bbox.make_bounding_box([0, 0, 0, 1, 1, 1]), np.nan).xfail(
    reason="NaNs/Infs not valid jitters",
    raises=ValueError,
)
@example(bbox.make_bounding_box([0, 0, 0, 1, 1, 1]), np.inf).xfail(
    reason="NaNs/Infs not valid jitters",
    raises=ValueError,
)
@example(bbox.make_bounding_box([0, 0, 0, 1, 1, 1]), -np.inf).xfail(
    reason="NaNs/Infs not valid jitters",
    raises=ValueError,
)
@given(
    ct.valid_bounding_boxes(),
    st.floats(allow_infinity=False, allow_nan=False),
)
def test_get_box_vertices_valid(bounding_box: bbox.BoundingBox, jitter: float):
    expected_vertices = np.zeros((8, 3))
    x, y, z, dx, dy, dz = bounding_box.box

    expected_vertices[0, :] = [x, y, z]
    expected_vertices[1, :] = [x + dx, y, z]
    expected_vertices[2, :] = [x, y + dy, z]
    expected_vertices[3, :] = [x + dx, y + dy, z]
    expected_vertices[4, :] = [x, y, z + dz]
    expected_vertices[5, :] = [x + dx, y, z + dz]
    expected_vertices[6, :] = [x, y + dy, z + dz]
    expected_vertices[7, :] = [x + dx, y + dy, z + dz]

    vertices = bounding_box.get_box_vertices(0)
    vertices_w_jitter = bounding_box.get_box_vertices(jitter)

    # check without jitter
    assert expected_vertices == pytest.approx(vertices)

    # check with jitter
    if jitter != 0:
        for i, (v, vj) in enumerate(zip(vertices, vertices_w_jitter, strict=True)):
            # need a better test here
            note(
                f"{i=} w/o jitter: {v} w/ jitter:{vj} "
                f"{'inside' if jitter > 0 else 'outside'}?:{bounding_box.contains(vj)}",
            )
            assert np.all(v != vj)
            assert bounding_box.contains(vj) != (jitter < 0)


#############################
# Test project_point_on_box
#############################
@given(
    ct.valid_bounding_boxes(),
    ct.invalid_positions().filter(lambda a: np.any(np.isnan(a))),
    st.floats(),
)
@settings(deadline=None)
def test_project_point_on_box_invalid_point_nan(
    bounding_box: bbox.BoundingBox,
    xyz: ArrayLike,
    jitter: float,
):
    for txyz in xyz:
        #  we only care about points with NaNs
        if not np.any(np.isnan(txyz)):
            continue
        with pytest.raises(ValueError, match="contains NaN"):
            bounding_box.project_point_on_box(txyz, jitter)


@example(
    bbox.make_bounding_box([0, 0, 0, 1, 1, 1]), np.array([0.0, 0.0, 0.0]), np.nan
).xfail(
    reason="Jitter must be a number",
    raises=ValueError,
)
@example(
    bbox.make_bounding_box([0, 0, 0, 1, 1, 1]), np.array([0.0, 0.0, 0.0]), -1
).xfail(
    reason="Negative jitter not supported yet",
    raises=NotImplementedError,
)
# infinite points are allowed
@example(bbox.make_bounding_box([0, 0, 0, 1, 1, 1]), np.array([np.inf, 0, 0]), 1)
@given(
    ct.valid_bounding_boxes(),
    ct.valid_positions(max_particles=1),
    st.floats(min_value=0, allow_nan=False),
)
@settings(deadline=None)
def test_project_point_on_box_valid(
    bounding_box: bbox.BoundingBox, xyz: ArrayLike, jitter: float
):
    xyzs = np.atleast_2d(xyz)
    assert xyzs.shape[1] == 3
    note(f"box={bounding_box.box}")

    for txyz in xyzs:
        note(f"{txyz=}")
        px, py, pz = bounding_box.project_point_on_box(txyz, 0)
        note(f"{px=}, {py=}, {pz=}")
        pxj, pyj, pzj = bounding_box.project_point_on_box(txyz, jitter)
        note(f"{pxj=}, {pyj=}, {pzj=}")
        assert bounding_box.contains_point(px, py, pz)
        # test we didn't mess up already contained points
        if bounding_box.contains(txyz):
            assert txyz == pytest.approx([px, py, pz])
        elif jitter:  # if xyz in box then jitter is ignored
            # terms should be different from 0-jitter unless already in box
            # in that dimension
            assert (
                (px != pxj)
                or np.isinf(txyz[0])
                or bounding_box.contains_point(txyz[0], py, pz)
            )
            assert (
                (py != pyj)
                or np.isinf(txyz[1])
                or bounding_box.contains_point(px, txyz[1], pz)
            )
            assert (
                (pz != pzj)
                or np.isinf(txyz[2])
                or bounding_box.contains_point(px, pz, txyz[2])
            )

            assert bounding_box.contains_point(pxj, pyj, pzj) != (jitter < 0)


#############################
# Test make_bounding_sphere
#############################
@given(ct.invalid_spheres())
def test_make_bounding_sphere_invalid_sphere(invalid_sph):
    center, radius = invalid_sph

    center = np.atleast_1d(center).astype(float)
    if len(center.flatten()) == 3:
        with np.errstate(invalid="ignore"):
            box = np.array(
                [
                    center[0] - radius,
                    center[1] - radius,
                    center[2] - radius,
                    radius * 2,
                    radius * 2,
                    radius * 2,
                ]
            )
        if bbox.check_valid(box, raise_error=False):
            note(box)
        assume(not bbox.check_valid(box, raise_error=False))

    with pytest.raises((bbox.BoundingBoxError, ValueError)):
        bbox.make_bounding_sphere(radius=radius, center=center)


@given(ct.valid_spheres())
def test_make_bounding_sphere_valid(valid_sphere):
    center, radius = valid_sphere

    default_center = bbox.make_bounding_sphere(radius)

    full_sphere = bbox.make_bounding_sphere(radius, center=center)

    assert isinstance(default_center, bbox.BoundingSphere)
    assert np.all(default_center.center == np.array([0, 0, 0]))
    assert default_center.radius == radius

    assert isinstance(full_sphere, bbox.BoundingSphere)
    assert np.all(full_sphere.center == center)
    assert full_sphere.radius == radius


#############################
# Test BoundingSphere contains
#############################
@given(ct.valid_bounding_spheres(), ct.invalid_positions())
@settings(deadline=None)
def test_bsph_contains_invalid_point(sph: bbox.BoundingSphere, xyz: ArrayLike):
    assert not np.any(sph.contains(xyz))


@given(ct.valid_bounding_spheres(), ct.valid_positions())
def test_bsph_contains_valid(sph: bbox.BoundingSphere, xyz: ArrayLike):
    center, radius = sph.center, sph.radius
    dist = np.sqrt(np.sum((xyz - center) ** 2, axis=1))
    inside_sph = dist <= radius
    assert np.all(inside_sph == sph.contains(xyz))


#############################
# Test BoundingSphere bounding_box
#############################
@given(ct.valid_bounding_spheres())
def test_bsph_bounding_box(sph: bbox.BoundingSphere):
    box = sph.bounding_box
    center, radius = sph.center, sph.radius

    assert np.all(box.box[:3] == center - radius)
    assert np.all(box.box[3:] == radius * 2)
