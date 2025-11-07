import logging

import numpy as np
import pytest
from hypothesis import assume, example, given, note
from hypothesis import strategies as st
from hypothesis.extra import numpy as hypnp
from numpy.typing import ArrayLike

import packingcubes.octree as octree

LOGGER = logging.getLogger(__name__)


#############################
# Hypothesis strategies
#############################
def valid_boxes():
    box = hypnp.arrays(
        float, 6, elements=st.floats(allow_infinity=False, allow_nan=False)
    ).filter(lambda a: np.all(a[3:] > 0))
    return box


@st.composite
def invalid_boxes(draw):
    box = draw(valid_boxes())

    @st.composite
    def error_list(draw):
        nan_inf_errors = draw(
            st.lists(st.integers(min_value=0, max_value=2), min_size=6, max_size=6)
        )
        neg_zero_errors = np.array(
            draw(st.lists(st.sampled_from([0, 3, 4]), min_size=6, max_size=6))
        )
        neg_zero_errors[:3] = 0
        return np.maximum(nan_inf_errors, neg_zero_errors)

    errors = draw(error_list().filter(lambda list_: sum(list_) > 0))
    for i, error in enumerate(errors):
        match error:
            case 1:
                box[i] = np.nan
            case 2:
                box[i] = np.inf
            case 3:
                box[i] = -box[i]
            case 4:
                box[i] = 0
    return box


def points():
    return st.tuples(st.floats(), st.floats(), st.floats())


def valid_points():
    return points().filter(lambda xyz: np.all(~np.isnan(xyz) & ~np.isinf(xyz)))


@st.composite
def invalid_points(draw):
    points = list(draw(valid_points()))
    bad_inds = draw(
        st.lists(st.integers(min_value=0, max_value=2), min_size=1, max_size=6)
    )
    bad_values = draw(
        st.lists(
            st.just(np.nan) | st.just(np.inf),
            min_size=len(bad_inds),
            max_size=len(bad_inds),
        )
    )
    for bi, bv in zip(bad_inds, bad_values):
        points[bi] = bv
    return tuple(points)


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


@given(invalid_boxes())
def test_get_child_box_invalid(box: ArrayLike):
    with pytest.raises(ValueError) as excinfo:
        child_box = octree._get_child_box(box, 0)
    assert "finite numbers" in str(excinfo.value) or "box dimensions" in str(
        excinfo.value
    )


@given(
    x=st.integers() | st.floats(allow_infinity=False, allow_nan=False),
    y=st.integers() | st.floats(allow_infinity=False, allow_nan=False),
    z=st.integers() | st.floats(allow_infinity=False, allow_nan=False),
    dx=st.integers(min_value=1)
    | st.floats(allow_infinity=False, allow_nan=False, min_value=0, exclude_min=True),
    dy=st.integers(min_value=1)
    | st.floats(allow_infinity=False, allow_nan=False, min_value=0, exclude_min=True),
    dz=st.integers(min_value=1)
    | st.floats(allow_infinity=False, allow_nan=False, min_value=0, exclude_min=True),
)
@pytest.mark.filterwarnings("ignore: overflow encountered")
def test_get_child_box_valid(
    x: float, y: float, z: float, dx: float, dy: float, dz: float
):
    # box should be of the form [x, y, z, dx, dy, dz]
    box = np.array([x, y, z, dx, dy, dz], dtype=float)  # default
    dx2, dy2, dz2 = dx / 2.0, dy / 2.0, dz / 2.0
    child_boxes = [
        [x, y, z],  # 1
        [x, y + dy2, z],  # 2
        [x + dx2, y, z],  # 3
        [x + dx2, y + dy2, z],  # 4
        [x, y, z + dz2],  # 5
        [x, y + dy2, z + dz2],  # 6
        [x + dx2, y, z + dz2],  # 7
        [x + dx2, y + dy2, z + dz2],  # 8
    ]
    dxbox = [dx2, dy2, dz2]
    for i in range(8):
        child_box = octree._get_child_box(box, i)
        assert child_box[:3] == pytest.approx(child_boxes[i])
        assert child_box[3:] == pytest.approx(dxbox)


#############################
# Test _in_box
#############################


@given(invalid_boxes(), valid_points())
@pytest.mark.filterwarnings("ignore: overflow encountered")
def test_in_box_invalid_box(box: ArrayLike, xyz: tuple[float]):
    # need to account for the cases where e.g. dx=x=0 s.t. xyz may be
    # *technically* inside the box...
    assert not octree._in_box(box, *xyz) or np.any(
        [(box[3 + i] == 0) & (xyz[i] - box[i] == 0) for i in range(3)]
    )


@given(valid_boxes(), invalid_points())
@pytest.mark.filterwarnings("ignore: overflow encountered")
def test_in_box_invalid_point(box: ArrayLike, xyz: tuple[float]):
    assert not octree._in_box(box, *xyz)


@given(valid_boxes(), valid_points())
@pytest.mark.filterwarnings("ignore: overflow encountered")
def test_in_box_valid(box: ArrayLike, xyz: tuple[float]):
    inside_box = np.all([box[i] <= xyz[i] <= box[i] + box[i + 3] for i in range(3)])
    assert inside_box == octree._in_box(box, *xyz)


#############################
# Test _box_neighbors_in_node
#############################
@pytest.mark.skip(reason="Not implemented yet")
def test_box_neighbors_in_node():
    pass


#############################
# Test _get_neighbor_boxes
#############################
@given(valid_boxes())
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

    neighbors = octree._get_neighbor_boxes(box)

    assert expected_neighbors == pytest.approx(neighbors)


#############################
# Test project_point_on_box
#############################
@pytest.mark.skip(reason="Not implemented yet")
def test_project_point_on_box():
    pass


#############################
# Test morton
#############################


@example(np.array([np.nan, 0, 0]), np.array([0, 0, 0, 1, 1, 1])).xfail(
    reason="NaNs/Infs propagate into indices"
)
@example(np.array([0, 0, np.inf]), np.array([0, 0, 0, 1, 1, 1])).xfail(
    reason="NaNs/Infs propagate into indices"
)
@example(np.array([0, 0, 0]), np.array([0, 0, 0, 1, -1, 1])).xfail(
    reason="Invalid boxes fail"
)
@given(
    hypnp.arrays(
        float, st.tuples(st.integers(min_value=0, max_value=1e3), st.just(3))
    ).filter(lambda a: np.all(~np.isnan(a) & ~np.isinf(a))),
    valid_boxes(),
)
@pytest.mark.filterwarnings("ignore: overflow encountered")
def test_morton(positions: ArrayLike, box: ArrayLike):
    midplane = np.array([box[i] + box[i + 3] / 2 for i in range(3)])
    note(f"Midplane for this test is {midplane}")

    mortons = octree.morton(positions, box)

    for morton, pos in zip(mortons, positions):
        assert 1 <= morton <= 8
        match morton:
            case 1:
                assert np.all(pos <= midplane)
            case 2:
                assert (
                    pos[0] <= midplane[0]
                    and pos[1] > midplane[1]
                    and pos[2] <= midplane[2]
                )
            case 3:
                assert (
                    pos[0] > midplane[0]
                    and pos[1] <= midplane[1]
                    and pos[2] <= midplane[2]
                )
            case 4:
                assert (
                    pos[0] > midplane[0]
                    and pos[1] > midplane[1]
                    and pos[2] <= midplane[2]
                )
            case 5:
                assert (
                    pos[0] <= midplane[0]
                    and pos[1] <= midplane[1]
                    and pos[2] > midplane[2]
                )
            case 6:
                assert (
                    pos[0] <= midplane[0]
                    and pos[1] > midplane[1]
                    and pos[2] > midplane[2]
                )
            case 7:
                assert (
                    pos[0] > midplane[0]
                    and pos[1] <= midplane[1]
                    and pos[2] > midplane[2]
                )
            case 8:
                assert np.all(pos > midplane)


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
