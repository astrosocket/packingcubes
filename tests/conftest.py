# File for fixtures that should be shared across the test files

from collections import namedtuple

import numpy as np
import pytest
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra import numpy as hypnp
from numpy.random import RandomState

from packingcubes.data_objects import Dataset


def fake_basic_dataset(num_particles: int = 10, seed: int = 0xDEADBEEF) -> Dataset:
    """
    Create a mock dataset for testing purposes
    """
    prng = RandomState(seed)
    # For 10 particles, this looks not too dissimilar from an actual extremely
    # low res cosmo sim...

    ds = Dataset(
        name="fake_basic",
        filepath="",
    )
    positions = prng.random_sample((num_particles, 3))

    Data = namedtuple(
        "Data",
        [
            "positions",
        ],
    )

    ds._data = Data(
        positions,
    )

    return ds


@pytest.fixture(scope="module", autouse=True)
def make_basic_data():
    def _basic_data(num_particles=10, seed=0xDEADBEEF):
        return fake_basic_dataset(num_particles, seed)

    yield _basic_data


#############################
# Hypothesis strategies
#############################
@st.composite
def valid_boxes(draw):
    coord = st.floats(min_value=-1e10, max_value=1e10, allow_subnormal=False)
    box_pos = draw(st.tuples(coord, coord, coord))

    def dx(x):
        return st.floats(
            min_value=2 * np.nextafter(np.abs(x), np.abs(x) + 1) - x,
            allow_infinity=False,
        )

    box_dx = draw(st.tuples(dx(box_pos[0]), dx(box_pos[1]), dx(box_pos[2])))
    box = np.array(box_pos + box_dx)
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
    return st.tuples(
        st.floats(min_value=-1e10, max_value=1e10, allow_subnormal=False),
        st.floats(min_value=-1e10, max_value=1e10, allow_subnormal=False),
        st.floats(min_value=-1e10, max_value=1e10, allow_subnormal=False),
    )


def valid_points():
    return points().filter(lambda xyz: np.all(np.isfinite(xyz)))


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


def valid_positions():
    return hypnp.arrays(
        float,
        st.tuples(st.integers(min_value=1, max_value=3e2), st.just(3)),
        elements=st.floats(min_value=-1e10, max_value=1e10),
    )


@st.composite
def basic_data_strategy(draw):
    ds = Dataset(name="basic_strategy", filepath="")
    positions = draw(valid_positions())
    extremes = np.array([np.min(positions, axis=0), np.max(positions, axis=0)])
    box = np.zeros(6)
    box[:3] = extremes[0, :]
    box[3:] = extremes[1, :] - extremes[0, :]
    assume(np.all(box[3:] > 0))
    ds.box = box

    Data = namedtuple(
        "Data",
        ["positions"],
    )

    ds._data = Data(
        positions,
    )

    return ds
