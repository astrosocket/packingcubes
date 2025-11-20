# File for fixtures that should be shared across the test files

import copy
import logging
from collections import namedtuple

import numpy as np
import pytest
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra import numpy as hypnp
from numpy.random import RandomState

from packingcubes.data_objects import Dataset

LOGGER = logging.getLogger(__name__)


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
def valid_coord():
    return st.floats(min_value=-1e10, max_value=1e10, allow_subnormal=False)


@st.composite
def valid_dx(draw, x: float):
    x = np.maximum(1, np.abs(x))
    min_value = x * np.finfo(float).eps * 100
    max_value = x / (10 * np.finfo(float).eps)
    if max_value / min_value < 10:
        raise ValueError(f"min and max values are too close! {min_value=} {max_value=}")
    return draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
        )
    )


@st.composite
def valid_boxes(draw):
    box_pos = draw(st.tuples(valid_coord(), valid_coord(), valid_coord()))
    box_dx = draw(
        st.tuples(
            valid_dx(x=box_pos[0]), valid_dx(x=box_pos[1]), valid_dx(x=box_pos[2])
        )
    )
    box = np.array(box_pos + box_dx)
    return box


@st.composite
def invalid_boxes_correct_shape(draw):
    box = draw(valid_boxes())

    @st.composite
    def error_list(draw):
        nan_inf_errors = draw(
            st.lists(st.integers(min_value=0, max_value=2), min_size=6, max_size=6)
        )
        neg_zero_small_errors = np.array(
            draw(st.lists(st.sampled_from([0, 3, 4, 5]), min_size=6, max_size=6))
        )
        neg_zero_small_errors[:3] = 0
        return np.maximum(nan_inf_errors, neg_zero_small_errors)

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
            case 5:
                # only in dx portion
                box[i] = box[i - 3] * np.finfo(float).eps / 4
    return box


@st.composite
def invalid_boxes(draw):
    box = draw(
        hypnp.arrays(float, hypnp.array_shapes().filter(lambda a: np.prod(a) != 6))
        | invalid_boxes_correct_shape()
    )
    return box


def valid_positions():
    return hypnp.arrays(
        float,
        st.tuples(st.integers(min_value=1, max_value=3e2), st.just(3)),
        elements=st.floats(min_value=-1e10, max_value=1e10),
    )


@st.composite
def invalid_positions(draw):
    positions = draw(valid_positions())
    # generate bad_inds as a array of flags corresponding to which indices are
    # bad., i.e. 3 = 1*2**0 + 1*2**1 + 0*2**2 = [1,1,0]
    bad_inds = draw(
        hypnp.arrays(
            int, (len(positions), 1), elements=st.integers(min_value=1, max_value=7)
        )
    )
    # convert to masking array
    mask = np.hstack((bad_inds & 1, (bad_inds & 2) >> 1, (bad_inds & 4) >> 2))
    num_bad = int(np.sum(mask))

    bad_values = draw(
        st.lists(
            st.just(np.nan) | st.just(np.inf),
            min_size=num_bad,
            max_size=num_bad,
        )
    )

    positions[np.nonzero(mask)] = bad_values
    return positions


@st.composite
def basic_data_strategy(draw):
    ds = Dataset(name="basic_strategy", filepath="")
    positions = draw(valid_positions())
    extremes = np.array([np.min(positions, axis=0), np.max(positions, axis=0)])
    box = np.zeros(6)
    box[:3] = extremes[0, :]
    box[3:] = extremes[1, :] - extremes[0, :]
    assume(np.all(box[3:] > box[:3] / 1e10))
    ds.box = box
    ds.MIN_PARTICLE_SPACING = np.min(box[3:]) / 1e10

    Data = namedtuple(
        "Data",
        ["positions"],
    )

    ds._data = Data(
        positions,
    )

    return ds


@st.composite
def data_with_duplicates(draw):
    data = draw(basic_data_strategy())
    # create list of data indices
    # list must have len in [len(data)+1, inf)
    # list elements are ints in [0, len(data))
    data_indices = draw(
        hypnp.arrays(
            int,
            st.integers(
                min_value=len(data) + 1,
                max_value=len(data) * 10,
            ),
            elements=st.integers(min_value=0, max_value=len(data) - 1),
        )
    )
    dup_data = copy.copy(data)
    Data = namedtuple(
        "Data",
        ["positions"],
    )

    dup_data._data = Data(
        data.positions[data_indices],
    )
    return dup_data


@st.composite
def valid_data_strategy(draw):
    data = draw(basic_data_strategy())

    valid_data = data
    return valid_data
