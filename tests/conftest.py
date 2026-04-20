# ruff: noqa: D100, D103
# File for fixtures that should be shared across the test files

import copy
import logging
import warnings

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis.extra import numpy as hypnp
from numpy.random import RandomState

from packingcubes.bounding_box import (
    make_bounding_box,
    make_bounding_sphere,
)
from packingcubes.data_objects import Dataset
from packingcubes.packed_tree import PackedTree

LOGGER = logging.getLogger(__name__)


def fake_basic_dataset(num_particles: int = 10, seed: int = 0xDEADBEEF) -> Dataset:
    """Create a mock dataset for testing purposes"""
    prng = RandomState(seed)
    # For 10 particles, this looks not too dissimilar from an actual extremely
    # low res cosmo sim...

    ds = Dataset(
        name="fake_basic",
        filepath="",
    )
    positions = prng.random_sample((num_particles, 3))

    ds._positions = positions

    ds._setup_index()

    return ds


@pytest.fixture(scope="package", autouse=True)
def make_basic_data():
    def _basic_data(num_particles=10, seed=0xDEADBEEF):
        return fake_basic_dataset(num_particles, seed)

    return _basic_data


#############################
# Numba pre-compilation
#############################
@pytest.fixture(scope="package", autouse=True)
def basic_bounding_box():
    return make_bounding_box([0, 0, 0, 1, 1, 1])


@pytest.fixture(scope="package", autouse=True)
def basic_bounding_sphere():
    return make_bounding_sphere(1, center=[0, 0, 0])


@pytest.fixture(scope="package", autouse=True)
def basic_data_container(make_basic_data):
    return make_basic_data().data_container


@pytest.fixture(scope="package", autouse=True)
def basic_optree(make_basic_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PackedTree(dataset=make_basic_data(), particle_threshold=1)


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
        ),
    )


@st.composite
def valid_boxes(draw):
    box_pos = draw(st.tuples(valid_coord(), valid_coord(), valid_coord()))
    box_dx = draw(
        st.tuples(
            valid_dx(x=box_pos[0]),
            valid_dx(x=box_pos[1]),
            valid_dx(x=box_pos[2]),
        ),
    )
    return np.array(box_pos + box_dx)


@st.composite
def valid_sphere_radius(draw, center):
    # use largest |coordinate| to specify radius
    large_coord = np.max(np.abs(center))
    return draw(valid_dx(x=large_coord))


@st.composite
def invalid_boxes_correct_shape(draw):
    box = draw(valid_boxes())

    @st.composite
    def error_list(draw):
        nan_inf_errors = draw(
            st.lists(st.integers(min_value=0, max_value=2), min_size=6, max_size=6),
        )
        neg_zero_small_errors = np.array(
            draw(st.lists(st.sampled_from([0, 3, 4, 5]), min_size=6, max_size=6)),
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
    return draw(
        hypnp.arrays(float, hypnp.array_shapes().filter(lambda a: np.prod(a) != 6))
        | invalid_boxes_correct_shape(),
    )


@st.composite
def invalid_spheres(draw):
    error_type = draw(st.integers(min_value=1, max_value=4))
    use_bad_center = error_type == 2 or error_type == 4
    use_bad_radii = error_type >= 3

    if error_type == 1:
        return draw(
            st.tuples(
                hypnp.arrays(
                    float, hypnp.array_shapes().filter(lambda a: np.prod(a) != 3)
                ),
                st.just(1),
            )
        )

    good_center = draw(valid_positions(max_particles=1))
    center = draw(invalid_positions(max_particles=1)) if use_bad_center else good_center
    center = center[0]

    @st.composite
    def bad_radii(draw, center):
        ind = draw(st.integers(min_value=0, max_value=2))
        return draw(
            st.just(np.inf)
            | st.just(np.nan)
            | st.just(center[ind] * np.finfo(float).eps / 16)
            | st.just(center[ind] / np.finfo(float).eps * 16)
        )

    if use_bad_radii:
        radii = draw(bad_radii(center=center))
    else:
        radii = draw(valid_sphere_radius(center=good_center[0]))
    return (center, radii)


@st.composite
def valid_bounding_boxes(draw):
    return make_bounding_box(draw(valid_boxes()))


@st.composite
def valid_spheres(draw):
    center = np.array(draw(st.tuples(valid_coord(), valid_coord(), valid_coord())))
    radius = draw(valid_sphere_radius(center=center))
    return (center, radius)


@st.composite
def valid_bounding_spheres(draw):
    center, radius = draw(valid_spheres())
    return make_bounding_sphere(radius=radius, center=center)


def valid_positions(max_particles=3e2):
    return hypnp.arrays(
        float,
        st.tuples(st.integers(min_value=1, max_value=max_particles), st.just(3)),
        elements=st.floats(min_value=-1e10, max_value=1e10),
    )


@st.composite
def invalid_positions(draw, max_particles=3e2):
    positions = draw(valid_positions(max_particles=max_particles))
    # generate bad_inds as a array of flags corresponding to which indices are
    # bad., i.e. 3 = 1*2**0 + 1*2**1 + 0*2**2 = [1,1,0]
    bad_inds = draw(
        hypnp.arrays(
            int,
            (len(positions), 1),
            elements=st.integers(min_value=1, max_value=7),
        ),
    )
    # convert to masking array
    mask = np.hstack((bad_inds & 1, (bad_inds & 2) >> 1, (bad_inds & 4) >> 2))
    num_bad = int(np.sum(mask))

    bad_values = draw(
        st.lists(
            st.just(np.nan) | st.just(np.inf),
            min_size=num_bad,
            max_size=num_bad,
        ),
    )

    positions[np.nonzero(mask)] = bad_values
    return positions


@st.composite
def basic_data_strategy(draw, max_particles=3e2):
    ds = Dataset(name="basic_strategy", filepath="")
    positions = draw(valid_positions(max_particles=max_particles))

    ds._positions = positions
    ds._set_bounding_box()

    ds._setup_index()

    return ds


@st.composite
def basic_data_container_strategy(draw, max_particles=3e2):
    ds = draw(basic_data_strategy(max_particles=max_particles))
    return ds.data_container


@st.composite
def data_with_duplicates(draw, max_particles=15):
    data = draw(basic_data_strategy(max_particles=max_particles))
    # create list of data indices
    # list must have len in [len(data)+1, inf) -> this guarantees duplicates
    # by the pigeonhole principle
    # list elements are ints in [0, len(data))
    data_indices = draw(
        hypnp.arrays(
            int,
            st.integers(
                min_value=len(data) + 1,
                max_value=len(data) * 10,
            ),
            elements=st.integers(min_value=0, max_value=len(data) - 1),
        ),
    )
    dup_data = copy.copy(data)

    dup_data._positions = data.positions[data_indices]

    del dup_data._index
    dup_data._setup_index()

    return dup_data


@st.composite
def valid_data_strategy(draw):
    raise NotImplementedError("valid_data_strategy is still in progress")
    data = draw(basic_data_strategy())

    valid_data = data
    return valid_data  # noqa RET504
