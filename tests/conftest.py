# File for fixtures that should be shared across the test files

from collections import namedtuple

import pytest
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
