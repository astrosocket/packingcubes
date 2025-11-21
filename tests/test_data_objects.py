import conftest as ct
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st


@st.composite
def data_ind(draw, *, min_data_len=5, max_data_len=20):
    data_len = draw(st.integers(min_value=min_data_len, max_value=max_data_len))
    start_ind = draw(st.integers(min_value=0, max_value=data_len - 1))
    stop_ind = draw(st.integers(min_value=0, max_value=data_len - 1))
    return data_len, start_ind, stop_ind


@example((10, 0, 11)).xfail(reason="Attempted OOB swap")
@example((10, -11, 8)).xfail(reason="Attempted (negative) OOB swap")
@given(data_ind(min_data_len=5, max_data_len=2e3))
def test_swap(make_basic_data, datalen_loc1_loc2):
    basic_data = make_basic_data(num_particles=datalen_loc1_loc2[0])
    loc1, loc2 = datalen_loc1_loc2[1:3]
    positions = basic_data.positions.copy()
    pos1 = positions[loc1]
    pos2 = positions[loc2]
    basic_data._swap(loc1, loc2)
    assert np.all(basic_data.positions[loc1] == pos2)
    assert np.all(basic_data.positions[loc2] == pos1)


@given(ct.basic_data_strategy())
def test_bounding_box(basic_data):
    box = basic_data.bounding_box
    positions = basic_data.positions

    assert np.all(box[:3] == np.min(positions, axis=0))
    assert box[3:] == pytest.approx(
        np.max(positions, axis=0) - box[:3],
    )


@given(st.integers(min_value=1, max_value=5e5))
def test_len(make_basic_data, data_len):
    basic_data = make_basic_data(num_particles=data_len)
    assert len(basic_data) == data_len
    assert len(basic_data) == len(basic_data._data.positions)
