import numpy as np
import pytest

from packingcubes import KDTree


# The following are modified from the scipy.KDTree.query example
@pytest.fixture
def scipy_query_example():
    x, y, z = np.mgrid[0:5, 2:8, 0:1]
    data = np.c_[x.ravel(), y.ravel(), z.ravel()]
    # our kdtree has a different leafsize by default than scipy's
    return KDTree(data=data, leafsize=10)


def test_query_example1(scipy_query_example):
    tree = scipy_query_example

    dd, ii = tree.query([[0, 0, 0], [2.2, 2.9, 0]], k=1)
    assert dd.shape == (2,)
    assert ii.shape == (2,)
    assert dd == pytest.approx([2.0, 0.2236068])
    assert ii == pytest.approx([0, 13])


def test_query_example2(scipy_query_example):
    tree = scipy_query_example

    dd, ii = tree.query([[0, 0, 0], [2.2, 2.9, 0]], k=[1])
    assert dd.shape == (2, 1)
    assert ii.shape == (2, 1)
    assert dd.flatten() == pytest.approx([2.0, 0.2236068])
    assert ii.flatten() == pytest.approx([0, 13])


def test_query_example3(scipy_query_example):
    tree = scipy_query_example

    dd, ii = tree.query([[0, 0, 0], [2.2, 2.9, 0]], k=[2])
    assert dd.shape == (2, 1)
    assert ii.shape == (2, 1)
    assert dd.flatten() == pytest.approx([2.23606798, 0.80622577])
    assert ii.flatten() == pytest.approx([6, 19])


def test_query_example4(scipy_query_example):
    tree = scipy_query_example

    dd, ii = tree.query([[0, 0, 0], [2.2, 2.9, 0]], k=2)
    assert dd.shape == (2, 2)
    assert ii.shape == (2, 2)
    assert dd.flatten() == pytest.approx([2.0, 2.23606798, 0.2236068, 0.80622577])
    assert ii.flatten() == pytest.approx([0, 6, 13, 19])


def test_query_example5(scipy_query_example):
    tree = scipy_query_example

    dd, ii = tree.query([[0, 0, 0], [2.2, 2.9, 0]], k=[1, 2])
    assert dd.shape == (2, 2)
    assert ii.shape == (2, 2)
    assert dd.flatten() == pytest.approx([2.0, 2.23606798, 0.2236068, 0.80622577])
    assert ii.flatten() == pytest.approx([0, 6, 13, 19])


# The following are modified from the scipy.KDTree.query_ball_point example
def scipy_query_ball_point_example_unwrapped():
    x, y, z = np.mgrid[0:5, 0:5, 0:1]
    data = np.c_[x.ravel(), y.ravel(), z.ravel()]
    # our kdtree has a different leafsize by default than scipy's
    return KDTree(data=data, leafsize=10)


@pytest.fixture
def scipy_query_ball_point_example():
    return scipy_query_ball_point_example_unwrapped()


def test_query_ball_point_example1(scipy_query_ball_point_example):
    tree = scipy_query_ball_point_example
    qbp = tree.query_ball_point([2, 0, 0], 1)
    qbps = sorted(qbp)
    true = [5, 10, 11, 15]
    for q, t in zip(qbps, true, strict=True):
        assert q == t


def test_query_ball_point_example2(scipy_query_ball_point_example):
    tree = scipy_query_ball_point_example
    qbp = tree.query_ball_point(([2, 0, 0], [3, 3, 0]), 1)

    assert qbp.shape == (2,)
    assert qbp.dtype == np.dtype("O")

    result1 = [5, 10, 11, 15]
    result2 = [13, 17, 18, 19, 23]
    for result, true in zip(qbp, [result1, result2], strict=True):
        assert isinstance(result, list)
        for r, rt in zip(result, true, strict=True):
            assert r == rt
