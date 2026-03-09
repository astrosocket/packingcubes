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
