import numpy as np
import pytest
from scipy.spatial import KDTree as SciTree

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


# The following are modified from the scipy.KDTree.query_ball_tree example
def scipy_query_ball_tree_example_unwrapped():
    rng = np.random.default_rng()
    points1_2d = rng.random((15, 2))
    points2_2d = rng.random((15, 2))
    points1 = np.zeros_like(points1_2d, shape=(points1_2d.shape[0], 3))
    points1[:, :2] = points1_2d
    points2 = np.zeros_like(points2_2d, shape=(points2_2d.shape[0], 3))
    points2[:, :2] = points2_2d
    # our kdtree has a different leafsize by default than scipy's
    tree1 = KDTree(
        data=points1,
        leafsize=10,
    )
    tree2 = KDTree(
        data=points2,
        leafsize=10,
    )
    stree1 = SciTree(data=points1, leafsize=10)
    stree2 = SciTree(data=points2, leafsize=10)
    return tree1, tree2, stree1, stree2, points1, points2


@pytest.fixture
def scipy_query_ball_tree_example():
    return scipy_query_ball_tree_example_unwrapped()[:4]


def test_query_ball_tree_example(scipy_query_ball_tree_example):
    tree1, tree2, stree1, stree2 = scipy_query_ball_tree_example

    # for performance, the index sublists might not be in sorted order
    # so turn on return_sorted for comparison
    indexes = tree1.query_ball_tree(tree2, r=0.2, return_sorted=True)
    sindexes = stree1.query_ball_tree(stree2, r=0.2)
    print("kdtree:", indexes)  # noqa
    print("scipy:", sindexes)  # noqa
    print("kdtree:", tree1._dataset.index)  # noqa

    assert len(indexes) == len(sindexes)

    for i in range(len(indexes)):
        assert len(indexes[i]) == len(sindexes[i])

        for ind, sind in zip(indexes[i], sindexes[i], strict=True):
            assert ind == sind


def plot_query_ball_tree_example():
    """
    Diagnostic method for test_query_ball_tree_example
    """
    tree1, tree2, stree1, stree2, points1, points2 = (
        scipy_query_ball_tree_example_unwrapped()
    )
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, layout="constrained")
    fig.set_figwidth(12)
    fig.set_figheight(6)
    for ax, t1, t2, title in zip(
        axs, [tree1, stree1], [tree2, stree2], ["kdtree", "scipy"], strict=True
    ):
        ax.plot(points1[:, 0], points1[:, 1], "xk", markersize=14)
        ax.plot(points2[:, 0], points2[:, 1], "og", markersize=14)
        for i, xyz in enumerate(points1):
            ax.annotate(f"{i}", xy=xyz[:2] + 0.01, color="k")
        for i, xyz in enumerate(points2):
            ax.annotate(f"{i}", xy=xyz[:2] + 0.01, color="g")
        indexes = t1.query_ball_tree(t2, r=0.2)
        if isinstance(t1, KDTree):
            indexes = [sorted(inds) for inds in indexes]
        for i in range(len(indexes)):
            for j in indexes[i]:
                ax.annotate(
                    "",
                    xytext=points1[i, :2],
                    xy=points2[j, :2],
                    arrowprops={"color": "r", "ls": "-", "lw": 0.5},
                )
        ax.set_title(title)
    plt.show(block=True)
