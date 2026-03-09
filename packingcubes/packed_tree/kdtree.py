from __future__ import annotations

import logging
import warnings

import numpy as np
from numpy.typing import NDArray

import packingcubes.bounding_box as bbox
import packingcubes.octree as octree
from packingcubes.data_objects import Dataset, InMemory
from packingcubes.packed_tree.packed_tree import PackedTree

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(capture=True)


class KDTreeWarning(octree.OctreeWarning):
    pass


class KDTreeError(octree.OctreeError):
    pass


class KDTreeAPI:
    """
    Class to mimic the SciPy KDTree API using PackedTrees

    Will provide identical API to SciPy's KDTree to the extent possible given
    that PackedTrees are fundamentally different. Where 1-1 matches for a
    requested method, argument, or functionality are not possible, raise an
    KDTreeError if there is nothing similar and emit a KDTreeWarning explaining
    the replacement otherwise.

    Warning! PackedTrees are not robust against large amounts of degenerate
    input data! Please sanitize data prior to usage if expecting data
    degeneracy levels above ~100 (i.e. 100 data points with the same values).
    Note that multiple degenerate regions are acceptable, assuming they are
    sufficiently separated.
    """

    _tree: PackedTree
    _dataset: Dataset
    data: NDArray
    """
    The n data points of dimension m to be indexed. The data is only copied if
    the "kd-tree" is built with copy_data=True.
    """
    n: int
    """
    The number of data points.
    """
    leafsize: int
    """
    The number of points at which the algorithm switches over to brute-force
    """
    maxs: NDArray
    """
    The maximum value in each dimension of the n data points
    """
    mins: NDArray
    """
    The minimum value in each dimension of the n data points
    """
    size: int
    """
    The number of nodes in the tree.
    """

    def __init__(
        self,
        data: NDArray,
        leafsize: int | None = None,
        compact_nodes: bool | None = None,  # noqa: FBT001
        copy_data: bool = False,  # noqa: FBT001, FBT002
        balanced_tree: bool | None = None,  # noqa: FBT001
        boxsize=None,
    ):
        if compact_nodes is not None:
            if boxsize is None:
                extra = (
                    "which is set from the outermost datapoints."
                    " So setting compact_nodes does nothing."
                )
            else:
                extra = f"which has been manually set to {boxsize}."
            warnings.warn(
                "Node sizes in PackedTrees are set by the geometry of the"
                "total bounding box, " + extra,
                KDTreeWarning,
                stacklevel=1,
            )
        if leafsize is None:
            LOGGER.info(
                "Using the default PackedTree leaf size "
                f"({octree._DEFAULT_PARTICLE_THRESHOLD}) instead of the KDTree's (10)"
            )
        if balanced_tree is not None and balanced_tree:
            warnings.warn(
                "PackedTree nodes are split at the middle of the bounding box, "
                "independent of the data contained. Setting balanced_tree does "
                "nothing.",
                KDTreeWarning,
                stacklevel=1,
            )
        data_shape = data.shape
        if len(data_shape) != 2 or data_shape[1] != 3:
            raise KDTreeError(
                "PackedTrees only support 3-dimensional data. Provided data "
                + (
                    f"was {data_shape[1]}-dimensional"
                    if len(data_shape) >= 2
                    else "was 1-dimensional."
                )
            )
        self._dataset = InMemory(positions=data.copy() if copy_data else data)
        self.data = self._dataset.positions
        self.n = len(self.data)
        data_box = self._dataset.bounding_box
        self.mins = data_box.box[:3]
        self.maxs = data_box.box[:3] + data_box.box[3:]

        if boxsize is not None:
            boxsize = np.atleast_1d(boxsize)
            box_warning = """
            PackedTrees do not need or expect data points to be normalized to the
            [0, 1) interval. We will assume that the data is within [0, L_i] with
            no wrapping. If you have negative or overly-large data values, simply
            use the full 6 terms and provide the full bounding box or pass None to
            generate it from the data extents. If you truly need the toroidal
            geometry, please impose that before calling the constructor. \n\n
            You can suppress this message by passing the full 6 terms (or None).
            """
            match len(boxsize):
                case 1:
                    warnings.warn(box_warning, KDTreeWarning, stacklevel=1)
                    box = np.array([0, 0, 0, boxsize, boxsize, boxsize])
                case 3:
                    warnings.warn(box_warning, KDTreeWarning, stacklevel=1)
                    box = np.hstack(([0, 0, 0], boxsize.flatten()))
                case 6:
                    box = boxsize
                case _:
                    raise KDTreeError(
                        f"Cannot handle boxsize argument with length {len(boxsize)}. "
                        "Supported options are 1, 3, and 6."
                    )
            bounding_box = bbox.make_bounding_box(box)
        else:
            bounding_box = data_box

        self.leafsize = (
            octree._DEFAULT_PARTICLE_THRESHOLD if leafsize is None else leafsize
        )
        self._tree = PackedTree(
            dataset=self._dataset,
            particle_threshold=leafsize,
            bounding_box=bounding_box,
        )
        # Each node is 5 fields, so number of nodes = length/5
        self.size = int(len(self._tree._tree.tree) / 5)

    def _query_ball_point(
        self,
        *,
        centers: NDArray,
        radius: float,
        return_length: bool = False,
        return_sorted: bool | None = False,
        strict: bool = False,
    ) -> list[int] | NDArray:
        """
        Private method to actually compute the query after inputs validated
        """

        if return_length:
            # psuedocode:
            # for center in centers:
            #     sph = sphere(center, radius)
            #     nodes = list of tree nodes that overlap sph
            #     sum = 0
            #     for node in nodes:
            #         sum += node.end-node.start + 1
            #     append sum to result
            return [
                sum(
                    e - s + 1
                    for (e, s) in self._tree.get_particle_indices_in_sphere(
                        center=center, radius=radius
                    )
                )
                for center in centers
            ]
        results = [
            self._tree.get_particle_index_list_in_sphere(
                dataset=self._dataset,
                center=center,
                radius=radius,
                strict=strict,
            )
            for center in centers
        ]
        #  results is now list of list of (unsorted) indices into shuffle list
        if return_sorted:
            for r in results:
                r.sort()

        if len(centers) > 1:
            return np.fromiter(results, dtype=np.object_)
        return results[0]

    def query_ball_point(
        self,
        x: NDArray,
        r: float,
        p: float | None = 2.0,
        eps: float | None = None,
        workers: int = 1,
        *,
        return_sorted: bool | None = None,
        return_length: bool = False,
        strict: bool | None = None,
    ) -> list[int] | NDArray:
        """
        Find all points within distance r of point(s) x.

        Args:
            x : array_like, shape tuple + (self.m,)
                The point or points to search for neighbors of.
            r : array_like, float
                The radius of points to return, must broadcast to the length of x.
            p : float, optional
                Which Minkowski p-norm to use.  Should be in the range [1, inf].
                A finite large p may cause a ValueError if overflow can occur.
            eps : nonnegative float, optional
                Approximate search. Branches of the tree are not explored if their
                nearest points are further than ``r / (1 + eps)``, and branches are
                added in bulk if their furthest points are nearer than
                ``r * (1 + eps)``.
            workers : int, optional
                Number of jobs to schedule for parallel processing. If -1 is given
                all processors are used. Default: 1.

            return_sorted : bool, optional
                Sorts returned indices if True and does not sort them if False. If
                None, does not sort single point queries, but does sort
                multi-point queries which was the behavior before this option
                was added.

            return_length : bool, optional
                Return the number of points inside the radius instead of a list
                of the indices. Note that this is much faster for large trees.

        Returns:
            results : list or array of lists
                If `x` is a single point, returns a list of the indices of the
                neighbors of `x`. If `x` is an array of points, returns an object
                array of shape tuple containing lists of neighbors.

        Notes:
        If you have many points whose neighbors you want to find, you may save
        substantial amounts of time by putting them in a KDTree and using
        query_ball_tree.

        Examples:
        >>> import numpy as np
        >>> from scipy import spatial
        >>> x, y = np.mgrid[0:5, 0:5]
        >>> points = np.c_[x.ravel(), y.ravel()]
        >>> tree = spatial.KDTree(points)
        >>> sorted(tree.query_ball_point([2, 0], 1))
        [5, 10, 11, 15]

        Query multiple points and plot the results:

        >>> import matplotlib.pyplot as plt
        >>> points = np.asarray(points)
        >>> plt.plot(points[:,0], points[:,1], '.')
        >>> for results in tree.query_ball_point(([2, 0], [3, 3]), 1):
        ...     nearby_points = points[results]
        ...     plt.plot(nearby_points[:,0], nearby_points[:,1], 'o')
        >>> plt.margins(0.1, 0.1)
        >>> plt.show()

        """
        x = np.atleast_2d(x)
        if len(x.shape) > 2 or x.shape[1] != 3:
            raise KDTreeError(
                "PackedTrees only support 3-dimensional query points. "
                + (
                    f"Provided point(s) were {x.shape[1]}-dimensional"
                    if len(x.shape[1]) != 3
                    else (
                        "Provided inputs were "
                        f"{'x'.join(f'{x1}' for x1 in x.shape[1:])}"
                    )
                )
            )

        p = 2 if p is None else p
        if p != 2:
            warnings.warn(
                (
                    "PackedTrees currently only support the Minkowski 2-norm "
                    f"(requested {p}). Continuing with p=2."
                ),
                KDTreeWarning,
                stacklevel=1,
            )

        if eps is not None:
            if eps < 0:
                raise ValueError("eps must be nonnegative.")
            if eps > 0:
                warnings.warn(
                    (
                        """
                        PackedTrees do not quantify the distance between points/nodes
                        in the same way as KDTrees. Setting eps>0 is equivalent to
                        strict=False (default).
                        """
                    ),
                    KDTreeWarning,
                    stacklevel=1,
                )
            else:
                strict = True if strict is None else strict
        strict = True if strict is None else strict

        if workers != 1:
            warnings.warn(
                """
                PackedTrees are single-threaded. For multi-threading consider
                switching to the Cubes API. Proceeding with workers=1.
                """,
                KDTreeWarning,
                stacklevel=1,
            )

        return self._query_ball_point(
            centers=x,
            radius=r,
            return_length=return_length,
            return_sorted=return_sorted,
            strict=strict,
        )
