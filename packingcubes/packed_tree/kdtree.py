from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

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


def _check_3d(x: ArrayLike) -> NDArray:
    x = np.atleast_2d(x)
    if len(x.shape) > 2 or x.shape[1] != 3:
        raise KDTreeError(
            "PackedTrees only support 3-dimensional query points. "
            + (
                f"Provided point(s) were {x.shape[1]}-dimensional"
                if x.shape[1] != 3
                else (f"Provided inputs were {'x'.join(f'{x1}' for x1 in x.shape[1:])}")
            )
        )
    return x


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

    Args:
        data : array_like, shape (n,3)
        The n 3-dimensional data points to be indexed. This array is
        not copied and will be sorted in place, so modifying this data will
        result in bogus results. The data are also copied if the kd-tree is
        built with copy_data=True.

        leafsize : positive int, optional
        The number of points at which the algorithm switches over to
        brute-force.  Default: 400.

        compact_nodes : bool, optional
        This parameter is irrelevant for PackedTrees and is only provided
        to match the KDTree API.

        copy_data : bool, optional
        If True the data is copied to protect the kd-tree against
        data corruption and to prevent the original data from being sorted.
        Default: False.

        balanced_tree : bool, optional
        PackedTrees are always split at the bounding box midpoint, so this
        option is only provided to match the KDTree API

        boxsize : array_like or scalar, optional
        Provide an explicit bounding box for the data in the form
        [x_min, y_min, z_min, dx, dy, dz]. If len(boxsize)==3, x_min = y_min =
        z_min = 0. If boxsize is a scalar, dx = dy = dz = boxsize. Other
        boxsize lengths are unsupported. Scipy's KDTree will impose a toroidal
        topology in addition; this functionality is currently unsupported.
    """

    _tree: PackedTree
    """
    The actual tree
    """
    _dataset: Dataset
    """
    Link to the dataset used. Needed for returning strict index lists.
    """
    _copied: bool
    """
    Whether the tree was constructed with copied data
    """
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
        self._copied = copy_data
        self._data_container = self._dataset.data_container
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

    @property
    def sort_index(self):
        """
        The shuffle list for the original data, aka self.data = data[self.sort_index]
        """
        if self._copied:
            warnings.warn(
                """
                Since the data was copied, the effective shuffle list is simply
                0:len(data) and this property is superfluous.
                """,
                KDTreeWarning,
                stacklevel=1,
            )
            return np.arange(len(self.dataset))
        index = self._dataset.index.view()
        index.flags.writeable = False
        return index

    def _query(
        self,
        *,
        x: NDArray,
        k: int | Sequence[int],
        distance_upper_bound: float,
        p: float,
        return_data_indices: bool,
    ) -> tuple[float | NDArray, int | NDArray]:
        """
        Private helper that wraps the PackedTree get_closest_particles method

        See query for argument definitions
        """
        k_max = k if not isinstance(k, Sequence) else max(k)

        d = np.full((len(x), k_max), np.inf)
        i = np.full((len(x), k_max), self.n)
        for ind, xyz in enumerate(x):
            dists, inds = self._tree.get_closest_particles(
                data=self._data_container,
                xyz=xyz,
                distance_upper_bound=distance_upper_bound,
                p=p,
                k=k_max,
            )
            d[ind, : len(dists)] = dists
            # kdtree needs the *original* indices if we copied the data and
            # might want them if we didn't
            i[ind, : len(inds)] = (
                inds if return_data_indices else self._dataset._index[inds]
            )
        if k_max == 1 and not isinstance(k, Sequence):
            return d.squeeze(), i.squeeze()
        if isinstance(k, Sequence):
            k_inds = np.fromiter(k, dtype=int) - 1
            return d[:, k_inds], i[:, k_inds]
        return d, i

    def query(
        self,
        x: ArrayLike,
        k: int | Sequence[int] = 1,
        eps: float | None = None,
        p: int | None = None,
        distance_upper_bound: float | None = None,
        workers: int | None = None,
        *,
        return_data_indices: bool | None = None,
    ) -> tuple[float | NDArray, int | NDArray]:
        """
        Query the KDTree for nearest neighbors

        Args:
            x: NDArray
            An array of points to query

            k: int | Sequence[int], optional
            Either the number of nearest neighbors to return or a list of the
            kth nearest neighbors to return, starting from 1. E.g., [2,3] will
            return the 2nd and 3rd nearest neighbors

            eps: nonnegative float, optional
            Return approximate nearest neighbors; Note that this parameter is
            unused

            p: 1<=p<=infinity, optional
            The Minkowski p-norm to use. 1 is the sum of absolute-values
            distance ("Manhattan" distance). 2 is the Euclidean distance.
            infinity is the maximum-coordinate-difference distance. Currently
            only p=2 is supported

            distance_upper_bound: nonnegative float, optional
            Return only neighbors from other nodes within this distance. This
            is used for tree pruning, so if you are doing a series of
            nearest-neighbor queries, it may help to supply the distance to the
            nearest neighbor of the most recent point.

            workers: int, optional
            Number of workers to use for parallel processing. Only 1 is
            supported, for more, see Cubes

            return_data_indices: bool | None, optional
            Return indices into the sorted data if True instead of into the
            original. Specify None to have this set by the copy_data argument
            used during tree construction.

        Returns:
            d: float or array of floats
            The distances to the nearest neighbors. If x has shape
            tuple+(self.m,), then d has shape tuple+(k,). When k==1, the last
            dimension of the output is squeezed. Missing neighbors are
            indicated with infinite distances. Hits are sorted by distance
            (nearest first)

            i: integer or array of integers
            The index of each neighbor in self.data. i is the same shape as d.
            Missing neighbors are indicated with self.n.

        Raises:
            NotImplementedError if p!=2

        """
        x = _check_3d(x)

        distance_upper_bound = (
            1e100 if distance_upper_bound is None else distance_upper_bound
        )
        p = 2 if p is None else p

        if eps is not None:
            if eps < 0:
                raise ValueError("eps must be nonnegative.")
            if eps > 0:
                warnings.warn(
                    (
                        """
                        PackedTrees do not quantify the distance between points/nodes
                        in the same way as KDTrees and setting this parameter has no
                        effect
                        """
                    ),
                    KDTreeWarning,
                    stacklevel=1,
                )

        if workers is not None and workers != 1:
            warnings.warn(
                """
                PackedTrees are single-threaded. For multi-threading consider
                switching to the Cubes API. Proceeding with workers=1.
                """,
                KDTreeWarning,
                stacklevel=1,
            )

        return_data_indices = (
            not self._copied if return_data_indices is None else return_data_indices
        )

        return self._query(
            x=x,
            k=k,
            distance_upper_bound=distance_upper_bound,
            p=p,
            return_data_indices=return_data_indices,
        )

    def _query_ball_point(
        self,
        *,
        centers: NDArray,
        radius: float,
        return_length: bool,
        return_sorted: bool,
        return_lists: bool,
        return_data_indices: bool,
        strict: bool = False,
    ) -> int | list[int] | NDArray:
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
            lengths = [
                sum(
                    e - s
                    for (s, e) in self._tree.get_particle_indices_in_sphere(
                        center=center, radius=radius
                    )
                )
                for center in centers
            ]
            return lengths[0] if len(lengths) == 1 else lengths
        results = [
            self._tree.get_particle_index_list_in_sphere(
                data=self._data_container,
                center=center,
                radius=radius,
                strict=strict,
            )
            for center in centers
        ]
        # get_particle_index_list_in_sphere returns the indices in the
        # reordered dataset. If we need the original indices, we need to use
        # the shuffle list
        if not return_data_indices:
            results = [self._dataset._index[r] for r in results]
        #  results is now list of list of (unsorted) indices
        if return_sorted:
            for r in results:
                r.sort()

        if return_lists:
            results = [r.tolist() for r in results]

        if len(centers) > 1:
            return np.fromiter(results, dtype=np.object_)
        return results[0]

    def query_ball_point(
        self,
        x: ArrayLike,
        r: float,
        p: float | None = 2.0,
        eps: float | None = None,
        workers: int = 1,
        *,
        return_sorted: bool | None = None,
        return_length: bool = False,
        return_lists: bool | None = None,
        return_data_indices: bool | None = None,
        strict: bool | None = None,
    ) -> int | list[int] | NDArray:
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

            return_lists : bool, optional
                Force returning lists instead of arrays. PackedTrees return
                arrays of indices by default, but this doesn't match the
                expected query_ball_point signature. For a slight performance
                increase, set this to False

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
        x = _check_3d(x)

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

        if strict and return_length:
            raise NotImplementedError(
                "For performance, strict is only valid when return_length=False"
            )

        if workers != 1:
            warnings.warn(
                """
                PackedTrees are single-threaded. For multi-threading consider
                switching to the Cubes API. Proceeding with workers=1.
                """,
                KDTreeWarning,
                stacklevel=1,
            )

        return_lists = True if return_lists is None else return_lists
        return_sorted = x.shape[0] > 1 if return_sorted is None else return_sorted

        return_data_indices = (
            not self._copied if return_data_indices is None else return_data_indices
        )

        return self._query_ball_point(
            centers=x,
            radius=r,
            return_length=return_length,
            return_sorted=return_sorted,
            return_lists=return_lists,
            return_data_indices=return_data_indices,
            strict=strict,
        )
