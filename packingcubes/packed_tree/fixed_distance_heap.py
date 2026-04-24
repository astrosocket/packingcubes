"""FixedDistance Heap Implementation"""

from __future__ import annotations

import collections
import logging

import numpy as np
from numba import (  # type: ignore
    float64,
    int64,
    njit,
    types,
)
from numba.experimental import jitclass
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(capture=True)


@jitclass([("distances", float64[:]), ("indices", int64[:])])
class FixedDistanceHeap:
    """
    Custom, fixed-size max-heap that explicitly stores distance/index tuples

    An array-based implementation of a data structure similar to a max-heap,
    with a few modifications:
     - Each node in the heap represents a (distance, index) tuple. These are
       stored as two separate arrays, so care needs to be taken when moving
       nodes around not to break their coupling
     - The heap is fixed size. Effectively, it is initialized with k-elements
       that look like (inf, index_fill_value). Thus, there are no add/remove
       operations, only replacement.
     - We don't actually care about any popped items or items larger than the
       max distance. So the primary public interface method will be try_replace
       and we won't implement any sort of push/pop/pushpop/append/etc
     - We will often want the distances to be sorted in ascending order
       afterwards, so we'll need to store the root at the **right** of the tree
       This will reduce the number of comparison/swap operations.

    The algorithm is based on the heapq library.
    """

    distances: NDArray[np.float64]
    """ The k distances of the elements in the heap """
    indices: NDArray[np.int_]
    """ The k indices of the elements in the heap """
    max_distance: float
    """ The maximum distance stored in the heap (distances[0]) """

    def __init__(self, k: int, index_fill_value: int):
        self.distances = np.full((k,), np.inf, dtype=np.float64)
        self.indices = np.full((k,), index_fill_value, dtype=np.int_)
        self.max_distance = np.inf

    def _siftdown(self, pos: int):
        """Max-, fixed-length-heap variant of sift-down"""
        newdist = self.distances[pos]
        newind = self.indices[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newdist fits
        while pos > 0:
            parentpos = (pos - 1) >> 1
            parent_dist = self.distances[parentpos]
            parent_ind = self.indices[parentpos]
            if parent_dist < newdist:
                self.distances[pos] = parent_dist
                self.indices[pos] = parent_ind
                pos = parentpos
                continue
            break
        self.distances[pos] = newdist
        self.indices[pos] = newind

    def _siftup(self):
        """Max-, fixed-length-heap variant of sift-up."""
        # Note we don't need start pos because pos is **always** initialized to 0
        pos = 0
        endpos = len(self.distances)
        newdist = self.distances[pos]
        newind = self.indices[pos]
        # Bubble up the larger child until hitting a leaf
        childpos = 1
        while childpos < endpos:
            # Set childpos to index of larger child
            rightpos = childpos + 1
            if (
                rightpos < endpos
                and self.distances[rightpos] >= self.distances[childpos]
            ):
                childpos = rightpos
            # Move the larger child up
            self.distances[pos] = self.distances[childpos]
            self.indices[pos] = self.indices[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        # The leaf at pos is empty now. Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down)
        self.distances[pos] = newdist
        self.indices[pos] = newind
        self._siftdown(pos)

    def try_replace(self, new_dist: float, new_ind: np.int_) -> bool:
        """Try to add a new distance/index pair to the heap

        Parameters
        ----------
            new_dist: float
            The distance to try adding

            new_ind: int
            The corresponding index

        Returns
        -------
            outcome: bool
            1 if the distance replaced an element, 0 otherwise
        """
        if new_dist > self.max_distance:
            return False

        self.distances[0] = new_dist
        self.indices[0] = new_ind
        self._siftup()

        self.max_distance = self.distances[0]
        return True

    def sorted(self) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        """Return sorted distance-index pairs"""
        # If we just did distances.sort, we'd break the coupling...
        # In numba, both sort and argsort use a median of three quicksort by
        # default, with insertion sort if n is small (15 currently)
        # inds = self.distances.argsort()
        # new_dists = np.empty_like(self.distances)
        # new_inds = np.empty_like(self.indices)
        # for i in range(len(inds)):
        #     new_dists[i] = self.distances[inds[i]]
        #     new_inds[i] = self.indices[inds[i]]
        if self.max_distance == 0:
            return self.distances, self.indices
        return _quicksort1(self.distances, self.indices)


Partition = collections.namedtuple("Partition", ("start", "stop"))

# Under this size, switch to a simple insertion sort
SMALL_QUICKSORT = 15

# Max "recursion" depth
MAX_STACK = 100

zero = types.intp(0)


@njit
def _insertion_sort(A, Ap, low, high):
    """Insertion sort A[low:high + 1].

    Note the inclusive bounds.

    This is modified from the numba quicksort implementation to preserve the
    A[i], Ap[i] coupling and to skip the genericity
    """  # noqa: D401 # because it *is* imperative...
    assert low >= 0
    if high <= low:
        return

    for i in range(low + 1, high + 1):
        v = A[i]
        vp = Ap[i]
        # Insert v into A[low:i]
        j = i
        while j > low and v < A[j - 1]:
            # Make place for moving A[i] downwards
            A[j] = A[j - 1]
            Ap[j] = Ap[j - 1]
            j -= 1
        A[j] = v
        Ap[j] = vp


@njit
def _partition(A, Ap, low, high):
    """Partition A[low:high + 1] around a chosen pivot and return index.

    This is modified from the numba quicksort implementation to preserve the
    A[i], Ap[i] coupling and to skip the genericity
    """
    assert low >= 0
    assert high > low

    mid = (low + high) >> 1
    # NOTE: the pattern of swaps below for the pivot choice and the
    # partitioning gives good results (i.e. regular O(n log n))
    # on sorted, reverse-sorted, and uniform arrays.  Subtle changes
    # risk breaking this property.

    # median of three {low, middle, high}
    if A[mid] < A[low]:
        A[low], A[mid] = A[mid], A[low]
        Ap[low], Ap[mid] = Ap[mid], Ap[low]
    if A[high] < A[mid]:
        A[high], A[mid] = A[mid], A[high]
        Ap[high], Ap[mid] = Ap[mid], Ap[high]
    if A[mid] < A[low]:
        A[low], A[mid] = A[mid], A[low]
        Ap[low], Ap[mid] = Ap[mid], Ap[low]
    pivot = A[mid]

    # Temporarily stash the pivot at the end
    A[high], A[mid] = A[mid], A[high]
    Ap[high], Ap[mid] = Ap[mid], Ap[high]
    i = low
    j = high - 1
    while True:
        while i < high and A[i] < pivot:
            i += 1
        while j >= low and pivot < A[j]:
            j -= 1
        if i >= j:
            break
        A[i], A[j] = A[j], A[i]
        Ap[i], Ap[j] = Ap[j], Ap[i]
        i += 1
        j -= 1
    # Put the pivot back in its final place (all items before `i`
    # are smaller than the pivot, all items at/after `i` are larger)
    A[i], A[high] = A[high], A[i]
    Ap[i], Ap[high] = Ap[high], Ap[i]
    return i


@njit
def _quicksort1(A: NDArray, Ap: NDArray) -> tuple[NDArray, NDArray]:
    """Quicksort a 1D-array A, coupled to array Ap, using insertion sort if A small"""
    if len(A) < 2:
        return A, Ap

    # the numba implementation initializes with the following, but I find a
    # 5-200% speedup by using an array instead of a list
    # stack = [Partition(zero, zero)] * MAX_STACK
    stack = np.empty((MAX_STACK, 2), dtype=np.intp)
    stack[0, 0:2] = Partition(zero, len(A) - 1)
    n = 1

    while n > 0:
        n -= 1
        low, high = stack[n, 0:2]
        # Partition until it becomes more efficient to do an insertion sort
        while high - low >= SMALL_QUICKSORT:
            assert n < MAX_STACK
            i = _partition(A, Ap, low, high)
            # Push largest partition on the stack
            if high - i > i - low:
                # Right is larger
                if high > i:
                    stack[n, 0:2] = Partition(i + 1, high)
                    n += 1
                high = i - 1
            else:
                if i > low:
                    stack[n, 0:2] = Partition(low, i - 1)
                    n += 1
                low = i + 1

        _insertion_sort(A, Ap, low, high)

    return A, Ap
