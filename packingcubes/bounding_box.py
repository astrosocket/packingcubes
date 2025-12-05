from __future__ import annotations

import logging
from dataclasses import InitVar, dataclass
from enum import Flag, auto

import numpy as np
from numpy.typing import ArrayLike

LOGGER = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    box: np.ndarray
    """ The left-front-bottom and length-width-height of the bounding box """
    force_skip_check: InitVar[bool] = False
    """ Skip validation checks """

    def __post_init__(self, force_skip_check):
        if not force_skip_check:
            check_valid(self.box)


type BoxLike = ArrayLike | BoundingBox


class BoundingBoxValidFlag(Flag):
    """
    Validations performed on a box
    """

    IS_BOX = auto(), lambda box: f"Box ({box}) is not a box!"
    CORRECT_SHAPE = (
        auto(),
        lambda box: f"Box has wrong shape. Required (6,), received {box.shape}",
    )
    FINITE = auto(), lambda box: f"Box ({box}) has inf/NaN values"
    POSITIVE = auto(), lambda box: f"Box has an invalid size: {box[3:]}"
    PRECISION = (
        auto(),
        lambda box: f"Provided box is too small ({box}). Precision will be lost",
    )
    # TODO: The following (e.g. IS_BOX[0]) is hacky and I don't like it...
    VALID = (
        IS_BOX[0] | CORRECT_SHAPE[0] | FINITE[0] | POSITIVE[0] | PRECISION[0],
        lambda box: "This box is valid",
    )

    def __new__(cls, value: int, message_fun: callable):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.message_fun = message_fun
        return obj

    def get_messages(self, box: ArrayLike):
        """
        Obtain the test results from this flag assuming the provided box
        """
        if self == 0:
            # If this happens, hoo boy...
            return [f"Something is very wrong with bbox: {box}"]
        return [
            flag.message_fun(box) for flag in BoundingBoxValidFlag if flag not in self
        ]


class BoundingBoxError(ValueError):
    """BoundingBox validation error"""

    def __init__(
        self,
        box: ArrayLike,
        errortype: BoundingBoxValidFlag,
    ):
        """
        Args:
            box: ArrayLike
            The invalid box responsible

            errortype: BoundingBoxValidFlag
            Flag listing the box definition violations
        """
        self.errortype = errortype
        self.box = box
        self.messages = self.errortype.get_messages(self.box)
        if len(self.messages):
            self.messages = "Bounding Box errors: \n" + "\n".join(self.messages)
        super().__init__(self.messages)


def check_valid(box: np.ndarray, *, raise_error: bool = True):
    """
    Check if a bounding box array is valid

    Tests that a numpy array has the following attributes:
        1. it has shape (6, )
        2. it has finite values
        3. Values 4-6 are positive
        4. Values 4-6 are larger than the floating point precision of the 1-3
            values (i.e. box[0] + box[3] > box[0])

    Args:
        box: numpy.ndarray
        Array to check

        raise_error: bool, optional
        How to deal with an invalid box. Raise error on True (default) or
        return test results flag

    Returns:
        flag: BoundingBoxValidFlag
        Flag of all test results

    Raises:
        BoundingBoxError if raise_error is True and box is invalid
    """
    flag = BoundingBoxValidFlag.VALID
    if not isinstance(box, np.ndarray):
        flag ^= BoundingBoxValidFlag.IS_BOX
        if raise_error:
            raise BoundingBoxError(box=box, errortype=flag)
        return flag
    if len(box) != 6 or box.shape != (6,):
        flag ^= BoundingBoxValidFlag.CORRECT_SHAPE
    else:
        if not np.all(np.isfinite(box)):
            flag ^= BoundingBoxValidFlag.FINITE
        if np.any(box[3:] <= 0):
            flag ^= BoundingBoxValidFlag.POSITIVE
        if np.any(
            np.isfinite(box[:3])
            & np.isfinite(box[3:])
            & (box[3:] > 0)
            & (box[3:] < (np.abs(box[:3]) + box[3:]) * np.finfo(float).eps),
        ):
            flag ^= BoundingBoxValidFlag.PRECISION
    if raise_error and flag != BoundingBoxValidFlag.VALID:
        raise BoundingBoxError(box=box, errortype=flag)
    return flag


def make_valid(bbox: BoxLike) -> BoundingBox:
    """
    Coerce a box-like object into a BoundingBox if possible. No-op if already.

    Args:
        bbox: BoxLike
        Something that looks like a box, i.e. a numpy array with in the form
        [x, y, z, dx, dy, dz] with the following conditions: finite, all
        |x[i]| * epsilon < dx[i] or a BoundingBox

    Returns:
        bbox: BoundingBox
        The input box as a BoundingBox

    Raises:
        BoundingBoxError if coercing is not possible

    See Also:
        check_valid
    """
    if isinstance(bbox, BoundingBox):
        return bbox
    return BoundingBox(np.atleast_1d(np.squeeze(np.asanyarray(bbox))))


def in_box(bbox: BoxLike, xyz: ArrayLike) -> np.ndarray:
    """
    Check if points are inside box

    Args:
        bbox: BoxLike
        Box to check

        xyz: ArrayLike
        Array of points with shape Nx3 to test. (3,) arrays will be converted

    Returns:
        in_box: ndarray[bool]
        Boolean array where True means point is inside box
    """
    bbox = make_valid(bbox)
    xyz = np.atleast_2d(xyz)
    return np.all((bbox.box[:3] <= xyz) & (xyz <= bbox.box[:3] + bbox.box[3:]), axis=1)


def midplane(bbox: BoxLike) -> ArrayLike:
    """
    Return the 3 coordinates specifying the midplane of the box
    """
    bbox = make_valid(bbox)
    # This is equivalent to
    # return bbox.box[:3] + bbox.box[3:] / 2
    # but we only need the three terms, not the full
    # array, so return a tuple
    return (
        bbox.box[0] + bbox.box[3] / 2,
        bbox.box[1] + bbox.box[4] / 2,
        bbox.box[2] + bbox.box[5] / 2,
    )


def max_depth(bbox: BoundingBox) -> int:
    """
    Get max depth supported by this box

    Max depth is defined as the maximum number of times this box can be split
    in half before x[i] + dx[i] == x[i].
    """
    min_box_sizes = (
        np.maximum(1, np.abs(bbox.box[:3]) + bbox.box[3:]) * np.finfo(float).eps
    )
    return np.ceil(np.log2(np.min(bbox.box[3:] / min_box_sizes))).astype(int)


def normalize_to_box(coordinates: ArrayLike, bbox: BoxLike) -> ArrayLike:
    """
    Rescale and shift the coordinates such that they are bounded by the unit cube
    """
    bbox = make_valid(bbox)
    # Need to deal with subnormal values
    fixed_subnormal = np.sign(coordinates) * np.clip(
        np.abs(coordinates),
        a_min=np.nextafter(bbox.box[3:], bbox.box[3:] + 1) - bbox.box[3:],
        a_max=None,
    )
    return np.clip(
        (fixed_subnormal - bbox.box[:3]) / bbox.box[3:],
        a_min=0.0,
        a_max=1.0,
    )


def get_neighbor_boxes(bbox: BoxLike) -> ArrayLike:
    """
    Return the 26 boxes that would be the neighbors of this box in a uniform grid

    Boxes are returned as a 26x6 array, where each row is a box. Order is
    z-order, so row 0 is the box at [x-dx,y-dy,z-dz], row 2 is [x+dx,y-dy,z-dz]
    and row 25 is [x+dx,y+dy,z+dz]
    """
    bbox = make_valid(bbox)
    # We generate all 27 boxes (so including box) and then remove box because
    # it makes the code logic *much* simpler and shouldn't significantly
    # increase the number of resources used
    neighbors = np.zeros((27, 6), dtype=bbox.box.dtype)
    dxv = np.zeros(6, dtype=bbox.box.dtype)
    for i in range(27):
        dxv[0] = ((i % 3) - 1) * bbox.box[3]  # dx
        dxv[1] = ((int(i / 3) % 3) - 1) * bbox.box[4]  # dy
        dxv[2] = ((int(i / 9) % 3) - 1) * bbox.box[5]  # dz
        neighbors[i, :] = bbox.box + dxv
    return np.vstack((neighbors[:13], neighbors[14:]))


def get_child_box(bbox: BoundingBox, ind: int) -> BoundingBox:
    """
    Get indth (0-indexed) new child box of current box

    New child box is defined as the suboctant described by position ind
    with size (box[3]/2, box[4]/2, box[5]/2). The child box is not currently
    guaranteed to be a proper BoundingBox

    Args:
        bbox: BoundingBox
        Parent box

        ind: int
        Z-order index (0-indexed) of child box

    Returns:
        child_box: BoundingBox
    """
    # Use z-index order for now, but other possibilities
    # like Hilbert curves exist - and see
    # https://math.stackexchange.com/questions/2411867/3d-hilbert-curve-without-double-length-edges
    # for a possible "Hilbert" curve that may be better?
    if not isinstance(ind, int) or ind < 0 or 7 < ind:
        raise ValueError(f"Octree code passed an invalid index: {ind}!")

    child_box = bbox.box.copy()
    child_box[3:] /= 2.0
    x, y, z = ((ind & 1) / 1, (ind & 2) / 2, (ind & 4) / 4)
    child_box[0] = child_box[0] + child_box[3] * x
    child_box[1] = child_box[1] + child_box[4] * y
    child_box[2] = child_box[2] + child_box[5] * z

    return BoundingBox(child_box, force_skip_check=True)


def get_box_center(bbox: BoxLike) -> ArrayLike:
    """
    Return the coordinates of the center of the box
    """
    bbox = make_valid(bbox)
    return bbox.box[:3] + bbox.box[3:] / 2


def get_box_vertex(bbox: BoxLike, index: int, *, jitter: float = 0) -> ArrayLike:
    """
    Return the coordinates of the vertex at z-order index (1-based)

    Note that a jitter can be applied. If so the coordinates will be the
    vertex of the box slightly (1%) smaller (larger) if jitter is positive
    (negative)

    Args:
        bbox: BoxLike
        Box to get vertex of

        index: int
        Z-order (0-based) index of vertex

        jitter: float, optional
        Jitter direction. Default 0 (no jitter)

    Returns:
        vertex: numpy.ndarray
        (3,) numpy array corresponding to the specified vertex
    """
    bbox = make_valid(bbox)
    if not isinstance(index, int):
        raise ValueError("Index must be an int!")
    if index < 1 or index > 8:
        raise ValueError(f"Index {index} is out of bounds")
    if not np.isfinite(jitter):
        raise ValueError(f"Jitter ({jitter}) must be finite!")
    index -= 1
    coord = bbox.box[:3].copy()
    for i in range(3):
        offset = (index & (1 << i)) >> i
        coord[i] += bbox.box[3 + i] * (
            offset + (1 - 2 * offset) * np.sign(jitter) / 100
        )
    return coord


def get_box_vertices(bbox: BoxLike, *, jitter: float = 0) -> ArrayLike:
    """
    Return the coordinates of the 8 box vertices in z-order

    Note that a jitter can be applied. If so the coordinates will be the
    vertices of the box slightly (1%) smaller (larger) if jitter is positive
    (negative)

    Args:
        bbox: BoxLike
        Box to get vertex of

        jitter: float, optional
        Jitter direction. Default 0 (no jitter)

    Returns:
        vertices: numpy.ndarray
        (6,3) numpy array corresponding to the box vertices in z-order
    """
    if not np.isfinite(jitter):
        raise ValueError("Jitter must be a finite value")

    bbox = make_valid(bbox)

    vertices = np.zeros((8, 3))

    jitter_amount = np.sign(jitter) * bbox.box[3:] / 100

    for k in range(2):
        for j in range(2):
            for i in range(2):
                ind = i + 2 * j + 4 * k
                vertices[ind, :] = [
                    bbox.box[0] + i * bbox.box[3] + jitter_amount[0] * (-1) ** i,
                    bbox.box[1] + j * bbox.box[4] + jitter_amount[1] * (-1) ** j,
                    bbox.box[2] + k * bbox.box[5] + jitter_amount[2] * (-1) ** k,
                ]

    return vertices


def project_point_on_box(
    bbox: BoxLike,
    xyz: ArrayLike,
    *,
    jitter: float = 0,
) -> np.ndarray:
    """
    Return coordinates of projection of (x, y, z) on nearest box face.

    This is the closest point on the box to (x, y, z). Can provide jitter to
    place point into/out of the box for determining sub-boxes.

    Note: There is no checking for whether points are already inside box

    Note: Jitter is not applied to points inside the box (so a point cannot
        be jittered *out*).

    Args:
        bbox: BoxLike
        Box to project on

        xyz: ArrayLike
        Point to project onto nearest box face

        jitter: nonnegative float
        Flag to move projected point 1% into the box. Negative values to move
        the point out of the box are not yet supported. Default is 0

    Returns:
        pxyz: numpy.ndarray
        Projected coordinates
    """
    bbox = make_valid(bbox)

    if np.any(np.isnan(xyz)):
        raise ValueError("Point contains NaN!")

    if np.isnan(jitter):
        raise ValueError("Jitter must be a number!")

    if jitter < 0:
        raise NotImplementedError()

    jitter = np.sign(jitter) * bbox.box[3:] / 100
    return np.clip(
        xyz,
        a_min=bbox.box[:3] + jitter,
        a_max=bbox.box[:3] + bbox.box[3:] - jitter,
    ).astype(float)
