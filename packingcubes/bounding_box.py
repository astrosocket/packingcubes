from __future__ import annotations

import logging
from enum import Flag, auto

import numpy as np
from numpy.typing import ArrayLike

LOGGER = logging.getLogger(__name__)


class BoundingBoxValidFlag(Flag):
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
    # TODO The following is hacky and I don't like it...
    VALID = (
        IS_BOX[0] | CORRECT_SHAPE[0] | FINITE[0] | POSITIVE[0] | PRECISION[0],
        lambda box: "This box is valid",
    )

    def __new__(cls, value: int, message_fun: callable):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.message_fun = message_fun
        return obj

    def get_Messages(self, box: ArrayLike):
        if self == 0:
            # If this happens, hoo boy...
            return [f"Something is very wrong with box: {box}"]
        messages = []
        for flag in BoundingBoxValidFlag:
            if flag not in self:
                messages.append(flag.message_fun(box))
        return messages


class BoundingBoxError(ValueError):
    def __init__(
        self,
        box: ArrayLike,
        errortype: BoundingBoxValidFlag = BoundingBoxValidFlag.VALID,
    ):
        self.errortype = errortype
        self.box = box
        self.messages = self.errortype.get_Messages(self.box)
        if len(self.messages):
            self.messages = "Bounding Box errors: \n" + "\n".join(self.messages)
        super().__init__(self.messages)


def check_valid(box: np.ndarray, *, raise_error: bool = True):
    """
    Check if a bounding box is valid

    Tests that a numpy array has the following attributes:
        1. it has shape (6, )
        2. it has finite values
        3. Values 4-6 are positive
        4. Values 4-6 are larger than the floating point precision of the 1-3
            values (i.e. box[0] + box[3] > box[0])
    """
    flag = BoundingBoxValidFlag.VALID
    if not isinstance(box, np.ndarray):
        flag ^= BoundingBoxValidFlag.IS_BOX
        if raise_error:
            raise BoundingBoxError(box=box, errortype=flag)
        else:
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
            & (box[3:] < (np.abs(box[:3]) + box[3:]) * np.finfo(float).eps)
        ):
            flag ^= BoundingBoxValidFlag.PRECISION
    if raise_error and flag != BoundingBoxValidFlag.VALID:
        raise BoundingBoxError(box=box, errortype=flag)
    return flag


def _make_valid(box: ArrayLike):
    """
    Coerce a box-like object into a box if possible
    """
    box = np.atleast_1d(np.squeeze(np.asanyarray(box)))
    check_valid(box, raise_error=True)
    return box


def in_box(box: ArrayLike, xyz: ArrayLike) -> np.ndarray:
    """
    Check if point is inside box
    """
    box = _make_valid(box)
    xyz = np.atleast_2d(xyz)
    return np.all((box[:3] <= xyz) & (xyz <= box[:3] + box[3:]), axis=1)


def midplane(box: ArrayLike) -> ArrayLike:
    """
    Return the 3 coordinates specifying the midplane of the box
    """
    box = _make_valid(box)
    return np.array([box[i] + box[i + 3] / 2 for i in range(3)])


def normalize_to_box(coordinates: ArrayLike, box: ArrayLike) -> ArrayLike:
    """
    Rescale and shift the coordinates such that they are bounded by the unit cube
    """
    box = _make_valid(box)
    # Need to deal with subnormal values
    fixed_subnormal = np.sign(coordinates) * np.clip(
        np.abs(coordinates),
        a_min=np.nextafter(box[3:], box[3:] + 1) - box[3:],
        a_max=None,
    )
    return np.clip((fixed_subnormal - box[:3]) / box[3:], a_min=0.0, a_max=1.0)


def get_neighbor_boxes(box: ArrayLike) -> ArrayLike:
    """
    Return the twenty-six boxes that would be the neighbors of this box in a uniform grid

    Boxes are returned as a 26x6 array, where each row is a box. Order is
    z-order, so row 0 is the box at [x-dx,y-dy,z-dz], row 2 is [x+dx,y-dy,z-dz]
    and row 25 is [x+dx,y+dy,z+dz]
    """
    box = _make_valid(box)
    # We generate all 27 boxes (so including box) and then remove box because
    # it makes the code logic *much* simpler and shouldn't significantly
    # increase the number of resources used
    neighbors = np.zeros((27, 6), dtype=box.dtype)
    dxv = np.zeros(6, dtype=box.dtype)
    for i in range(27):
        dxv[0] = ((i % 3) - 1) * box[3]  # dx
        dxv[1] = ((int(i / 3) % 3) - 1) * box[4]  # dy
        dxv[2] = ((int(i / 9) % 3) - 1) * box[5]  # dz
        neighbors[i, :] = box + dxv
    return np.vstack((neighbors[:13], neighbors[14:]))


def get_box_center(box: ArrayLike) -> ArrayLike:
    """
    Return the coordinates of the center of the box
    """
    box = _make_valid(box)
    return box[:3] + box[3:] / 2


def get_box_vertex(box: ArrayLike, index: int, *, jitter: float = 0) -> ArrayLike:
    """
    Return the coordinates of the vertex at z-order index (1-based)

    Note that a jitter can be applied. If so the coordinates will be the
    vertex of the box slightly (1%) smaller (larger) if jitter is positive
    (negative)
    """
    box = _make_valid(box)
    if not isinstance(index, int):
        raise ValueError("Index must be an int!")
    if index < 1 or index > 8:
        raise ValueError(f"Index {index} is out of bounds")
    if not np.isfinite(jitter):
        raise ValueError(f"Jitter ({jitter}) must be finite!")
    index -= 1
    coord = box[:3].copy()
    for i in range(3):
        offset = (index & (1 << i)) >> i
        coord[i] += box[3 + i] * (offset + (1 - 2 * offset) * np.sign(jitter) / 100)
    return coord


def get_box_vertices(box: ArrayLike, *, jitter: float = 0) -> ArrayLike:
    """
    Return the coordinates of the 8 box vertices in z-order

    Note that a jitter can be applied. If so the coordinates will be the
    vertices of the box slightly (1%) smaller (larger) if jitter is positive
    (negative)
    """
    if not np.isfinite(jitter):
        raise ValueError("Jitter must be a finite value")

    box = _make_valid(box)

    vertices = np.zeros((8, 3))

    jitter_amount = np.sign(jitter) * box[3:] / 100

    for k in range(2):
        for j in range(2):
            for i in range(2):
                ind = i + 2 * j + 4 * k
                vertices[ind, :] = [
                    box[0] + i * box[3] + jitter_amount[0] * (-1) ** i,
                    box[1] + j * box[4] + jitter_amount[1] * (-1) ** j,
                    box[2] + k * box[5] + jitter_amount[2] * (-1) ** k,
                ]

    return vertices


def project_point_on_box(
    box: ArrayLike, xyz: ArrayLike, *, jitter: float = 0
) -> np.ndarray:
    """
    Return coordinates of projection of (x, y, z) on nearest box face.

    This is the closest point on the box to (x, y, z). Can provide jitter to
    place point into/out of the box for determining sub-boxes.

    Note: There is no checking for whether points are already inside box

    Note: Jitter is not applied to points inside the box (so a point cannot
        be jittered *out*).

    Inputs:
        box: ArrayLike
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
    box = _make_valid(box)

    if np.any(np.isnan(xyz)):
        raise ValueError("Point contains NaN!")

    if np.isnan(jitter):
        raise ValueError("Jitter must be a number!")

    if jitter < 0:
        raise NotImplementedError()

    jitter = np.sign(jitter) * box[3:] / 100
    clamped_xyz = np.clip(
        xyz, a_min=box[:3] + jitter, a_max=box[:3] + box[3:] - jitter
    ).astype(float)

    return clamped_xyz
