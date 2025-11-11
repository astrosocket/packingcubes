from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

LOGGER = logging.getLogger(__name__)


def in_box(box: ArrayLike, x: float, y: float, z: float) -> bool:
    """
    Check if point is inside box
    """
    return (
        (box[0] <= x <= box[0] + box[3])
        & (box[1] <= y <= box[1] + box[4])
        & (box[2] <= z <= box[2] + box[5])
    )


def midplane(box: ArrayLike) -> ArrayLike:
    """
    Return the 3 coordinates specifying the midplane of the box
    """
    return np.array([box[i] + box[i + 3] / 2 for i in range(3)])


def normalize_to_box(coordinates: ArrayLike, box: ArrayLike) -> ArrayLike:
    """
    Rescale and shift the coordinates such that they are bounded by the unit cube
    """
    # Need to deal with subnormal values
    fixed_subnormal = np.sign(coordinates) * np.clip(
        np.abs(coordinates),
        a_min=np.nextafter(box[3:], box[3:] + 1) - box[3:],
        a_max=None,
    )
    return np.clip((fixed_subnormal - box[:3]) / box[3:], a_min=0.0, a_max=1.0)


def get_neighbor_boxes(box: ArrayLike) -> ArrayLike:
    """
    Return the six boxes that would be the neighbors of this box in a uniform grid

    We are currently not considering any diagonal boxes. This may be something
    to consider in the future.

    Boxes are returned as a 6x6 array, where each row is a box
    """
    neighbors = np.zeros((6, 6))
    for i in range(3):
        for j in range(2):
            neighbor_ind = 2 * i + j
            neighbors[neighbor_ind, :] = box
            neighbors[neighbor_ind, i] += (2 * j - 1) * neighbors[
                neighbor_ind, i + 3
            ]  # x ±= dx
    return neighbors


def project_point_on_box(
    box: ArrayLike, x: float, y: float, z: float, jitter: float = 0
) -> tuple[float]:
    """
    Return coordinates of projection of (x, y, z) on nearest box face.

    This is the closest point on the box to (x, y, z). Can provide jitter to
    place point into/out of the box for determining sub-boxes.

    Inputs:
        box: ArrayLike
        Box to project on

        x, y, z: float
        Point to project onto nearest box face

        jitter: float
        Amount to move projected point into(positive values) or out of
        (negative values) the box. Default is 0

    Returns:
        px, py, pz: float
        Projected coordinates
    """
    raise NotImplementedError()
