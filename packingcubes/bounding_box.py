from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Flag, auto
from typing import TYPE_CHECKING

import numpy as np
from numba import TypingError, float64  # type: ignore
from numba.experimental import jitclass
from numba.extending import as_numba_type
from numpy.typing import ArrayLike, NDArray

LOGGER = logging.getLogger(__name__)


def check_floating_point(
    box: NDArray,
) -> tuple[BoundingBoxValidFlag, str]:
    flag = ~BoundingBoxValidFlag.VALID
    fpmessage = ""
    with np.errstate(all="raise"):
        try:
            if np.any(
                np.isfinite(box[:3])
                & np.isfinite(box[3:])
                & (box[3:] > 0)
                & (
                    box[3:]
                    < (np.maximum(np.abs(box[:3]) + box[3:], 1.0)) * np.finfo(float).eps
                ),
            ):
                flag ^= BoundingBoxValidFlag.PRECISION
        except FloatingPointError as fpe:
            flag ^= BoundingBoxValidFlag.NOFPERROR
            fpmessage = fpe.args[0]
    return flag, fpmessage


class BoundingBoxValidFlag(Flag):
    """
    Validations performed on a box
    """

    IS_BOX = auto(), lambda box: f"Box ({box}, type={type(box)}) is not a box!"
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
    NOFPERROR = (
        auto(),
        lambda box: (
            f"Floating Point Error with box ({box}): {check_floating_point(box)[1]}"
        ),
    )
    # TODO: The following (e.g. IS_BOX[0]) is hacky and I don't like it...
    VALID = (
        IS_BOX[0]
        | CORRECT_SHAPE[0]
        | FINITE[0]
        | POSITIVE[0]
        | PRECISION[0]
        | NOFPERROR[0],
        lambda box: "This box is valid",
    )

    message_fun: Callable

    def __new__(cls, value: int, message_fun: Callable):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.message_fun = message_fun
        return obj

    def get_messages(self, box: BoxLike):
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
        box: BoxLike,
        errortype: BoundingBoxValidFlag,
    ):
        """
        Args:
            box: BoxLike
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


def check_valid(box: BoxLike, *, raise_error: bool = True) -> BoundingBoxValidFlag:
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
    if TYPE_CHECKING:
        if not isinstance(box, np.ndarray):
            raise TypeError("Box is not an NDArray!")
    if len(box) != 6 or box.shape != (6,):
        flag ^= BoundingBoxValidFlag.CORRECT_SHAPE
    else:
        if not np.all(np.isfinite(box)):
            flag ^= BoundingBoxValidFlag.FINITE
        if np.any(box[3:] <= 0):
            flag ^= BoundingBoxValidFlag.POSITIVE
        flag ^= check_floating_point(box)[0]
    if raise_error and flag != BoundingBoxValidFlag.VALID:
        raise BoundingBoxError(box=box, errortype=flag)
    return flag


class BoundingVolume:
    def contains(self, xyz: NDArray) -> NDArray[np.bool_]:
        raise NotImplementedError("Must be called from an implementing class")

    def contains_point(self, x: float, y: float, z: float) -> bool:
        raise NotImplementedError("Must be called from an implementing class")

    def contains_pointlist(self, xyz: NDArray) -> NDArray[np.bool_]:
        raise NotImplementedError("Must be called from an implementing class")

    def count_inside(self, xyz: NDArray) -> int:
        raise NotImplementedError("Must be called from an implementing class")


# KW-only arguments with defaults seem to be just completely broken. Likely
# related to https://github.com/numba/numba/issues/5903
# So the methods with those arguments have been changed to positional. Beware!
@jitclass(
    [
        ("box", float64[:]),
    ]
)
class BoundingBox(BoundingVolume):
    box: np.ndarray

    def __init__(self, box: NDArray):
        self.box = box

    @property
    def x(self):
        return self.box[0]

    @property
    def y(self):
        return self.box[1]

    @property
    def z(self):
        return self.box[2]

    @property
    def dx(self):
        return self.box[3]

    @property
    def dy(self):
        return self.box[4]

    @property
    def dz(self):
        return self.box[5]

    @property
    def position(self):
        return self.box[:3]

    @property
    def size(self):
        return self.box[3:]

    def __str__(self):
        return f"BBN:[{self.x}, {self.y}, {self.z}, {self.dx}, {self.dy}, {self.dz}]"

    def copy(self) -> BoundingBox:
        """
        Return a deep copy of this bounding box
        """
        return BoundingBox(self.box.copy())

    def get_box_center(self) -> NDArray:
        """
        Return the coordinates of the center of the box
        """
        return self.box[:3] + self.box[3:] / 2

    def get_box_vertex(self, index: int, jitter: float = 0) -> NDArray:
        """
        Return the coordinates of the vertex at z-order index (1-based)

        Note that a jitter can be applied. If so the coordinates will be the
        vertex of the box slightly (1%) smaller (larger) if jitter is positive
        (negative)

        Args:
            index: int
            Z-order (0-based) index of vertex

            jitter: float, optional
            Jitter direction. Default 0 (no jitter)

        Returns:
            vertex: numpy.ndarray
            (3,) numpy array corresponding to the specified vertex
        """
        if not isinstance(index, int):
            raise ValueError("Index must be an int!")
        if index < 1 or index > 8:
            raise ValueError(f"Index {index} is out of bounds")
        if not np.isfinite(jitter):
            raise ValueError(f"Jitter ({jitter}) must be finite!")
        index -= 1
        coord = self.box[:3].copy()
        for i in range(3):
            offset = (index & (1 << i)) >> i
            coord[i] += self.box[3 + i] * (
                offset + (1 - 2 * offset) * np.sign(jitter) / 100
            )
        return coord

    def get_box_vertices(self, jitter: float = 0) -> NDArray:
        """
        Return the coordinates of the 8 box vertices in z-order

        Note that a jitter can be applied. If so the coordinates will be the
        vertices of the box slightly (1%) smaller (larger) if jitter is positive
        (negative)

        Args:
            jitter: float, optional
            Jitter direction. Default 0 (no jitter)

        Returns:
            vertices: numpy.ndarray
            (6,3) numpy array corresponding to the box vertices in z-order
        """
        if not np.isfinite(jitter):
            raise ValueError("Jitter must be a finite value")

        vertices = np.zeros((8, 3))

        jitter_amount = np.sign(jitter) * self.box[3:] / 100

        x0 = self.box[0] + jitter_amount[0]
        x1 = self.box[0] + self.box[3] - jitter_amount[0]
        y0 = self.box[1] + jitter_amount[1]
        y1 = self.box[1] + self.box[4] - jitter_amount[1]
        z0 = self.box[2] + jitter_amount[2]
        z1 = self.box[2] + self.box[5] - jitter_amount[2]

        vertices[0, 0] = x0
        vertices[0, 1] = y0
        vertices[0, 2] = z0

        vertices[1, 0] = x1
        vertices[1, 1] = y0
        vertices[1, 2] = z0

        vertices[2, 0] = x0
        vertices[2, 1] = y1
        vertices[2, 2] = z0

        vertices[3, 0] = x1
        vertices[3, 1] = y1
        vertices[3, 2] = z0

        vertices[4, 0] = x0
        vertices[4, 1] = y0
        vertices[4, 2] = z1

        vertices[5, 0] = x1
        vertices[5, 1] = y0
        vertices[5, 2] = z1

        vertices[6, 0] = x0
        vertices[6, 1] = y1
        vertices[6, 2] = z1

        vertices[7, 0] = x1
        vertices[7, 1] = y1
        vertices[7, 2] = z1

        return vertices

    def get_child_box(self, ind: int) -> BoundingBox:
        """
        Get indth (0-indexed) new child box of current box

        New child box is defined as the suboctant described by position ind
        with size (box[3]/2, box[4]/2, box[5]/2). The child box is not currently
        guaranteed to be a proper BoundingBox

        Args:
            ind: int
            Z-order index (0-indexed) of child box (i.e. 0<=ind<=7)

        Returns:
            child_box: BoundingBox
        """
        # Use z-index order for now, but other possibilities
        # like Hilbert curves exist - and see
        # https://math.stackexchange.com/questions/2411867/3d-hilbert-curve-without-double-length-edges
        # for a possible "Hilbert" curve that may be better?
        if not isinstance(ind, int) or ind < 0 or 7 < ind:
            raise ValueError(f"Octree code passed an invalid index: {ind}!")

        child_box = self.box.copy()
        child_box[3:] /= 2.0
        x, y, z = ((ind & 1) / 1, (ind & 2) / 2, (ind & 4) / 4)
        child_box[0] = child_box[0] + child_box[3] * x
        child_box[1] = child_box[1] + child_box[4] * y
        child_box[2] = child_box[2] + child_box[5] * z

        return BoundingBox(child_box)

    def get_neighbor_boxes(self) -> NDArray:
        """
        Return the 26 boxes that would be the neighbors of this box in a uniform grid

        Boxes are returned as a 26x6 array, where each row is a box. Order is
        z-order, so row 0 is the box at [x-dx,y-dy,z-dz], row 2 is [x+dx,y-dy,z-dz]
        and row 25 is [x+dx,y+dy,z+dz]
        """
        # We generate all 27 boxes (so including box) and then remove box because
        # it makes the code logic *much* simpler and shouldn't significantly
        # increase the number of resources used
        neighbors = np.zeros((27, 6), dtype=self.box.dtype)
        dxv = np.zeros(6, dtype=self.box.dtype)
        for i in range(27):
            dxv[0] = ((i % 3) - 1) * self.box[3]  # dx
            dxv[1] = ((int(i / 3) % 3) - 1) * self.box[4]  # dy
            dxv[2] = ((int(i / 9) % 3) - 1) * self.box[5]  # dz
            neighbors[i, :] = self.box + dxv
        return np.vstack((neighbors[:13], neighbors[14:]))

    def contains(self, xyz: NDArray) -> NDArray[np.bool_]:
        """
        Check if points are inside box

        Args:
            xyz: ArrayLike
            Array of points with shape Nx3 to test. (3,) arrays will be converted

        Returns:
            in_box: NDArray[np.bool_]
            Boolean array where True means point is inside box
        """
        xyz = np.atleast_2d(xyz)
        # The following is the original python implementation. numba does not support
        # the optional axis argument, however, so we need to split the call up
        # return np.all(
        #     (self.box[:3] <= xyz) & (xyz <= self.box[:3] + self.box[3:]), axis=1
        # )
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        bx, by, bz = self.box[:3]
        dx, dy, dz = self.box[3:]
        return (
            ((bx <= x) & (x <= bx + dx))
            & ((by <= y) & (y <= by + dy))
            & ((bz <= z) & (z <= bz + dz))
        )

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """
        Check if point is inside box

        Args:
            x, y, z: float
            3D coordinates of the point

        Returns:
            in_box: bool
            True if point is inside box

        See Also:
            contains_pointlist, count_inside
        """
        bx, by, bz = self.box[:3]
        dx, dy, dz = self.box[3:]
        return (
            ((bx <= x) & (x <= bx + dx))
            & ((by <= y) & (y <= by + dy))
            & ((bz <= z) & (z <= bz + dz))
        )

    def contains_pointlist(self, xyz: NDArray) -> NDArray[np.bool_]:
        """
        Check if points are inside box

        Args:
            xyz: NDArray
            Array of points with shape Nx3 to test.

        Returns:
            in_box: NDArray[np.bool_]
            Boolean array where True means point is inside box

        See Also:
            contains_point, count_inside
        """
        assert len(xyz.shape) == 2
        assert xyz.shape[1] == 3
        in_box = np.empty((xyz.shape[0],), dtype=np.bool_)
        bx, by, bz = self.box[:3]
        dx, dy, dz = self.box[3:]
        for i in range(xyz.shape[0]):
            x, y, z = xyz[i, :]
            in_box[i] = (
                ((bx <= x) & (x <= bx + dx))
                & ((by <= y) & (y <= by + dy))
                & ((bz <= z) & (z <= bz + dz))
            )
        return in_box

    def count_inside(self, xyz: NDArray) -> int:
        """
        Return a count of how many points in xyz are inside the box

        Prefer this function to sum(contains_pointlist) or similar to prevent
        unnecessary array creation.

        Args:
            xyz: NDArray
            Array of points with shape Nx3 to test.

        Returns:
            count: int
            The count of points that are inside this box

        See Also:
            contains_point, contains_pointlist
        """
        assert len(xyz.shape) == 2
        assert xyz.shape[1] == 3
        in_box = 0
        bx, by, bz = self.box[:3]
        dx, dy, dz = self.box[3:]
        for i in range(xyz.shape[0]):
            x, y, z = xyz[i, :]
            in_box += (
                ((bx <= x) & (x <= bx + dx))
                & ((by <= y) & (y <= by + dy))
                & ((bz <= z) & (z <= bz + dz))
            )
        return in_box

    def max_depth(self) -> int:
        """
        Get max depth supported by this box

        Max depth is defined as the maximum number of times this box can be split
        in half before x[i] + dx[i] == x[i].
        """
        min_box_sizes = (
            np.maximum(1.0, np.abs(self.box[:3]) + self.box[3:])
            * np.finfo(np.float64).eps
        )
        return np.ceil(np.log2(self.box[3:] / min_box_sizes)).astype(np.int_).min()

    def midplane(self) -> tuple[float, float, float]:
        """
        Return the 3 coordinates specifying the midplane of the box
        """
        # This is equivalent to
        # return self.box[:3] + self.box[3:] / 2
        # but we only need the three terms, not the full
        # array, so return a tuple
        return (
            self.box[0] + self.box[3] / 2,
            self.box[1] + self.box[4] / 2,
            self.box[2] + self.box[5] / 2,
        )

    def normalize_to_box(self, coordinates: NDArray) -> NDArray:
        """
        Rescale and shift the coordinates such that they are bounded by the unit cube
        """
        # Need to deal with subnormal values

        fixed_subnormal = np.sign(coordinates) * np.clip(
            np.abs(coordinates),
            a_min=np.nextafter(self.box[3:], self.box[3:] + 1) - self.box[3:],
            a_max=None,
        )
        return np.clip(
            (fixed_subnormal - self.box[:3]) / self.box[3:],
            a_min=0.0,
            a_max=1.0,
        )

    def project_point_on_box(self, xyz: NDArray, jitter: float = 0) -> NDArray:
        """
        Return coordinates of projection of (x, y, z) on nearest box face.

        This is the closest point on the box to (x, y, z). Can provide jitter to
        place point into/out of the box for determining sub-boxes.

        Note: There is no checking for whether points are already inside box

        Note: Jitter is not applied to points inside the box (so a point cannot
            be jittered *out*).

        Args:
            xyz: ArrayLike
            Point to project onto nearest box face. Expected to be shape (3,)

            jitter: nonnegative float
            Flag to move projected point 1% into the box. Negative values to move
            the point out of the box are not yet supported. Default is 0

        Returns:
            pxyz: numpy.ndarray
            Projected coordinates
        """
        if np.any(np.isnan(xyz)):
            raise ValueError("Point contains NaN!")

        if np.isnan(jitter):
            raise ValueError("Jitter must be a number!")

        if jitter < 0:
            raise NotImplementedError()

        box_pos = self.box[:3]
        box_size = self.box[3:]
        assert len(box_pos) == 3
        assert len(box_size) == 3
        assert len(xyz) == 3
        jitter_arr = np.sign(jitter) * box_size / 100
        assert len(jitter_arr) == 3
        return np.clip(
            xyz,
            a_min=box_pos + jitter_arr,
            a_max=box_pos + box_size - jitter_arr,
        ).astype(np.float64)


try:
    bbn_type = as_numba_type(BoundingBox)
except TypingError:
    bbn_type = type(BoundingBox)

type BoxLike = ArrayLike | BoundingBox


def make_bounding_box(box: BoxLike) -> BoundingBox:
    """
    Convert a Boxlike object into an BoundingBox.

    A valid boxlike object is one that passes the check_valid function.
    (i.e. a numpy array with in the form [x, y, z, dx, dy, dz] with the
    following conditions: finite, all |x[i]| * epsilon < dx[i]).

    Args:
        box: BoxLike
        The boxlike object to convert. Object will attempt to be coerced into
        a valid form (e.g. a (1, 6) int array will be converted to a (6,)
        float64 array)

    Returns:
        bbn: BoundingBox
        A BoundingBox object

    Raises:
        BoundingBoxError if box is not a valid bounding box

    See Also:
        check_valid
    """
    if isinstance(box, (BoundingBox)):
        check_valid(box.box, raise_error=True)
        return BoundingBox(box.box)
    bbox = np.atleast_1d(np.squeeze(np.asanyarray(box))).astype(np.float64)
    check_valid(bbox, raise_error=True)
    return BoundingBox(bbox)


@jitclass([("center", float64[:])])
class BoundingSphere(BoundingVolume):
    center: NDArray
    radius: float

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def contains(self, xyz: NDArray) -> NDArray[np.bool_]:
        """
        Return true if point is closer than (<=) radius to center. Vectorizable
        """
        # We don't need to calculate the actual distance, we only need to compare dist^2
        dist2 = np.sum(np.atleast_2d((xyz - self.center) ** 2), axis=1)
        return dist2 <= self.radius**2

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """
        Check if point is inside sphere

        Args:
            x, y, z: float
            3D coordinates of the point

        Returns:
            in_sph: bool
            True if point is inside sphere

        See Also:
            contains_pointlist, count_inside
        """
        return (self.center[0] - x) ** 2 + (self.center[1] - y) ** 2 + (
            self.center[2] - z
        ) ** 2 < self.radius * self.radius

    def contains_pointlist(self, xyz: NDArray) -> NDArray[np.bool_]:
        """
        Check if points are inside sphere

        Args:
            xyz: NDArray
            Array of points with shape Nx3 to test.

        Returns:
            in_sph: NDArray[np.bool_]
            Boolean array where True means point is inside sphere

        See Also:
            contains_point, count_inside
        """
        assert len(xyz.shape) == 2
        assert xyz.shape[1] == 3
        r2 = self.radius * self.radius
        cx, cy, cz = self.center
        in_sph = np.empty((xyz.shape[0],), dtype=np.bool_)
        for i in range(xyz.shape[0]):
            x, y, z = xyz[i, :]
            in_sph[i] = (cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2 < r2
        return in_sph

    def count_inside(self, xyz: NDArray) -> int:
        """
        Return a count of how many points in xyz are inside the sphere

        Prefer this function to sum(contains_pointlist) or similar to prevent
        unnecessary array creation.

        Args:
            xyz: NDArray
            Array of points with shape Nx3 to test.

        Returns:
            count: int
            The count of points that are inside this sphere

        See Also:
            contains_point, contains_pointlist
        """
        assert len(xyz.shape) == 2
        assert xyz.shape[1] == 3
        in_sph = 0
        r2 = self.radius * self.radius
        cx, cy, cz = self.center
        for i in range(xyz.shape[0]):
            x, y, z = xyz[i, :]
            in_sph += (cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2 < r2
        return in_sph

    @property
    def bounding_box(self) -> BoundingBox:
        center = self.center
        radius = self.radius
        return BoundingBox(
            np.array(
                [
                    center[0] - radius,
                    center[1] - radius,
                    center[2] - radius,
                    2 * radius,
                    2 * radius,
                    2 * radius,
                ],
            )
        )


try:
    bs_type = as_numba_type(BoundingSphere)
except TypingError:
    bs_type = type(BoundingSphere)


def make_bounding_sphere(
    radius: float, *, center: ArrayLike | None = None
) -> BoundingSphere:
    """
    Convert a spherelike object into an BoundingBox

    Args:
        radius: float
        Radius of the sphere

        center: ArrayLike | None
        Center coordinates of the sphere. [0,0,0] if not provided

    Returns:
        sph: BoundingSphere
        A BoundingSphere object

    Raises:
        BoundingBoxError if the sphere's bounding box is not a valid bounding
        box, e.g. the radius is too small.
    """
    if center is None:
        center = [0, 0, 0]

    center = np.atleast_1d(center).astype(np.float64)

    if len(center) != 3:
        raise ValueError("Center should be a 3 element array")

    # sphere bounding box
    bounding_box = make_bounding_box(
        np.array(
            [
                center[0] - radius,
                center[1] - radius,
                center[2] - radius,
                2 * radius,
                2 * radius,
                2 * radius,
            ],
        ),
    )

    return BoundingSphere(center, float(radius))
