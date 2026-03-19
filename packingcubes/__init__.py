import logging
from importlib.metadata import PackageNotFoundError, version

from packingcubes.cubes import Cubes as Cubes
from packingcubes.cubes import make_cubes as make_cubes
from packingcubes.data_objects import GadgetishHDF5Dataset as HDF5Dataset
from packingcubes.packed_tree import KDTree as KDTree
from packingcubes.packed_tree import PackedTree as Optree

try:
    __version__ = version("packingcubes")
except PackageNotFoundError:
    __version__ = "Not Found"

__all__ = [
    "Optree",
    "HDF5Dataset",
    "KDTree",
    "make_cubes",
    "Cubes",
    "__version__",
]


logging.getLogger("packingcubes")
