import logging
from importlib.metadata import PackageNotFoundError, version

from packingcubes.cubes import (
    Cubes as Cubes,
)
from packingcubes.cubes import (
    MultiCubes as MultiCubes,
)
from packingcubes.cubes import (
    ParticleCubes as ParticleCubes,
)
from packingcubes.cubes import make_cubes as make_cubes
from packingcubes.cubes import (
    make_ParticleCubes as make_ParticleCubes,
)
from packingcubes.data_objects import (
    GadgetishHDF5Dataset as GadgetishHDF5Dataset,
)
from packingcubes.data_objects import (
    InMemory as InMemory,
)
from packingcubes.packed_tree import KDTree as KDTree
from packingcubes.packed_tree import PackedTree as PackedTree

try:
    __version__ = version("packingcubes")
except PackageNotFoundError:
    __version__ = "Not Found"

__all__ = [
    "PackedTree",
    "GadgetishHDF5Dataset",
    "InMemory",
    "KDTree",
    "make_cubes",
    "Cubes",
    "ParticleCubes",
    "MultiCubes",
    "make_ParticleCubes",
    "__version__",
]


logging.getLogger("packingcubes")
