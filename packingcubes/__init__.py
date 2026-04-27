"""The packingcubes package

`packingcubes` aims to provide a fast, minimal-memory-usage octree
implementation, specialized for use in astronomical/astrophysical contexts.
It's written in pure python, with [Numba](https://numba.pydata.org/)-based
acceleration of the critical code paths.

`packingcubes` can be used from the command line, via the `packcubes`
command, but will generally be used programmatically.

The following classes and methods represent the primary public interface of
`packingcubes`, though we expect most users will rely on only a small
portion.

Classes
-------
ParticleCubes
    Workhorse class. Performs parallel sorting of datasets and `PackedTree`
    creation. Expected to be the primary interface for users.

GadgetishHDF5Dataset
    Use to load HDF5 Datasets that look like Gadget-2 snapshots

InMemory
    Use to convert an in-memory array into a dataset

PackedTree
    Actual Octree implementation.

OpTree
    Use as a drop-in replacement for SciPy's KDTree. Not all functionality is
    implemented, but `query` and `query_ball_point` are. Tree creation and the
    `query_ball_point` implementation should be **significantly** faster than
    SciPy's for larger (`> 10_000`) particle balls, and tree size should be
    substantially smaller.

Methods
-------
Cubes
    Intended ParticleCubes creation method. Can accept snaphsot file paths,
    Dataset objects, and position arrays and returns a ParticleCubes or
    dictionary of ParticleCubes, depending on the number of particle types

make_ParticleCubes
    Effectively `Cubes` but raise an error if more than one particle type is
    present (useful for typing)


"""

import logging
from importlib.metadata import PackageNotFoundError, version

from packingcubes.cubes import (
    Cubes as Cubes,
)
from packingcubes.cubes import (
    ParticleCubes as ParticleCubes,
)
from packingcubes.cubes import make_cubes as make_cubes
from packingcubes.data_objects import (
    GadgetishHDF5Dataset as GadgetishHDF5Dataset,
)
from packingcubes.data_objects import (
    InMemory as InMemory,
)
from packingcubes.packed_tree import OpTree as OpTree
from packingcubes.packed_tree import PackedTree as PackedTree

try:
    __version__ = version("packingcubes")
except PackageNotFoundError:
    __version__ = "Not Found"

__all__ = [
    "PackedTree",
    "GadgetishHDF5Dataset",
    "InMemory",
    "OpTree",
    "make_cubes",
    "Cubes",
    "ParticleCubes",
    "__version__",
]


logging.getLogger("packingcubes")
