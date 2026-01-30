import contextlib
import logging
import warnings
from pathlib import Path

import h5py  # type: ignore
import numpy as np
from numba import boolean, float32, int64  # type: ignore
from numba.experimental import jitclass
from numba.extending import as_numba_type
from numpy.typing import NDArray

import packingcubes.bounding_box as bbox

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(capture=True)


class DatasetError(Exception):
    pass


class DatasetWarning(UserWarning):
    pass


@jitclass(
    [
        ("_positions", float32[:, :]),
        ("_index", int64[:]),
        ("_box", bbox.bbn_type),
        ("_index_dirty", boolean),
    ]
)
class DataContainer:
    _positions: NDArray
    _index: NDArray
    _box: bbox.BoundingBox
    _index_dirty: bool

    def __init__(self, positions: NDArray, index: NDArray, box: bbox.BoundingBox):
        self._positions = positions
        self._index = index
        self._box = box
        self._index_dirty = False

    @property
    def positions(self) -> NDArray[np.float64]:
        return self._positions

    def _setup_index(self):
        if not hasattr(self, "_positions"):
            raise DatasetError("Dataset has no data!")
        if hasattr(self, "_index"):
            warnings.warn(
                "Dataset already has an index. Overwriting!",
                DatasetWarning,
                stacklevel=2,
            )
        self._index = np.arange(len(self.positions))
        self._index_dirty = False

    @property
    def index(self) -> NDArray[np.int_]:
        if not hasattr(self, "_index"):
            self._setup_index()
        return self._index

    def _swap(self, first: int, second: int) -> None:
        # unrolling loop, to use faster numba code
        temp = self._positions[first, 0]
        self._positions[first, 0] = self._positions[second, 0]
        self._positions[second, 0] = temp
        temp = self._positions[first, 1]
        self._positions[first, 1] = self._positions[second, 1]
        self._positions[second, 1] = temp
        temp = self._positions[first, 2]
        self._positions[first, 2] = self._positions[second, 2]
        self._positions[second, 2] = temp
        temp = self._index[first]
        self._index[first] = self._index[second]
        self._index[second] = temp
        self._index_dirty = True

    def __len__(self) -> int:
        return len(self._positions)

    @property
    def bounding_box(self) -> bbox.BoundingBox:
        return self._box.copy()


dc_type = as_numba_type(DataContainer)


class Dataset:
    name: str
    filepath: Path
    _positions: np.ndarray

    def __init__(
        self,
        *,
        name: str | None = None,
        filepath: str | Path,
    ) -> None:
        filepath = Path(filepath)
        if name is None:
            name = filepath.name
        self.filepath = filepath
        self.name = name

        # the following will need to be set by the data loader
        self._box = bbox.make_bounding_box(np.array([0, 0, 0, 1, 1, 1], dtype=float))

    @property
    def positions(self) -> NDArray:
        return self._positions

    def _setup_index(self):
        if not hasattr(self, "_positions"):
            raise DatasetError("Dataset has no data!")
        if hasattr(self, "_index"):
            warnings.warn(
                "Dataset already has an index. Overwriting!",
                DatasetWarning,
                stacklevel=2,
            )
        self._index = np.arange(len(self.positions))
        self._index_dirty = False

    @property
    def index(self) -> NDArray:
        if not hasattr(self, "_index"):
            self._setup_index()
        return self._index

    def _swap(self, first: int, second: int) -> None:
        temp = self._positions[first, :].copy()
        self._positions[first, :] = self._positions[second, :]
        self._positions[second, :] = temp
        temp = self._index[first]
        self._index[first] = self._index[second]
        self._index[second] = temp
        self._index_dirty = True

    def __len__(self) -> int:
        return len(self._positions)

    def __repr__(self) -> str:
        return f"Dataset with {len(self)} particles and box {self.bounding_box}"

    @property
    def bounding_box(self) -> bbox.BoundingBox:
        return self._box.copy()

    @property
    def data_container(self) -> DataContainer:
        return DataContainer(self._positions.astype(np.float32), self._index, self._box)


class HDF5Dataset(Dataset):
    """
    HDF5 Dataset

    Base class for using HDF5 datasets. We will assume the entire **positions**
    array can be loaded into memory. We do **not** need to be able to load the
    entire dataset since this is for purely spatial sorting.

    Note that for simplicity, only one particle type is available at a time.
    You can use the get_particle_types() and switch_particle_type() methods to
    change particle types.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        filepath: str | Path,
    ):
        super().__init__(name=name, filepath=filepath)

        self._preload()
        self._set_bounding_box()
        self._setup_index()

    def _preload(self):
        raise NotImplementedError(
            "You are trying to instantiate a base HDF5 class.\nUse a subclass instead.",
        )

    def _set_bounding_box(self):
        """
        Compute bounding box from data
        """
        # sadly no numpy extrema function...
        min_bounds = np.min(self.positions, axis=0)
        max_bounds = np.max(self.positions, axis=0)
        self._box = bbox.make_bounding_box(
            np.hstack((min_bounds, max_bounds - min_bounds))
        )

    @property
    def particle_type(self):
        """
        Current particle type
        """
        return self._particle_type

    @particle_type.setter
    def particle_type(self, new_type):
        if new_type not in self._particle_types:
            raise DatasetError(f"{new_type} is not a valid particle_type")
        # save old index if necessary
        if hasattr(self, "_particle_type") and self._index_dirty:
            with h5py.File(self._cache_file_name, "a") as _cache_file:
                if self._particle_type in _cache_file:
                    _cache_file[self._particle_type] = self._index
                else:
                    _cache_file.create_dataset(self._particle_type, data=self._index)
        self._particle_type = new_type
        self._load_positions()

    @property
    def particle_types(self):
        """
        List of particle types in this dataset
        """
        return self._particle_types

    def _load_positions(self):
        """
        Load particle positions from file for the current particle type
        """
        with h5py.File(self.filepath, "r") as file:
            positions = file[self._particle_type][self._positions_field]
            self._positions = np.array(positions)
        with h5py.File(self._cache_file_name, "r") as _cache_file:
            if self._particle_type in _cache_file:
                self._index = np.array(_cache_file[self._particle_type])
                self._index_dirty = False
                self._positions = self._positions[self._index, :]
            else:
                with contextlib.suppress(AttributeError):
                    del self._index
                self._setup_index()


class GadgetishHDF5Dataset(HDF5Dataset):
    """
    HDF5 dataset with Gadget-2 like header

    Represents an HDF5 dataset that at least has the fields from the Gadget-2
    header specification [here](https://wwwmpa.mpa-garching.mpg.de/gadget/html/structio__header.html)

    """

    def _preload(self):
        # TODO handle case where particles are split across multiple files...
        particle_types = []
        self._positions_field = "Coordinates"
        with h5py.File(self.filepath) as file:
            self._header = dict(file["Header"].attrs)
            groups = file.keys()
            particle_types.extend([p for p in groups if "Part" in p])
        if not particle_types:
            raise DatasetError(
                "No particle types found in dataset. Looking for groups named Part*",
            )
        self._particle_types = particle_types
        self._cache_file_name = self.filepath.parent / (
            "." + str(self.filepath.name) + ".cache"
        )
        if self._cache_file_name.exists():
            if not h5py.is_hdf5(self._cache_file_name):
                raise DatasetError(
                    f"{self._cache_file_name} already exists but is not an hdf5 file!",
                )
        else:
            with h5py.File(self._cache_file_name, "w") as file:
                file.create_dataset("Header", dtype=float)

        # set initial particle type and load data
        self.particle_type = particle_types[0]

    def _set_bounding_box(self):
        # Use BoxLen from header if provided
        if "BoxSize" in self._header:
            boxSize = np.atleast_1d(self._header["BoxSize"])
            match len(boxSize):
                case 1:
                    boxSize = boxSize[0]
                    box = np.array([0, 0, 0, boxSize, boxSize, boxSize], dtype=float)
                case 3:
                    box = np.array([0, 0, 0, *boxSize], dtype=float)
                case 6:
                    box = boxSize
                case _:
                    raise DatasetError(
                        "Don't know how to deal with a header "
                        f"BoxSize of {len(boxSize)}!",
                    )
            self._box = bbox.make_bounding_box(box)
        else:
            super()._set_bounding_box()


class TranslatedHDF5Dataset(HDF5Dataset):
    """
    Already translated HDF5 dataset

    Represents an HDF5 dataset that has already been translated by the
    translation function. Thus it is guaranteed to already have certain
    fields.

    """

    def _preload(self):
        raise NotImplementedError(
            "Not yet implemented. Use GadgetishHDF5Dataset for now.",
        )
