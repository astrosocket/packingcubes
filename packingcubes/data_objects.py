"""
Particle Datasets

Classes
-------
GadgetishHDF5Dataset
    Use to load HDF5 Datasets that look like Gadget-2 snapshots

InMemory
    Use to convert an in-memory array into a dataset

HDF5Dataset
    Abstract dataset, use as a base for custom subclassing for loading HDF5 data
    (see e.g. [GadgetishHDF5Dataset][GadgetishHDF5Dataset])

Dataset
    Generic dataset class, use for typing

MultiParticleDataset
    Generic dataset class with multiple particle types, use for typing

DataContainer
    Effectively a view of a Dataset, only use within `jit`ted code


"""

from __future__ import annotations

import contextlib
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import h5py  # type: ignore
import numpy as np
from numba import TypingError, boolean, float64, int64, njit  # type: ignore
from numba.experimental import jitclass
from numba.extending import as_numba_type
from numpy.typing import NDArray

import packingcubes.bounding_box as bbox

LOGGER = logging.getLogger(__name__)
logging.captureWarnings(capture=True)


class DatasetError(Exception):
    """Error when loading or manipulating data in Datasets"""

    pass


class DatasetWarning(UserWarning):
    """Warning when loading or manipulating data in Datasets"""

    pass


@jitclass(
    [
        ("_positions", float64[:, :]),
        ("_index", int64[:]),
        ("_box", bbox.bbn_type),
        ("_index_dirty", boolean),
    ]
)
class DataContainer:
    """Jitclass for particle position information

    This class is intended for use in jitted Numba code and is effectively a
    view on a [Dataset][Dataset]'s postion and index data.

    See Also
    --------
    [Dataset][Dataset], [Dataset.data_container][Dataset.data_container]
    """

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
        """Return the particle position data"""
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
        """Return the shuffle list, creating if necessary"""
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
        """Return the number of particles in the DataContainer"""
        return len(self._positions)

    @property
    def bounding_box(self) -> bbox.BoundingBox:
        """Return the BoundingBox of the DataContainer"""
        return self._box.copy()


try:
    dc_type = as_numba_type(DataContainer)
except TypingError:
    dc_type = type(DataContainer)


@njit
def subview(data: DataContainer, start_index: int, end_index: int) -> DataContainer:
    """Return a view of the DataContainer from start_index to end_index"""
    return DataContainer(
        data._positions[start_index:end_index],
        data._index[start_index:end_index],
        data._box,
    )


class Dataset:
    """
    Base class for holding particle position data and associated shuffle list

    This class is intended to be the primary interface for octree access to the
    position data. Essentially, it abstracts where the data is and what it
    looks like so the octrees only care about the position data (and more
    specifically, its order)
    """

    name: str
    """A name for this dataset (can be empty)"""
    filepath: Path
    """The path to this dataset (can be empty)"""
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
        """Return the particle position data"""
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
        """Return the shuffle list, creating if necessary"""
        if not hasattr(self, "_index"):
            self._setup_index()
        return self._index

    def reorder(self, new_order):
        """Impose a new order on the position data and shuffle list"""
        self._positions = self._positions[new_order, :]
        self._index = self._index[new_order]

    def _swap(self, first: int, second: int) -> None:
        temp = self._positions[first, :].copy()
        self._positions[first, :] = self._positions[second, :]
        self._positions[second, :] = temp
        temp = self._index[first]
        self._index[first] = self._index[second]
        self._index[second] = temp
        self._index_dirty = True

    def __len__(self) -> int:
        """Return the number of particles in the dataset"""
        return len(self._positions)

    def __repr__(self) -> str:
        """Return a string representation of this dataset"""
        return f"Dataset with {len(self)} particles and box {self.bounding_box}"

    @property
    def bounding_box(self) -> bbox.BoundingBox:
        """Return a copy of the bounding box for this dataset"""
        return self._box.copy()

    def _set_bounding_box(self):
        """Compute bounding box from data"""
        if len(self.positions):
            # sadly no numpy extrema function...
            extremes = np.array(
                [np.min(self.positions, axis=0), np.max(self.positions, axis=0)]
            )
            box = np.zeros(6)
            box[:3] = extremes[0, :] - np.abs(extremes[0, :]) * np.finfo(np.float32).eps
            box[3:] = extremes[1, :] - extremes[0, :]
            box[3:] += (
                np.maximum(box[3:], np.maximum(np.abs(box[:3]), 1))
                * 2
                * np.finfo(np.float32).eps
            )
        else:
            box = np.array([0, 0, 0, 1, 1, 1])
        self._box = bbox.make_bounding_box(box)

    @property
    def data_container(self) -> DataContainer:
        """Return the DataContainer wrapping this dataset"""
        if not hasattr(self, "_data"):
            self._data = DataContainer(self._positions, self._index, self._box)
        return self._data


class MultiParticleDataset(Dataset, ABC):
    """
    Dataset containing multiple particle types

    Multiple particle types are handled by exposing only one particle type at a
    time.
    """

    @property
    @abstractmethod
    def particle_type(self) -> str:
        """Currently selected particle type"""
        ...

    @particle_type.setter
    @abstractmethod
    def particle_type(self, new_type: str): ...
    @property
    @abstractmethod
    def particle_types(self) -> list[str]:
        """List of particle types in this dataset"""
        ...

    @property
    @abstractmethod
    def particle_numbers(self) -> dict[str, int]:
        """Number of particles of each type"""
        ...

    @abstractmethod
    def save(
        self,
        *,
        output_file: str | Path | None = None,
        force_overwrite: bool | None = None,
        particle_type: str | None = None,
    ):
        """Save this dataset to disk

        It will be up to the subclass to decide what that means

        Parameters
        ----------
        output_file: str | Path, optional
            The name of the output file. Note this field is optional because
            there might be an obvious default.

        force_overwrite: bool, optional
            Force overwriting position and index data if the output file
            already contains it under the specified particle type

        particle_type: str, optional
            Save positions under a different particle type than
            `self.particle_type`
        """
        ...


class InMemory(MultiParticleDataset):
    """In-memory Dataset

    Class for datasets where the positions data is entirely in-memory. These
    datasets generally are not expected to have a name or filepath and may
    consist solely of positions data.
    """

    _particle_type: str

    def __init__(
        self,
        *,
        positions: NDArray,
        name: str = "",
        filepath: str = "",
        particle_type: str | None = None,
    ):
        """
        Initialize an InMemory Dataset

        Parameters
        ----------
        positions: NxM NDArray
            Array containing particle position data.

        particle_type: str, optional
            Particle type these positions belong to. Default is `"PartTypeIM"`

        """
        positions = np.atleast_2d(positions)
        if positions.shape[1] != 3 or len(positions.shape) > 2:
            raise ValueError(
                "Only Nx3 arrays are allowed. "
                + (
                    f"You provided an {positions.shape[0]}x{positions.shape[1]} array."
                    if len(positions.shape) == 2
                    else "You provided an "
                    + "x".join(f"{i}" for i in positions.shape)
                    + " array."
                )
            )
        self._positions = positions.astype(np.float64, copy=False)
        super().__init__(name=name, filepath=filepath)
        self._set_bounding_box()
        self._setup_index()
        self._particle_type = "PartTypeIM" if particle_type is None else particle_type

    @property
    def particle_type(self) -> str:
        """Currently selected particle type"""
        return self._particle_type

    @particle_type.setter
    def particle_type(self, new_type: str):
        self._particle_type = new_type

    @property
    def particle_types(self) -> list[str]:
        """List of particle types in this dataset"""
        return [self._particle_type]

    @property
    def particle_numbers(self) -> dict[str, int]:
        """Number of particles of each type"""
        return {self._particle_type: len(self)}

    def save(
        self,
        *,
        output_file: str | Path | None = None,
        force_overwrite: bool | None = None,
        particle_type: str | None = None,
    ):
        """Save sorted particle data and shuffle-list to disk in an HDF5 file

        Parameters
        ----------
        output_file: str | Path, optional
            The name of the output file. Note this field is optional to match
            the superclass, however, specifying None will raise a ValueError.

        force_overwrite: bool, optional
            Force overwriting position and index data if the output file
            already contains it under the specified particle type

        particle_type: str, optional
            Save positions under a different particle type than
            `self.particle_type`

        Raises
        ------
            ValueError
                If no output_file is specified
        """
        if output_file is None:
            raise ValueError(
                "InMemory datasets have no default output file. Please specify"
                " output_file."
            )

        particle_type = self.particle_type if particle_type is None else particle_type

        with h5py.File(output_file, "a") as output:
            for field, array in zip(
                ["Coordinates", "index"], [self._positions, self._index], strict=True
            ):
                name = f"{particle_type}/{field}"
                if name in output:
                    if force_overwrite:
                        del output[name]
                    else:
                        raise DatasetError(
                            f"{output_file} already contains {particle_type} "
                            "and force_overwrite=False"
                        )
                output[name] = array
            output[particle_type].attrs["use_sorted"] = True


def _load_data_from_slices(dataset: h5py.Dataset, slices: list | None) -> NDArray:
    """Load dataset possibly by slices

    Parameters
    ----------
    dataset: h5py Dataset
        The dataset to load

    slices: list of slices | None
        The list of slices into the dataset

    Returns
    -------
    data: NDArray
        The (possibly) sliced dataset
    """
    if slices is None:
        return dataset[:, :]

    width = dataset.shape[1]
    number = sum(s.stop - s.start for s in slices)
    data = np.empty((number, width), dtype=dataset.dtype)
    offset = 0
    for s in slices:
        size = s.stop - s.start
        data[offset : offset + size] = dataset[s]
        offset += size

    return data


class HDF5Dataset(MultiParticleDataset):
    """HDF5 Dataset

    Base class for using HDF5 datasets. We will assume the entire **positions**
    array can be loaded into memory. We do **not** need to be able to load the
    entire dataset since this is for purely spatial sorting.

    Note that for simplicity, only one particle type is available at a time.
    You can use the particle_type and particle_types attributes to change
    particle type and get a list of valid particle types.
    """

    _positions_field: str
    """ Name of the positions dataset in the hdf5 file (e.g. "Coordinates") """
    _particle_types: list[str]
    """ List of all particle types in the hdf5 file """
    _particle_numbers: dict[str, int]
    """ Per particle type mapping of total number of particles """
    _top_level_groups: list[str]
    """ 
    The top level groups in the file in lowercase (e.g. header, parttype0, cosmology)
    """
    _sorted_file_name: Path
    """ Name of the sorted positions file """
    _prefer_sorted: bool
    """ Whether to load sorted versions of positions if possible """
    _particle_type: str
    """ The currently loaded particle type """
    _data_slices: dict

    def __init__(
        self,
        *,
        name: str | None = None,
        filepath: str | Path,
        sorted_filepath: str | Path | None = None,
        data_slices=None,
    ):
        """Initialize an HDF5Dataset

        Parameters
        ----------
        sorted_filepath: str | Path, optional
            Optional file to store sorted position and shuffle-list data

        data_slices: np.s_ | dict[str, np.s_], optional
            A numpy slice object or dictionary of slice objects per particle
            type. This can be used to load only a portion of the dataset.
            Effectively, the dataset will be loaded as
            `data = data[data_slice[0]:data_slice[1]:data_slice[3]]`
        """
        super().__init__(name=name, filepath=filepath)

        self._preload(sorted_filepath)

        self._process_slices(data_slices)

        self._check_loading_strategy()
        self._load_positions()
        self._set_bounding_box()

    def _preload(self, sorted_filepath: str | Path | None):
        """Load certain attributes at initialization

        Must set _positions_field, _particle_types, _particle_numbers,
        _top_level_groups, _sorted_file_name and _particle_type
        """
        raise NotImplementedError(
            "You are trying to instantiate a base HDF5 class.\nUse a subclass instead.",
        )

    def _process_slices(self, data_slice):
        data_slices = {}
        for pt in self._particle_types:
            if isinstance(data_slice, dict):
                data_slices[pt] = data_slice.get(pt, None)
            elif not data_slice:
                data_slices[pt] = None
            else:
                try:
                    len(data_slice)
                    data_slices[pt] = data_slice
                except TypeError:
                    data_slices[pt] = [data_slice]

        self._data_slices = data_slices

    def _check_loading_strategy(self):
        if not Path(self._sorted_file_name).is_file():
            self._prefer_cache = False
            return
        with h5py.File(self._sorted_file_name) as file, contextlib.suppress(KeyError):
            self._prefer_cache = (
                file[self._particle_type].attrs["use_sorted"]
                and self._particle_type + "/positions" in file
            )
            return
        self._prefer_cache = False

    @property
    def particle_type(self):
        """Current particle type"""
        return self._particle_type

    @particle_type.setter
    def particle_type(self, new_type):
        if new_type not in self._particle_types:
            raise DatasetError(f"{new_type} is not a valid particle_type")
        if self._particle_type == new_type:
            # do nothing
            return
        self._particle_type = new_type
        self._check_loading_strategy()
        self._load_positions()

    @property
    def particle_types(self):
        """List of particle types in this dataset"""
        return self._particle_types

    @property
    def particle_numbers(self):
        """Map of particle types to numbers in this dataset"""
        return self._particle_numbers

    def _load_positions(self):
        """Load particle positions from file for the current particle type"""
        filepath = self._sorted_file_name if self._prefer_cache else self.filepath
        with h5py.File(filepath, "r") as file:
            if self._prefer_cache:
                pt = self._particle_type + "/"
                self._positions = _load_data_from_slices(
                    file[pt + self._positions_field],
                    self._data_slices[self._particle_type],
                )
                self._index = file[pt + "index"]
            else:
                positions = _load_data_from_slices(
                    file[self._particle_type][self._positions_field],
                    self._data_slices[self._particle_type],
                )
                self._positions = np.array(positions, dtype=np.float64)
                with contextlib.suppress(AttributeError):
                    del self._index
                self._setup_index()

        if hasattr(self, "_data"):
            del self._data

    def save(
        self,
        *,
        output_file: str | Path | None = None,
        force_overwrite: bool | None = None,
        particle_type: str | None = None,
    ):
        """Save sorted particle positions and shuffle list to provided file

        Parameters
        ----------
        output_file: str | Path |None, optional
            File to save information to. Default is `self.sorted_filepath`

        force_overwrite: bool, optional
            Force overwriting position and index data if the output file
            already contains it under the specified particle type

        particle_type: str, optional
            Save positions under a different particle type than
            `self.particle_type`
        """
        output_file = self._sorted_file_name if output_file is None else output_file

        particle_type = self.particle_type if particle_type is None else particle_type

        with h5py.File(output_file, "a") as output:
            for field, array in zip(
                [self._positions_field, "index"],
                [self._positions, self._index],
                strict=True,
            ):
                name = f"{particle_type}/{field}"
                if name in output and force_overwrite:
                    del output[name]
                else:
                    raise DatasetError(
                        f"{output_file} already contains {particle_type} "
                        "and force_overwrite=False"
                    )
                output[name] = array
            output[particle_type].attrs["use_sorted"] = True


class GadgetishHDF5Dataset(HDF5Dataset):
    """HDF5 dataset with Gadget-2 like header

    Represents an HDF5 dataset that at least has the fields from the Gadget-2
    header specification [here](https://wwwmpa.mpa-garching.mpg.de/gadget/html/structio__header.html)

    """

    def _preload(self, sorted_filepath: str | Path | None):
        # TODO handle case where particles are split across multiple files...
        particle_types = []
        groups = []
        self._positions_field = "Coordinates"
        with h5py.File(self.filepath) as file:
            self._header = dict(file["Header"].attrs)
            groups.extend(file.keys())
            particle_types.extend([p for p in groups if "Part" in p])
        if not groups:
            raise DatasetError("This dataset appears to be empty")
        self._top_level_groups = [g.lower() for g in groups]
        if not particle_types:
            raise DatasetError(
                "No particle types found in dataset. Looking for groups named Part*",
            )
        self._particle_types = particle_types
        self._sorted_file_name = Path(
            self.filepath.parent / ("." + str(self.filepath.name) + "_sorted.hdf5")
            if sorted_filepath is None
            else sorted_filepath
        )
        if self._sorted_file_name.exists() and not h5py.is_hdf5(self._sorted_file_name):
            raise DatasetError(
                f"{self._sorted_file_name} already exists but is not an hdf5 file!",
            )

        # set initial particle type and load data
        self._particle_type = particle_types[0]
        _particle_numbers = self._header["NumPart_Total"]
        _particle_numbers = _particle_numbers[_particle_numbers > 0]
        self._particle_numbers = dict(
            zip(self._particle_types, _particle_numbers, strict=True)
        )

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
