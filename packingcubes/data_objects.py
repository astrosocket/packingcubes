import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

LOGGER = logging.get_logger(__name__)


class Dataset:
    name: str
    filepath: Path
    _data: Any

    def __init__(
        self,
        *,
        name: str = None,
        filepath: str | Path,
    ) -> None:
        filepath = Path(filepath)
        if name is None:
            name = filepath.name
        self.filepath = filepath
        self.name = name
        self.box = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    @property
    def positions(self) -> ArrayLike:
        return self._data.positions

    def _swap(self, first: int, second: int) -> None:
        LOGGER.debug(f"Swapping {first} and {second}")
        temp = self._data.positions[first, :].copy()
        self._data.positions[first, :] = self._data.positions[second, :]
        self._data.positions[second, :] = temp

    def __len__(self) -> int:
        return len(self._data.positions)

    @property
    def bounding_box(self):
        return self.box
