import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

LOGGER = logging.getLogger(__name__)


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

        # the following will need to be set by the data loader
        self._box = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    @property
    def positions(self) -> ArrayLike:
        return self._positions

    def _swap(self, first: int, second: int) -> None:
        temp = self._positions[first, :].copy()
        self._positions[first, :] = self._positions[second, :]
        self._positions[second, :] = temp

    def __len__(self) -> int:
        return len(self._positions)

    def __repr__(self) -> str:
        return f"Dataset with {len(self)} particles and box {self.bounding_box}"

    @property
    def bounding_box(self):
        return self._box.copy()
