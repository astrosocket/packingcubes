from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


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
        pass

    def __len__(self) -> int:
        return len(self._data.positions)

    @property
    def bounding_box(self):
        return self.box
