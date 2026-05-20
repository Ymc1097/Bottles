"""
2x4 bottle rack: hole poses from rack origin, column step dx, row step dy.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DX = 0.042  # m, hole spacing along rack +x
DEFAULT_DY = 0.031  # m, hole spacing along rack +y


class BottleRack:
    """2 rows x 4 columns. Origin at hole (0, 0)."""

    N_ROWS = 2
    N_COLS = 4
    N_HOLES = N_ROWS * N_COLS

    def __init__(
        self,
        dx: float = DEFAULT_DX,
        dy: float = DEFAULT_DY,
        pose: np.ndarray | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dx : float
            Spacing along rack +x from hole (i, j) to (i + 1, j).
        dy : float
            Spacing along rack +y from hole (i, j) to (i, j + 1).
        pose : (4, 4), optional
            Initial rack pose (transform of hole (0, 0) in world frame).
        """
        self._dx = float(dx)
        self._dy = float(dy)
        self._pose = (
            np.eye(4, dtype=np.float64)
            if pose is None
            else self._validate_pose(pose).copy()
        )

    @staticmethod
    def _validate_pose(T: np.ndarray) -> np.ndarray:
        arr = np.asarray(T, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError("pose must be shape (4, 4)")
        return arr

    @staticmethod
    def _index_to_rc(index: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(index, tuple):
            row, col = index
        else:
            if not 0 <= index < BottleRack.N_HOLES:
                raise IndexError(
                    f"hole index {index} out of range [0, {BottleRack.N_HOLES})"
                )
            row, col = divmod(index, BottleRack.N_COLS)
        if not (0 <= row < BottleRack.N_ROWS and 0 <= col < BottleRack.N_COLS):
            raise IndexError(f"hole ({row}, {col}) out of range for 2x4 rack")
        return row, col

    def set_pose(self, pose: np.ndarray) -> None:
        """Set rack pose: 4x4 transform of hole (0, 0) in world frame."""
        self._pose = self._validate_pose(pose).copy()

    def get_pose(self) -> np.ndarray:
        """Rack origin pose (hole (0, 0))."""
        return self._pose.copy()

    def _local_translation(self, row: int, col: int) -> np.ndarray:
        return np.array([row * self._dx, col * self._dy, 0.0], dtype=np.float64)

    def hole_transform(self, index: int | tuple[int, int]) -> np.ndarray:
        """4x4 transform of hole index in rack frame (origin at hole (0, 0))."""
        row, col = self._index_to_rc(index)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = self._local_translation(row, col)
        return T

    def get_bottle_pose(self, index: int | tuple[int, int]) -> np.ndarray:
        """
        World pose of the given hole as a 4x4 homogeneous matrix.

        Parameters
        ----------
        index : int or (row, col)
            Flat index 0..7 (row-major: col varies fastest) or (row, col).
        """
        return self._pose @ self.hole_transform(index)
