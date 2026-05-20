"""
4x2 rack container: each rack is a 2x4 BottleRack grid cell.
"""

from __future__ import annotations

import numpy as np

from bottle_rack import DEFAULT_DX, DEFAULT_DY, BottleRack

DEFAULT_MX = 0.065  # m, rack (0,0) hole (0,0) in container frame
DEFAULT_MY = 0.045
DEFAULT_TX = 0.085  # m, rack spacing along container +x
DEFAULT_TY = 0.128  # m, rack spacing along container +y


class RackContainer:
    """4 racks along +x (tx), 2 along +y (ty). Origin is the container frame."""

    N_RACK_ROWS = 4
    N_RACK_COLS = 2
    N_RACKS = N_RACK_ROWS * N_RACK_COLS

    def __init__(
        self,
        mx: float = DEFAULT_MX,
        my: float = DEFAULT_MY,
        tx: float = DEFAULT_TX,
        ty: float = DEFAULT_TY,
        dx: float = DEFAULT_DX,
        dy: float = DEFAULT_DY,
        pose: np.ndarray | None = None,
    ) -> None:
        """
        Parameters
        ----------
        mx, my : float
            Position of rack (0, 0) hole (0, 0) in the container frame.
        tx, ty : float
            Spacing between rack (i, j) and (i + 1, j) / (i, j + 1) in container frame.
        dx, dy : float
            Hole spacing within each BottleRack (passed to BottleRack).
        pose : (4, 4), optional
            Container pose in world frame.
        """
        self._mx = float(mx)
        self._my = float(my)
        self._tx = float(tx)
        self._ty = float(ty)
        self._rack = BottleRack(dx, dy)
        self._pose = (
            np.eye(4, dtype=np.float64)
            if pose is None
            else BottleRack._validate_pose(pose).copy()
        )
        self._hole_positions_container = self._build_hole_positions_container()

    @staticmethod
    def _rack_index_to_rc(index: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(index, tuple):
            row, col = index
        else:
            if not 0 <= index < RackContainer.N_RACKS:
                raise IndexError(
                    f"rack index {index} out of range [0, {RackContainer.N_RACKS})"
                )
            row, col = divmod(index, RackContainer.N_RACK_COLS)
        if not (
            0 <= row < RackContainer.N_RACK_ROWS
            and 0 <= col < RackContainer.N_RACK_COLS
        ):
            raise IndexError(
                f"rack ({row}, {col}) out of range for {RackContainer.N_RACK_ROWS}x"
                f"{RackContainer.N_RACK_COLS} container"
            )
        return row, col

    def _rack_origin_in_container(self, row: int, col: int) -> np.ndarray:
        return np.array(
            [self._mx + row * self._tx, self._my + col * self._ty, 0.0],
            dtype=np.float64,
        )

    @property
    def n_holes_total(self) -> int:
        return self.N_RACKS * BottleRack.N_HOLES

    def _build_hole_positions_container(self) -> np.ndarray:
        """All hole centers in container frame, shape (N, 3). Vectorized."""
        n = self.n_holes_total
        idx = np.arange(n, dtype=np.int32)
        rack_idx = idx // BottleRack.N_HOLES
        hole_idx = idx % BottleRack.N_HOLES
        r_row, r_col = np.divmod(rack_idx, self.N_RACK_COLS)
        h_row, h_col = np.divmod(hole_idx, BottleRack.N_COLS)
        x = self._mx + r_row * self._tx + h_row * self._rack._dx
        y = self._my + r_col * self._ty + h_col * self._rack._dy
        z = np.zeros(n, dtype=np.float64)
        return np.stack([x, y, z], axis=1)

    def all_hole_positions_in_container(self) -> np.ndarray:
        """(N, 3) hole centers in container frame."""
        return self._hole_positions_container.copy()

    def all_hole_positions_in_camera(
        self, R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        (N, 3) hole centers in camera frame.

        R, t : board/container pose from solvePnP (cam_X = R @ container_X + t).
        """
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        return (R @ self._hole_positions_container.T).T + t

    def all_hole_transforms(self, T_container: np.ndarray) -> np.ndarray:
        """
        (N, 4, 4) homogeneous transforms for every hole in world/camera frame.

        T_container : 4x4 pose of container frame (same as set_pose input).
        """
        T = BottleRack._validate_pose(T_container)
        R = T[:3, :3]
        local = np.tile(np.eye(4, dtype=np.float64), (self.n_holes_total, 1, 1))
        local[:, :3, 3] = self._hole_positions_container
        return np.einsum("ab,nbc->nac", T, local)

    def _rack_transform_in_container(self, row: int, col: int) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = self._rack_origin_in_container(row, col)
        return T

    def set_pose(self, pose: np.ndarray) -> None:
        """Set container pose (4x4) in world frame."""
        self._pose = BottleRack._validate_pose(pose).copy()

    def get_container_pose(self) -> np.ndarray:
        """Container pose in world frame."""
        return self._pose.copy()

    def get_pose(
        self,
        rack_index: int | tuple[int, int],
        hole_index: int | tuple[int, int],
    ) -> np.ndarray:
        """
        World pose of a hole in a given rack.

        Parameters
        ----------
        rack_index : int or (row, col)
            Rack in the 4x2 grid (row along tx, col along ty).
        hole_index : int or (row, col)
            Hole in the 2x4 BottleRack (same convention as BottleRack).
        """
        r_row, r_col = self._rack_index_to_rc(rack_index)
        T_rack = self._rack_transform_in_container(r_row, r_col)
        T_hole = self._rack.hole_transform(hole_index)
        return self._pose @ T_rack @ T_hole
