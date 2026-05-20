"""
从 p2.py 提炼：仅根据图像与内参求 marker/工位板在相机坐标系下的位姿。

位姿为 OpenCV solvePnP 惯用含义：R,t 将板子/模型坐标系中的点变到相机坐标系（cam_X = R @ obj_X + t）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

# 与 p2 一致：32mm 码半边长 (m)
_DEFAULT_MARKER_HALF_M = 0.016
# 与 p2 中 get_location_configs 的 off 一致 (m)
_DEFAULT_BOARD_OFFSET_M = 0.01865
_DEFAULT_MAX_STATIONS = 6


def _object_points_by_marker_id(
    marker_half_m: float = _DEFAULT_MARKER_HALF_M,
    board_offset_m: float = _DEFAULT_BOARD_OFFSET_M,
    num_stations: int = _DEFAULT_MAX_STATIONS,
) -> dict[int, np.ndarray]:
    """每个 ArUco ID 对应 4 个角点在「该工位板」模型平面 (Z=0) 上的 3D 坐标。"""
    m_half = float(marker_half_m)
    off = float(board_offset_m)
    rel = np.array(
        [
            [-m_half, m_half, 0],
            [m_half, m_half, 0],
            [m_half, -m_half, 0],
            [-m_half, -m_half, 0],
        ],
        dtype=np.float32,
    )
    cfg: dict[int, np.ndarray] = {}
    for loc_idx in range(num_stations):
        base = loc_idx * 4
        cfg[base + 0] = rel + np.array([-off, off, 0], dtype=np.float32)
        cfg[base + 1] = rel + np.array([off, off, 0], dtype=np.float32)
        cfg[base + 3] = rel + np.array([off, -off, 0], dtype=np.float32)
        cfg[base + 2] = rel + np.array([-off, -off, 0], dtype=np.float32)
    return cfg


def _aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant = 7
    return cv2.aruco.ArucoDetector(dictionary, params)


@dataclass
class BoardPoseInCamera:
    """单个可见工位（由同一 loc 下 1~多个 marker 共同解算）在相机系下的位姿。"""

    location_index: int
    rvec: np.ndarray  # (3, 1) float64
    tvec: np.ndarray  # (3, 1) float64，单位与 board 参数一致（p2 为米）


def estimate_board_poses_in_camera(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    marker_half_m: float = _DEFAULT_MARKER_HALF_M,
    board_offset_m: float = _DEFAULT_BOARD_OFFSET_M,
    num_stations: int = _DEFAULT_MAX_STATIONS,
) -> List[BoardPoseInCamera]:
    """
    输入 BGR 或灰度图与相机内参，输出各可见工位板在相机系下的 (rvec, tvec)。

    - 与 p2 相同：同 loc_idx 的多个 marker 角点一起喂给 solvePnP（SQPNP）。
    - 若未检测到受支持的 ID 或 PnP 失败，该工位不返回。

    参数
    ----
    image : 任意可传给 ArUco 的 2/3 通道图
    camera_matrix : 3x3
    dist_coeffs : (N,) 与 calibrate 时维数一致（p2 为 8 维）
    """
    K = np.asarray(camera_matrix, dtype=np.float64)
    D = np.asarray(dist_coeffs, dtype=np.float64)
    board_config = _object_points_by_marker_id(
        marker_half_m=marker_half_m,
        board_offset_m=board_offset_m,
        num_stations=num_stations,
    )

    detector = _aruco_detector()
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or len(ids) == 0:
        return []

    loc_groups: dict[int, dict] = {}
    for i, m_id in enumerate(ids.flatten().tolist()):
        if m_id not in board_config:
            continue
        loc_idx = m_id // 4
        if loc_idx not in loc_groups:
            loc_groups[loc_idx] = {"obj": [], "img": []}
        loc_groups[loc_idx]["obj"].append(board_config[m_id])
        loc_groups[loc_idx]["img"].append(corners[i][0])

    out: List[BoardPoseInCamera] = []
    for loc_idx in sorted(loc_groups.keys()):
        data = loc_groups[loc_idx]
        obj_pts = np.vstack(data["obj"])
        img_pts = np.vstack(data["img"])
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_SQPNP
        )
        if not ok:
            continue
        out.append(
            BoardPoseInCamera(
                location_index=int(loc_idx),
                rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            )
        )
    return out


def rvec_tvec_to_Rt(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """辅助：将旋转向量 + 平移变为 R(3,3) 与 t(3,1)。"""
    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    return R, t
