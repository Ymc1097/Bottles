#!/usr/bin/env python3
"""
Real-time check: marker board pose on container -> project all rack holes on RGB.

Uses RealSense (data_acquisition), ArUco board pose (marker_board_pose),
vectorized hole layout (rack_container).
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from data_acquisition import RealSenseCamera
from marker_board_pose import estimate_board_poses_in_camera, rvec_tvec_to_Rt
from rack_container import RackContainer

HOLE_RADIUS_M = 0.016
N_CIRCLE_PTS = 32


def project_hole_circles(
    centers_cam: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    radius_m: float,
    n_pts: int = N_CIRCLE_PTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized projection of hole rim circles.

    Returns
    -------
    centers_uv : (N, 2)
    radii_px : (N,)
    rim_uv : (N, n_pts, 2)  NaN where Z <= 0
    """
    N = centers_cam.shape[0]
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False, dtype=np.float64)
    circle = np.stack(
        [radius_m * np.cos(theta), radius_m * np.sin(theta), np.zeros(n_pts)],
        axis=1,
    )  # (n_pts, 3) in container/board xy plane

    rim_cam = centers_cam[:, np.newaxis, :] + circle[np.newaxis, :, :]
    rim_cam = rim_cam.reshape(-1, 3).astype(np.float64)

    valid = centers_cam[:, 2] > 1e-6
    rim_uv, _ = cv2.projectPoints(
        rim_cam,
        np.zeros(3),
        np.zeros(3),
        K,
        D,
    )
    rim_uv = rim_uv.reshape(N, n_pts, 2)

    centers_uv, _ = cv2.projectPoints(
        centers_cam.astype(np.float64),
        np.zeros(3),
        np.zeros(3),
        K,
        D,
    )
    centers_uv = centers_uv.reshape(N, 2)
    fx = float(K[0, 0])
    radii_px = np.where(valid, fx * radius_m / centers_cam[:, 2], np.nan)
    rim_uv[~valid] = np.nan
    return centers_uv, radii_px, rim_uv


def draw_holes(
    image: np.ndarray,
    centers_uv: np.ndarray,
    radii_px: np.ndarray,
    rim_uv: np.ndarray,
) -> np.ndarray:
    vis = image.copy()
    for i in range(centers_uv.shape[0]):
        u, v = centers_uv[i]
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        r = radii_px[i]
        if np.isfinite(r) and r > 0.5:
            cv2.circle(vis, (int(round(u)), int(round(v))), int(round(r)), (0, 255, 0), 1)
        poly = rim_uv[i]
        if np.all(np.isfinite(poly)):
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(
            vis,
            (int(round(u)), int(round(v))),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            6,
            1,
        )
    return vis


def pick_container_pose(
    poses: list,
    location_index: int | None,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    if not poses:
        return None
    if location_index is not None:
        for p in poses:
            if p.location_index == location_index:
                R, t = rvec_tvec_to_Rt(p.rvec, p.tvec)
                return R, t, p.location_index
        return None
    p = poses[0]
    R, t = rvec_tvec_to_Rt(p.rvec, p.tvec)
    return R, t, p.location_index


def run(args: argparse.Namespace) -> None:
    cam = RealSenseCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
        outdir=args.outdir,
    )
    D = np.zeros(8, dtype=np.float64)
    container = RackContainer()
    win = "Hole projection check"

    print(
        f"Holes: {container.n_holes_total} | radius {HOLE_RADIUS_M} m | "
        f"container loc={args.container_loc} | Q/ESC quit"
    )

    try:
        while True:
            color, _ = cam.get_aligned_images()
            if color is None:
                continue

            board_poses = estimate_board_poses_in_camera(color, cam.K, D)
            vis = color.copy()
            status = "no marker"

            loc = None if args.container_loc < 0 else args.container_loc
            picked = pick_container_pose(board_poses, loc)
            if picked is None and board_poses:
                status = f"loc {args.container_loc} not visible ({len(board_poses)} other)"
            elif picked is not None:
                R, t, loc = picked
                centers_cam = container.all_hole_positions_in_camera(R, t)
                centers_uv, radii_px, rim_uv = project_hole_circles(
                    centers_cam, cam.K, D, HOLE_RADIUS_M
                )
                vis = draw_holes(vis, centers_uv, radii_px, rim_uv)
                n_vis = int(np.sum(np.isfinite(centers_uv[:, 0])))
                status = f"loc={loc} holes drawn={n_vis}/{container.n_holes_total}"

            cv2.putText(
                vis,
                status,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                status,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(win, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cam.pipeline.stop()
        cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Project container holes from marker pose")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", type=str, default=None)
    p.add_argument("--outdir", type=str, default="./out")
    p.add_argument(
        "--container-loc",
        type=int,
        default=0,
        help="marker board location_index for container; use -1 for first visible",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
