import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
import os


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, serial=None, outdir=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.outdir = outdir if outdir is not None else './out'

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)

        self._select_device()
        self._start_pipeline()
        self._save_intrinsics()

        self.record = False

    def _select_device(self):
        ctx = rs.context()
        devices = ctx.query_devices()

        print("Connected RealSense devices:")
        for i, dev in enumerate(devices):
            print(f"[{i}] {dev.get_info(rs.camera_info.name)} | "
                  f"Serial: {dev.get_info(rs.camera_info.serial_number)}")

        if len(devices) == 0:
            raise RuntimeError("No RealSense device found")

        if self.serial is None:
            self.serial = devices[0].get_info(rs.camera_info.serial_number)
            print(f"Using default device: {self.serial}")
        else:
            print(f"Using selected device: {self.serial}")

        self.config.enable_device(self.serial)

    def _start_pipeline(self):
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )

        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth scale:", self.depth_scale)

    def _save_intrinsics(self):
        os.makedirs("out", exist_ok=True)

        color_profile = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
        )
        intr = color_profile.get_intrinsics()

        self.K = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        K_path = os.path.join(self.outdir, 'cam_K.txt')
        with open(K_path, "w") as f:
            for row in self.K:
                f.write(" ".join(map(str, row)) + "\n")

        print(f"Saved intrinsics to {K_path}")

    def get_aligned_images(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())

        return color, depth

    def run(self):
        rgb_dir = os.path.join(self.outdir, 'rgb')
        depth_dir = os.path.join(self.outdir, 'depth')
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        print("SPACE: start/stop recording | Q / ESC: quit")

        try:
            while True:
                color, depth = self.get_aligned_images()
                if color is None:
                    continue

                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                vis = np.hstack((color, depth_vis))
                cv2.imshow("Aligned RGB | Depth", vis)

                key = cv2.waitKey(1)

                if key & 0xFF == ord(" "):
                    self.record = not self.record
                    print("Recording:", self.record)
                    time.sleep(0.2)

                if self.record:
                    ts = int(time.time() * 1000)
                    cv2.imwrite(os.path.join(rgb_dir, f"{ts}.png"), color)
                    cv2.imwrite(os.path.join(depth_dir, f"{ts}.png"), depth)

                if key & 0xFF == ord("q") or key == 27:
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


# -------------------------
# Usage
# -------------------------
if __name__ == "__main__":
    outdir = './data/0508_08'
    os.makedirs(os.path.join(outdir, 'all_masks'), exist_ok=True)
    cam = RealSenseCamera(
        width=640,
        height=480,
        serial='420122071863',   # or "420122071863" "347622072186" "339322073657"
        outdir=outdir,
    )
    cam.run()
