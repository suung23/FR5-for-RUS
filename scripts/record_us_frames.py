#!/usr/bin/env python3
"""Record ultrasound frames from /us/image to disk.

    python3 scripts/record_us_frames.py              # until Ctrl+C
    python3 scripts/record_us_frames.py -n 500       # stop after 500 frames
    python3 scripts/record_us_frames.py -n 500 --out ~/datasets/us_captures/my_run

Needs the ROS environment (source install/setup.bash) and a running
`us_frame` node. This script only subscribes; it never connects to the
probe, so it does not take the probe's single client slot.

Output folder (default ~/datasets/us_captures/us_<date>_<time>/):
    frames.npy       uint8, (N, H, W) — raw 160 × 512 line data from the C10UR
    png/frame_NNNN.png
    timestamps.csv   index, header_stamp_s, recv_unix_s

Ctrl+C saves what was captured so far.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Recorder(Node):
    def __init__(self, topic: str, limit: int | None):
        super().__init__("us_recorder")
        self.limit = limit
        self.frames: list[np.ndarray] = []
        self.rows: list[tuple[int, float, float]] = []
        self.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)

    def done(self) -> bool:
        return self.limit is not None and len(self.frames) >= self.limit

    def _on_image(self, msg: Image) -> None:
        if self.done():
            return
        if msg.encoding not in ("mono8", "8UC1"):
            self.get_logger().error(f"unexpected encoding {msg.encoding}; skipping")
            return
        buf = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
        i = len(self.frames)
        self.frames.append(buf[:, : msg.width].copy())
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.rows.append((i, stamp, time.time()))
        if (i + 1) % 50 == 0:
            total = f"/{self.limit}" if self.limit else ""
            print(f"  {i + 1}{total} frames", flush=True)


def save(rec: Recorder, out: str) -> None:
    if not rec.frames:
        print("no frames captured — is the us_frame node running?", file=sys.stderr)
        return
    os.makedirs(os.path.join(out, "png"), exist_ok=True)
    stack = np.stack(rec.frames)
    np.save(os.path.join(out, "frames.npy"), stack)
    for i, frame in enumerate(rec.frames):
        PILImage.fromarray(frame).save(os.path.join(out, "png", f"frame_{i:04d}.png"))
    with open(os.path.join(out, "timestamps.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "header_stamp_s", "recv_unix_s"])
        writer.writerows(rec.rows)
    span = rec.rows[-1][2] - rec.rows[0][2]
    rate = (len(stack) - 1) / span if span > 0 else float("nan")
    print(f"saved {len(stack)} frames {stack.shape[1:]} over {span:.1f} s "
          f"({rate:.2f} Hz) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", "--count", type=int, default=None,
                    help="frames to record (default: until Ctrl+C)")
    ap.add_argument("--out", default=None, help="output folder")
    ap.add_argument("--topic", default="/us/image")
    args = ap.parse_args()

    out = os.path.expanduser(args.out or os.path.join(
        "~/datasets/us_captures", time.strftime("us_%Y%m%d_%H%M%S")))
    if os.path.exists(os.path.join(out, "frames.npy")):
        sys.exit(f"{out} already holds a recording — pick another --out")

    rclpy.init()
    rec = Recorder(args.topic, args.count)
    what = f"{args.count} frames" if args.count else "until Ctrl+C"
    print(f"recording {args.topic} ({what}) -> {out}", flush=True)
    t0 = time.time()
    try:
        while rclpy.ok() and not rec.done():
            rclpy.spin_once(rec, timeout_sec=1.0)
            if not rec.frames and time.time() - t0 > 10:
                print("no frames after 10 s — is the us_frame node running?",
                      file=sys.stderr)
                t0 = time.time()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        save(rec, out)
        rec.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
