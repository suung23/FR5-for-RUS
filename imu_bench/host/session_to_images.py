#!/usr/bin/env python3
"""세션의 초음파 프레임을 PNG 로 꺼내 본다 — 부채꼴(B-mode) 또는 극좌표 원본, 그리고 한 장짜리 요약 시트.

    python imu_bench\\host\\session_to_images.py <session_dir>                  # 요약 시트 1 장 (프레임 12 장 + IMU)
    python imu_bench\\host\\session_to_images.py <session_dir> --every 10       # 10 프레임마다 PNG 로 (fan)
    python imu_bench\\host\\session_to_images.py <session_dir> --every 1 --polar --out D:\\dump
    python imu_bench\\host\\session_to_images.py <session_dir> --frame 512      # 프레임 하나만

기본 출력 폴더: <session_dir>\\images\\  (fan_000123.png, polar_000123.png, sheet.png)
부채꼴 변환은 세션 메타의 us.fan_geometry (GUI 가 기록) 를 쓴다. 저장 원본은 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, "..", "..", "policy_learning"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load(session_dir: Path, shape: tuple[int, int] | None = None):
    meta = json.loads((session_dir / "session.meta.json").read_text(encoding="utf-8"))
    meta_shape = tuple(meta["us"].get("frame_shape") or (256, 256))
    if shape is None:
        shape = meta_shape
    elif shape[0] * shape[1] != meta_shape[0] * meta_shape[1]:
        raise SystemExit("--shape %s 는 메타 frame_shape %s 와 바이트 수가 다릅니다" % (shape, meta_shape))
    frames = np.fromfile(str(session_dir / "us_frames.bin"), np.uint8).reshape(-1, *shape)
    return meta, frames


def make_converter(meta, shape, fan: bool):
    if not fan:
        return lambda f: f
    from us_scan_convert import FanGeometry, ScanConverter

    g = (meta.get("us") or {}).get("fan_geometry") or {}
    geo = FanGeometry(radius_mm=float(g.get("radius_mm", 59.0)), half_angle_deg=float(g.get("half_angle_deg", 28.0)),
                      depth_mm=float(g.get("depth_mm", 220.0)), flip_lines=bool(g.get("flip_lines", False)))
    conv = ScanConverter(geo, shape[0], shape[1], out_h=512)
    return conv.convert


def contact_sheet(session_dir: Path, meta, frames, conv, out: Path, tag: str, n_tiles: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.linspace(0, len(frames) - 1, n_tiles).astype(int)
    cols = 6
    rows = int(np.ceil(n_tiles / cols))
    fig = plt.figure(figsize=(3.0 * cols, 2.6 * rows + 3.2))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [1.2])
    t0 = None
    idx_csv = session_dir / "us_index.csv"
    t = np.loadtxt(idx_csv, delimiter=",", skiprows=1, usecols=0) if idx_csv.is_file() else np.arange(len(frames)) / 10.0
    t0 = t[0]
    for k, i in enumerate(idx):
        ax = fig.add_subplot(gs[k // cols, k % cols])
        ax.imshow(conv(frames[i]), cmap="gray", vmin=0, vmax=255)
        ax.set_title("#%d  t=%.1fs" % (i, t[i] - t0), fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(gs[rows, :])
    sync = session_dir / "sync.npz"
    if sync.is_file():
        z = np.load(sync)
        tt = z["frame_t_acq"] - z["frame_t_acq"][0]
        ax.plot(tt, z["acc"], lw=0.7, label=["ax", "ay", "az"])
        ax2 = ax.twinx()
        ax2.plot(tt, z["gyr"], lw=0.5, ls="--")
        ax2.set_ylabel("gyro rad/s (dashed)")
        if "still" in z:
            ax.fill_between(tt, *ax.get_ylim(), where=z["still"], color="g", alpha=0.12)
        ax.set_ylabel("accel m/s²"); ax.set_xlabel("t [s]  (green = still)")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "sync.npz 없음", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("%s — %d frames %s, %s" % (session_dir.name, len(frames), "x".join(map(str, frames.shape[1:])), tag), fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--every", type=int, default=0, help="N 프레임마다 PNG (0 = 안 함, 요약 시트만)")
    ap.add_argument("--frame", type=int, default=None, help="이 프레임 하나만 PNG")
    ap.add_argument("--polar", action="store_true", help="극좌표 원본으로 저장 (기본 fan)")
    ap.add_argument("--no-sheet", action="store_true")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본 <session>/images)")
    ap.add_argument("--shape", type=int, nargs=2, metavar=("LINES", "SAMPLES"), default=None,
                    help="저장 바이트를 이 (라인, 깊이 표본) 으로 재해석 (기본 메타 frame_shape)")
    args = ap.parse_args()

    from PIL import Image

    sd = Path(args.session)
    meta, frames = load(sd, tuple(args.shape) if args.shape else None)
    out = Path(args.out) if args.out else sd / "images"
    out.mkdir(parents=True, exist_ok=True)
    fan = not args.polar
    conv = make_converter(meta, frames.shape[1:], fan)
    tag = "fan" if fan else "polar"
    n = 0
    if args.frame is not None:
        Image.fromarray(conv(frames[args.frame])).save(out / f"{tag}_{args.frame:06d}.png"); n += 1
    if args.every > 0:
        for i in range(0, len(frames), args.every):
            Image.fromarray(conv(frames[i])).save(out / f"{tag}_{i:06d}.png"); n += 1
    if not args.no_sheet:
        contact_sheet(sd, meta, frames, conv, out / "sheet.png", tag)
        print("요약 시트:", out / "sheet.png")
    print("PNG %d 장 → %s   (프레임 %d 장, 형상 %s, %s)" % (n, out, len(frames), frames.shape[1:], tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
