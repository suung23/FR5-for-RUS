#!/usr/bin/env python3
"""세션 목록 (sessions.csv) → 학습 샘플 HDF5.

    python scripts/build_dataset.py                                   # configs/policy_default.yaml
    python scripts/build_dataset.py --sessions data/sessions.csv --out data/policy_dataset.h5
    python scripts/build_dataset.py --set perception.backend=none     # U-Net 없이 (Q̂ 항 마스크)
    python scripts/build_dataset.py --set split.strategy=subject_random

sessions.csv 열: session_dir, subject, source(freehand|teleop), split(train|val|test|빈칸), note
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import add_config_args, config_from_args

from rus_policy.dataset import build_dataset
from rus_policy.session import read_sessions_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--sessions", default=None, help="sessions.csv (기본: paths.sessions_manifest)")
    parser.add_argument("--out", default=None, help="출력 HDF5 (기본: paths.dataset)")
    parser.add_argument("--compress", action="store_true", help="프레임 gzip 압축 (느리지만 작다)")
    args = parser.parse_args()
    cfg = config_from_args(args)

    manifest = Path(args.sessions) if args.sessions else cfg.resolve(cfg.paths.sessions_manifest)
    records = read_sessions_manifest(manifest, root=manifest.parent)
    out = Path(args.out) if args.out else cfg.resolve(cfg.paths.dataset)
    result = build_dataset(cfg, out_path=out, compress=args.compress, records=records)
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n데이터셋: {result['path']}  샘플 {result['n_samples']}")
    for s in result["sessions"]:
        print(f"  {s['name']:40s} {s['split']:5s} {s['source']:8s} 샘플 {s['samples']:4d}  "
              f"이동 {s['n_move_segments']:3d}/정지 {s['n_still_segments']:3d}  "
              f"US {s['us_fps']:.1f} fps  IMU {s['imu_hz']:.0f} Hz  규약 {s['convention']}"
              f"{' (모호)' if s.get('convention_ambiguous') else ''}  지각 {s['perception']}")
    print(f"요약: {summary_path}")
    if result["n_samples"] == 0:
        print("경고: 샘플이 0 개입니다. scripts/inspect_session.py 로 정지 판정 임계를 확인하십시오.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
