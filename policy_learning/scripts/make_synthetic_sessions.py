#!/usr/bin/env python3
"""합성 세션 여러 개 + sessions.csv 를 만든다 (하드웨어 없는 파이프라인 검증용).

    python scripts/make_synthetic_sessions.py --out data/synthetic --n 4 --duration 60

생성 후:
    python scripts/build_dataset.py --sessions data/synthetic/sessions.csv --out data/synthetic/dataset.h5 \
        --set perception.backend=none
    python scripts/train_policy.py --dataset data/synthetic/dataset.h5 --out runs/synthetic --set train.epochs=5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import PKG_ROOT, setup_logging

from rus_policy.session import SessionRecord, write_sessions_manifest
from rus_policy.synth import SynthConfig, generate_session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(PKG_ROOT / "data" / "synthetic"))
    parser.add_argument("--n", type=int, default=4, help="세션 수 (피험자 = 세션)")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--accel-noise", type=float, default=0.03)
    parser.add_argument("--us-latency", type=float, default=0.0)
    args = parser.parse_args()
    setup_logging()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    splits = ["train"] * max(1, args.n - 2) + ["val", "test"] if args.n >= 3 else ["train", "val"][: args.n]
    for i in range(args.n):
        cfg = SynthConfig(duration_s=args.duration, seed=args.seed + i, image_size=args.image_size,
                          accel_noise=args.accel_noise, us_latency_s=args.us_latency,
                          t0_unix=1_700_000_000.0 + 3600.0 * i)
        root = generate_session(out, cfg, name=f"synth_{i:02d}")
        records.append(SessionRecord(session_dir=root.name, subject=f"S{i:02d}", source="freehand",
                                     split=splits[i], note="synthetic"))
        print(f"  {root}  ({splits[i]})")
    manifest = out / "sessions.csv"
    write_sessions_manifest(manifest, records)
    print(f"세션 목록: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
