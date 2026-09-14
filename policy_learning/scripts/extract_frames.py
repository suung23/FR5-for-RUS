#!/usr/bin/env python3
"""에피소드의 초음파 프레임을 그림으로 뽑는다 — 논문 그림과 눈으로 하는 확인.

    python3 scripts/extract_frames.py --list runs/main_rule
    python3 scripts/extract_frames.py runs/main_rule --out results/figures
    python3 scripts/extract_frames.py runs/main_rule/ep0001_pose01_rule --every 10s

무엇을 뽑는가
-------------
기본은 에피소드마다 **네 장**이다. 가운데를 기계적으로 자르면 아무 일도 안 일어난 순간이
걸리기 쉬우므로, 이야기가 되는 시점을 고른다:

    start    시작 (0 s) — 방광이 안 보이는 자세에서 출발했다는 증거
    found    면적비가 성공 문턱(0.04)을 처음 넘은 순간. 없으면 면적 최대의 절반 지점
    best     면적이 가장 컸던 순간
    end      마지막 프레임

파일 이름에 시각과 면적비가 들어간다 (`ep0001_pose01_rule__found_t31.2s_a0.058.png`) —
그림을 고를 때 파일 목록만 보고도 고를 수 있다.

원자료(`runs/`, 4.1 GB)는 리포에 없다. 이 스크립트는 **로컬에 원자료가 있을 때** 돌리고,
뽑은 그림만 `results/figures/` 로 커밋한다.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

AREA_MIN = 0.04          # 성공 판정의 면적 문턱


def episodes(root: Path) -> list[Path]:
    if (root / "frames.npz").is_file():
        return [root]
    return sorted(p.parent for p in root.glob("ep*/frames.npz"))


def picks(t: np.ndarray, area: np.ndarray) -> list[tuple[str, int]]:
    """뽑을 시점 — (이름, 인덱스)."""
    out = [("start", 0)]
    hit = np.where(area >= AREA_MIN)[0]
    if hit.size:
        out.append(("found", int(hit[0])))
    elif area.max() > 0:
        half = np.where(area >= 0.5 * area.max())[0]
        if half.size:
            out.append(("half", int(half[0])))
    out.append(("best", int(np.argmax(area))))
    out.append(("end", len(t) - 1))
    seen, uniq = set(), []
    for name, i in out:
        if i not in seen:
            seen.add(i)
            uniq.append((name, i))
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="세션 폴더 또는 에피소드 폴더")
    ap.add_argument("--out", default=None, help="그림을 쓸 곳 (기본: <target>/figures)")
    ap.add_argument("--list", action="store_true", help="무엇이 있는지만 보여 준다")
    ap.add_argument("--every", default=None, help="이 간격으로도 뽑는다 (예: 10s)")
    ap.add_argument("--scale", type=int, default=2, help="확대 배수 (기본 2 → 512 px)")
    args = ap.parse_args()

    root = Path(args.target)
    eps = episodes(root)
    if not eps:
        print(f"프레임이 없습니다: {root}  (frames.npz 를 찾지 못했습니다)")
        return 1
    if args.list:
        print(f"{'에피소드':32s} {'프레임':>7s} {'길이':>7s} {'면적최대':>8s}")
        for d in eps:
            z = np.load(d / "frames.npz")
            st = np.load(d / "states.npz")["state"] if (d / "states.npz").is_file() else None
            a = st[:, 4].max() if st is not None else float("nan")
            print(f"  {d.name:32s} {len(z['t']):7d} {z['t'][-1] - z['t'][0]:6.1f}s {a:8.4f}")
        return 0

    from PIL import Image
    out = Path(args.out) if args.out else root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    every = None
    if args.every:
        m = re.fullmatch(r"([0-9.]+)s?", args.every)
        every = float(m.group(1)) if m else None

    n_written = 0
    for d in eps:
        zf = np.load(d / "frames.npz")
        ft, frames = zf["t"], zf["frames"]
        st = np.load(d / "states.npz") if (d / "states.npz").is_file() else None
        if st is not None:
            area = np.interp(ft, st["t"], np.asarray(st["state"])[:, 4])
        else:
            area = np.zeros(len(ft))
        chosen = picks(ft, area)
        if every:
            t0 = ft[0]
            # 이름을 0 으로 채운다 — 안 그러면 목록에서 t100 이 t20 앞에 온다.
            chosen += [(f"at{int(round(s)):04d}", int(np.argmin(np.abs(ft - (t0 + s)))))
                       for s in np.arange(every, ft[-1] - t0, every)]
        sub = out / d.name
        sub.mkdir(exist_ok=True)
        for name, i in chosen:
            img = Image.fromarray(frames[i])
            if args.scale != 1:
                img = img.resize((img.width * args.scale, img.height * args.scale), Image.NEAREST)
            # 간격 프레임(at####)은 이름에 이미 시각이 있으므로 중복을 뺀다.
            stamp = "" if name.startswith("at") else f"_t{ft[i] - ft[0]:05.1f}s"
            img.save(sub / f"{name}{stamp}_a{area[i]:.3f}.png")
            n_written += 1
        print(f"  {d.name}: {len(chosen)} 장")
    print(f"\n{n_written} 장 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
