#!/usr/bin/env python3
"""내보낸 프레임으로 논문 그림을 만든다. **이 PC 가 없어도 돈다.**

    # 어떤 프레임이 있는지 본다
    python3 Paper/icra_bladder/make_figure.py --list
    python3 Paper/icra_bladder/make_figure.py --list ep0010_pose04_rule --best 8

    # 고른 프레임으로 그림 한 장
    python3 Paper/icra_bladder/make_figure.py \
        --pick ep0010_pose04_rule:42 ep0012_pose04_hold:42 \
        --labels "rule (t=42 s)" "hold (t=42 s)" \
        --overlay --roi --out figures/fig_rule_vs_hold.png

필요한 것은 ``numpy`` 와 ``opencv-python`` 뿐이다 — torch 도, ROS 도, 원시 세션도
필요 없다. 마스크가 PNG 로 함께 나가 있기 때문이다 (``export_results.py`` 참조).

왜 이렇게 나눠 놨는가
---------------------
실험 PC 에서 ``export_results.py`` 를 한 번 돌리면 필요한 것이 전부 저장소로 들어온다.
그 뒤의 논문 작업 — 프레임 고르기, 그림 만들기, 표 옮기기 — 은 노트북에서 저장소만
클론해서 끝까지 할 수 있다. 실험 PC 와의 연결이 끊겨도 막히지 않는다.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).with_name("data")

#: 겹쳐 그리는 색 (BGR). 마스크는 빨강, 부채꼴 경계는 주황 — 회색조 위에서 둘 다 읽힌다.
MASK_BGR = (0, 0, 255)
ROI_BGR = (0, 165, 255)


def episodes() -> list[str]:
    return sorted(p.name for p in (DATA / "frames").iterdir()
                  if p.is_dir() and (p / "index.csv").is_file())


def index_of(ep: str) -> list[dict]:
    with open(DATA / "frames" / ep / "index.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def cmd_list(args) -> None:
    if not args.list_ep:
        print(f"{'에피소드':26s} {'장수':>5s} {'길이 s':>7s} {'Q_seg 최대':>10s} {'Q_seg 중앙':>10s}")
        for ep in episodes():
            rows = index_of(ep)
            q = np.array([_num(r, "q_seg") for r in rows])
            print(f"{ep:26s} {len(rows):5d} {_num(rows[-1],'t'):7.0f} "
                  f"{np.nanmax(q):10.3f} {np.nanmedian(q):10.3f}")
        print("\n한 에피소드 안을 보려면:  --list <에피소드> [--best N]")
        return
    rows = index_of(args.list_ep)
    if args.best:
        rows = sorted(rows, key=lambda r: -_num(r, "q_seg"))[:args.best]
    print(f"{'n':>4s} {'t [s]':>7s} {'Q_seg':>7s} {'면적':>7s} {'dx':>7s} {'잘림':>6s} {'F [N]':>6s}")
    for r in rows:
        print(f"{r['n']:>4s} {_num(r,'t'):7.1f} {_num(r,'q_seg'):7.3f} {_num(r,'area'):7.4f} "
              f"{_num(r,'centroid_dx'):+7.3f} {_num(r,'border'):6.3f} {_num(r,'force_n'):6.2f}")
    print(f"\n고르려면:  --pick {args.list_ep}:<n>")


def tile(ep: str, n: int, *, overlay: bool, roi: bool, label: str | None,
         scale: int) -> np.ndarray:
    d = DATA / "frames" / ep
    img = cv2.imread(str(d / f"f{n:04d}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"프레임이 없다: {ep}:{n}  (--list {ep} 로 범위를 확인할 것)")
    v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if roi:
        r = cv2.imread(str(DATA / "frames" / "roi_mask.png"), cv2.IMREAD_GRAYSCALE)
        if r is not None:
            edge = cv2.morphologyEx((r > 0).astype(np.uint8), cv2.MORPH_GRADIENT,
                                    np.ones((3, 3), np.uint8))
            v[edge > 0] = ROI_BGR
    if overlay:
        m = cv2.imread(str(d / f"m{n:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is not None and m.max() > 0:
            c, _ = cv2.findContours((m > 127).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, c, -1, MASK_BGR, 1)
    if scale != 1:
        v = cv2.resize(v, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if label:
        # 흰 글자 + 검은 테두리 — 초음파는 밝은 곳과 어두운 곳이 함께 있다.
        org = (6 * scale, v.shape[0] - 6 * scale)
        for color, th in ((( 0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(v, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, color, th,
                        cv2.LINE_AA)
    return v


def cmd_pick(args) -> None:
    tiles = []
    for k, spec in enumerate(args.pick):
        if ":" not in spec:
            raise SystemExit(f"--pick 는 '에피소드:번호' 형식이다: {spec}")
        ep, n = spec.rsplit(":", 1)
        label = args.labels[k] if args.labels and k < len(args.labels) else None
        tiles.append(tile(ep, int(n), overlay=args.overlay, roi=args.roi,
                          label=label, scale=args.scale))
    h = max(t.shape[0] for t in tiles)
    gap = args.gap
    pads = []
    for t in tiles:
        if t.shape[0] < h:
            t = np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), t.dtype)])
        pads.append(t)
        pads.append(np.full((h, gap, 3), 255, np.uint8))
    out = np.hstack(pads[:-1]) if gap else np.hstack(pads[::2])
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"{dst}  ({out.shape[1]}×{out.shape[0]})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", dest="list_ep", nargs="?", const="", default=None,
                   help="에피소드 목록, 또는 한 에피소드의 프레임 목록")
    p.add_argument("--best", type=int, default=0, help="--list 에서 Q_seg 상위 N 장만")
    p.add_argument("--pick", nargs="+", help="'에피소드:번호' 를 나열한다")
    p.add_argument("--labels", nargs="+", help="타일마다 붙일 설명 (--pick 과 같은 순서)")
    p.add_argument("--overlay", action="store_true", help="U-Net 마스크 윤곽을 겹친다")
    p.add_argument("--roi", action="store_true", help="부채꼴 시야 경계를 그린다")
    p.add_argument("--scale", type=int, default=2, help="확대 배수 (256² 는 논문에 작다)")
    p.add_argument("--gap", type=int, default=6, help="타일 사이 흰 여백 [px]")
    p.add_argument("--out", default=str(Path(__file__).with_name("figures") / "picked.png"))
    a = p.parse_args()
    if a.pick:
        cmd_pick(a)
    elif a.list_ep is not None:
        cmd_list(a)
    else:
        p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
