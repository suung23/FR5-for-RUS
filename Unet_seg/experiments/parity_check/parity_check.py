#!/usr/bin/env python3
"""±d 패리티 검사 — elevational 오프셋의 부호가 영상에서 구별되는가.

배경
----
`docs/IMAGE_SERVOING_MATH.md` §6 명제 4: 목표 단면에서 해부구조가 거울대칭이면
`o(+d) = o(−d)` 이고, 그러면 **어떤 함수 f 를 써도** `f(o(+d)) = f(o(−d))` 다.
부호를 담은 특징이 존재하지 않는다. 그 경우 `docs/POLICY_LEARNING_MATH.md` §3.3 의
`Q̂` 모드 선택이 성립하지 않는다 (L11).

대칭이 얼마나 정확한지는 **경험적 문제**이며 이 스크립트가 그것을 잰다.

방법
----
면적 `A(d)` 는 `d` 의 우함수이므로, **면적으로 짝지으면 `|d|` 가 같은 쌍**이 된다
(인덱스로 짝짓는 것과 달리 스윕 속도가 일정하지 않아도 성립).

    D_cross(g)  면적이 같은, peak 반대쪽 두 프레임 (간격 g)
    D_same(g)   같은 쪽, 같은 간격 g                     ← 대조
    D_adj       인접 프레임                               ← 잡음 바닥

정확한 대칭이면 cross 쌍은 '같은 슬라이스' 이므로 `D_cross ≈ D_adj ≪ D_same`.

⚠️ 위약 대조가 필수다
--------------------
peak 를 **임의 지점**으로 가짜 지정해도 같은 신호가 나오는지 반드시 확인한다.
2026-08-25 실행에서 부호 일관성이 cross 1.00 / 대조 0.62 로 강해 보였으나,
위약에서 1.00 이 나와 **간격 분포 편향**임이 드러났다. 이 대조 없이는 없는 효과를
설계에 넣게 된다.

⚠️ 이 데이터셋의 한계
-------------------
PFUS 는 임상 프리핸드 스캔이고 **elevational 스윕이 아니다.** 자세 측정도 없다.
따라서 "면적 peak 의 양쪽" 이 elevational 최적점의 양쪽이라는 보장이 없고,
결과는 대칭성에 대해 **지지도 반증도 하지 못한다.**

제대로 답하려면 **로봇 제어 하 스윕**(FK 라벨)이 필요하다. 그때는 면적으로 짝지을
필요 없이 `d` 를 직접 알므로 이 스크립트의 `--pose` 경로를 쓰면 된다 (미구현).

사용
----
    python3 parity_check.py --manifest ~/datasets/pfus/manifest.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

import numpy as np
from PIL import Image

MIN_FRAMES = 24          # 환자당 최소 유효 프레임
MIN_SIDE = 6             # peak 양쪽 최소 프레임
MAX_SIDE_MIN = 0.92      # 양쪽 모두 peak 대비 이 값 이하로 내려가야 '스윕'
AREA_TOL = 0.03          # 면적 매칭 허용 상대오차
N_PLACEBO = 12           # 환자당 가짜 peak 시행 수


def load_features(root, mask_path):
    """면적비 · 정규화 centroid · 경계 접촉비."""
    m = np.array(Image.open(os.path.join(root, mask_path))) > 0
    h, w = m.shape
    n = int(m.sum())
    if n < 30:
        return None
    ys, xs = np.nonzero(m)
    border = ((xs < 2) | (xs > w - 3) | (ys < 2) | (ys > h - 3)).sum() / n
    return dict(area=n / (h * w), cx=xs.mean() / w, cy=ys.mean() / h, border=border)


def load_vector(root, image_path, size=128):
    a = np.asarray(Image.open(os.path.join(root, image_path)).convert("L")
                   .resize((size, size)), np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def matched_pairs(feats, valid, t):
    """면적이 같은 (peak 이전, peak 이후) 쌍."""
    left = [i for i in valid if i < t - 2]
    right = [i for i in valid if i > t + 2]
    for i in left:
        cand = [j for j in right
                if abs(feats[j]["area"] - feats[i]["area"]) < AREA_TOL * feats[i]["area"]]
        if cand:
            yield i, min(cand, key=lambda j: abs(feats[j]["area"] - feats[i]["area"]))


def sign_consistency(v):
    """가장 흔한 부호의 비율. 0.5 = 무작위, 1.0 = 완전 일관."""
    v = np.asarray(v)
    return float(max((v > 0).mean(), (v < 0).mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.expanduser("~/datasets/pfus/manifest.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(args.manifest))
    rng = np.random.default_rng(args.seed)

    by_patient = collections.defaultdict(list)
    for r in csv.DictReader(open(args.manifest)):
        by_patient[r["patient_id"]].append(r)
    for v in by_patient.values():
        v.sort(key=lambda r: int(r["frame_index"]))

    cross_d, same_d, adj_d = [], [], []
    real_sign, placebo_sign, border, accepted = [], [], [], []

    for pid, rs in by_patient.items():
        feats = [load_features(root, r["mask_path"]) for r in rs]
        valid = [i for i, f in enumerate(feats) if f]
        if len(valid) < MIN_FRAMES:
            continue
        area = np.array([feats[i]["area"] if feats[i] else 0.0 for i in range(len(feats))])
        t = int(np.argmax(np.convolve(area, np.ones(5) / 5, mode="same")))
        left = [i for i in valid if i < t - 2]
        right = [i for i in valid if i > t + 2]
        if len(left) < MIN_SIDE or len(right) < MIN_SIDE:
            continue
        if max(area[left].min(), area[right].min()) > MAX_SIDE_MIN * area[t]:
            continue

        accepted.append(pid)
        border.append(np.mean([feats[i]["border"] for i in valid]))
        vec = {i: load_vector(root, rs[i]["image_path"]) for i in valid}

        for i, j in zip(valid, valid[1:]):
            if j == i + 1:
                adj_d.append((np.hypot(feats[i]["cx"] - feats[j]["cx"],
                                       feats[i]["cy"] - feats[j]["cy"]),
                              float(np.abs(vec[i] - vec[j]).mean())))

        dx = []
        for i, j in matched_pairs(feats, valid, t):
            g = j - i
            ctl = [(p, q) for p, q in ((i - g, i), (i, i + g), (j - g, j), (j, j + g))
                   if p in vec and q in vec and ((p < t and q < t) or (p > t and q > t))]
            if not ctl:
                continue
            cross_d.append((np.hypot(feats[j]["cx"] - feats[i]["cx"],
                                     feats[j]["cy"] - feats[i]["cy"]),
                            float(np.abs(vec[j] - vec[i]).mean())))
            for p, q in ctl:
                same_d.append((np.hypot(feats[q]["cx"] - feats[p]["cx"],
                                        feats[q]["cy"] - feats[p]["cy"]),
                               float(np.abs(vec[p] - vec[q]).mean())))
            dx.append(feats[j]["cx"] - feats[i]["cx"])
        if len(dx) >= 8:
            real_sign.append(sign_consistency(dx))

        # 위약 — peak 를 임의 지점으로
        n = len(feats)
        for _ in range(N_PLACEBO):
            tf = int(rng.integers(int(0.25 * n), int(0.75 * n)))
            if abs(tf - t) < 0.10 * n:
                continue
            d = [feats[j]["cx"] - feats[i]["cx"] for i, j in matched_pairs(feats, valid, tf)]
            if len(d) >= 8:
                placebo_sign.append(sign_consistency(d))

    if not cross_d:
        print("스윕으로 인정된 환자가 없습니다.")
        return 1

    C, S, A = np.array(cross_d), np.array(same_d), np.array(adj_d)
    print("환자 %d명 / %d,  cross %d,  대조 %d,  인접 %d"
          % (len(accepted), len(by_patient), len(C), len(S), len(A)))
    print("\n                      centroid    영상")
    for name, arr in (("인접 (잡음 바닥)", A), ("cross (건너편)", C), ("same  (대조)", S)):
        print("  %-16s  %.4f     %.4f" % (name, np.median(arr[:, 0]), np.median(arr[:, 1])))
    print("\n  D_cross / D_same    %.2f        %.2f"
          % (np.median(C[:, 0]) / np.median(S[:, 0]), np.median(C[:, 1]) / np.median(S[:, 1])))
    print("  D_cross / D_adj     %.2f        %.2f"
          % (np.median(C[:, 0]) / np.median(A[:, 0]), np.median(C[:, 1]) / np.median(A[:, 1])))

    r, p = np.median(real_sign), np.median(placebo_sign)
    pct = 100.0 * (np.asarray(placebo_sign) < r).mean()
    print("\n부호 일관성 (cx)      진짜 peak %.3f   위약 %.3f (%d 시행)"
          % (r, p, len(placebo_sign)))
    print("  진짜가 위약 분포의 %.0f 백분위" % pct)
    print("  판정: %s" % ("peak 특이적 신호 있음" if pct > 90 else
                          "⚠️ 위약과 구별 안 됨 — peak 와 무관한 인공물"))
    print("\n경계 접촉비 중앙값 %.4f  (0 이면 border_penalty 항이 무력)" % np.median(border))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
