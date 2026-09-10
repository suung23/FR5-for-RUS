#!/usr/bin/env python3
"""강성 점의 수집·적합·저장.

**측정 원리.** 프로브를 팬텀에 대고 축 방향으로 조금씩 파고들며, 매 단계마다
변위가 멎기를 기다린 뒤 (침하·응력완화) 힘을 평균 낸다. 접촉력 F 와 압입 깊이 δ 의
기울기가 강성 k = dF/dδ 다.

**왜 단계마다 기다리는가.** 팬텀은 점탄성이라 같은 깊이에서도 힘이 시간에 따라
내려앉는다. 멈추지 않고 훑으면 재는 것은 강성이 아니라 그 속도에서의 **겉보기**
임피던스다. 그래서 각 점은 (평균, 표준편차, 표본 수, 침하량) 을 함께 남기고, 침하가
큰 점은 적합에서 걸러낼 수 있게 한다.

**왕복.** 눌렀다 되돌아오면 이력(hysteresis) 이 보인다. 점마다 `phase` 로 loading /
unloading 을 구분해 적재 구간만으로 k 를 적합하고, 이력은 따로 보고한다 — 둘을 섞으면
같은 깊이에 힘이 둘 있는 자료에 직선을 맞추게 된다.

기존 `force_hold_validation` 의 강성(정상상태 평형점 8 개, 0.374 N/mm) 과 같은 양을
재지만 경로가 다르다: 그쪽은 힘 목표를 바꿔 admittance 가 앉은 자리를 읽고, 이쪽은
깊이를 직접 주고 힘을 읽는다. 같은 팬텀에서 두 값이 어긋나면 admittance 의 정상상태
오차를 의심할 근거가 된다.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

#: NumPy 2 에서 ``trapz`` 가 빠졌다. 두 버전 모두에서 도는 이름을 하나 쓴다.
_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class StiffnessPoint:
    """단계 하나. 지령이 아니라 **실측**만 담는다."""

    index: int
    t_pc: float
    depth_mm: float                 # 접촉 기준에서 프로브 축 방향 [mm], 양수 = 파고듦
    force_n: float                  # 정착 창 평균 접촉력 [N], 양수 = 압축
    force_sd: float                 # 같은 창의 표준편차 — 흔들리면 크다
    samples: int                    # 평균에 들어간 표본 수
    relax_n: float = 0.0            # 창 앞절반 − 뒷절반 평균 = 응력완화량 [N]
    phase: str = "load"             # load | unload
    wrench: list = field(default_factory=list)      # 6 축 원시 (영점 적용) 평균
    pose_xyz: list = field(default_factory=list)
    pose_quat: list = field(default_factory=list)
    us_seq: int = -1                # 이 시점에 저장된 US 프레임 번호 (-1 = 없음)
    note: str = ""


def fit_line(depth_mm: np.ndarray, force_n: np.ndarray) -> dict:
    """F = k·δ + b 를 최소제곱으로 맞춘다.

    Returns:
        `k_n_per_mm`, `intercept_n`, `r_squared`, `points`, `depth_span_mm`,
        `force_span_n`. 점이 2 개 미만이면 `ok=False`.
    """
    d = np.asarray(depth_mm, float)
    f = np.asarray(force_n, float)
    good = np.isfinite(d) & np.isfinite(f)
    d, f = d[good], f[good]
    if d.size < 2 or np.ptp(d) < 1e-9:
        return {"ok": False, "reason": "점이 2 개 미만이거나 깊이가 한 자리다", "points": int(d.size)}
    k, b = np.polyfit(d, f, 1)
    pred = k * d + b
    ss_res = float(np.sum((f - pred) ** 2))
    ss_tot = float(np.sum((f - f.mean()) ** 2))
    return {
        "ok": True,
        "points": int(d.size),
        "k_n_per_mm": float(k),
        "k_n_per_m": float(k) * 1000.0,
        "intercept_n": float(b),
        "r_squared": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "depth_span_mm": float(np.ptp(d)),
        "force_span_n": float(np.ptp(f)),
    }


def tangent_stiffness(points: list[StiffnessPoint]) -> list[dict]:
    """이웃한 두 점 사이의 접선 강성. 팬텀이 굳어지는지(비선형) 를 본다."""
    out = []
    for a, b in zip(points, points[1:]):
        dd = b.depth_mm - a.depth_mm
        if abs(dd) < 1e-6:
            continue
        out.append({
            "from_index": a.index, "to_index": b.index,
            "depth_mid_mm": 0.5 * (a.depth_mm + b.depth_mm),
            "force_mid_n": 0.5 * (a.force_n + b.force_n),
            "k_n_per_mm": (b.force_n - a.force_n) / dd,
        })
    return out


def hysteresis(points: list[StiffnessPoint]) -> dict:
    """적재/제하 곡선의 벌어짐. 겹치는 깊이 구간에서만 본다."""
    load = [p for p in points if p.phase == "load"]
    unload = [p for p in points if p.phase == "unload"]
    if len(load) < 2 or len(unload) < 2:
        return {"ok": False, "reason": "적재·제하 각각 2 점 이상 필요"}
    ld = np.array([p.depth_mm for p in load]); lf = np.array([p.force_n for p in load])
    ud = np.array([p.depth_mm for p in unload]); uf = np.array([p.force_n for p in unload])
    lo, hi = max(ld.min(), ud.min()), min(ld.max(), ud.max())
    if hi <= lo:
        return {"ok": False, "reason": "적재·제하 깊이 구간이 겹치지 않는다"}
    grid = np.linspace(lo, hi, 32)
    li = np.interp(grid, ld[np.argsort(ld)], lf[np.argsort(ld)])
    ui = np.interp(grid, ud[np.argsort(ud)], uf[np.argsort(ud)])
    gap = li - ui
    return {
        "ok": True,
        "overlap_mm": [float(lo), float(hi)],
        "mean_gap_n": float(np.mean(gap)),
        "max_gap_n": float(np.max(np.abs(gap))),
        # 적재 곡선 아래 넓이에 대한 이력 넓이의 비 — 감쇠의 크기 척도
        "loop_area_n_mm": float(_trapz(np.abs(gap), grid)),
    }


class StiffnessRun:
    """한 번의 압입 시험. 점을 모으고, 즉석 적합을 주고, 세션 폴더로 저장한다."""

    def __init__(self, label: str, out_dir: str, meta: dict | None = None) -> None:
        self.label = label
        self.out_dir = out_dir
        self.meta = dict(meta or {})
        self.points: list[StiffnessPoint] = []
        self.started = time.time()

    def add(self, point: StiffnessPoint) -> StiffnessPoint:
        point.index = len(self.points)
        self.points.append(point)
        return point

    def undo(self) -> StiffnessPoint | None:
        return self.points.pop() if self.points else None

    def arrays(self, phase: str | None = "load") -> tuple[np.ndarray, np.ndarray]:
        pts = [p for p in self.points if phase is None or p.phase == phase]
        return (np.array([p.depth_mm for p in pts]), np.array([p.force_n for p in pts]))

    def fit(self, phase: str | None = "load") -> dict:
        return fit_line(*self.arrays(phase))

    def summary(self) -> dict:
        return {
            "label": self.label,
            "points": len(self.points),
            "fit_loading": self.fit("load"),
            "fit_all": self.fit(None),
            "tangent": tangent_stiffness([p for p in self.points if p.phase == "load"]),
            "hysteresis": hysteresis(self.points),
        }

    def save(self, extra_meta: dict | None = None) -> str:
        """`stiff_<label>_<시각>/` 에 점·요약·메타를 쓴다. 폴더 경로를 준다."""
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.started))
        path = os.path.join(self.out_dir, f"stiff_{self.label}_{stamp}")
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "points.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "t_pc", "phase", "depth_mm", "force_n", "force_sd", "samples",
                        "relax_n", "fx", "fy", "fz", "mx", "my", "mz",
                        "x", "y", "z", "qw", "qx", "qy", "qz", "us_seq", "note"])
            for p in self.points:
                wr = list(p.wrench) + [float("nan")] * (6 - len(p.wrench))
                po = list(p.pose_xyz) + [float("nan")] * (3 - len(p.pose_xyz))
                qu = list(p.pose_quat) + [float("nan")] * (4 - len(p.pose_quat))
                w.writerow([p.index, f"{p.t_pc:.6f}", p.phase, f"{p.depth_mm:.4f}",
                            f"{p.force_n:.5f}", f"{p.force_sd:.5f}", p.samples, f"{p.relax_n:.5f}",
                            *[f"{v:.6f}" for v in wr], *[f"{v:.6f}" for v in po],
                            *[f"{v:.6f}" for v in qu], p.us_seq, p.note])

        meta = {**self.meta, **(extra_meta or {})}
        meta.update({
            "label": self.label,
            "started": stamp,
            "points": [asdict(p) for p in self.points],
            "summary": self.summary(),
            "caveat": "깊이는 프로브 축 투영 (fh/analysis.axial_travel 과 같은 규약). "
                      "힘 부호·축 배정은 PX6D AXIS_ORDER 의 잠정값을 따른다.",
        })
        with open(os.path.join(path, "session.meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        return path
