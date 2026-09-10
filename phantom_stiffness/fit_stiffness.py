#!/usr/bin/env python3
"""저장된 압입 시험에서 강성을 적합하고 그림을 낸다.

    python3 phantom_stiffness/fit_stiffness.py runs/stiff_phantom_b_20260910_161200
    python3 phantom_stiffness/fit_stiffness.py runs/stiff_* --compare
    python3 phantom_stiffness/fit_stiffness.py runs/stiff_b_* --force-band 0.5 4.0

GUI 가 이미 적재 구간의 k 를 화면에 보여 주지만, 여기서는 **어느 구간을 믿을지**
고를 수 있다. 팬텀은 얕게 눌렀을 때와 깊이 눌렀을 때 기울기가 다르고 (접촉 면적이
자라는 구간 + 재료가 굳는 구간), 제어가 실제로 도는 힘 대역이 정해져 있다면 그
대역에서 잰 k 가 admittance 설계에 쓸 값이다. `--force-band` 가 그 창이다.

내는 것: `stiffness_fit.json`, `stiffness_fit.csv`, `stiffness_fit.png`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stiffness import fit_line, hysteresis, tangent_stiffness, StiffnessPoint   # noqa: E402


def load_points(session_dir: str) -> tuple[list[StiffnessPoint], dict]:
    """`points.csv` 를 읽는다. 메타가 있으면 함께 준다."""
    path = os.path.join(session_dir, "points.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 가 없다 — 강성 세션 폴더가 맞습니까?")
    pts: list[StiffnessPoint] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            def f(key, default=float("nan")):
                v = row.get(key, "")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default
            pts.append(StiffnessPoint(
                index=int(row["index"]), t_pc=f("t_pc"), depth_mm=f("depth_mm"),
                force_n=f("force_n"), force_sd=f("force_sd"), samples=int(f("samples", 0)),
                relax_n=f("relax_n", 0.0), phase=row.get("phase", "load"),
                wrench=[f(k) for k in ("fx", "fy", "fz", "mx", "my", "mz")],
                pose_xyz=[f(k) for k in ("x", "y", "z")],
                pose_quat=[f(k) for k in ("qw", "qx", "qy", "qz")],
                us_seq=int(f("us_seq", -1)), note=row.get("note", ""),
            ))
    meta = {}
    meta_path = os.path.join(session_dir, "session.meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    return pts, meta


def select(points: list[StiffnessPoint], phase: str | None, band: tuple[float, float] | None,
           max_sd: float | None, max_relax: float | None) -> list[StiffnessPoint]:
    """적합에 쓸 점을 고른다. 거른 이유는 호출한 쪽이 보고한다."""
    out = []
    for p in points:
        if phase and p.phase != phase:
            continue
        if band and not (band[0] <= p.force_n <= band[1]):
            continue
        if max_sd is not None and p.force_sd > max_sd:
            continue
        if max_relax is not None and abs(p.relax_n) > max_relax:
            continue
        out.append(p)
    return out


def analyse(session_dir: str, args) -> dict:
    points, meta = load_points(session_dir)
    band = tuple(args.force_band) if args.force_band else None
    kept = select(points, args.phase, band, args.max_sd, args.max_relax)
    # 인덱스로 비교한다. dataclass 의 값 비교로 거르면 값이 우연히 같은 두 점을 하나로 본다.
    kept_ix = {p.index for p in kept}
    dropped = [p.index for p in points
               if p.index not in kept_ix and (not args.phase or p.phase == args.phase)]

    fit = fit_line([p.depth_mm for p in kept], [p.force_n for p in kept])
    result = {
        "session": os.path.basename(session_dir),
        "label": meta.get("label", ""),
        "points_total": len(points),
        "points_used": len(kept),
        "points_dropped": dropped,
        "selection": {"phase": args.phase, "force_band_n": list(band) if band else None,
                      "max_sd_n": args.max_sd, "max_relax_n": args.max_relax},
        "fit": fit,
        "tangent": tangent_stiffness(kept),
        "hysteresis": hysteresis(points),
        "force_axis": (meta.get("force") or {}).get("axis"),
        "depth_source": (meta.get("depth") or {}).get("source"),
    }
    if fit.get("ok"):
        # 잔차의 최대 절대값 — 직선이 실제로 자료를 설명하는지 R² 말고 한 번 더 본다.
        d = np.array([p.depth_mm for p in kept]); f = np.array([p.force_n for p in kept])
        resid = f - (fit["k_n_per_mm"] * d + fit["intercept_n"])
        result["residual_max_n"] = float(np.max(np.abs(resid)))
        result["residual_rms_n"] = float(np.sqrt(np.mean(resid ** 2)))
    return result


def plot(session_dirs: list[str], results: list[dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from korean_font import use_korean_font
    use_korean_font()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax, ax_t = axes
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (sd, res) in enumerate(zip(session_dirs, results)):
        pts, _ = load_points(sd)
        c = colors[i % len(colors)]
        load = [p for p in pts if p.phase == "load"]
        unload = [p for p in pts if p.phase == "unload"]
        lbl = res.get("label") or os.path.basename(sd)
        ax.errorbar([p.depth_mm for p in load], [p.force_n for p in load],
                    yerr=[p.force_sd for p in load], fmt="o", ms=5, color=c,
                    capsize=2, lw=1, label=f"{lbl} 적재")
        if unload:
            ax.plot([p.depth_mm for p in unload], [p.force_n for p in unload],
                    "s", ms=5, mfc="none", color=c, label=f"{lbl} 제하")
        fit = res["fit"]
        if fit.get("ok"):
            used = [p for p in pts if p.index not in res["points_dropped"]]
            xs = np.linspace(min(p.depth_mm for p in used), max(p.depth_mm for p in used), 2)
            ax.plot(xs, fit["k_n_per_mm"] * xs + fit["intercept_n"], "-", color=c, lw=1.6)
        for t in res["tangent"]:
            ax_t.plot(t["force_mid_n"], t["k_n_per_mm"], "o", ms=5, color=c)
        if res["tangent"]:
            ax_t.plot([t["force_mid_n"] for t in res["tangent"]],
                      [t["k_n_per_mm"] for t in res["tangent"]], "-", color=c, lw=1.0, label=lbl)

    ax.set_xlabel("압입 깊이 δ [mm]"); ax.set_ylabel("접촉력 F [N]")
    ax.set_title("강성 곡선"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax_t.set_xlabel("접촉력 F [N]"); ax_t.set_ylabel("접선 강성 dF/dδ [N/mm]")
    ax_t.set_title("접선 강성 — 팬텀이 굳어지는가"); ax_t.grid(alpha=0.3)
    if ax_t.get_legend_handles_labels()[0]:
        ax_t.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"그림 → {out_path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="강성 세션 폴더 (glob 가능)")
    ap.add_argument("--phase", default="load", choices=("load", "unload", "all"),
                    help="적합에 쓸 구간 (기본 %(default)s). all 은 이력을 섞으므로 권하지 않는다")
    ap.add_argument("--force-band", type=float, nargs=2, metavar=("LO", "HI"),
                    help="이 힘 대역의 점만 적합한다 [N] — 제어가 실제로 도는 대역")
    ap.add_argument("--max-sd", type=float, default=None, help="정착 창 표준편차 상한 [N]")
    ap.add_argument("--max-relax", type=float, default=None, help="응력완화량 상한 [N]")
    ap.add_argument("--compare", action="store_true", help="여러 세션을 한 그림에 겹친다")
    ap.add_argument("--out", default=None, help="그림 경로 (기본: 첫 세션 폴더 안)")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    if args.phase == "all":
        args.phase = None

    dirs: list[str] = []
    for spec in args.sessions:
        hits = sorted(glob.glob(spec)) if any(c in spec for c in "*?[") else [spec]
        dirs.extend(d for d in hits if os.path.isdir(d))
    if not dirs:
        print("세션 폴더를 찾지 못했다.", file=sys.stderr)
        return 1

    results = []
    for d in dirs:
        res = analyse(d, args)
        results.append(res)
        fit = res["fit"]
        head = f"── {res['session']}  ({res['points_used']}/{res['points_total']} 점)"
        print(head)
        if fit.get("ok"):
            print(f"   k = {fit['k_n_per_mm']:.4f} N/mm  ({fit['k_n_per_m']:.1f} N/m)"
                  f"   절편 {fit['intercept_n']:+.3f} N   R² {fit['r_squared']:.4f}")
            print(f"   깊이 {fit['depth_span_mm']:.2f} mm   힘 {fit['force_span_n']:.2f} N"
                  f"   잔차 rms {res.get('residual_rms_n', float('nan')):.4f} N"
                  f" / max {res.get('residual_max_n', float('nan')):.4f} N")
        else:
            print(f"   적합 불가: {fit.get('reason')}")
        h = res["hysteresis"]
        if h.get("ok"):
            print(f"   이력: 평균 간격 {h['mean_gap_n']:.3f} N, 최대 {h['max_gap_n']:.3f} N,"
                  f" 고리 넓이 {h['loop_area_n_mm']:.2f} N·mm")
        if res["points_dropped"]:
            print(f"   거른 점: {res['points_dropped']}")
        with open(os.path.join(d, "stiffness_fit.json"), "w") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        with open(os.path.join(d, "stiffness_fit.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["session", "points_used", "k_n_per_mm", "k_n_per_m", "intercept_n",
                        "r_squared", "depth_span_mm", "force_span_n", "residual_rms_n"])
            w.writerow([res["session"], res["points_used"], fit.get("k_n_per_mm", ""),
                        fit.get("k_n_per_m", ""), fit.get("intercept_n", ""),
                        fit.get("r_squared", ""), fit.get("depth_span_mm", ""),
                        fit.get("force_span_n", ""), res.get("residual_rms_n", "")])

    if len(dirs) > 1 and not args.compare:
        print("\n(여러 세션을 한 그림에 겹치려면 --compare)")
    out = args.out or os.path.join(dirs[0], "stiffness_fit.png")
    plot(dirs if args.compare else dirs[:1], results if args.compare else results[:1], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
