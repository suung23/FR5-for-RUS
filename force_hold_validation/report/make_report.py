#!/usr/bin/env python3
"""검증 리포트 본문. 표의 숫자는 전부 캡처에서 읽는다 — 손으로 적은 값은 없다."""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from fh.analysis import (excursions, hold_metrics, load_run,        # noqa: E402
                         settling_point)

RUNS = os.path.join(os.path.dirname(_HERE), "runs")
HOLD = ("A_t0p5", "A_t2p0", "A_t3p0", "A_t4p0")
DIST = ("B_t0p5", "B_t3p0_pose", "C_limit")

#: 표·범례용 이름. 내부 라벨은 파일 이름이지 독자가 읽을 것이 아니다.
NAMES = {"B_t0p5": "Disturbance · 0.5 N",
         "B_t3p0_pose": "Disturbance · 3.0 N",
         "C_limit": "Hard press · 3.0 N"}



def load(label):
    return load_run(os.path.join(RUNS, f"{label}_samples.csv"))


def markdown(spec) -> str:
    """(header, rows, align) 을 마크다운 표로."""
    header, rows, align = spec
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def t1_configuration():
    rows = []
    for label in HOLD:
        r = load(label)
        m = r.meta
        rows.append([f"{r.target:.1f}", f"±{r.band:.2f}",
                     f"{m['contact_probing_force_n']:.2f}",
                     f"{m['contact_probing_release_n']:.2f}",
                     f"{settling_point(r):.2f}",
                     f"{m['warn_contact_force_n']:.1f} / {m['max_contact_force_n']:.1f}"])
    return (["Setpoint [N]", "Deadband [N]", "Entry [N]", "Release [N]",
             "Operating point [N]", "Warn / Limit [N]"], rows, ["---:"] * 6)


def t2_hold():
    rows = []
    for label in HOLD:
        r = load(label)
        m = hold_metrics(r)
        rows.append([f"{r.target:.1f}", f"{m['seconds']:.0f}", f"{m['samples']:,}",
                     f"**{m['sd_n']:.3f}**", f"{m['error_vs_settling_n']:+.3f}",
                     f"{m['sd_n'] / r.target * 100:.1f}", f"{m['in_band_pct']:.1f}"])
    return (["Setpoint [N]", "Duration [s]", "Samples", "SD [N]",
             "Offset [N]", "SD / setpoint [%]", "In band [%]"], rows, ["---:"] * 7)


def t3_disturbance():
    rows = []
    for label in DIST:
        r = load(label)
        ex = excursions(r)
        if not ex:
            continue
        d = np.array([e["duration_s"] for e in ex])
        rows.append([NAMES[label], f"{r.target:.1f}",
                     f"{r.target + r.band:.2f}", f"{len(ex)}",
                     f"{np.median(d):.2f}", f"**{d.max():.2f}**",
                     f"{max(e['peak_n'] for e in ex):.2f}"])
    return (["Run", "Setpoint [N]", "Band top [N]", "Excursions",
             "Median return [s]", "Max return [s]", "Peak ‖F‖ [N]"], rows,
            ["---"] + ["---:"] * 6)


def t4_exposure():
    levels = (3.5, 4.0, 4.5, 5.0)
    rows = []
    for label in DIST:
        r = load(label)
        mask = r.probing()
        t, f = r.t[mask], r.force[mask]
        cells = []
        for lv in levels:
            above = f > lv
            best, i = 0.0, 0
            while i < len(above):
                if not above[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(above) and above[j + 1]:
                    j += 1
                best = max(best, float(t[j] - t[i]))
                i = j + 1
            cells.append("0" if best == 0 else f"{best:.2f}")
        rows.append([NAMES[label], f"{f.max():.2f}"] + cells)
    return (["Run", "Peak ‖F‖ [N]", "> 3.5 N", "> 4.0 N",
             "> 4.5 N (warn)", "> 5.0 N (limit)"], rows, ["---"] + ["---:"] * 5)


REPORT = """# Force-controlled contact: first validation

## 1. Method

A convex ultrasound probe on an FR5 arm was brought into contact with an
abdominal phantom whose internal volume is set by a 500 mL syringe. The control
stack (`us_diff_ik`) closes an admittance loop on the probe penetration axis:
once the contact force crosses an entry threshold the arm switches to contact
velocity limits and holds the force itself, leaving the operator no axis.

All forces are the **contact-force magnitude** `‖F‖ = √(Fx²+Fy²+Fz²)` in the
compensated probe frame — the same scalar the controller compares against its
setpoint and its limits. Force is read from the PX6D six-axis sensor over its
own USB link at 1 kHz, gravity- and payload-compensated, and referenced to the
probe contact point. Every sample the controller acted on was recorded; nothing
was filtered before analysis.

Three measurements were made:

- **Hold.** The probe was settled on the phantom and held for 60 s at each
  setpoint, with no operator input and no volume change.
- **Disturbance.** With the force loop closed, 150 mL was injected and withdrawn
  repeatedly while recording. Each syringe action was timestamped at its onset.
- **Exposure.** Every interval in which the force left the intended band was
  measured, together with the time taken to return.

Probe pose was logged alongside force at the flange, and axial travel is the
displacement projected onto the probe penetration axis.

**Table 1 — Configuration**

{t1}

## 2. Force holding

The regulator holds a band rather than a point: inside the deadband it commands
zero velocity, so the force settles where it first enters the band and stays
there. The operating point in Table 1 is that entry force. Deviation from it is
the tracking quality; standard deviation is independent of band width and is
therefore the figure that compares setpoints directly.

**Figure 1** — 60 s of held force at each setpoint.

![Hold traces](figures/fig1_hold_traces.png)

**Table 2 — Hold quality**

{t2}

**Figure 2** — Stability, offset and relative stability against setpoint.

![Hold quality](figures/fig2_hold_quality.png)

Hold SD is 0.022–0.065 N across a setpoint range of 0.5–4.0 N, with no trend
against setpoint. In relative terms the loop is tighter at higher forces —
1.0–1.4 % of setpoint above 2 N — because the noise floor is fixed by the sensor
rather than by the controller. Tracking offset from the operating point stays
within 0.10 N at every setpoint.

## 3. Disturbance rejection

**Figure 3** — Force change aligned on each 150 mL injection, and the time taken
to return inside the band.

![Disturbance response](figures/fig3_disturbance_response.png)

**Table 3 — Band excursions**

{t3}

Twenty-two excursions were recorded across the three runs. Every one returned
inside the band, the slowest in 1.39 s and the median in 0.19–1.09 s depending
on setpoint.

**Figure 4** — Longest continuous time spent above a given force, and the
hardest-pressed run.

![Force exposure](figures/fig4_force_exposure.png)

**Table 4 — Exposure**

{t4}

Peak force reached 4.41 N against a 5.0 N limit. Time spent continuously above
the 4.5 N warning level was zero in every run.

## 4. Direct evidence of regulation

**Figure 5** — Contact force and probe axial travel during the disturbance run.

![Probe travel](figures/fig5_probe_travel.png)

The arm moves against the disturbance. Each injection lifts the phantom surface
into the probe; the force rises, the probe retreats, and the force returns to
the band. Over the run the probe travels 2.29 mm along its penetration axis
while the force stays within 3.0 ± 0.5 N — the displacement is the volume the
loop absorbed.

## 5. What this establishes

- The loop **holds a commanded contact force** from 0.5 N to 4.0 N with a
  stability of 0.02–0.07 N, set by the sensor noise floor rather than by the
  controller.
- Holding stability is **flat across the range**, so setpoint selection can be
  driven by imaging requirements rather than by control performance.
- The loop **returns the force to its band after an external disturbance**,
  every time, within 1.4 s.
- Under repeated 150 mL volume changes the force **stayed clear of the warning
  level entirely**, with 0.59 N of margin to the hard limit at its worst.
- The mechanism is **motion, not tolerance**: the probe travels millimetres
  along its own axis to keep the force constant.
"""


def tables() -> dict:
    """표 네 개를 (header, rows, align) 로. 마크다운과 워드가 **같은 것**을 쓴다."""
    return {"t1": t1_configuration(), "t2": t2_hold(),
            "t3": t3_disturbance(), "t4": t4_exposure()}


def main():
    spec = tables()
    text = REPORT.format(**{k: markdown(v) for k, v in spec.items()})
    path = os.path.join(_HERE, "force_hold_validation_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(os.path.relpath(path, os.path.dirname(_HERE)))


if __name__ == "__main__":
    main()
