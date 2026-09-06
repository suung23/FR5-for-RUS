# Constant-force hold — validation summary

2026-09-06. Data: `runs_sweep500/` (lap 1 + the `1h` hold re-run) and `runs_lap2/`.
Figures and tables: `outputs_lap2/`.

Two things are claimed here and they are not equally supported. The hold claim is
strong. The disturbance-rejection claim **is not supported by this data**, and that
is a result, not a gap — the placebo arm was built to detect exactly this and did.

---

## 1. Steady-state force hold — validated

`B_z = 4500 N·s/m`, deadband 0.05 N, contact force `‖F‖`. Sixteen 30 s holds:
eight bands in the `1h` set (2026-09-04) and eight in lap 2 (2026-09-06), **two
sessions with two independently measured calibrations**.

| target [N] | SD `1h` | error `1h` | SD `r2` | error `r2` | session-to-session |Δerror| |
|---|---|---|---|---|---|
| 0.5 | 0.024 | −0.004 | 0.021 | −0.030 | 0.026 |
| 1.0 | 0.022 | −0.038 | 0.024 | +0.008 | 0.046 |
| 1.5 | 0.021 | −0.020 | 0.023 | −0.005 | 0.015 |
| 2.0 | 0.028 | −0.014 | 0.024 | −0.021 | 0.008 |
| 2.5 | 0.025 | −0.013 | 0.032 | −0.047 | 0.034 |
| 3.0 | 0.030 | −0.019 | 0.024 | −0.013 | 0.006 |
| 3.5 | 0.026 | −0.006 | 0.028 | −0.015 | 0.009 |
| 4.0 | 0.070 | −0.024 | 0.025 | −0.024 | 0.001 |

- **Every band, both sessions, lands inside the ±0.05 N deadband.** Worst
  |error| 0.047 N; mean signed error −0.018 N.
- Mean SD **0.028 N**, worst 0.070 N. probe.yaml records the PX6D noise floor as
  ±0.05 N per axis — the hold is at the sensor's own resolution, so this number
  is bounded by the instrument, not the controller.
- Session-to-session error agreement ≤ 0.046 N across a recalibration.

The negative mean error is the deadband working as designed, not a tracking
failure: the regulator outputs `v_z = 0` once inside the band and stops. See
`outputs_lap2/hold_by_target.csv` (`error_vs_settling_n`) for the tracking error
measured against the settling point rather than the target.

### The tuning that got here

`B_z` was raised twice, each time against measured contact stiffness `k_c`:

| `B_z` | hold SD | in-band % | open-loop gain \|L\| at the limit-cycle frequency |
|---|---|---|---|
| 1000 | 0.28–0.33 N | 36–46 % | 0.42–0.93 |
| 3000 | 0.18–0.21 N | 33–49 % | 0.84–1.02 |
| **4500** | **0.021–0.070 N** | **85–100 %** | **0.01–0.20** |

At 1000 and 3000 the loop sat on the stability boundary (|L| ≈ 1 at ω ≈ 0.74 rad/s)
and ran a sustained limit cycle: period 8–16 s, ±0.2–0.35 N, with tip travel in
phase. `k_c` rises steeply with indentation (0.28 → 1.8 N/mm from 0.5 to 4 N), which
is why 3000 was enough at 2 N and not at 3.5 N. 4500 removes the oscillation across
the whole range.

---

## 2. Disturbance rejection — NOT demonstrated

Lap 2 ran a placebo arm: every band got the same syringe routine twice, once with
force hold on (`B_`) and once with it off (`C_`, robot frozen in contact, safety
layer still live). Without that arm, "force stayed near target" cannot be
attributed — it may just be what the disturbance does unopposed.

**Figure: `outputs_lap2/fig7_placebo.{png,pdf}`**

Peak |F − target| per disturbance event, 80 events per arm:

| | force hold ON | force hold OFF |
|---|---|---|
| median | 0.744 N | 0.703 N |
| p90 | 1.167 N | 3.008 N |

**Rank-sum two-sided p = 0.11.** Band by band the OFF/ON ratio of medians sits
between 0.72 and 1.07 across 0.5–3.5 N — no consistent direction, let alone an
effect. Turning the controller off did not measurably worsen the force excursion.

### Why, and what it means

The loop time constant is `B_z / k_c` ≈ 4500 / 700 ≈ **6.4 s**. The injections ran
every **4–5 s** (`i2 w6 i12 w18 …` in the event marks). The disturbance is faster
than the loop can act, by the same choice of `B_z` that removed the limit cycle.

This is a real trade, and lap 2 measured both ends of it:

> The controller holds a commanded force to within ±0.05 N across 0.5–4.0 N. Its
> disturbance rejection at 4–5 s forcing is indistinguishable from none.

The right claim to make from this dataset is the first sentence. The second is a
bandwidth statement about this tuning, not a defect in the implementation.

---

## 3. Safety layer — exercised and held

`safety.warn_contact_force_n` 4.5 N, `safety.max_contact_force_n` 5.0 N.

| run | peak | over warn | samples over limit |
|---|---|---|---|
| B_t3p5_r2 | 4.75 N | 2.2 s | 0 |
| C_t3p5_r2 | 4.77 N | 1.0 s | 0 |
| B_t4p0_r2 | 5.38 N | 8.4 s | 1788 |
| C_t4p0_r2 | 5.37 N | 2.0 s | 1453 |

The forced retreat fired at 4.0 N in both arms and capped the force. It also fires
in the control arm by design — "no force control" is not "no safety layer"
(`fr5_ik/test/test_force_hold_toggle.py` pins this).

---

## 4. Known limits of this dataset

Both accepted deliberately; stated so the claims stay inside them.

1. **The 4.0 N band is excluded from the placebo comparison.** Both arms crossed
   the 5.0 N hard limit, so the retreat fired and those events are no longer
   open-loop. `C_t4p0_r2` lost contact outright (mean 1.62 N against a 4.0 N
   target, min −0.32 N). Shown greyed in fig7 rather than deleted. **The
   disturbance-rejection result covers 0.5–3.5 N.**
2. **n = 1 contact point and angle.** `run_repeat.sh` closes each lap by telling
   the operator to move the contact point for the next one. Lap 1's holds are
   split across three `B_z` values, so the usable evidence is lap 2 plus the `1h`
   hold set — both at one placement. No between-placement variance estimate, and
   `k_c` is known to vary with indentation, so it will vary with site too.
3. Lap 1's `B_` disturbance captures were taken at `B_z` 1000/3000 with no control
   arm. They are pilot data and are not pooled with lap 2.

## 5. What would extend it

Not required for the claims above; listed so the next session doesn't rediscover them.

- A slower disturbance (one step over tens of seconds) would sit inside the loop's
  bandwidth and is what the physiological case — breathing, tissue relaxation,
  perfusion — actually looks like. That is the experiment that could turn §2 into a
  positive result.
- Capping the sweep at 3.5 N, or dropping the step volume at 4.0 N, keeps the
  safety layer from contaminating the top band.
- Repeats at other contact sites for between-placement variance.
