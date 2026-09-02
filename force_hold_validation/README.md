# Constant-force hold validation against a volume-changing phantom

Measures how well `us_diff_ik`'s force loop holds a commanded contact force, and
whether it stays under the safety limit, using a phantom whose abdominal volume
is changed with a 500 mL syringe.

Three questions, three run types:

| Run | Asks | Protocol |
|---|---|---|
| **A · hold** | Which force band does it regulate best? | Settle on the phantom, hold still, record 60 s. |
| **B · disturbance** | Does it hold against the surface moving? | Step **150 mL** in and out while recording, three of each. |
| **C · safety** | Does it ever cross the limit? | Drive the force up deliberately and check the forced retreat at 5 N. |

**A and B run at every target, back to back.** How a loop rejects a surface that
moves is a property of the band it is holding — reading it at 3 N says nothing
about 0.5 N. Taking both captures before changing the target also removes the
commonest way this goes wrong: a capture recorded against the previous setpoint
because the `param set` was missed.

The capture step **reads the existing control stack and nothing else**. It
subscribes to the wrench the robot is actually regulating on and to the mode the
robot declares; it does not re-derive either.

**Everything here is judged on the contact force `‖F‖`, not the normal
component.** That is the scalar `us_diff_ik` acts on — `ft_sensor.contact_force_mode`
is `magnitude`, `_control_force()` returns `±‖F‖` signed by the normal
component, and `ForceRegulator.update` compares that one number against the
target, the deadband and the limit. The safety parameters say so in their names
as of 2026-09-02: `safety.max_contact_force_n` and `warn_contact_force_n`, both
applied to `‖F‖`. They used to read `*_normal_force_n`, which had never been
what they governed.

The normal component is recorded beside it in `fn_n` as evidence, never used in
a metric. A contact carrying shear separates the two, and that separation is
worth being able to look up afterwards.

```
operator ──stylus──┐
                   ├─ us_diff_ik ── v_z ── robot ── probe ── phantom
syringe ──volume───┘        │                                   │
                            └──────── wrench_px6d ──────────────┘
                                          │
                                   what this tool records
```

## ⚠️ Read this before running: the sweep needs a parameter set

**The deployed configuration cannot reach targets below 2 N.** Contact probing
engages at `‖F‖ ≥ teleop.contact_probing_force_n` (2.0 N today), so the robot
does not start regulating until 2 N and can never settle at 0.5. Two more
constraints bind:

- `target − deadband > teleop.contact_probing_release_n`, or the mode drops back
  to approach while the force is being held correctly (`us_diff_ik` warns at
  startup when this is violated).
- `target < warn ≤ max`, enforced in `ForceRegulator.__init__`.

Set three parameters once. After that **only `target_force_n` changes** between
runs:

```bash
ros2 param set /us_diff_ik_node teleop.contact_probing_force_n 0.40
ros2 param set /us_diff_ik_node teleop.contact_probing_release_n 0.20
ros2 param set /us_diff_ik_node contact_control.deadband_n 0.15
```

| target | band | enter | release | invariants |
|---|---|---|---|---|
| 0.5 N | 0.35 – 0.65 | 0.40 | 0.20 | ok |
| 1.0 N | 0.85 – 1.15 | 0.40 | 0.20 | ok |
| 1.5 N | 1.35 – 1.65 | 0.40 | 0.20 | ok |
| 2.0 N | 1.85 – 2.15 | 0.40 | 0.20 | ok |
| 3.0 N | 2.85 – 3.15 | 0.40 | 0.20 | ok |
| 4.0 N | 3.85 – 4.15 | 0.40 | 0.20 | last one — 0.35 N under the warning |

Four invariants bind and all four hold at every target: `release < enter`,
`enter < target`, `target − band > release`, `target < warn ≤ max`. 4.5 N fails
the last, which is where the sweep stops. Raise `warn_contact_force_n` and
`max_contact_force_n` together to go higher, and say in the report that you did.

**The entry threshold is fixed, and that is what makes the bands comparable.**
Contact probing engages at the same 0.4 N in every run, so every run hands the
regulator the same entry transient and the same approach history. An entry that
tracked the target would give each band a different overshoot on the way in —
the arm is still at approach speed when the switch fires — and the sweep would
read that difference as if it were a difference in holding.

The deadband is held constant for the same reason. The question is which
absolute force the loop holds best, and a band that grew with the target would
flatter the high end by widening the target it has to hit.

**The regulator drives the last stretch itself.** Entry at 0.4 N is below every
target, so after the switch fires the loop closes the remaining gap on its own —
2.5 mm/s at the 3 N band, slowing as it loads. The first 3 s after entry are
discarded from the holding metrics for that reason; fig1 keeps them visible.

**The release stays at 0.20 N for every target.** Holding is what is being
compared, and a release that moved with the target would give each band a
different hysteresis. One consequence is worth predicting: at 0.5 N a full
150 mL withdrawal will probably take the force through 0.20 N and end contact
probing altogether. That is a finding about the low end, not a setup error — the
analysis will show the mode leaving `contact_probing` and the event marked as
not recovered.

**These are experiment settings, not deployment settings.** A 0.4 N entry is
only defensible on a bench where the pose does not change: it sits just above
what a fresh working tare leaves, and **below** the 0.34 N mean residual the
compensation carries across poses. Put `probe.yaml` back afterwards.

## ⚠️ And take a working zero first, at the test pose

Below about 1 N the measurement is dominated by compensation error rather than
control error. Across the nineteen calibration poses the deployed no-contact
residual is 0.34 N mean and 0.61 N max — comparable to the whole 0.5 N target.
A working tare taken at the pose you are about to test removes that at that
pose, and it is the only thing that makes the low end of the sweep mean
anything.

Press **Zero** on the console with the probe clear of the phantom, in the pose
you will run from, before every session. The event log says whether it was
taken or refused.

## 1. Capture

```bash
source ~/FR5-for-RUS/install/setup.bash

# per target: one param set, then both captures
ros2 param set /us_diff_ik_node contact_control.target_force_n 1.0
python3 capture_force_hold.py --label A_t1p0 --run-type hold --seconds 60
python3 capture_force_hold.py --label B_t1p0 --run-type disturbance --step-ml 150

# once, at the end
python3 capture_force_hold.py --label C_limit --run-type safety
```

Thirteen runs: six targets × (hold + disturbance), plus one safety run.

Records every wrench sample the control stack sees, at sensor rate, with the
mode the robot declares beside it. Keys during a run:

| key | marks |
|---|---|
| `i` | syringe **in** — volume rising, surface lifting into the probe |
| `w` | syringe **withdrawn** — volume falling, surface dropping away |
| `s` | settled — the disturbance has finished moving |
| `space` | generic mark |
| `u` | undo the last mark |
| `q` | save and stop |

The commanded parameters are read from the running `us_diff_ik_node` and stored with
the samples, so the analysis knows what the loop was asked to do rather than
what a config file said hours earlier.

**Refusals are recorded, not hidden.** A run that never reached contact probing,
or whose calibration went invalid partway, is written with that stated and the
analysis excludes it by name.

## 2. Analyse

```bash
python3 run_force_hold_analysis.py --runs runs --output-dir outputs
```

Outputs:

```
outputs/
├── hold_by_target.csv            per-target holding metrics
├── disturbance_events.csv        one row per syringe step
├── safety_margin.csv             worst force seen against the limits
├── analysis_report.md
├── fig1_force_traces.png/.pdf    force vs time per target, band and limits drawn
├── fig2_hold_quality.png/.pdf    error distribution and time-in-band per target
├── fig3_disturbance.png/.pdf     responses aligned on the step
└── fig4_safety_margin.png/.pdf   worst force per target against warn and limit
```

## What this tool will not do

**It does not smooth the force before measuring it.** The regulator acts on
every sample; an analysis that judges it on a filtered signal would report a
loop that does not exist.

**It does not drop a run silently.** Every run in `runs/` appears in the report,
with `included` and, when excluded, the reason.

**It does not invent the disturbance size.** `--step-ml` is what you tell it you
injected. If you varied it, run separate captures rather than averaging over a
number that was never true.

## Limitations

Phantom mechanics are not tissue mechanics: a water-filled abdominal phantom is
far more linear and far less viscoelastic than the abdomen, so settling times
here are a lower bound on what tissue will give.

The limit is now **5 N on `‖F‖`**, chosen so that it is reachable: at 15 N a
soft phantom cannot be pressed hard enough to cross it, and a limit that is
never approached yields "we did not exceed it" rather than "it holds". Testing
the mechanism at a reachable limit is evidence; not testing it because the real
one is out of reach is not.

The invariant `target < warn ≤ max` binds — 3.0 < 4.5 ≤ 5.0 today, and
`ForceRegulator.__init__` refuses any combination that breaks it. **This bounds
the top of the sweep**: a 4 N target leaves its band's top 0.35 N under the
warning, and above that the regulator is not allowed to advance at all. Stop the
sweep there, or raise `warn_contact_force_n` and `max_contact_force_n` together
and say in the report that you did.
