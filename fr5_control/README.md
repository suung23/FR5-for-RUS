# fr5_control

ROS 2 nodes for the FR5 right-arm ultrasound probing cell: force/torque
acquisition, contact staging, and the PX6D calibration workflow.

## PX6D force/torque calibration

The PX6D sits **between the FR5 flange and the probe mount**, so what it measures
is not the force at the probe tip. Two things stand between the two: a fixed
rotation, and the weight of everything hanging below the sensor. Both have to be
removed before the reading means anything.

### Frames

| Frame | What it is |
|---|---|
| `{B}` | FR5 base |
| `{F}` | FR5 flange (J6) |
| `{S}` | PX6D measurement frame, manufacturer's axis convention |
| `{M}` | probe mount |
| `{P}` | probe control frame, at the acoustic/contact reference point |

The controller operates in `{P}`:

```
+z_P   probe axial — surface normal, the compression direction
+x_P   lateral, in the ultrasound image plane
+y_P   elevational, normal to the image plane
```

`F_normal = F_zP`, positive is compression.

### Mounting rotation

The sensor is **not** aligned with the image or base axes. As installed, `+y_S`
points about +45° from the horizontal reference and `+x_S` about −45°. Those two
are orthogonal and share `z`, so the registration is a single rotation about z:

```
{P}R{S} = Rz(-θ),  θ = mounting angle, default 45°
```

Raw sensor channels are never used as probe-frame channels. The angle is a
calibration parameter with a field on the Sensor calibration page — set it from
the physical measurement, not from an assumption.

**The sign convention lives in that matrix and nowhere else.** An assembly
mounted the other way up is expressed as `axial_flip`, which composes `Rx(180°)`
— a proper rotation, so force and moment flip together. A scalar sign would have
been applied to the force and forgotten on the moment.

### Calibration sequence

Two procedures, deliberately separate:

1. **Electronic zero.** Probe assembly attached, touching nothing, robot still,
   3–5 s of samples. Rejected if the variance is too high — a noisy capture means
   something *is* touching, and averaging that would bake the contact into the
   zero.

2. **Multi-pose gravity.** At least 12 static poses that point gravity in
   substantially different directions in `{S}`. Solves payload mass, centre of
   mass, and residual bias by robust least squares.

The zero alone is **not enough**: it is only correct at the pose it was taken.
Tip the flange 90° and an uncompensated reading is off by several newtons.

**The robot is never moved by the calibration workflow.** The operator moves it
to each pose by hand; the bridge accepts only capture, fit, save, and reset, and
rejects anything else.

### Runtime

```
{S}w_ext = {S}w_raw − {S}b − {S}w_g(q)
{P}w_ext = blkdiag({P}R{S}, {P}R{S}) · {S}w_ext
{P}τ_contact = {P}τ_ext − {P}r_{S→P} × {P}f_ext
```

Gravity is removed **in the sensor frame**, because that is where the mass and
COM were identified. Removing it after rotating into `{P}` mixes the rotation in
and gives the wrong answer.

The lever term matters more than it looks: with `r ≈ 0.1 m` and `F_z ≈ 5 N` the
offset is 0.5 N·m, which buries an alignment signal of tens of mN·m.

### Gating

Contact control does not open unless the calibration is valid. The bridge
publishes `/fr5_right/calibration_valid` and `us_diff_ik_node` releases the force
axis only on a true — a profile that is missing, rejected, incomplete, or more
than a day old counts as invalid, and the reason is shown rather than the
calibration being silently used.

### Where it stands

Measured 2026-08-27, 19 poses:

| | |
|---|---|
| Payload | 200 g |
| Centre of mass in `{S}` | −2.0, −1.4, 54.2 mm |
| Sensor alignment about the stack axis | 136.25° |
| Fit residual | 0.152 N force, 0.0023 N·m torque |
| **Unloaded probe, three orientations** | **0.235 N worst case** (10.53 N uncompensated) |

The alignment is solved from the data rather than assumed. The model is

    f = m · ({S}R{F} · g_F) + b

which is linear in `A = m·{S}R{F}`; solve for A, then split it into a mass and
a rotation by SVD. Assuming the rotation instead produced a **negative mass**
on real data — the magnitude was right and the direction 60° wrong, and that is
the shape such an error takes.

The rotation is constrained to the stack axis. That is not an extra assumption:
axial compression lands cleanly on Fz (separation 4.9), which is what "coaxial"
means. It matters because the payload is light — 200 g of gravity signal under
an 8.5 N sensor offset — and three free rotational degrees of freedom do not
survive that ratio. Constraining to one took the residual from 0.183 to
0.047 N on the same poses.

### Limitations

- **Phantom-only research use.** Nothing here is validated for patient contact.
- The force limits were raised to 9/10 N when the contact-probing transition was
  set at 8 N. **The original 7 N had a tissue-safety rationale (§2.1) and the
  raise does not yet have one** — revisit before phantom work.
- The hold force of 5 N is a placeholder for the Stage 1 force search.
- **Compensation leaves up to 0.235 N.** Against a hold band of 5.0 ± 0.5 N that
  is 47% of the band, so contact force should be read as 4.5–5.5 N rather than
  4.8–5.2 N. The limit is the signal ratio — 200 g of payload under an 8.5 N
  offset — not the number of poses; adding six poses moved the alignment spread
  but not the residual.
- Verification is operator-led: the tool names a pose and waits for a keypress.
  Three attempts to have it decide when the arm was still were all wrong, in
  the same way each time — a joint that has stopped moving is not the same as a
  hand that has let go, and that difference is not visible in the telemetry.
- `tool.j6_to_probe` is still unmeasured, so `r_{S→P}` must be entered from the
  mount drawing before the moment channels mean anything.
- The sensor's own axis assignment is provisional; `px6d_axis_id` confirms it.
