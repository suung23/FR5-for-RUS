# Ultrasound probe robot mount v2

Re-design of the upper probe clamp of the FR5 probe mount (`본체3.stl`) for the new probe
(handle 99.01 × 19.14 mm). The lower body (robot-flange disc, mounting holes, tapered column,
cable slot) is the **untouched original mesh**; only the part above z = 79 mm is new.

| File | What |
|---|---|
| `ultrasound_probe_mount_v2.stl` | **print file**, same coordinate frame as the original STL (base centre at x = −165, y = 0) |
| `ultrasound_probe_mount_v2_aligned.stl` | same part, base centre at the origin and clamp walls axis-aligned (handy in the slicer) |
| `probe_mount_v2.py` | editable source (Python + manifold3d CSG). Edit the `P = dict(...)` block, re-run |
| `verify_mount_v2.py` | pre-print checks (watertight, manifold edges, pocket size, wall skins, lower body preserved) |
| `render_stl.py`, `make_figures.py` | renders / section figures |
| `reference/bonche3_original.stl`, `reference/bonche3_aligned.stl` | source STL and its aligned copy |
| `analysis/` | slice tables of the original, `frame.json` (aligned ↔ original transform), `verify_report.txt`, section sheets |
| `renders/` | `v2_front.png`, `v2_side.png`, `v2_iso.png` (+ top / below / comparison with the original) |

Environment note: CadQuery/OCP and VTK are blocked by Smart App Control on this laptop, so the build uses
manifold3d (works) and the renders are a small numpy rasteriser. Fillets are therefore built from 2-D rounded
profiles (gusset root, vertical edges), not 3-D CAD fillets.

## 1. What the original mount is (measured from the STL)

Aligned frame: base centre at the origin, +Z up, X = probe width direction, Y = clamp thickness direction.
In the original file the clamp is rotated 135° about Z and sits at x = −165 (see `analysis/frame.json`).

| Feature | Measured |
|---|---|
| Flange disc | Ø 60.0, z 0–12 |
| Mounting holes | 4 × Ø 5.0 through (z 0–6) with Ø 10.0 × 6 deep counterbore from the top, on PCD 45 (at ±15.9, ±15.9 in the aligned frame = 0°/90°/180°/270° in the original frame) |
| Column | tapered loft z 12–79, Ø 35 round at the bottom → 45 × 33 rectangle (r ≈ 1) at the top |
| Cable slot | 12.0 wide (\|y\| < 6), open on the −X face, runs from the face to x = +6 (round end), full column height |
| Old clamp | z 82–152, two 3 mm "hat" flexure plates forming a 40 × 27 pocket, joined by **4 × M4** (Ø 3.3 tap side / Ø 4.0 clearance side) at x = ±27, z = 112 & 142, screws along Y |
| Volume | 132.6 cm³, watertight |

The old clamp closes by flexing two thin plates toward each other; that mechanism cannot be widened to 99 mm, so
the screw positions were **not reused**. The screw *size* (M4) and *direction* (along Y, through the wide faces) are kept.

## 2. New clamp (all in mm)

| Item | Value |
|---|---|
| Pocket | **99.30 (X) × 20.00 (Y) × 60.00 (Z)**, open at the top, 1.5 × 45° lead-in chamfer at the mouth |
| Pocket floor | z = 84 (5 mm floor on top of the original column top at z = 79) |
| Walls | end walls 5.0 (X); front/back walls 6.5 (Y) so the sleeve is flush with the 33 mm column top |
| Sleeve outer | 109.3 × 33 × 60 (+ 5 mm floor block), vertical edges r 4 |
| Overall part | 109.3 × 60 (flange) × 144 tall; 258 cm³ solid volume |
| Cable | the 12 mm slot continues up through the floor into the pocket (x −23.5 … +6); the −X gusset is split into two bands leaving a 12.4 mm channel so the slot mouth stays open over the whole height |
| Gussets | full-width (33 mm) triangular gussets on both sides from the column face to the sleeve ends, 50° from horizontal, r 5 fillet at the root, root blended into the column taper |
| Screws | **4 × M4**, opposing pairs at x = ±30, z = 114 (30 mm above the floor), entering along Y through the front and back walls |
| Through hole | Ø 4.4 (spec 4.3–4.5) |
| Fastener pocket | captive **M4 hex-nut slot** 7.4 × 3.6, open from the top of the boss, 6.0 mm outside the pocket face (5.9 mm outer skin, 4.5 mm inner skin under the pad recess). Alternative: set `FASTENER='insert'` → Ø 5.7 × 7 pocket for an M4 heat-set insert |
| Bosses | 20 wide × 24 tall × +9 proud of the wall, r 3 vertical edges, 45° chamfer underneath (no support) |
| Pad recess | 16 × 16 × 1.5 (r 2) on the inner face at every screw, for a 15 × 15 × 1.5 TPU / silicone pad between screw tip and probe (`PAD_DEPTH=2.0` if you use 2 mm pads) |
| Hardware | 4 × M4 × 20 (socket cap or thumb screw), 4 × M4 hex nuts (DIN 934, 7 A/F × 3.2) or 4 × M4 heat-set inserts; 4 pads |

Screw length: boss face → pad face is 14 mm; M4 × 16 works, M4 × 20 leaves adjustment travel.
Tighten in two opposing pairs alternately so the handle stays centred (0.43 mm clearance per side).

## 3. Verification (`verify_mount_v2.py`, see `analysis/verify_report.txt`)

All checks pass: watertight, consistent winding, 0 non-manifold edges of 16 245, single body; pocket
min 99.300 × 20.000, depth 60.00 from floor z = 84.00 to rim z = 144.00; solid skins on both sides of every hole /
nut slot / pad recess and the boss outer face unbroken; flange disc radius and the four holes identical to the
original at z = 3 and z = 9 (Ø 5.0 / Ø 9.99); no original material removed at z = 20 / 35 / 60 / 78 (0.0 mm²); cable
slot, floor opening and gusset channel open.

## 4. Printing

* **Orientation:** as modelled, flange disc flat on the bed, pocket opening up. Every overhang is ≥ 50° from
  horizontal (gussets 50°, boss undersides 45°); the only bridges are the 12.4 mm cable-channel roof under the floor
  and the 2.5 mm counterbore steps – no supports needed. Nut slots are vertical, screw holes are horizontal Ø 4.4.
* **Material:** PETG or ASA (ASA preferred if the robot cell gets warm; PETG is fine indoors). Avoid PLA (creep under
  clamping load).
* **Settings:** 0.2 mm layers, 4–5 perimeters, 5 top/bottom layers, 30–40 % gyroid/cubic infill (the gussets and
  bosses are solid in the STL; infill keeps the printed weight around 170–200 g). Print the pads in TPU 95A or cut
  from 1.5 mm silicone sheet.
* **Post-print:** run an M4 tap or bolt through the four Ø 4.4 holes, drop the nuts into the slots from the top,
  stick the pads into the recesses, then insert the probe from the top and tighten the four screws alternately.
* The robot-flange screws sit in the Ø 10 counterbores exactly as before; access is unchanged from the original
  (the column shadows the counterbores above z ≈ 40 in both versions), so use the same angled / ball-end key as today.

## 5. Open points to confirm on the real probe

1. Cable exit: the floor opening reproduces the original 12 mm slot (x −23.5 … +6). If the new probe's strain relief
   is thicker than 12 mm or exits off-centre, change `SLOT_W` / `SLOT_END_X` (the column slot below is fixed at 12 mm).
2. Handle taper: the pocket is a straight 99.30 × 20.00 prism over the 60 mm grip length; if the handle tapers, the
   pads take up the difference, or lower `POCKET_H`.
3. If you want the screws at two heights instead of one row, change `SCREW_Z_FROM_FLOOR` into a list in the source
   (one boss per height); the current row at mid-grip was chosen for symmetric clamping.
