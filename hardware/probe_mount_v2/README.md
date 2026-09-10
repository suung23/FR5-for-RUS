# Ultrasound probe robot mount v2

The original FR5 probe mount (`본체3.stl`) with its clamp pocket resized for the new probe handle
(**42.5 wide × 19.05 thick**, measured 2026-09-10). Everything else is the original idea: two 3 mm
flexure plates whose end flanges are pulled together by four M4 screws so the plates grip the handle
directly. No pads, no pressure screws.

| File | What |
|---|---|
| `ultrasound_probe_mount_v2.stl` | **print file**, same coordinate frame as the original STL (base centre at x = −165, y = 0) |
| `ultrasound_probe_mount_v2_aligned.stl` | same part, base centre at the origin and clamp walls axis-aligned (handy in the slicer) |
| `probe_mount_v2.py` | editable source (Python + manifold3d). Edit the `P = dict(...)` block, re-run |
| `verify_mount_v2.py` | pre-print checks; report in `analysis/verify_report.txt` |
| `render_stl.py`, `make_figures.py` | renders and section figures |
| `reference/` | source STL (`bonche3_original.stl`) and its axis-aligned copy |
| `analysis/` | slice tables of the original, `frame.json` (aligned ↔ original transform), section sheets, `v2_dimensions.png` |
| `renders/` | `v2_front.png`, `v2_side.png`, `v2_iso.png`, top / below views, `compare_original_vs_v2.png` |
| `_obsolete_99mm_pressure_screws/` | earlier build for the wrong 99 mm handle width – do not print |

Tooling note: CadQuery/OCP and VTK are blocked by Smart App Control on this laptop, so the part is built with
manifold3d (CSG) + trimesh and rendered with a small numpy rasteriser. Rounds are built into the 2-D plate profile.

## 1. Original mount, as measured from the STL

Aligned frame: base centre at the origin, +Z up, X = probe width, Y = clamp thickness (screw direction).
In the original file the clamp is rotated 135° about Z and sits at x = −165 (transform in `analysis/frame.json`).

| Feature | Measured |
|---|---|
| Flange disc | Ø 60.0 × 12, 4 × Ø 5.0 through + Ø 10.0 × 6 counterbore from the top, PCD 45 |
| Column | tapered z 12–79, Ø 35 round at the bottom → 45 × 33 (r 1) at the top |
| Cable slot | 12 wide (\|y\| < 6), open on the −X face, to x = +6, full column height, open into the pocket floor |
| Root | z 79–82 transition from the column top to the plate outline; flange gap closes in a 45° V over 3 mm |
| Plates | two 3 mm hat plates z 82–152, pocket 39 × 27, side wall → flange round R3, flange 9 long, gap 6 |
| Screws | 4 × M4 at x = ±27 (4.5 outboard of the side wall), z = 112 & 142; Ø 4.0 in the −Y flange, Ø 3.3 thread-forming in the +Y flange |

## 2. What changed (all in mm)

| Item | Original | v2 |
|---|---|---|
| Pocket W × T | 39 × 27 | **42.80 × 19.85** (= 42.5 + 0.30 fit, 19.05 + 0.80 closed by the screws) |
| Plate height / pocket depth | 70 (z 82–152) | **60** (z 82–142) – `POCKET_H` |
| Clamp outer W × T | 63 × 33 | 66.8 × 25.9 |
| Screw positions | x = ±27, z = 112 / 142 | x = ±28.9, z = 102 / 132 (20 and 50 above the floor, top one 10 below the rim) |
| Pocket top corners | R3 | R1 (so a sharp-cornered handle still seats) |
| Plate-end underside | 6 mm flat overhang | 45° chamfer from the column edge (no support needed) |
| Root | column top → 33 wide plates | column top tapers 16.5 → 12.95 in Y over the 3 mm root |
| Volume | 132.6 cm³ | 125.2 cm³ |

Unchanged: plate thickness 3, flange 3 × 9, gap 6, R3 rounds, root V, screw sizes, everything below z = 79
(the original mesh itself), the cable slot (continues through the root block into the pocket floor).

Hardware: 4 × M4 × 16 (through 3 + 6 + 3 = 12 mm; original hardware fits). Screw heads on the −Y side, threads
form into the Ø 3.3 holes of the +Y flange as before. Tighten the four screws evenly; the plates need to close
only 0.4 mm per side.

## 3. Verification (`verify_mount_v2.py` – all pass)

Watertight, consistent winding, 0 non-manifold edges of 15 735, single body. Pocket min 42.800 × 19.850,
depth 60.00 (floor z = 82.00, rim z = 142.00). Top plate 3.00, flange 2.99, gap 6.00. Four holes Ø 4.00 / Ø 3.30
with material beside them, side walls and top plates intact at the screw heights, chamfered undersides open.
Flange disc radius and the 4 holes identical to the original at z = 3 and 9; column sections at z = 20 / 35 / 60 / 78
identical (0.00 mm² removed or added). Cable slot open from the column through the root into the pocket.

## 4. Printing

* Orientation as modelled: flange disc on the bed, pocket opening up (same as the original).
* No supports: the plate-end undersides are 45° chamfers, the root taper is 52°, the only bridges are the
  2.5 mm counterbore steps and the 12 mm slot roof at the pocket floor.
* Material: PETG or ASA (ASA if the cell gets warm). Avoid PLA – the flexure plates creep under clamping load.
* 0.2 mm layers, 3–4 perimeters (the plates are 3 mm = fully solid walls), 5 top/bottom layers, 30 % infill in
  the column. Printed weight ≈ 120–140 g.
* After printing, run an M4 screw into the four Ø 3.3 holes once before fitting the probe.

## 5. Parameters worth knowing

`PROBE_W`, `PROBE_T` (measured handle), `CLR_W`, `CLR_T` (clearances), `POCKET_H` (set 70 to restore the original
height), `R_TOP_IN` (pocket corner radius), `SCREW_Z_FROM_FLOOR`, `FLANGE_L`, `PLATE_T`. The script asserts the pocket
stays narrower than the 45 mm column top so the floor is always supported.
