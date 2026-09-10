"""Pre-print verification of ultrasound_probe_mount_v2 (run after probe_mount_v2.py).

Checks (all in the aligned frame):
 1. watertight, consistent winding, no non-manifold edges (every edge shared by exactly 2 faces)
 2. pocket >= POCKET_W x POCKET_T x POCKET_H, open at the top
 3. screw holes / nut slots / pad recesses stay inside the walls (solid skins on both sides)
 4. lower flange disc + 4 counterbored holes + column identical to the original (section compare)
 5. cable slot / floor opening / gusset channel are open
Exit code 1 if any check fails.  Writes analysis/verify_report.txt
"""
import json, sys
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon, Point

ROOT = Path(__file__).resolve().parent
P = json.load(open(ROOT / 'build/params.json')); D = P['derived']
new = trimesh.load(ROOT / 'ultrasound_probe_mount_v2_aligned.stl', force='mesh')
orig = trimesh.load(ROOT / 'reference/bonche3_aligned.stl', force='mesh')
report, ok_all = [], True


def check(name, ok, detail=''):
    global ok_all
    ok_all &= bool(ok)
    report.append(f'[{"PASS" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))


def polys(mesh, axis, v):
    normal = np.zeros(3); normal[axis] = 1; origin = np.zeros(3); origin[axis] = v
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return []
    p2, T = sec.to_2D(); keep = [i for i in range(3) if i != axis]
    out = []
    for poly in p2.polygons_full:
        f = lambda c: trimesh.transform_points(np.column_stack([np.asarray(c), np.zeros(len(c))]), T)[:, keep]
        out.append(Polygon(f(poly.exterior.coords), [f(r.coords) for r in poly.interiors]))
    return out


# ---------------------------------------------------------------- 1. mesh integrity
check('watertight', new.is_watertight)
check('winding consistent', new.is_winding_consistent)
edges = new.edges_sorted
_, counts = np.unique(edges, axis=0, return_counts=True)
check('no non-manifold edges (every edge has exactly 2 faces)', bool((counts == 2).all()),
      f'{int((counts != 2).sum())} bad edges of {len(counts)}')
check('single connected body', len(new.split(only_watertight=False)) == 1)
check('positive volume', new.volume > 0, f'{new.volume:.0f} mm3')

# ---------------------------------------------------------------- 2. pocket size
zf, zt = D['Z_FLOOR'], D['Z_TOP']
inner_min = None
for z in np.linspace(zf + 0.5, zt - P['LEAD_IN'] - 0.5, 9):
    if abs(z - D['SCREW_Z']) < P['HOLE_D'] / 2 + 0.5:
        continue                                         # plane through the screw holes: cavity joins the outside
    P_ = polys(new, 2, z)
    holes = [Polygon(r) for p in P_ for r in p.interiors]
    holes = [h for h in holes if h.area > 500]          # the pocket
    if len(holes) != 1:
        check(f'pocket section at z={z:.1f} is a single cavity', False, f'{len(holes)} cavities'); continue
    b = holes[0].bounds; w, t = b[2] - b[0], b[3] - b[1]
    inner_min = (w, t) if inner_min is None else (min(inner_min[0], w), min(inner_min[1], t))
check('pocket width >= %.2f' % P['POCKET_W'], inner_min[0] >= P['POCKET_W'] - 1e-3, f'min {inner_min[0]:.3f}')
check('pocket thickness >= %.2f' % P['POCKET_T'], inner_min[1] >= P['POCKET_T'] - 1e-3, f'min {inner_min[1]:.3f}')
# floor height: ray straight down from above at a point away from the cable opening
hit, _, _ = new.ray.intersects_location([[35.0, 4.0, zt + 20]], [[0, 0, -1]], multiple_hits=True)
floor_z = hit[:, 2].max()
check('pocket depth >= %.1f (open top)' % P['POCKET_H'], (new.bounds[1][2] - floor_z) >= P['POCKET_H'] - 1e-3,
      f'floor z={floor_z:.2f}, rim z={new.bounds[1][2]:.2f}, depth {new.bounds[1][2] - floor_z:.2f}')

# ---------------------------------------------------------------- 3. walls around fasteners
sx, sz = P['SCREW_X'], D['SCREW_Z']; y_in = P['POCKET_T'] / 2; y_out = P['BODY_Y'] / 2 + P['BOSS_OUT']
solid_pts, void_pts = [], []
for x in (sx, -sx):
    for s in (+1, -1):
        y_slot = y_in + P['NUT_INNER_GAP']; t = P['NUT_T'] + P['NUT_CLR']
        # solid: inner skin (between pad recess and nut slot), outer skin (between nut slot and boss face), beside the hole
        solid_pts += [(x, s * (y_in + P['PAD_DEPTH'] + (y_slot - y_in - P['PAD_DEPTH']) / 2), sz + 4.0),
                      (x, s * (y_slot + t + (y_out - y_slot - t) / 2), sz + 4.0),
                      (x + 5.5, s * (y_in + 3.0), sz), (x, s * (y_in + 3.0), sz + 5.5)]
        # void: hole axis in the wall, nut slot centre, pad recess
        void_pts += [(x, s * (y_in + 3.0), sz), (x, s * (y_slot + t / 2), sz), (x, s * (y_in + P['PAD_DEPTH'] / 2), sz + 6.0)]
inside = new.contains(np.array(solid_pts + void_pts))
check('fastener region: skins are solid', bool(inside[:len(solid_pts)].all()),
      f'{int(inside[:len(solid_pts)].sum())}/{len(solid_pts)} solid probes inside')
check('fastener region: hole / nut slot / pad recess are open', bool((~inside[len(solid_pts):]).all()),
      f'{int((~inside[len(solid_pts):]).sum())}/{len(void_pts)} void probes outside')
# section at screw height: the boss outline must be unbroken (min |y| of the exterior near the screw = boss face)
P_ = polys(new, 2, sz); ext = np.array(max(P_, key=lambda p: p.area).exterior.coords)
near = ext[np.abs(np.abs(ext[:, 0]) - sx) < 2.5]
check('nut slot does not break the boss outer face', near[:, 1].max() > y_out - 0.05 and near[:, 1].min() < -y_out + 0.05,
      f'exterior |y| at screws: {np.abs(near[:, 1]).min():.2f}..{np.abs(near[:, 1]).max():.2f}')

# ---------------------------------------------------------------- 4. lower body preserved
for z in (3.0, 9.0):
    a = max(polys(new, 2, z), key=lambda p: p.area); b = max(polys(orig, 2, z), key=lambda p: p.area)
    ra = np.linalg.norm(np.array(a.exterior.coords), axis=1).mean(); rb = np.linalg.norm(np.array(b.exterior.coords), axis=1).mean()
    ha = sorted((np.array(r.coords).mean(0).round(1).tolist(), round(np.ptp(np.array(r.coords)[:, 0]), 2)) for r in a.interiors)
    hb = sorted((np.array(r.coords).mean(0).round(1).tolist(), round(np.ptp(np.array(r.coords)[:, 0]), 2)) for r in b.interiors)
    check(f'flange disc at z={z}: radius & 4 holes identical to original', abs(ra - rb) < 0.02 and len(ha) == 4 and
          all(abs(x[1] - y[1]) < 0.05 and np.hypot(x[0][0] - y[0][0], x[0][1] - y[0][1]) < 0.2 for x, y in zip(ha, hb)),
          f'r={ra:.2f} vs {rb:.2f}; holes {[h[1] for h in ha]} vs {[h[1] for h in hb]}')
for z in (20.0, 35.0, 60.0, 78.0):
    a = max(polys(new, 2, z), key=lambda p: p.area); b = max(polys(orig, 2, z), key=lambda p: p.area)
    removed = b.difference(a).area; added = a.difference(b).area
    check(f'column section at z={z}: no original material removed', removed < 3.0,
          f'removed {removed:.1f} mm2, added by gussets {added:.0f} mm2 (orig {b.area:.0f})')

# ---------------------------------------------------------------- 5. cable path open
open_pts = [(-10.0, 0.0, 50.0), (-19.0, 0.0, 30.0), (0.0, 0.0, 81.5), (-20.0, 0.0, 83.0), (-40.0, 0.0, 70.0), (-30.0, 0.0, 78.0)]
check('cable slot / floor opening / gusset channel are open', bool((~new.contains(np.array(open_pts))).all()))

# ---------------------------------------------------------------- summary
mass_petg = new.volume / 1000 * 1.27; mass_asa = new.volume / 1000 * 1.07
report.append(f'volume {new.volume / 1000:.1f} cm3 -> solid-mass estimate PETG {mass_petg:.0f} g / ASA {mass_asa:.0f} g '
              f'(printed weight lower with infill)')
report.append(f'bounds (aligned) {np.round(new.bounds, 2).tolist()}  faces {len(new.faces)}')
txt = '\n'.join(report); print(txt)
(ROOT / 'analysis/verify_report.txt').write_text(txt, encoding='utf-8')
print('\nALL CHECKS PASSED' if ok_all else '\nSOME CHECKS FAILED')
sys.exit(0 if ok_all else 1)
