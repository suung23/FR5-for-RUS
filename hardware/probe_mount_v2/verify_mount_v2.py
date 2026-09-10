"""Pre-print verification of ultrasound_probe_mount_v2 (flexure-plate clamp). Run after probe_mount_v2.py.

 1. watertight, consistent winding, no non-manifold edges, single body
 2. pocket >= POCKET_W x POCKET_T x POCKET_H (ray-measured), open at the top
 3. plate / flange geometry: plate thickness, flange gap, 4 screw holes through the flanges only
 4. lower body preserved (flange disc + 4 holes identical, no column material removed)
 5. cable slot open from the column up into the pocket floor
Exit 1 on failure.  Writes analysis/verify_report.txt
"""
import json, sys
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon

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


def ray_dist(origin, direction):
    loc, _, _ = new.ray.intersects_location([origin], [direction], multiple_hits=False)
    return np.linalg.norm(loc[0] - np.array(origin)) if len(loc) else np.inf


XI, XO, XE, YI, YO, YF_IN, YF_OUT, XB = (D[k] for k in ('XI', 'XO', 'XE', 'YI', 'YO', 'YF_IN', 'YF_OUT', 'XB'))
PW, PT = D['POCKET_W'], D['POCKET_T']
zf, zt = D['Z_FLOOR'], D['Z_TOP']

# ---------------------------------------------------------------- 1. mesh integrity
check('watertight', new.is_watertight)
check('winding consistent', new.is_winding_consistent)
_, counts = np.unique(new.edges_sorted, axis=0, return_counts=True)
check('no non-manifold edges (every edge has exactly 2 faces)', bool((counts == 2).all()), f'{int((counts != 2).sum())} bad of {len(counts)}')
check('single connected body', len(new.split(only_watertight=False)) == 1)

# ---------------------------------------------------------------- 2. pocket (ray-measured from the pocket centre line)
wmin, tmin = np.inf, np.inf
for z in np.linspace(zf + 0.5, zt - P['LEAD_IN'] - 0.5, 13):
    w = ray_dist((0, 0, z), (1, 0, 0)) + ray_dist((0, 0, z), (-1, 0, 0))
    t = ray_dist((0, 4.0, z), (0, 1, 0)) + ray_dist((0, -4.0, z), (0, -1, 0)) + 8.0     # start outside the flange-gap plane
    wmin, tmin = min(wmin, w), min(tmin, t)
check('pocket width >= %.2f' % PW, wmin >= PW - 1e-3, f'min {wmin:.3f}')
check('pocket thickness >= %.2f' % PT, tmin >= PT - 1e-3, f'min {tmin:.3f}')
floor_z = zt + 20 - ray_dist((XI - 4.0, 4.0, zt + 20), (0, 0, -1))
check('pocket depth >= %.1f (open top)' % P['POCKET_H'], (new.bounds[1][2] - floor_z) >= P['POCKET_H'] - 1e-3,
      f'floor z={floor_z:.2f}, rim z={new.bounds[1][2]:.2f}, depth {new.bounds[1][2] - floor_z:.2f}')

# ---------------------------------------------------------------- 3. plates and flanges
zmid = (zf + zt) / 2
sec = polys(new, 2, zmid)
big = max([p for p in sec if p.centroid.y > 0], key=lambda p: p.area)     # the +Y hat plate
ext = np.array(big.exterior.coords)
top_out = ext[np.abs(ext[:, 0]) < XI - 3][:, 1].max()
check('top plate thickness = %.1f' % P['PLATE_T'], abs((top_out - YI) - P['PLATE_T']) < 0.05, f'{top_out - YI:.2f}')
gap = ray_dist((XB, 0, zmid), (0, 1, 0)) + ray_dist((XB, 0, zmid), (0, -1, 0))
check('flange gap = %.1f' % P['FLANGE_GAP'], abs(gap - P['FLANGE_GAP']) < 0.05, f'{gap:.2f}')
fl = ray_dist((XB + 2.5, YF_IN + 0.01, zmid), (0, 1, 0))
check('flange thickness = %.1f' % P['PLATE_T'], abs(fl - P['PLATE_T']) < 0.05, f'{fl:.2f}')
holes_ok, det = True, []
for x in (XB, -XB):
    for z in D['SCREW_Z']:
        d_clr = ray_dist((x, -(YF_IN + 1.5), z), (1, 0, 0)) + ray_dist((x, -(YF_IN + 1.5), z), (-1, 0, 0))
        d_tap = ray_dist((x, +(YF_IN + 1.5), z), (1, 0, 0)) + ray_dist((x, +(YF_IN + 1.5), z), (-1, 0, 0))
        beside = new.contains([[x + 2.6, -(YF_IN + 1.5), z], [x - 2.6, (YF_IN + 1.5), z], [x, (YF_IN + 1.5), z + 2.6]])
        ok = abs(d_clr - P['HOLE_CLR']) < 0.06 and abs(d_tap - P['HOLE_TAP']) < 0.06 and beside.all()
        holes_ok &= ok; det.append(f'x={x:+.2f} z={z:.0f}: clr {d_clr:.2f} tap {d_tap:.2f}')
check('4 flange screw holes: O%.1f clearance (-Y) / O%.1f tap (+Y), material beside them' % (P['HOLE_CLR'], P['HOLE_TAP']), holes_ok, '; '.join(det))
side_ok = all(new.contains([[XI + 1.5, 8.0, z], [-(XI + 1.5), -8.0, z], [0.0, YI + 1.5, z], [0.0, -(YI + 1.5), z]]).all() for z in D['SCREW_Z'])
check('side walls / top plates intact at the screw heights', side_ok)
# the plate ends overhang the column: their underside must be chamfered (no flat overhang)
under = new.contains([[XE - 1.0, -(YF_IN + 1.5), zf + 0.5], [-(XE - 1.0), (YF_IN + 1.5), zf + 0.5]])
check('overhanging plate ends chamfered (void just above the floor level under the flange ends)', bool((~under).all()))

# ---------------------------------------------------------------- 4. lower body preserved
for z in (3.0, 9.0):
    a = max(polys(new, 2, z), key=lambda p: p.area); b = max(polys(orig, 2, z), key=lambda p: p.area)
    ra = np.linalg.norm(np.array(a.exterior.coords), axis=1).mean(); rb = np.linalg.norm(np.array(b.exterior.coords), axis=1).mean()
    ha = sorted((np.array(r.coords).mean(0).round(1).tolist(), round(float(np.ptp(np.array(r.coords)[:, 0])), 2)) for r in a.interiors)
    hb = sorted((np.array(r.coords).mean(0).round(1).tolist(), round(float(np.ptp(np.array(r.coords)[:, 0])), 2)) for r in b.interiors)
    check(f'flange disc at z={z}: radius & 4 holes identical to original', abs(ra - rb) < 0.02 and len(ha) == 4 and
          all(abs(x[1] - y[1]) < 0.05 and np.hypot(x[0][0] - y[0][0], x[0][1] - y[0][1]) < 0.2 for x, y in zip(ha, hb)),
          f'r={ra:.2f} vs {rb:.2f}; holes {[h[1] for h in ha]} vs {[h[1] for h in hb]}')
for z in (20.0, 35.0, 60.0, 78.0):
    a = max(polys(new, 2, z), key=lambda p: p.area); b = max(polys(orig, 2, z), key=lambda p: p.area)
    removed = b.difference(a).area; added = a.difference(b).area
    check(f'column section at z={z}: identical to original', removed < 0.5 and added < 0.5, f'removed {removed:.2f} mm2, added {added:.2f} mm2')

# ---------------------------------------------------------------- 5. cable path open
open_pts = [(-10.0, 0.0, 50.0), (-19.0, 0.0, 30.0), (-10.0, 0.0, 78.6), (-15.0, 0.0, 78.9), (0.0, 0.0, 79.1),
            (0.0, 0.0, 80.5), (-15.0, 0.0, zf - 0.5), (0.0, 0.0, zf + 1.0)]
check('cable slot open from the column up into the pocket floor', bool((~new.contains(np.array(open_pts))).all()))

report.append(f'volume {new.volume / 1000:.1f} cm3 (original 132.6) -> solid-mass estimate PETG {new.volume / 1000 * 1.27:.0f} g / ASA {new.volume / 1000 * 1.07:.0f} g')
report.append(f'bounds (aligned) {np.round(new.bounds, 2).tolist()}  faces {len(new.faces)}')
txt = '\n'.join(report); print(txt)
(ROOT / 'analysis/verify_report.txt').write_text(txt, encoding='utf-8')
print('\nALL CHECKS PASSED' if ok_all else '\nSOME CHECKS FAILED')
sys.exit(0 if ok_all else 1)
