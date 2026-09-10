"""Ultrasound probe robot mount v2 - parametric rebuild of the upper clamp.

The lower body (O60 robot-flange disc with 4 counterbored holes, tapered column with the
12 mm cable slot) is KEPT from the original STL: the original mesh is cut by a plane at
Z_CUT and unioned with a new parametric sleeve clamp built with manifold3d (CSG kernel).

Frames
------
* "aligned" frame: base-disc centre at the origin, +Z up, clamp thickness direction = Y,
  probe width direction = X (the original clamp plates are rotated 135 deg about Z from this).
* "original" frame: the frame of the source STL (base centre at (-165, 0, 0)).
  analysis/frame.json holds the transform; the deliverable STL is exported in the
  ORIGINAL frame so it drops into the same place as the old part.

Edit the PARAMETERS block and re-run:  python probe_mount_v2.py
Outputs: ultrasound_probe_mount_v2.stl (original frame),
         ultrasound_probe_mount_v2_aligned.stl (base at origin, walls axis-aligned),
         build/*.stl helper meshes, build/params.json
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import trimesh
import manifold3d as m3
from manifold3d import Manifold, Mesh, CrossSection, JoinType

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / 'build'; BUILD.mkdir(exist_ok=True)
m3.set_circular_segments(96)

# ============================== PARAMETERS (mm) ==============================
P = dict(
    # --- probe handle (measured) and pocket (with clearances) ---
    PROBE_W=99.01, PROBE_T=19.14,
    POCKET_W=99.30,          # X: 99.01 + 0.29 FDM assembly clearance
    POCKET_T=20.00,          # Y: 19.14 + screw-tightening clearance
    POCKET_H=60.00,          # Z: gripped length of the handle
    WALL_X=5.0,              # left/right end walls (>= 5.0)
    BODY_Y=33.0,             # outer front-back size -> walls (33-20)/2 = 6.5 (>= 5.0), flush with column top
    FLOOR_T=5.0,             # pocket floor thickness above the column top
    BODY_R=4.0,              # vertical outer edge radius of the sleeve
    LEAD_IN=1.5,             # 45 deg lead-in chamfer at the pocket mouth
    # --- retained original lower body ---
    Z_CUT=79.0,              # everything below this plane is the untouched original mesh
    COL_TOP_X=22.5, COL_TOP_Y=16.5,   # column top = 45 x 33 rectangle (measured)
    SLOT_W=12.0, SLOT_END_X=6.0,      # cable slot |y|<6, from -X face to x=+6 (round end centred x=0)
    # column face lines (bbox of the tapered column, measured): x = a + b*(z-12.25)
    COL_FACE_POS=(17.52, (22.50 - 17.52) / (79.25 - 12.25)),
    COL_FACE_NEG=(17.01, (22.50 - 17.01) / (79.25 - 12.25)),
    # --- gussets supporting the overhanging sleeve ends ---
    GUSSET_ANGLE=50.0,       # slope from horizontal (>= 45 deg -> support-free printing upright)
    GUSSET_FILLET=5.0,       # fillet radius at the gusset/column root
    GUSSET_INSET=2.5,        # root polygon sits this far inside the column face line (no ledge on the taper)
    CHANNEL_W=12.4,          # cable channel through the -X gusset (slot width + 0.4)
    # --- side pressure screws (M4), 2 per wall, opposing pairs ---
    SCREW_X=30.0,            # +-X position of the screws
    SCREW_Z_FROM_FLOOR=30.0, # height above the pocket floor (mid-grip)
    HOLE_D=4.4,              # M4 clearance (spec 4.3-4.5)
    BOSS_W=20.0, BOSS_OUT=9.0, BOSS_HALF_H=12.0, BOSS_R=3.0,
    FASTENER='nut',          # 'nut' = captive M4 hex-nut slot from the top; 'insert' = M4 heat-set insert
    NUT_AF=7.0, NUT_T=3.2, NUT_CLR=0.4, NUT_INNER_GAP=6.0,   # slot starts NUT_INNER_GAP outside the pocket face
    INSERT_D=5.7, INSERT_DEPTH=7.0,                          # typical M4 x 6 heat-set insert (OD 5.6-6.0)
    # --- probe-protection pad recess on the inner face at every screw ---
    PAD_SIZE=16.0, PAD_DEPTH=1.5, PAD_R=2.0,   # for a 15 x 15 x 1.5 TPU/silicone pad
)
# ============================================================================

# derived
BODY_X = P['POCKET_W'] + 2 * P['WALL_X']
X_OUT = BODY_X / 2
Y_OUT = P['BODY_Y'] / 2
Z_FLOOR = P['Z_CUT'] + P['FLOOR_T']
Z_TOP = Z_FLOOR + P['POCKET_H']
SCREW_Z = Z_FLOOR + P['SCREW_Z_FROM_FLOOR']
TAN_G = math.tan(math.radians(P['GUSSET_ANGLE']))
Z_G0 = P['Z_CUT'] - (X_OUT - P['COL_TOP_X']) * TAN_G      # gusset root height on the column face


def col_face(z, side):
    a, b = P['COL_FACE_POS'] if side > 0 else P['COL_FACE_NEG']
    return side * (a + b * (z - 12.25))


def rounded_rect(w, h, r, center=True):
    cs = CrossSection.square([w, h], center)
    if r > 0:
        cs = cs.offset(-r, JoinType.Miter).offset(r, JoinType.Round)
    return cs


def M(mat3x3, t=(0, 0, 0)):
    """3x4 affine for Manifold.transform (rotation part must be proper, det = +1)."""
    m = np.zeros((3, 4)); m[:, :3] = mat3x3; m[:, 3] = t
    assert np.linalg.det(m[:, :3]) > 0
    return m


def to_trimesh(man: Manifold) -> trimesh.Trimesh:
    mesh = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)


def from_trimesh(tm: trimesh.Trimesh) -> Manifold:
    return Manifold(Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32)))


# ---------------------------------------------------------------- lower body (original)
orig = trimesh.load(ROOT.parent / "reference/bonche3_aligned.stl", force='mesh')
lower_tm = trimesh.intersections.slice_mesh_plane(orig, plane_normal=[0, 0, -1], plane_origin=[0, 0, P['Z_CUT']], cap=True)
lower_tm.merge_vertices()
assert lower_tm.is_watertight, 'planar cut of the original is not watertight'
lower = from_trimesh(lower_tm)
assert lower.status() == m3.Error.NoError, lower.status()

# ---------------------------------------------------------------- sleeve body
footprint = rounded_rect(BODY_X, P['BODY_Y'], P['BODY_R'])
sleeve = footprint.extrude(Z_TOP - P['Z_CUT']).translate([0, 0, P['Z_CUT']])


# ---------------------------------------------------------------- gussets (XZ profile, extruded along Y)
def gusset(side):
    """Triangular gusset from the column face up to the sleeve end (XZ profile).
    The root part sits GUSSET_INSET inside the measured column face line so that the
    polygon never sticks out of the tapered column; its lower end is chamfered at 45 deg."""
    ins = P['GUSSET_INSET']
    z0 = Z_G0 - 4.0
    zt = P['Z_CUT'] + 2.0
    xf0 = col_face(z0, side) - side * ins          # face line, moved inside the column
    xfg = col_face(Z_G0, side) - side * ins
    pts = [(0.0, z0 - 3.0), (xf0 - side * 3.0, z0 - 3.0), (xf0, z0), (xfg, Z_G0),
           (side * X_OUT, P['Z_CUT']), (side * X_OUT, zt), (0.0, zt)]
    if side < 0:
        pts = pts[::-1]
    cs = CrossSection([np.array(pts, dtype=np.float64)])
    r = P['GUSSET_FILLET']
    cs = cs.offset(r, JoinType.Round).offset(-r, JoinType.Round)       # closing -> fillets the concave root
    # the -X gusset is built as two Y-bands that leave the cable slot / channel (|y| < CHANNEL_W/2) open,
    # so the original slot walls are never touched by a boolean cut
    bands = [(-Y_OUT, Y_OUT)] if side > 0 else [(-Y_OUT, -P['CHANNEL_W'] / 2), (P['CHANNEL_W'] / 2, Y_OUT)]
    out = None
    for ya, yb in bands:
        slab = cs.extrude(yb - ya)                                      # (u=x, v=z, w in [0, yb-ya])
        slab = slab.transform(M([[1, 0, 0], [0, 0, -1], [0, 1, 0]], (0, yb, 0)))   # (u,v,w) -> (u, yb-w, v)
        out = slab if out is None else out + slab
    return out


body = sleeve + gusset(+1) + gusset(-1)
# round the vertical corners of the whole footprint (sleeve + gussets)
body = body ^ footprint.extrude(Z_TOP + 5).translate([0, 0, -1])


# ---------------------------------------------------------------- screw bosses (outer faces, chamfered underside)
def boss(xc, ysign):
    zlo, zhi = SCREW_Z - P['BOSS_HALF_H'], SCREW_Z + P['BOSS_HALF_H']
    yin, yout = Y_OUT - 0.5, Y_OUT + P['BOSS_OUT']
    prof = [(yin, zlo - (yout - yin)), (yout, zlo), (yout, zhi), (yin, zhi)]      # 45 deg chamfer below
    cs = CrossSection([np.array(prof, dtype=np.float64)])
    s = cs.extrude(P['BOSS_W'])                                                    # (u=y, v=z, w=x)
    s = s.transform(M([[0, 0, 1], [1, 0, 0], [0, 1, 0]], (xc - P['BOSS_W'] / 2, 0, 0)))
    rr = rounded_rect(P['BOSS_W'], (yout - yin) + 2, P['BOSS_R']).extrude(200).translate([xc, (yin + yout) / 2, 0])
    s = s ^ rr
    if ysign < 0:
        s = s.mirror([0, 1, 0])
    return s


solid = lower + body
for sx in (P['SCREW_X'], -P['SCREW_X']):
    for ys in (+1, -1):
        solid = solid + boss(sx, ys)

# ---------------------------------------------------------------- cuts
cuts = []
# pocket (open top) + lead-in chamfer
cuts.append(Manifold.cube([P['POCKET_W'], P['POCKET_T'], 100], True).translate([0, 0, Z_FLOOR + 50]))
li = P['LEAD_IN']
cuts.append(CrossSection.square([P['POCKET_W'], P['POCKET_T']], True)
            .extrude(li, scale_top=[(P['POCKET_W'] + 2 * li) / P['POCKET_W'], (P['POCKET_T'] + 2 * li) / P['POCKET_T']])
            .translate([0, 0, Z_TOP - li]))
cuts.append(Manifold.cube([P['POCKET_W'] + 2 * li, P['POCKET_T'] + 2 * li, 10], True).translate([0, 0, Z_TOP + 5 - 0.01]))
# floor opening = continuation of the column cable slot up into the pocket
sw = P['SLOT_W']
slot_cs = (CrossSection.square([P['COL_TOP_X'] + 1.0, sw], False).translate([-(P['COL_TOP_X'] + 1.0), -sw / 2])
           + CrossSection.circle(P['SLOT_END_X']))
cuts.append(slot_cs.extrude(P['FLOOR_T'] + 2).translate([0, 0, P['Z_CUT']]))   # starts exactly at the cut plane
# (the cable channel through the -X gusset is left open by construction, see gusset())
# screw through-holes (Y direction); one cylinder per X position cuts both walls
for sx in (P['SCREW_X'], -P['SCREW_X']):
    cyl = Manifold.cylinder(2 * (Y_OUT + P['BOSS_OUT']) + 4, P['HOLE_D'] / 2, -1.0, 0, True)
    cuts.append(cyl.rotate([90, 0, 0]).translate([sx, 0, SCREW_Z]))
# fastener pockets + pad recesses
y_in = P['POCKET_T'] / 2
for sx in (P['SCREW_X'], -P['SCREW_X']):
    for ys in (+1, -1):
        if P['FASTENER'] == 'nut':
            w = P['NUT_AF'] + P['NUT_CLR']; t = P['NUT_T'] + P['NUT_CLR']
            zb = SCREW_Z - (P['NUT_AF'] / math.cos(math.radians(30)) / 2 + 0.6)   # below the nut's lower corner
            y0 = y_in + P['NUT_INNER_GAP']
            c = Manifold.cube([w, t, (SCREW_Z + P['BOSS_HALF_H'] + 2) - zb], False).translate([sx - w / 2, y0, zb])
        else:
            y_face = Y_OUT + P['BOSS_OUT']
            c = (Manifold.cylinder(P['INSERT_DEPTH'] + 1, P['INSERT_D'] / 2).rotate([-90, 0, 0])
                 .translate([sx, y_face - P['INSERT_DEPTH'], SCREW_Z]))
        # pad recess: rounded square in XZ, extruded along -Y from y_in+PAD_DEPTH to below y_in
        pad = rounded_rect(P['PAD_SIZE'], P['PAD_SIZE'], P['PAD_R']).extrude(P['PAD_DEPTH'] + 0.5)   # (u=x, v=z, w)
        pad = pad.transform(M([[1, 0, 0], [0, 0, -1], [0, 1, 0]], (sx, y_in + P['PAD_DEPTH'], SCREW_Z)))
        if ys < 0:
            c = c.mirror([0, 1, 0]); pad = pad.mirror([0, 1, 0])
        cuts += [c, pad]

for c in cuts:
    solid = solid - c
assert solid.status() == m3.Error.NoError, solid.status()

# ---------------------------------------------------------------- export
final_aligned = to_trimesh(solid)
final_aligned.export(ROOT / 'ultrasound_probe_mount_v2_aligned.stl')
frame = json.load(open(ROOT.parent / "analysis/frame.json"))
T = np.array(frame['T_aligned_from_original'])
final_orig = final_aligned.copy(); final_orig.apply_transform(np.linalg.inv(T))
final_orig.export(ROOT / 'ultrasound_probe_mount_v2.stl')

# helper meshes for renders / checks
to_trimesh(lower).export(BUILD / 'lower_original_cut_aligned.stl')
to_trimesh(solid - lower).export(BUILD / 'new_upper_only_aligned.stl')
json.dump({**P, 'derived': dict(BODY_X=BODY_X, Z_FLOOR=Z_FLOOR, Z_TOP=Z_TOP, SCREW_Z=SCREW_Z, Z_G0=round(Z_G0, 2),
                                 volume_mm3=round(final_aligned.volume, 1))},
          open(BUILD / 'params.json', 'w'), indent=1)
print(f'volume {final_aligned.volume:.0f} mm3  faces {len(final_aligned.faces)}  watertight {final_aligned.is_watertight}')
print('bounds aligned', np.round(final_aligned.bounds, 2).tolist())
print('bounds original frame', np.round(final_orig.bounds, 2).tolist())
