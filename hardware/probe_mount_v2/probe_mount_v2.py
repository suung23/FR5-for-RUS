"""Ultrasound probe robot mount v2 - the ORIGINAL clamp, resized for the new probe handle.

Concept (identical to the source STL): two 3 mm "hat" plates rise from the column top, their end
flanges face each other across a 6 mm gap, and four M4 screws (O4.0 clearance in the -Y flange,
O3.3 thread-forming hole in the +Y flange) pull the flanges together so the plates squeeze the
probe handle directly.  Only the pocket size changes: 39 x 27 -> POCKET_W x POCKET_T.

Below Z_CUT everything is the untouched original mesh (planar cut, unioned).  The 3 mm root
transition (column top -> plate outline) and the 45-degree chamfer under the overhanging plate ends
are rebuilt the way the original had them, so the part prints upright without support.

Frames: "aligned" = base-disc centre at the origin, X = probe width, Y = clamp thickness (screw
direction), Z up.  The deliverable STL is exported in the ORIGINAL frame (base centre at
x = -165, clamp rotated 135 deg about Z) via analysis/frame.json.

Edit the PARAMETERS block and re-run:  python probe_mount_v2.py
Outputs: ultrasound_probe_mount_v2.stl (original frame), ultrasound_probe_mount_v2_aligned.stl,
         build/*.stl helper meshes, build/params.json
"""
from __future__ import annotations
import json
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
    # --- probe handle (measured 2026-09-10) and pocket clearances ---
    PROBE_W=42.5, PROBE_T=19.05,
    CLR_W=0.30,              # width clearance (FDM fit)      -> POCKET_W = 42.80
    CLR_T=0.80,              # thickness clearance, closed by the flange screws -> POCKET_T = 19.85
    POCKET_H=60.0,           # plate height above the pocket floor (original: 70)
    # --- hat-plate profile, copied from the original clamp ---
    PLATE_T=3.0,             # plate / flange thickness (original 3.0)
    FLANGE_GAP=6.0,          # gap between the two flanges (original 6.0)
    FLANGE_L=9.0,            # flange length beyond the side wall (original 9.0)
    R_IN_BOT=3.0,            # convex round side wall -> flange inner face (original R3)
    R_OUT=3.0,               # concave fillet flange top -> side wall outer face (original R3)
    R_TOP_IN=1.0,            # pocket top corners (original R3; R1 so a sharp-cornered handle still fits)
    R_TOP_OUT=1.0,           # outer top corners (original R1)
    R_END=1.0,               # flange end corners (original R1)
    ROOT_V=3.0,              # the flange gap closes in a 45-deg V over this height at the root (original 3)
    ROOT_T=3.0,              # root transition height column top -> plate outline (original z 79-82)
    END_CHAMFER=45.0,        # chamfer under the plate ends that overhang the column (deg from horizontal)
    LEAD_IN=1.0,             # chamfer on the pocket's top inner edges
    # --- flange screws: M4, heads on the -Y side, thread-forming into the +Y flange (as original) ---
    SCREW_OFFSET=4.5,        # screw axis outboard of the side wall outer face (original 27 - 22.5)
    SCREW_Z_FROM_FLOOR=(20.0, 50.0),   # original: 30 & 60 above the floor with 70 tall plates (top one 10 below the rim)
    HOLE_CLR=4.0, HOLE_TAP=3.3,        # original: O4.0 clearance / O3.3 thread-forming
    # --- retained original lower body ---
    Z_CUT=79.0,              # everything below this plane is the untouched original mesh
    COL_TOP_X=22.5, COL_TOP_Y=16.5, COL_TOP_R=1.0,   # column top = 45 x 33 rectangle, r1 corners (measured)
    SLOT_W=12.0, SLOT_END_X=6.0,      # cable slot |y|<6, from -X face to x=+6 (round end centred x=0)
)
# ============================================================================

# derived
T = P['PLATE_T']
POCKET_W = P['PROBE_W'] + P['CLR_W']; POCKET_T = P['PROBE_T'] + P['CLR_T']
XI = POCKET_W / 2;  XO = XI + T;  XE = XO + P['FLANGE_L']            # pocket half-width, side wall outer, flange end
YI = POCKET_T / 2;  YO = YI + T                                      # pocket half-thickness, top plate outer
YF_IN = P['FLANGE_GAP'] / 2;  YF_OUT = YF_IN + T                     # flange inner / outer face
XC, YC = P['COL_TOP_X'], P['COL_TOP_Y']
Z_FLOOR = P['Z_CUT'] + P['ROOT_T']
Z_TOP = Z_FLOOR + P['POCKET_H']
XB = XO + P['SCREW_OFFSET']
SCREW_Z = [Z_FLOOR + h for h in P['SCREW_Z_FROM_FLOOR']]
assert XI <= XC, 'pocket wider than the column top: the floor would be unsupported'


def rounded_rect(w, h, r, center=True):
    cs = CrossSection.square([w, h], center)
    if r > 0:
        cs = cs.offset(-r, JoinType.Miter).offset(r, JoinType.Round)
    return cs


def M(mat3x3, t=(0, 0, 0)):
    m = np.zeros((3, 4)); m[:, :3] = mat3x3; m[:, 3] = t
    assert np.linalg.det(m[:, :3]) > 0
    return m


def arc(cx, cy, r, a0, a1, n=12):
    a = np.radians(np.linspace(a0, a1, n))
    return [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in a]


def to_trimesh(man: Manifold) -> trimesh.Trimesh:
    mesh = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)


def from_trimesh(tm: trimesh.Trimesh) -> Manifold:
    return Manifold(Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32)))


# ---------------------------------------------------------------- lower body (original)
orig = trimesh.load(ROOT / 'reference/bonche3_aligned.stl', force='mesh')
lower_tm = trimesh.intersections.slice_mesh_plane(orig, plane_normal=[0, 0, -1], plane_origin=[0, 0, P['Z_CUT']], cap=True)
lower_tm.merge_vertices()
assert lower_tm.is_watertight, 'planar cut of the original is not watertight'
lower = from_trimesh(lower_tm)
assert lower.status() == m3.Error.NoError, lower.status()


# ---------------------------------------------------------------- hat plate profile (+Y half), CCW
def hat_half_profile():
    r1, r2, r3, r4, r5 = P['R_IN_BOT'], P['R_OUT'], P['R_TOP_IN'], P['R_TOP_OUT'], P['R_END']
    pts = [(-XE, YF_IN), (-XO, YF_IN)]
    pts += arc(-XO, YF_IN + r1, r1, -90, 0)                 # convex round to the inner side wall
    pts += [(-XI, YI - r3)]
    pts += arc(-XI + r3, YI - r3, r3, 180, 90)              # pocket top-left corner
    pts += [(XI - r3, YI)]
    pts += arc(XI - r3, YI - r3, r3, 90, 0)                 # pocket top-right corner
    pts += [(XI, YF_IN + r1)]
    pts += arc(XO, YF_IN + r1, r1, 180, 270)                # convex round to the right flange
    pts += [(XE, YF_IN), (XE, YF_OUT - r5)]
    pts += arc(XE - r5, YF_OUT - r5, r5, 0, 90)             # flange end
    pts += [(XO + r2, YF_OUT)]
    pts += arc(XO + r2, YF_OUT + r2, r2, 270, 180)          # concave fillet flange -> outer wall
    pts += [(XO, YO - r4)]
    pts += arc(XO - r4, YO - r4, r4, 0, 90)                 # outer top-right corner
    pts += [(-XO + r4, YO)]
    pts += arc(-XO + r4, YO - r4, r4, 90, 180)              # outer top-left corner
    pts += [(-XO, YF_OUT + r2)]
    pts += arc(-XO - r2, YF_OUT + r2, r2, 0, -90)           # concave fillet
    pts += [(-XE + r5, YF_OUT)]
    pts += arc(-XE + r5, YF_OUT - r5, r5, 90, 180)          # flange end
    return np.array(pts, dtype=np.float64)


# plates run from the cut plane up, so they merge with the root block below the pocket floor
plate_pos = CrossSection([hat_half_profile()]).extrude(Z_TOP - P['Z_CUT']).translate([0, 0, P['Z_CUT']])
plate_neg = plate_pos.mirror([0, 1, 0])

# root V: the flange gap closes at 45 deg over ROOT_V above the floor (as in the original).
# 1 mm wider than the gap and 0.5 below the floor so it overlaps the flanges/floor as a solid.
v = P['ROOT_V']
wedge_cs = CrossSection([np.array([(-(YF_IN + 1.0), -0.5), (YF_IN + 1.0, -0.5), (0.0, v + 0.5)])])   # (u=y, v=z)
wedges = [wedge_cs.extrude(XE - XI).transform(M([[0, 0, 1], [1, 0, 0], [0, 1, 0]], (x0, 0, Z_FLOOR)))
          for x0 in (XI, -XE)]

# root block: column top outline tapering (in Y) to the plate outline over ROOT_T; its top face is the
# pocket floor.  It starts exactly at the cut plane (so it cannot plug the cable slot below it) and ends
# 0.05 inside the plate face (a taper ending exactly on the plate face would leave a zero-thickness sliver).
root = (rounded_rect(2 * XC, 2 * YC, P['COL_TOP_R']).extrude(P['ROOT_T'], scale_top=[1.0, (YO - 0.05) / YC])
        .translate([0, 0, P['Z_CUT']]))

# 45-deg chamfer under the plate ends that overhang the column (|x| <= XC + (z - Z_CUT) * tan)
tan_c = np.tan(np.radians(P['END_CHAMFER']))
zc = P['Z_CUT'] + (XE + 1 - XC) / tan_c
cham_pts = np.array([(-XC, P['Z_CUT'] - 1), (XC, P['Z_CUT'] - 1), (XC, P['Z_CUT']), (XE + 1, zc), (XE + 1, Z_TOP + 5),
                     (-XE - 1, Z_TOP + 5), (-XE - 1, zc), (-XC, P['Z_CUT'])], dtype=np.float64)
chamfer = (CrossSection([cham_pts]).extrude(2 * YC + 10)
           .transform(M([[1, 0, 0], [0, 0, -1], [0, 1, 0]], (0, YC + 5, 0))))          # (u=x, v=z) extruded along Y

upper = plate_pos + plate_neg
for w in wedges:
    upper = upper + w
upper = (upper ^ chamfer) + root
solid = lower + upper

# ---------------------------------------------------------------- cuts
cuts = []
# cable slot continues through the root block into the pocket floor
sw = P['SLOT_W']
slot_cs = (CrossSection.square([XC + 0.1, sw], False).translate([-(XC + 0.1), -sw / 2]) + CrossSection.circle(P['SLOT_END_X']))
cuts.append(slot_cs.extrude(P['ROOT_T'] + 0.5).translate([0, 0, P['Z_CUT']]))
# lead-in chamfer at the pocket's top inner edges (frustum starts 0.02 inside the walls: no coplanar faces)
li = P['LEAD_IN']; e = 0.02
cuts.append(CrossSection.square([2 * XI - e, 2 * YI - e], True)
            .extrude(li, scale_top=[(2 * XI + 2 * li) / (2 * XI - e), (2 * YI + 2 * li) / (2 * YI - e)])
            .translate([0, 0, Z_TOP - li]))
cuts.append(Manifold.cube([2 * XI + 2 * li, 2 * YI + 2 * li, 5], True).translate([0, 0, Z_TOP + 2.5 - 0.01]))
# flange screw holes: clearance in the -Y flange, thread-forming hole in the +Y flange (axis along Y)
for x in (XB, -XB):
    for z in SCREW_Z:
        cuts.append(Manifold.cylinder(YF_OUT + 2, P['HOLE_CLR'] / 2).rotate([90, 0, 0]).translate([x, 0.0, z]))    # y < 0
        cuts.append(Manifold.cylinder(YF_OUT + 2, P['HOLE_TAP'] / 2).rotate([-90, 0, 0]).translate([x, 0.0, z]))   # y > 0

for c in cuts:
    solid = solid - c
assert solid.status() == m3.Error.NoError, solid.status()

# ---------------------------------------------------------------- export
final_aligned = to_trimesh(solid)
final_aligned.export(ROOT / 'ultrasound_probe_mount_v2_aligned.stl')
frame = json.load(open(ROOT / 'analysis/frame.json'))
Tm = np.array(frame['T_aligned_from_original'])
final_orig = final_aligned.copy(); final_orig.apply_transform(np.linalg.inv(Tm))
final_orig.export(ROOT / 'ultrasound_probe_mount_v2.stl')

to_trimesh(lower).export(BUILD / 'lower_original_cut_aligned.stl')
to_trimesh(solid - lower).export(BUILD / 'new_upper_only_aligned.stl')
json.dump({**P, 'derived': dict(POCKET_W=POCKET_W, POCKET_T=POCKET_T, XI=XI, XO=XO, XE=XE, YI=YI, YO=YO,
                                 YF_IN=YF_IN, YF_OUT=YF_OUT, XB=XB, Z_FLOOR=Z_FLOOR, Z_TOP=Z_TOP, SCREW_Z=SCREW_Z,
                                 volume_mm3=round(final_aligned.volume, 1))},
          open(BUILD / 'params.json', 'w'), indent=1)
print(f'pocket {POCKET_W:.2f} x {POCKET_T:.2f} x {P["POCKET_H"]:.0f}; clamp {2 * XE:.1f} x {2 * YO:.1f}; screws x=+-{XB:.2f} z={SCREW_Z}')
print(f'volume {final_aligned.volume:.0f} mm3  faces {len(final_aligned.faces)}  watertight {final_aligned.is_watertight}')
print('bounds aligned', np.round(final_aligned.bounds, 2).tolist())
print('bounds original frame', np.round(final_orig.bounds, 2).tolist())
