"""Analysis figures: v2 section sheet, original-vs-v2 comparison, dimensioned front/top drawing."""
import json
from pathlib import Path
import numpy as np
import trimesh
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent
P = json.load(open(ROOT / 'build/params.json')); D = P['derived']
new = trimesh.load(ROOT / 'ultrasound_probe_mount_v2_aligned.stl', force='mesh')
orig = trimesh.load(ROOT / 'reference/bonche3_aligned.stl', force='mesh')


def polys(mesh, axis, v):
    normal = np.zeros(3); normal[axis] = 1; origin = np.zeros(3); origin[axis] = v
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return []
    p2, T = sec.to_2D(); keep = [i for i in range(3) if i != axis]
    out = []
    for poly in p2.polygons_full:
        f = lambda c: trimesh.transform_points(np.column_stack([np.asarray(c), np.zeros(len(c))]), T)[:, keep]
        out.append((f(poly.exterior.coords), [f(r.coords) for r in poly.interiors]))
    return out


def draw(ax, mesh, axis, v, color='#b8bcc4'):
    for ext, holes in polys(mesh, axis, v):
        ax.fill(ext[:, 0], ext[:, 1], color=color, ec='k', lw=0.7)
        for h in holes:
            ax.fill(h[:, 0], h[:, 1], color='white', ec='k', lw=0.7)
    ax.set_aspect('equal'); ax.grid(True, lw=0.3)


# ---------------------------------------------------------------- 1. v2 section sheet
sz = D['SCREW_Z']
levels = [(2, 3, 'z = 3  flange disc, 4x O5 through'), (2, 9, 'z = 9  O10 counterbores'), (2, 50, 'z = 50  column + cable slot'),
          (2, 70, 'z = 70  column with gussets'), (2, 81.5, 'z = 81.5  floor block, cable opening'),
          (2, 100, 'z = 100  pocket 99.30 x 20.00'), (2, sz, f'z = {sz:.0f}  screw axis (bosses, nut slots, pad recesses)'),
          (2, sz + 8, f'z = {sz + 8:.0f}  pad recesses + nut slots'), (2, 143.5, 'z = 143.5  lead-in chamfer'),
          (1, 0.0, 'y = 0  (XZ) cable path'), (1, P['POCKET_T'] / 2 + 3.0, f'y = {P["POCKET_T"] / 2 + 3:.0f}  (XZ) through the front wall'),
          (0, P['SCREW_X'], f'x = {P["SCREW_X"]:.0f}  (YZ) through the screw axis')]
fig, axs = plt.subplots(3, 4, figsize=(22, 15))
for ax, (axis, v, title) in zip(axs.ravel(), levels):
    draw(ax, new, axis, v)
    ax.set_title(title, fontsize=10)
    labels = ['x', 'y', 'z']; keep = [i for i in range(3) if i != axis]
    ax.set_xlabel(labels[keep[0]]); ax.set_ylabel(labels[keep[1]])
fig.suptitle('ultrasound_probe_mount_v2 - sections (aligned frame, mm)', fontsize=14)
fig.tight_layout(); fig.savefig(ROOT / 'analysis/v2_section_sheet.png', dpi=100); plt.close(fig)

# ---------------------------------------------------------------- 2. dimensioned front (XZ, y=0 section) + top drawing
def dim(ax, p0, p1, text, off=(0, 0), fs=9):
    p0 = np.array(p0, float); p1 = np.array(p1, float); o = np.array(off, float)
    ax.add_patch(FancyArrowPatch(p0 + o, p1 + o, arrowstyle='<->', mutation_scale=10, lw=0.8, color='#c0392b'))
    ax.text(*((p0 + p1) / 2 + o), text, ha='center', va='bottom', fontsize=fs, color='#c0392b',
            bbox=dict(fc='white', ec='none', pad=0.5))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 11), gridspec_kw={'width_ratios': [1.15, 1]})
draw(a1, new, 1, 0.0); a1.set_title('front section y = 0 (XZ)  - grey = solid, white = void')
Xo = D['BODY_X'] / 2; zf, zt = D['Z_FLOOR'], D['Z_TOP']
dim(a1, (-Xo, zt + 6), (Xo, zt + 6), f'{D["BODY_X"]:.1f}')
dim(a1, (-P['POCKET_W'] / 2, zt - 6), (P['POCKET_W'] / 2, zt - 6), f'pocket {P["POCKET_W"]:.2f}')
dim(a1, (Xo + 6, zf), (Xo + 6, zt), f'{P["POCKET_H"]:.0f}', fs=9)
dim(a1, (Xo + 14, 0), (Xo + 14, zt), f'{zt:.0f} overall')
dim(a1, (-Xo - 6, P['Z_CUT']), (-Xo - 6, zf), f'floor {P["FLOOR_T"]:.0f}')
dim(a1, (-30, 0), (30, 0), 'O60 flange', off=(0, -8))
dim(a1, (-22.5, P['Z_CUT']), (22.5, P['Z_CUT']), 'column top 45', off=(0, -14))
dim(a1, (-Xo - 6, round(D['Z_G0'], 1)), (-Xo - 6, P['Z_CUT']), f'gusset {P["GUSSET_ANGLE"]:.0f} deg')
a1.plot([-P['SCREW_X'], P['SCREW_X']], [sz, sz], 'o', color='#c0392b', ms=4)
a1.text(P['SCREW_X'] + 3, sz + 2, f'M4 screw axis z={sz:.0f}', color='#c0392b', fontsize=9)
a1.set_xlim(-Xo - 22, Xo + 22); a1.set_ylim(-14, zt + 14); a1.set_xlabel('x'); a1.set_ylabel('z')

draw(a2, new, 2, sz); a2.set_title(f'top section z = {sz:.0f} (screw axis) - bosses, nut slots, pad recesses')
Yo = P['BODY_Y'] / 2; Yb = Yo + P['BOSS_OUT']
dim(a2, (-P['POCKET_T'] / 2 * 0 - 50, Yo), (-50, -Yo), f'{P["BODY_Y"]:.0f}', off=(-8, 0))
dim(a2, (-P['SCREW_X'], Yb), (P['SCREW_X'], Yb), f'screw pitch {2 * P["SCREW_X"]:.0f}', off=(0, 3))
dim(a2, (P['SCREW_X'] + 12, -P['POCKET_T'] / 2), (P['SCREW_X'] + 12, P['POCKET_T'] / 2), f'{P["POCKET_T"]:.2f}', off=(2, 0))
dim(a2, (P['SCREW_X'] + 16, Yo), (P['SCREW_X'] + 16, Yb), f'boss +{P["BOSS_OUT"]:.0f}')
a2.text(0, -Yb - 6, f'hole O{P["HOLE_D"]:.1f} through wall; nut slot {P["NUT_AF"] + P["NUT_CLR"]:.1f} x {P["NUT_T"] + P["NUT_CLR"]:.1f} (M4 nut) '
        f'from top; pad recess {P["PAD_SIZE"]:.0f} x {P["PAD_SIZE"]:.0f} x {P["PAD_DEPTH"]:.1f}', ha='center', fontsize=9)
a2.set_xlim(-Xo - 10, Xo + 24); a2.set_ylim(-Yb - 12, Yb + 10); a2.set_xlabel('x'); a2.set_ylabel('y')
fig.suptitle('ultrasound_probe_mount_v2 - key dimensions (mm, aligned frame: base centre at origin)', fontsize=14)
fig.tight_layout(); fig.savefig(ROOT / 'analysis/v2_dimensions.png', dpi=100); plt.close(fig)

# ---------------------------------------------------------------- 3. original vs v2 (rendered) side by side
imgs = [Image.open(ROOT / 'renders' / n) for n in ('orig_iso_clean.png', 'v2_iso.png')]
w = sum(i.width for i in imgs); h = max(i.height for i in imgs)
sheet = Image.new('RGB', (w, h), 'white'); x = 0
for i in imgs:
    sheet.paste(i, (x, 0)); x += i.width
sheet.save(ROOT / 'renders/compare_original_vs_v2.png')
print('figures written')
