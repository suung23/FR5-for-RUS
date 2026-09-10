"""Analysis figures for the resized flexure clamp: section sheet, dimensioned drawing, original-vs-v2 comparison."""
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


def dim(ax, p0, p1, text, off=(0, 0), fs=9):
    p0 = np.array(p0, float) + np.array(off, float); p1 = np.array(p1, float) + np.array(off, float)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='<->', mutation_scale=10, lw=0.8, color='#c0392b'))
    ax.text(*((p0 + p1) / 2), text, ha='center', va='bottom', fontsize=fs, color='#c0392b', bbox=dict(fc='white', ec='none', pad=0.5))


XI, XO, XE, YI, YO, YF_IN, YF_OUT, XB = (D[k] for k in ('XI', 'XO', 'XE', 'YI', 'YO', 'YF_IN', 'YF_OUT', 'XB'))
PW, PT = D['POCKET_W'], D['POCKET_T']
zf, zt = D['Z_FLOOR'], D['Z_TOP']; z1, z2 = D['SCREW_Z']

# ---------------------------------------------------------------- 1. section sheet (original on top row, v2 below)
levels = [(2, 3, 'z = 3  flange disc, 4x O5'), (2, 9, 'z = 9  O10 counterbores'), (2, 50, 'z = 50  column + cable slot'),
          (2, 80.5, 'z = 80.5  root transition + slot'), (2, zf + 1.5, f'z = {zf + 1.5:.1f}  root V, chamfered ends'),
          (2, 100, f'z = 100  plates, pocket {PW:.2f} x {PT:.2f}'), (2, z1, f'z = {z1:.0f}  lower screw row'), (2, z2, f'z = {z2:.0f}  upper screw row'),
          (1, 0.0, 'y = 0  (XZ) cable path'), (1, -(YF_IN + 1.5), f'y = {-(YF_IN + 1.5):.1f}  (XZ) -Y flange, O4.0'),
          (1, +(YF_IN + 1.5), f'y = {(YF_IN + 1.5):.1f}  (XZ) +Y flange, O3.3 tap'), (0, 0.0, 'x = 0  (YZ) plates + root')]
fig, axs = plt.subplots(3, 4, figsize=(22, 15))
for ax, (axis, v, title) in zip(axs.ravel(), levels):
    draw(ax, new, axis, v); ax.set_title(title, fontsize=10)
    labels = ['x', 'y', 'z']; keep = [i for i in range(3) if i != axis]
    ax.set_xlabel(labels[keep[0]]); ax.set_ylabel(labels[keep[1]])
fig.suptitle('ultrasound_probe_mount_v2 - sections, aligned frame, mm', fontsize=14)
fig.tight_layout(); fig.savefig(ROOT / 'analysis/v2_section_sheet.png', dpi=100); plt.close(fig)

# ---------------------------------------------------------------- 2. dimensioned drawing: original vs new section + front
fig, axs = plt.subplots(1, 3, figsize=(24, 9), gridspec_kw={'width_ratios': [1, 1, 1.1]})
a0, a1, a2 = axs
draw(a0, orig, 2, 100.0, color='#d9d9d9'); a0.set_title('ORIGINAL clamp section z = 100')
dim(a0, (-19.5, 13.5), (19.5, 13.5), 'pocket 39', off=(0, 2))
dim(a0, (-31.5, -20), (31.5, -20), '63')
dim(a0, (36, -13.5), (36, 13.5), '27'); dim(a0, (-36, 3), (-36, 6), '3', fs=8); dim(a0, (40, -3), (40, 3), 'gap 6', fs=8)
a0.plot([27, -27, 27, -27], [4.5, 4.5, -4.5, -4.5], 'o', color='#c0392b', ms=3)
a0.text(27, 7.5, 'M4 x = ±27', color='#c0392b', fontsize=8, ha='center')
a0.set_xlim(-46, 46); a0.set_ylim(-26, 22); a0.set_xlabel('x'); a0.set_ylabel('y')

draw(a1, new, 2, 100.0); a1.set_title(f'NEW clamp section z = 100 (same profile, pocket {PW:.2f} x {PT:.2f})')
dim(a1, (-XI, YI), (XI, YI), f'pocket {PW:.2f}', off=(0, 2))
dim(a1, (-XE, -20), (XE, -20), f'{2 * XE:.1f}')
dim(a1, (XE + 3, -YI), (XE + 3, YI), f'{PT:.2f}'); dim(a1, (XE + 8, -YO), (XE + 8, YO), f'{2 * YO:.1f}')
dim(a1, (-XE - 3, YF_IN), (-XE - 3, YF_OUT), f'{P["PLATE_T"]:.0f}', fs=8); dim(a1, (-XE - 8, -YF_IN), (-XE - 8, YF_IN), f'gap {P["FLANGE_GAP"]:.0f}', fs=8)
dim(a1, (XO, YO + 5), (XE, YO + 5), f'{P["FLANGE_L"]:.0f}', fs=8)
a1.plot([XB, -XB, XB, -XB], [YF_IN + 1.5, YF_IN + 1.5, -YF_IN - 1.5, -YF_IN - 1.5], 'o', color='#c0392b', ms=3)
a1.text(XB, YF_OUT + 1.5, f'M4 x = ±{XB:.1f}', color='#c0392b', fontsize=8, ha='center')
a1.set_xlim(-46, 46); a1.set_ylim(-26, 22); a1.set_xlabel('x'); a1.set_ylabel('y')

draw(a2, new, 1, -(YF_IN + 1.5)); a2.set_title(f'front section through the -Y flange (y = {-(YF_IN + 1.5):.1f})')
dim(a2, (XE + 4, zf), (XE + 4, zt), f'plates {P["POCKET_H"]:.0f}')
dim(a2, (XE + 12, 0), (XE + 12, zt), f'{zt:.0f} overall')
dim(a2, (-XE - 4, P['Z_CUT']), (-XE - 4, zf), f'root {P["ROOT_T"]:.0f}', fs=8)
dim(a2, (-XB - 5, z1), (-XB - 5, z2), f'{z2 - z1:.0f}', fs=8); dim(a2, (-XB - 5, zf), (-XB - 5, z1), f'{z1 - zf:.0f}', fs=8)
dim(a2, (-30, -6), (30, -6), 'O60 flange (PCD 45, 4x O5 / O10 cbore)')
a2.text(0, zt + 5, f'4x M4: O{P["HOLE_CLR"]:.1f} in this flange / O{P["HOLE_TAP"]:.1f} thread-forming in the +Y flange, z = {z1:.0f} & {z2:.0f}',
        ha='center', fontsize=9, color='#c0392b')
a2.set_xlim(-XE - 16, XE + 20); a2.set_ylim(-12, zt + 12); a2.set_xlabel('x'); a2.set_ylabel('z')
fig.suptitle('ultrasound_probe_mount_v2 - key dimensions (mm, aligned frame: base centre at origin)', fontsize=14)
fig.tight_layout(); fig.savefig(ROOT / 'analysis/v2_dimensions.png', dpi=100); plt.close(fig)

# ---------------------------------------------------------------- 3. original vs v2 rendered side by side (same scale)
imgs = [Image.open(ROOT / 'renders' / n) for n in ('orig_iso_clean.png', 'v2_iso.png')]
w = sum(i.width for i in imgs); h = max(i.height for i in imgs)
sheet = Image.new('RGB', (w, h), 'white'); x = 0
for i in imgs:
    sheet.paste(i, (x, 0)); x += i.width
sheet.save(ROOT / 'renders/compare_original_vs_v2.png')
print('figures written')
