"""Find the clamp-plate orientation, re-express the original in an aligned frame
(base centre at origin, plates parallel to X), and slice/plot/render it."""
import numpy as np, trimesh, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
m = trimesh.load(ROOT / 'reference/bonche3_original.stl', force='mesh')

# --- 1. dominant horizontal normal direction (area-weighted angle histogram)
n = m.face_normals; a = m.area_faces
horiz = np.abs(n[:, 2]) < 0.01
ang = np.degrees(np.arctan2(n[horiz, 1], n[horiz, 0])) % 180.0
hist, edges = np.histogram(ang, bins=180, range=(0, 180), weights=a[horiz])
top = np.argsort(hist)[::-1][:6]
print('dominant horizontal normal angles (deg, area mm2):', [(round(float(edges[i]), 1), round(float(hist[i]), 1)) for i in top])
theta = float(edges[top[0]]) + 0.5  # normal angle of the plates
# refine: mean angle of faces within +-2 deg
sel = horiz.copy(); sel[horiz] = np.abs(((ang - theta + 90) % 180) - 90) < 2
vec = (n[sel][:, :2] * a[sel, None]); vec[np.dot(n[sel][:, :2], [np.cos(np.radians(theta)), np.sin(np.radians(theta))]) < 0] *= -1
theta = float(np.degrees(np.arctan2(vec.sum(0)[1], vec.sum(0)[0])))
print('refined plate-normal angle: %.3f deg' % theta)

# base centre from the 4 counterbored holes: (-165, 0)
BASE_C = np.array([-165.0, 0.0, 0.0])
# aligned frame: plate normal -> +Y  (so plates lie in XZ, probe width along X)
rot = trimesh.transformations.rotation_matrix(np.radians(90 - theta), [0, 0, 1])
T = rot @ trimesh.transformations.translation_matrix(-BASE_C)
ma = m.copy(); ma.apply_transform(T)
ma.export(ROOT / 'reference/bonche3_aligned.stl')
json.dump({'plate_normal_angle_deg_in_original': theta, 'base_centre_original': BASE_C.tolist(),
           'T_aligned_from_original': T.tolist()}, open(ROOT / 'analysis/frame.json', 'w'), indent=1)
print('aligned bounds\n', np.round(ma.bounds, 2))

def polys_at(mesh, axis, v):
    normal = np.zeros(3); normal[axis] = 1; origin = np.zeros(3); origin[axis] = v
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None: return [], None
    p2, Tm = sec.to_2D()
    keep = [i for i in range(3) if i != axis]
    out = []
    for poly in p2.polygons_full:
        def to3(c):
            c = np.asarray(c); return trimesh.transform_points(np.column_stack([c, np.zeros(len(c))]), Tm)[:, keep]
        out.append((to3(poly.exterior.coords), [to3(r.coords) for r in poly.interiors], poly.area))
    return out, keep

def report(mesh, axis, values, fname):
    lines = []
    for v in values:
        polys, keep = polys_at(mesh, axis, v)
        parts = []
        for ext, holes, area in polys:
            bmin, bmax = ext.min(0), ext.max(0)
            s = f'bbox [{bmin[0]:7.2f},{bmax[0]:7.2f}]x[{bmin[1]:7.2f},{bmax[1]:7.2f}] area {area:7.1f}'
            hs = []
            for h in holes:
                ctr = h.mean(0); r = np.linalg.norm(h - ctr, axis=1); e = h.max(0) - h.min(0)
                hs.append(f'hole@({ctr[0]:7.2f},{ctr[1]:7.2f}) {e[0]:5.2f}x{e[1]:5.2f}')
            parts.append(s + ((' | ' + '; '.join(hs)) if hs else ''))
        lines.append(f'{v:7.2f} | n={len(polys)} | ' + ' || '.join(parts))
    (ROOT / 'analysis' / fname).write_text('\n'.join(lines), encoding='utf-8')
    return lines

print('\n=== Z slices (aligned frame; axes x,y) ===')
for l in report(ma, 2, np.arange(0.25, 152, 1.0), 'aligned_slices_z.txt'):
    print(l)
print('\n=== Y slices (aligned; axes x,z) ===')
for l in report(ma, 1, np.arange(-30, 30, 1.0), 'aligned_slices_y.txt'):
    print(l)
print('\n=== X slices (aligned; axes y,z) ===')
for l in report(ma, 0, np.arange(-33, 34, 1.0), 'aligned_slices_x.txt'):
    print(l)

# --- section sheet
levels = [3, 9, 14, 30, 50, 70, 76, 79, 81, 83, 100, 112, 142, 151]
fig, axs = plt.subplots(2, 7, figsize=(24, 7.5))
for ax, z in zip(axs.ravel(), levels):
    polys, _ = polys_at(ma, 2, z)
    for ext, holes, area in polys:
        ax.fill(ext[:, 0], ext[:, 1], color='#bbbbbb', ec='k', lw=0.8)
        for h in holes: ax.fill(h[:, 0], h[:, 1], color='white', ec='k', lw=0.8)
    ax.set_title(f'z = {z} mm'); ax.set_aspect('equal'); ax.grid(True, lw=0.3)
    ax.set_xlim(-35, 35); ax.set_ylim(-35, 35)
fig.suptitle('bonche3_original.stl  Z-sections (aligned frame: base centre at origin, clamp plates parallel to X)')
fig.tight_layout(); fig.savefig(ROOT / 'analysis/orig_section_sheet.png', dpi=110)

# --- 3-view renders (orthographic, simple lambert)
def render(mesh, elev, azim, fname, title):
    fig = plt.figure(figsize=(7, 9)); ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    v = mesh.vertices; f = mesh.faces
    light = np.array([0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(mesh.face_normals @ light, 0, 1)
    pc = Poly3DCollection(v[f], facecolors=np.column_stack([shade * 0.75, shade * 0.78, shade * 0.85]), edgecolor='none')
    ax.add_collection3d(pc)
    lo, hi = mesh.bounds; c = (lo + hi) / 2; r = (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z'); ax.set_title(title)
    fig.tight_layout(); fig.savefig(ROOT / 'renders' / fname, dpi=110); plt.close(fig)

render(ma, 0, -90, 'orig_front.png', 'original – front (looking along +y, plate normal)')
render(ma, 0, 0, 'orig_side.png', 'original – side (looking along -x)')
render(ma, 28, -50, 'orig_iso.png', 'original – isometric')
render(ma, 90, -90, 'orig_top.png', 'original – top')
print('done')
