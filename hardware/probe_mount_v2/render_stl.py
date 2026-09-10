"""Tiny orthographic z-buffer renderer (numpy + Pillow) - no OpenGL needed.

usage: render_stl.py out.png VIEW mesh1.stl[:#rrggbb] [mesh2.stl[:#rrggbb] ...] [--title TEXT]
VIEW = front | back | side | side2 | top | iso | iso2 | "elev,azim"
Camera looks along the rotated -Y axis; screen u = x', v = z'.
"""
import sys
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

VIEWS = {'front': (0, 0), 'back': (0, 180), 'side': (0, 90), 'side2': (0, -90),
         'top': (90, 0), 'iso': (28, 35), 'iso2': (28, -145), 'iso_low': (-25, 35)}


def rot(elev, azim):
    e, a = np.radians(elev), np.radians(azim)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


def render(meshes, view, out, W=1400, H=1600, title=None, edges=True, margin=1.12):
    elev, azim = VIEWS[view] if view in VIEWS else tuple(map(float, view.split(',')))
    R = rot(elev, azim)
    allv = np.vstack([m.vertices for m, _ in meshes]) @ R.T
    lo, hi = allv.min(0), allv.max(0); c = (lo + hi) / 2
    span = max((hi[0] - lo[0]) * H / W, hi[2] - lo[2]) * margin
    scale = H / span
    zbuf = np.full((H, W), np.inf); img = np.full((H, W, 3), 255, np.uint8)
    light = np.array([0.35, -0.75, 0.55]); light /= np.linalg.norm(light)
    for m, col in meshes:
        col = np.array(col, float)
        v = m.vertices @ R.T - c
        u = v[:, 0] * scale + W / 2; w = H / 2 - v[:, 2] * scale; d = v[:, 1]
        n = m.face_normals @ R.T
        shade = 0.28 + 0.72 * np.clip(n @ light, 0, 1)
        front = n[:, 1] < 0            # camera sits at -y' looking toward +y'
        for fi in np.nonzero(front)[0]:
            f = m.faces[fi]; pu, pw, pd = u[f], w[f], d[f]
            x0, x1 = int(max(np.floor(pu.min()), 0)), int(min(np.ceil(pu.max()), W - 1))
            y0, y1 = int(max(np.floor(pw.min()), 0)), int(min(np.ceil(pw.max()), H - 1))
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            det = (pu[1] - pu[0]) * (pw[2] - pw[0]) - (pu[2] - pu[0]) * (pw[1] - pw[0])
            if abs(det) < 1e-9:
                continue
            l1 = ((pu[1] - gx) * (pw[2] - gy) - (pu[2] - gx) * (pw[1] - gy)) / det
            l2 = ((pu[2] - gx) * (pw[0] - gy) - (pu[0] - gx) * (pw[2] - gy)) / det
            l0 = 1 - l1 - l2
            inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            if not inside.any():
                continue
            depth = l0 * pd[0] + l1 * pd[1] + l2 * pd[2]
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]
            upd = inside & (depth < sub)
            sub[upd] = depth[upd]
            img[y0:y1 + 1, x0:x1 + 1][upd] = np.clip(col * shade[fi], 0, 255).astype(np.uint8)
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    if edges:
        tol = 0.35 * span / H * 3
        for m, col in meshes:
            v = m.vertices @ R.T - c; u = v[:, 0] * scale + W / 2; w = H / 2 - v[:, 2] * scale; d = v[:, 1]
            ang = m.face_adjacency_angles
            keep = ang > np.radians(25)
            ed = m.face_adjacency_edges[keep]
            # an edge is drawn only if at least one of its two faces looks at the camera
            fn = m.face_normals @ R.T
            fa = m.face_adjacency[keep]
            facing = (fn[fa[:, 0], 1] < 0) | (fn[fa[:, 1], 1] < 0)
            t = np.linspace(0.05, 0.95, 12)
            for (a, b), fc in zip(ed, facing):
                if not fc:
                    continue
                px = u[a] + (u[b] - u[a]) * t; py = w[a] + (w[b] - w[a]) * t; pz = d[a] + (d[b] - d[a]) * t
                ix = np.clip(px.astype(int), 0, W - 1); iy = np.clip(py.astype(int), 0, H - 1)
                vis = pz <= zbuf[iy, ix] + tol
                if vis.all():
                    dr.line([(u[a], w[a]), (u[b], w[b])], fill=(45, 45, 45), width=1)
    if title:
        try:
            font = ImageFont.truetype('arial.ttf', 30)
        except Exception:
            font = ImageFont.load_default()
        dr.text((20, 15), title, fill=(0, 0, 0), font=font)
    im.save(out)
    return im


if __name__ == '__main__':
    args = sys.argv[1:]
    title = None
    if '--title' in args:
        i = args.index('--title'); title = args[i + 1]; del args[i:i + 2]
    out, view = args[0], args[1]
    meshes = []
    for spec in args[2:]:
        path, _, col = spec.partition(':')
        rgb = tuple(int(col.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) if col else (180, 186, 196)
        meshes.append((trimesh.load(path, force='mesh'), rgb))
    render(meshes, view, out, title=title)
