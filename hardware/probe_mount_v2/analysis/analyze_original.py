"""Slice-by-slice analysis of the original mount STL (bonche3_original.stl).

Writes analysis/slices_z.txt (one line per Z slice), analysis/slices_xy.txt (side holes),
and renders/orig_*.png views + analysis/slice_sheet.png.
"""
import sys, numpy as np, trimesh
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
m = trimesh.load(ROOT / 'reference/bonche3_original.stl', force='mesh')

def describe_ring(coords):
    c = np.asarray(coords)[:, :2]
    ctr = c.mean(axis=0)
    r = np.linalg.norm(c - ctr, axis=1)
    return ctr, r.min(), r.max(), c.min(axis=0), c.max(axis=0)

def slice_report(axis, values, fname):
    normal = np.zeros(3); normal[axis] = 1
    lines = []
    for v in values:
        origin = np.zeros(3); origin[axis] = v
        sec = m.section(plane_origin=origin, plane_normal=normal)
        if sec is None:
            lines.append(f'{v:7.2f} | empty'); continue
        p2, T = sec.to_2D()
        polys = p2.polygons_full
        parts = []
        for poly in polys:
            ext = np.asarray(poly.exterior.coords)
            # map 2D back to 3D to report in world coords
            ext3 = trimesh.transform_points(np.column_stack([ext, np.zeros(len(ext))]), T)
            keep = [i for i in range(3) if i != axis]
            e = ext3[:, keep]
            bmin, bmax = e.min(axis=0), e.max(axis=0)
            s = f'outer bbox {keep}: [{bmin[0]:7.2f},{bmax[0]:7.2f}]x[{bmin[1]:7.2f},{bmax[1]:7.2f}] area {poly.area:8.1f}'
            holes = []
            for ring in poly.interiors:
                h = np.asarray(ring.coords)
                h3 = trimesh.transform_points(np.column_stack([h, np.zeros(len(h))]), T)[:, keep]
                ctr = h3.mean(axis=0); r = np.linalg.norm(h3 - ctr, axis=1)
                ext_ = h3.max(axis=0) - h3.min(axis=0)
                holes.append(f'hole ctr({ctr[0]:7.2f},{ctr[1]:7.2f}) size {ext_[0]:5.2f}x{ext_[1]:5.2f} r[{r.min():4.2f}..{r.max():4.2f}]')
            parts.append(s + ((' | ' + '; '.join(holes)) if holes else ''))
        lines.append(f'{v:7.2f} | n={len(polys)} | ' + ' || '.join(parts))
    (ROOT / 'analysis' / fname).write_text('\n'.join(lines), encoding='utf-8')
    return lines

zs = np.arange(0.25, 152, 1.0)
lz = slice_report(2, zs, 'slices_z.txt')
print('\n'.join(lz))
