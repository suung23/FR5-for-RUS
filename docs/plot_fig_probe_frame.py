#!/usr/bin/env python3
"""프로브 프레임 그림 — 위치 · 축 정렬 · roll/pitch/yaw.

    python3 docs/plot_fig_probe_frame.py

형상은 **지어내지 않는다.** 브래킷과 프로브는 시뮬레이터가 쓰는 그 솔리드
(`teleop_gui/public/models/probe/*.stl`) 를 직교투영으로 직접 래스터라이즈하고,
치수·각도는 `fr5_control/fr5_control/config/probe.yaml` 의 값을 그대로 읽어 적는다.
어댑터와 PX6D 만은 솔리드가 없어 실측 길이의 원통으로 세운다 (10 / 23 mm).

패널
  (a) 적층 입면 — 플랜지 → 어댑터 → PX6D → 마운트 → 프로브, 10+23+152+49 = 234 mm
  (b) 축방향 등록 — 플랜지 +x 기준 45° (클램프 CAD) · +47° (센서 채널) · +90° (배열)
  (c) 프로브 프레임 {P} 와 영상면 Π, 그리고 콘솔이 이 축을 그리는 방식
  (d,e,f) roll ω_x · pitch ω_y · yaw ω_z — 회전 중심은 배열 면 (DESIGN_NOTES §4.5)

근거: DESIGN_NOTES §4.1 · §4.2 · §4.3 · §4.5 · §10.4,
      teleop_gui/src/components/ProbeAssembly.tsx (STACK, PROBE_YAW_DEG).
"""
from __future__ import annotations

import os
import struct

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Arc, Circle, FancyArrowPatch       # noqa: E402
from matplotlib.path import Path                                  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_MESHDIR = os.path.join(_REPO, "teleop_gui", "public", "models", "probe")
OUT = os.path.join(_HERE, "figures", "fig_probe_frame_axes")

# ---------------------------------------------------------------- 팔레트
INK, SUB = "#141A21", "#5A6068"
EDGE, GHOST = "#4F5459", "#A8AEB6"
GREEN, ACCENT, CRIT = "#17683a", "#4b8b5a", "#8b1e1e"
ALU, GRAPHITE, SHELL = "#b4b7b2", "#5c5f5a", "#d7d9d6"
PANEL = "#F5F5F5"

# ------------------------------------------------- probe.yaml / STACK 값
ADAPTER, SENSOR, MOUNT, PROBE_EXP = 0.010, 0.023, 0.152, 0.049
Z_SENSOR = ADAPTER                      # 0.010  플랜지 → 센서 바닥
Z_MOUNT = ADAPTER + SENSOR              # 0.033  = ft_sensor.j6_to_sensor_xyz[2]
Z_PROBE = ADAPTER + SENSOR + MOUNT      # 0.185  클램프 면
Z_TOTAL = Z_PROBE + PROBE_EXP           # 0.234  = tool.j6_to_probe_xyz[2]

PROBE_YAW = 90.0                        # tool.j6_to_probe_rpy[2] = 1.5708 rad
SENSOR_YAW = 47.0                       # ft_sensor.j6_to_sensor_rpy[2] = 0.8203
CLAMP_CAD = 45.0                        # 본체3.stl 클램프 장축, 실측 아님 — CAD
LEVER = Z_TOTAL - Z_MOUNT               # 0.201  r_{S→P}

ARRAY_R = 0.0826                        # 볼록 배열 곡률, CAD 피팅 (잔차 0.14 mm)
ARRAY_HALF = float(np.arcsin(0.040 / ARRAY_R))

FIG_W, FIG_H = 16.4, 9.6
AX = [0.020, 0.014, 0.962, 0.972]

# GUI ToolFrame 축 길이 (Fr5Model.tsx) — 색이 아니라 길이로 축을 가른다
GUI_AXIS_LEN = (0.075, 0.055, 0.038)


# ==================================================================== STL
def load_stl(path: str) -> np.ndarray:
    """이진 STL → (N,3,3) 삼각형 배열, 단위 m."""
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(84)[80:84])[0]
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    return raw[:, 12:48].copy().view(np.float32).reshape(n, 3, 3).astype(np.float64)


def rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    return rz(deg)


def xform(tris: np.ndarray, R: np.ndarray | None = None,
          t: np.ndarray | None = None) -> np.ndarray:
    out = tris if R is None else tris @ R.T
    return out if t is None else out + np.asarray(t, float)


def cylinder(r: float, z0: float, z1: float, n: int = 72) -> np.ndarray:
    """축이 +z 인 닫힌 원통. 솔리드가 없는 어댑터·PX6D 용."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p0 = np.stack([r * np.cos(a), r * np.sin(a), np.full(n, z0)], 1)
    p1 = np.stack([r * np.cos(a), r * np.sin(a), np.full(n, z1)], 1)
    j = np.roll(np.arange(n), -1)
    tris = [np.stack([p0, p1, p1[j]], 1), np.stack([p0, p1[j], p0[j]], 1)]
    for z, cap in ((z1, np.array([0.0, 0.0, z1])), (z0, np.array([0.0, 0.0, z0]))):
        ring = p1 if z == z1 else p0
        tris.append(np.stack([np.repeat(cap[None], n, 0), ring, ring[j]], 1))
    return np.concatenate(tris, 0)


# ============================================================= 직교 래스터
def basis(eye: np.ndarray, up_hint=(0.0, 0.0, -1.0)) -> np.ndarray:
    """eye = 장면에서 카메라를 향하는 단위벡터. 행 = [right, up, eye].

    ⚠ right × up = eye 여야 한다. 이 부호를 틀리면 그림이 좌우로 뒤집히는데,
    프로브가 거의 대칭이라 **눈으로는 안 보인다** — (b) 의 +47° 가 조용히
    반대로 돌 뿐이다. 그래서 규약을 여기 한 곳에서만 만든다.
    """
    e = np.asarray(eye, float)
    e /= np.linalg.norm(e)
    right = np.cross(np.asarray(up_hint, float), e)
    right /= np.linalg.norm(right)
    up = np.cross(e, right)
    return np.stack([right, up, e])


def proj(p, B: np.ndarray):
    """세계 좌표 → 화면 (u, w)."""
    v = np.asarray(p, float) @ B.T
    return (v[..., 0], v[..., 1])


#: 화면 좌표계(u, w, 카메라 쪽) 기준 조명. 좌상 전방.
_LIGHT = np.array([-0.38, 0.52, 0.76])
_LIGHT /= np.linalg.norm(_LIGHT)


def rasterize(tris: np.ndarray, B: np.ndarray, extent, shape, color,
              ambient: float = 0.36):
    """z-버퍼 평면 셰이딩. (rgba, mask) 반환.

    벡터 그래픽으로 3만 면을 넣으면 PDF 가 감당이 안 되고 면 사이 이음선이 남는다.
    래스터로 굽고 imshow 로 얹으면 파일도 작고 실루엣도 깨끗하다.
    """
    H, W = shape
    x0, x1, y0, y1 = extent
    v = tris @ B.T                                  # (N,3,3) → u, w, d
    px = (v[:, :, 0] - x0) / (x1 - x0) * W - 0.5
    py = (y1 - v[:, :, 1]) / (y1 - y0) * H - 0.5
    d = v[:, :, 2]

    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    ln = np.linalg.norm(n, axis=1)
    keep = ln > 1e-12
    n = n[keep] / ln[keep, None]
    n[n[:, 2] < 0] *= -1.0                          # 보이는 쪽으로 뒤집는다
    shade = ambient + (1.0 - ambient) * np.clip(n @ _LIGHT, 0.0, 1.0)
    px, py, d = px[keep], py[keep], d[keep]

    zbuf = np.full((H, W), -np.inf)
    val = np.zeros((H, W))
    hit = np.zeros((H, W), bool)

    xi0 = np.clip(np.floor(px.min(1)).astype(int), 0, W - 1)
    xi1 = np.clip(np.ceil(px.max(1)).astype(int) + 1, 0, W)
    yi0 = np.clip(np.floor(py.min(1)).astype(int), 0, H - 1)
    yi1 = np.clip(np.ceil(py.max(1)).astype(int) + 1, 0, H)

    for k in range(px.shape[0]):
        cx0, cx1, cy0, cy1 = xi0[k], xi1[k], yi0[k], yi1[k]
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        ax_, ay_ = px[k, 0], py[k, 0]
        bx, by = px[k, 1], py[k, 1]
        cx, cy = px[k, 2], py[k, 2]
        area = (bx - ax_) * (cy - ay_) - (by - ay_) * (cx - ax_)
        if abs(area) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(cx0, cx1), np.arange(cy0, cy1))
        w0 = ((bx - ax_) * (gy - ay_) - (by - ay_) * (gx - ax_)) / area
        w1 = ((cx - bx) * (gy - by) - (cy - by) * (gx - bx)) / area
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        if not inside.any():
            continue
        # 무게중심: w1→A, (1-w0-w1)→B, w0→C 순서가 위 부호와 맞는다
        la, lb, lc = w1[inside], 1.0 - w0[inside] - w1[inside], w0[inside]
        z = la * d[k, 0] + lb * d[k, 1] + lc * d[k, 2]
        yy, xx = gy[inside], gx[inside]
        better = z > zbuf[yy, xx]
        if not better.any():
            continue
        yy, xx, z = yy[better], xx[better], z[better]
        zbuf[yy, xx] = z
        val[yy, xx] = shade[k]
        hit[yy, xx] = True

    base = np.array(matplotlib.colors.to_rgb(color))
    rgba = np.zeros((H, W, 4))
    rgba[..., :3] = np.clip(val[..., None] * base, 0, 1)
    rgba[..., 3] = hit.astype(float)
    return rgba, hit


def show(ax, tris, B, extent, color, res=1000, zorder=2, alpha=1.0):
    x0, x1, y0, y1 = extent
    W = res
    H = max(8, int(round(res * (y1 - y0) / (x1 - x0))))
    rgba, mask = rasterize(tris, B, extent, (H, W), color)
    rgba[..., 3] *= alpha
    ax.imshow(rgba, extent=extent, origin="upper", interpolation="bilinear",
              zorder=zorder)
    return mask


def silhouette(ax, tris, B, extent, res=800, lw=1.0, ec=EDGE, fc=None,
               ls="-", zorder=3, alpha=1.0, zorder_edge=None):
    """실루엣 외곽선(+선택적 면). 겹쳐 그릴 유령 자세에 쓴다.

    yaw 처럼 회전이 실루엣을 **줄이는** 경우 유령이 실물 뒤로 완전히 숨는다.
    그때는 zorder_edge 로 외곽선만 위에 얹어야 자세 변화가 보인다.
    """
    x0, x1, y0, y1 = extent
    W = res
    H = max(8, int(round(res * (y1 - y0) / (x1 - x0))))
    _, mask = rasterize(tris, B, extent, (H, W), "#000000")
    gx = np.linspace(x0, x1, W)
    gy = np.linspace(y1, y0, H)
    m = mask.astype(float)
    if fc is not None:
        ax.contourf(gx, gy, m, levels=[0.5, 1.5], colors=[fc], alpha=alpha,
                    zorder=zorder - 0.1)
    ax.contour(gx, gy, m, levels=[0.5], colors=[ec], linewidths=lw,
               linestyles=[ls],
               zorder=zorder if zorder_edge is None else zorder_edge)


# ============================================================== 그림 헬퍼
def font() -> str:
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for fam in ("Arial", "Helvetica", "Inter", "Liberation Sans", "DejaVu Sans"):
        if fam in have:
            return fam
    return "DejaVu Sans"


def fig_xy(x, y):
    return AX[0] + AX[2] * x / 100.0, AX[1] + AX[3] * y / 100.0


def sub_ax(fig, dbox, extent):
    x0, y0 = fig_xy(dbox[0], dbox[1])
    x1, y1 = fig_xy(dbox[2], dbox[3])
    a = fig.add_axes([x0, y0, x1 - x0, y1 - y0])
    a.set_aspect("equal")
    a.axis("off")
    a.set_xlim(extent[0], extent[1])
    a.set_ylim(extent[2], extent[3])
    return a


def t(ax, x, y, s, size=9.0, color=INK, weight="normal", ha="left",
      va="center", halo=False):
    bb = dict(fc="white", ec="none", pad=0.8, alpha=0.82) if halo else None
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha,
            va=va, zorder=6, bbox=bb)


def tag(bg, x, y, letter, title, sub=None):
    t(bg, x, y, letter, 11.2, INK, "bold")
    t(bg, x + 3.0, y, title, 10.4, INK, "bold")
    if sub:
        t(bg, x + 3.0, y - 2.5, sub, 8.5, SUB)


def arrow(ax, p0, p1, color=INK, lw=1.1, style="-|>", ms=7, zorder=5, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=zorder,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def arc3d(ax, B, centre, axis, radius, a0, a1, color=GREEN, lw=1.5, n=48,
          head=True, zorder=5):
    """3-D 원호를 투영해 그린다. 회전축 둘레의 화살표용."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(axis @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    th = np.radians(np.linspace(a0, a1, n))
    pts = (np.asarray(centre, float)[None]
           + radius * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2))
    u, w = proj(pts, B)
    ax.plot(u[:-2], w[:-2], color=color, lw=lw, zorder=zorder,
            solid_capstyle="round")
    if head:
        arrow(ax, (u[-3], w[-3]), (u[-1], w[-1]), color=color, lw=lw, ms=8,
              zorder=zorder)


def sector(R=ARRAY_R, half=ARRAY_HALF, depth=0.050, n=40):
    """볼록 배열의 B-mode 부채꼴, 프로브 좌표계. 곡률 중심은 z = -R."""
    th = np.linspace(-half, half, n)
    inner = np.stack([R * np.sin(th), np.zeros(n), -R + R * np.cos(th)], 1)
    ro = R + depth
    outer = np.stack([ro * np.sin(th[::-1]), np.zeros(n),
                      -R + ro * np.cos(th[::-1])], 1)
    return np.concatenate([inner, outer], 0)


def poly3d(ax, pts, B, fc, ec=None, alpha=0.5, lw=0.8, zorder=1):
    u, w = proj(pts, B)
    ax.fill(u, w, facecolor=fc, edgecolor=ec or "none", alpha=alpha, lw=lw,
            zorder=zorder)


# ============================================================== 지오메트리
MOUNT_STL = load_stl(os.path.join(_MESHDIR, "probe_mount.stl"))
PROBE_STL = load_stl(os.path.join(_MESHDIR, "probe_4c_rs.stl"))

#: 프로브 솔리드는 배열 면이 원점이고 축이 이미 §4.1 규약이다 (+z 조직, +x 배열).
PROBE_HEAD = PROBE_STL[(PROBE_STL[:, :, 2] > -0.075).all(1)]

#: 플랜지 프레임의 적층. 마운트는 CAD 45° 를 영상면 위로 돌려 앉힌다.
STACK_TRIS = [
    (cylinder(0.0405, 0.0, ADAPTER), ALU),
    (cylinder(0.0375, Z_SENSOR, Z_MOUNT), GRAPHITE),
    (cylinder(0.0345, Z_SENSOR + 0.0075, Z_MOUNT - 0.0075), "#43463f"),
    (xform(MOUNT_STL, rz(PROBE_YAW - CLAMP_CAD), [0, 0, Z_MOUNT]), ALU),
    (xform(PROBE_STL, rz(PROBE_YAW), [0, 0, Z_TOTAL]), SHELL),
]

# ================================================================= (a) 입면
def panel_a(fig, bg, box):
    # 영상면을 지면에 둔다. 화면 오른쪽 = 프로브 +x, 화면 위 = 플랜지 -z,
    # 카메라는 프로브 +y (elevational) 쪽에 선다.
    R = rz(PROBE_YAW)
    B = np.stack([R @ np.array([1.0, 0, 0]),
                  np.array([0.0, 0, -1.0]),
                  R @ np.array([0.0, 1.0, 0])])
    ext = (-0.116, 0.116, -0.262, 0.028)
    ax = sub_ax(fig, box, ext)

    for tris, c in STACK_TRIS:
        show(ax, tris, B, ext, c, res=780)

    # 공통 축 — "적층이 전부 동축" 이 이 그림의 전제다
    ax.plot([0, 0], [0.020, -0.250], color=GHOST, lw=0.7, ls=(0, (7, 2, 1, 2)),
            zorder=4)

    # 플랜지 면
    ax.plot([-0.052, 0.052], [0, 0], color=INK, lw=1.4, zorder=5)
    for x in np.linspace(-0.050, 0.046, 9):
        ax.plot([x, x + 0.008], [0.0, 0.008], color=SUB, lw=0.7, zorder=5)

    # 치수 사슬
    xd = 0.072
    stops = [(0.0, ""), (-ADAPTER, "10"), (-Z_MOUNT, "23"), (-Z_PROBE, "152"),
             (-Z_TOTAL, "49")]
    ax.plot([xd, xd], [0, -Z_TOTAL], color=SUB, lw=0.7, zorder=5)
    prev = 0.0
    for z, lab in stops:
        ax.plot([0.044, xd + 0.004], [z, z], color=SUB, lw=0.5, ls=":", zorder=4)
        ax.plot([xd - 0.003, xd + 0.003], [z, z], color=SUB, lw=1.0, zorder=5)
        if lab:
            t(ax, xd + 0.006, (z + prev) / 2, lab, 8.4, INK)
            prev = z

    xt = 0.104
    arrow(ax, (xt, 0.0), (xt, -Z_TOTAL), color=INK, lw=1.0, style="<|-|>", ms=7)
    ax.text(xt - 0.004, -Z_TOTAL / 2, "234 mm", fontsize=10.0, color=INK,
            ha="right", va="center", rotation=90, zorder=6, fontweight="bold")

    # 프레임 세 개.  {P} 만 점 위쪽에 적는다 — 아래는 축 삼각대가 쓴다
    def mark(z, name, note, above=False):
        ax.plot([0], [-z], marker="o", ms=4.0, mfc="white", mec=INK, mew=1.1,
                zorder=6)
        y0 = -z + (0.015 if above else -0.006)
        t(ax, -0.058, y0, name, 9.2, INK, "bold", ha="right")
        t(ax, -0.058, y0 - 0.008, note, 7.8, SUB, ha="right")

    mark(0.0, "{F}", "J6 flange")
    mark(Z_MOUNT, "{S}", "PX6D, 33 mm")
    mark(Z_TOTAL, "{P}", "array face, 234 mm", above=True)

    # 레버암 — wrench 기준점 이동 (§4.3)
    xl = -0.104
    arrow(ax, (xl, -Z_MOUNT), (xl, -Z_TOTAL), color=CRIT, lw=1.0,
          style="<|-|>", ms=7)
    ax.text(xl - 0.004, -(Z_MOUNT + Z_TOTAL) / 2, "r = 201 mm", fontsize=8.4,
            color=CRIT, ha="right", va="center", rotation=90, zorder=6)

    # {P} 축 — 영상면 안의 두 축은 실선, elevational 은 지면 밖이라 ⊗
    o = np.array([0.0, -Z_TOTAL])
    arrow(ax, o, o + [0.040, 0], color=GREEN, lw=1.8, ms=9)
    t(ax, 0.043, -Z_TOTAL + 0.001, "+x", 9.2, GREEN, "bold")
    arrow(ax, o, o + [0, -0.020], color=GREEN, lw=1.8, ms=9)
    t(ax, -0.005, -Z_TOTAL - 0.017, "+z", 9.2, GREEN, "bold", ha="right")
    ax.plot([o[0]], [o[1]], marker="o", ms=8.0, mfc="white", mec=GREEN, mew=1.3,
            zorder=6)
    ax.plot([o[0]], [o[1]], marker="x", ms=4.5, color=GREEN, mew=1.3, zorder=7)
    t(ax, 0.008, -Z_TOTAL - 0.009, "+y", 8.6, GREEN, "bold")

    t(ax, -0.114, -0.246, "measured   10 · 23 · 49", 7.9, SUB)
    t(ax, -0.114, -0.256, "CAD          152", 7.9, SUB)
    t(ax, -0.114, 0.022, "image plane in the page;  elevational into it",
      7.9, SUB)


# ============================================================ (b) 축방향 등록
def panel_b(fig, bg, box):
    # 배열 면을 정면으로 본다: +z 가 지면 밖으로 나오므로 +47° 가 반시계다
    B = np.stack([np.array([1.0, 0, 0]), np.array([0.0, 1.0, 0]),
                  np.array([0.0, 0, 1.0])])
    ext = (-0.122, 0.122, -0.076, 0.076)
    ax = sub_ax(fig, box, ext)

    ax.add_patch(Circle((0, 0), 0.0405, fc="none", ec=GHOST, lw=0.9, ls=":",
                        zorder=1))
    ax.add_patch(Circle((0, 0), 0.0375, fc="none", ec=GRAPHITE, lw=1.0,
                        zorder=1))
    silhouette(ax, xform(MOUNT_STL, rz(PROBE_YAW - CLAMP_CAD)), B, ext,
               fc="#eaece9", ec=GHOST, lw=0.9, zorder=2)
    silhouette(ax, xform(PROBE_HEAD, rz(PROBE_YAW)), B, ext,
               fc="#d7d9d6", ec=EDGE, lw=1.1, zorder=3)

    rays = [(0.0, 0.076, "flange +x", INK, 1.2, "-"),
            (CLAMP_CAD, 0.078, None, GHOST, 1.1, (0, (5, 2))),
            (SENSOR_YAW, 0.062, "PX6D +x", GRAPHITE, 1.3, "-"),
            (PROBE_YAW, 0.070, "probe +x", GREEN, 1.7, "-")]
    for a, r, lab, c, lw, ls in rays:
        p = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) * r
        arrow(ax, (0, 0), tuple(p), color=c, lw=lw, ms=8, ls=ls)
        if lab:
            q = p * (1 + 0.007 / r)
            t(ax, q[0], q[1], lab, 8.8, c, "bold",
              ha="left" if q[0] > -0.004 else "center")

    for r, a0, a1, lab, c in ((0.050, 0.0, SENSOR_YAW, "+47°", GRAPHITE),
                              (0.060, SENSOR_YAW, PROBE_YAW, "+43°", GREEN)):
        ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=a0, theta2=a1, color=c,
                         lw=1.1, zorder=5))
        m = np.radians((a0 + a1) / 2)
        t(ax, (r + 0.006) * np.cos(m), (r + 0.006) * np.sin(m), lab, 8.8, c,
          "bold", ha="center")

    ax.plot([0], [0], marker="+", ms=9, color=INK, mew=1.2, zorder=6)

    leg = [(GRAPHITE, "-", "PX6D channel +x — derived, +47°"),
           (GHOST, (0, (5, 2)), "bracket clamp axis — CAD, 45°")]
    for i, (c, ls, lab) in enumerate(leg):
        y = 0.068 - i * 0.010
        ax.plot([-0.118, -0.106], [y, y], color=c, lw=1.2, ls=ls, zorder=5)
        t(ax, -0.102, y, lab, 8.0, SUB)

    t(ax, -0.118, -0.056,
      r"$^{F}\!R_{S} = R_z(+90^\circ)\cdot R_z(-43^\circ) = R_z(+47^\circ)$",
      9.6, INK)
    t(ax, -0.118, -0.068,
      "CAD clamp 45° vs. as-built 43°: the 2° is assembly, not design", 8.0, SUB)
    t(ax, 0.118, 0.068, "array face toward the reader,  +z out of the page",
      8.0, SUB, ha="right")


# ========================================================== (c) 프로브 프레임
BISO = basis(np.array([0.45, 0.80, 0.32]))


def panel_c(fig, bg, box):
    ext = (-0.205, 0.110, -0.072, 0.084)
    ax = sub_ax(fig, box, ext)

    poly3d(ax, sector(depth=0.050), BISO, "#dfe7e1", ec=ACCENT, alpha=0.85,
           lw=0.9, zorder=1)
    show(ax, PROBE_HEAD, BISO, ext, SHELL, res=980, zorder=2)

    axes = [(np.array([1.0, 0, 0]), GUI_AXIS_LEN[0], "+x", "left"),
            (np.array([0.0, 1.0, 0]), GUI_AXIS_LEN[1], "+y", "right"),
            (np.array([0.0, 0, 1.0]), GUI_AXIS_LEN[2], "+z", "left")]
    for d, L, lab, ha in axes:
        u, w = proj(np.stack([np.zeros(3), d * L]), BISO)
        arrow(ax, (u[0], w[0]), (u[1], w[1]), color=GREEN, lw=2.0, ms=10,
              zorder=6)
        dx = 0.006 if ha == "left" else -0.006
        t(ax, u[1] + dx, w[1] + 0.002, lab, 10.2, GREEN, "bold", ha=ha,
          halo=True)

    ax.plot([0], [0], marker="o", ms=5.2, mfc="white", mec=GREEN, mew=1.4,
            zorder=7)
    t(ax, 0.011, -0.009, "{P}", 9.8, GREEN, "bold", halo=True)

    t(ax, -0.006, -0.062, r"$\Pi$   image plane = $x$–$z$", 9.0, ACCENT,
      "bold")

    t(ax, -0.201, -0.020, "console triad — told apart by length, not colour",
      8.2, INK, "bold")
    for i, (sym, txt, ln) in enumerate(
            [("+x", "lateral, inside the image plane", 75),
             ("+y", "elevational, out of the plane", 55),
             ("+z", "axial, into the tissue", 38)]):
        y = -0.032 - i * 0.011
        t(ax, -0.201, y, sym, 9.0, GREEN, "bold")
        t(ax, -0.187, y, txt, 8.2, INK)
        t(ax, -0.070, y, f"{ln} mm", 8.2, SUB, ha="right")

    notes = [(0.078, "{P} origin = array face centre, 234 mm from J6", INK,
              "bold"),
             (0.067, "convex array R = 82.6 mm, sector ±29° (from the solid)",
              SUB, "normal"),
             (0.050, "[!]  Fr5Model.tsx draws this triad in the flange", CRIT,
              "normal"),
             (0.039, "      frame — the +90° is missing from it", CRIT,
              "normal")]
    for y, line, c, wt in notes:
        t(ax, -0.201, y, line, 8.2, c, wt)


# ====================================================== (d,e,f) roll·pitch·yaw
def panel_rpy(fig, bg, box, axis, ang, note):
    ext = (-0.138, 0.138, -0.070, 0.090)
    ax = sub_ax(fig, box, ext)
    d = {"x": np.array([1.0, 0, 0]), "y": np.array([0.0, 1.0, 0]),
         "z": np.array([0.0, 0, 1.0])}[axis]
    R = rot(axis, ang)

    poly3d(ax, sector(depth=0.050) @ R.T, BISO, GHOST, ec=GHOST, alpha=0.28,
           lw=0.7, zorder=0)
    silhouette(ax, PROBE_HEAD @ R.T, BISO, ext, ec=SUB, lw=1.0,
               ls=(0, (4, 2)), fc="#edefec", zorder=1, zorder_edge=4.5)
    poly3d(ax, sector(depth=0.050), BISO, "#dfe7e1", ec=ACCENT, alpha=0.85,
           lw=0.9, zorder=2)
    show(ax, PROBE_HEAD, BISO, ext, SHELL, res=820, zorder=3)

    L = 0.064
    u, w = proj(np.stack([-d * 0.020, d * L]), BISO)
    ax.plot(u, w, color=GREEN, lw=1.3, ls=(0, (5, 2)), zorder=6)
    t(ax, u[1] + 0.005, w[1] + 0.003, "+" + axis, 9.6, GREEN, "bold")

    a0 = {"x": -75.0, "y": -75.0, "z": -115.0}[axis]
    arc3d(ax, BISO, d * 0.032, d, 0.028, a0, a0 + 175.0, color=GREEN, lw=1.7)

    ax.plot([0], [0], marker="o", ms=4.8, mfc="white", mec=GREEN, mew=1.3,
            zorder=7)
    ax.plot([-0.062, -0.010], [-0.026, -0.003], color=SUB, lw=0.6, zorder=6)
    t(ax, -0.064, -0.027, "rotation centre", 7.8, SUB, ha="right")

    t(ax, -0.135, -0.062, note, 8.4, INK)
    t(ax, 0.135, -0.062, f"drawn at {ang:.0f}°", 7.9, GHOST, ha="right")


# ==================================================================== 조립
def build():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 100)
    bg.set_ylim(0, 100)
    bg.axis("off")

    tag(bg, 0.6, 98.2, "(a)", "Tool stack, seen in the image plane",
        "flange → adapter → PX6D → bracket → probe,  all coaxial")
    panel_a(fig, bg, (0.5, 40.0, 24.0, 95.0))

    tag(bg, 26.0, 98.2, "(b)", "Axial registration",
        "who is turned by how much about the common +z")
    panel_b(fig, bg, (25.5, 61.5, 57.5, 95.0))

    tag(bg, 60.0, 98.2, "(c)", "Probe frame {P} and the image plane Π",
        "DESIGN_NOTES §4.1 — +z into tissue, +x along the array")
    panel_c(fig, bg, (59.5, 61.5, 99.5, 95.0))

    # ---- 중간 띠: 프레임 사슬 + 유보 사항 -----------------------------
    bg.add_patch(plt.Rectangle((25.5, 40.0), 74.0, 18.5, fc=PANEL, ec=EDGE,
                               lw=0.9, zorder=0))
    t(bg, 27.0, 55.8, "Frame chain", 10.2, INK, "bold")
    chain = [(27.0, "{base}", 4.6), (36.0, "{F}  J6 flange", 9.5),
             (52.0, "{S}  PX6D", 7.0), (68.0, "{P}  array face", 9.8)]
    for i, (x, lab, wdt) in enumerate(chain):
        t(bg, x, 51.6, lab, 9.4, INK, "bold")
        if i:
            bg.annotate("", xy=(x - 1.2, 51.6),
                        xytext=(chain[i - 1][0] + chain[i - 1][2] + 1.2, 51.6),
                        arrowprops=dict(arrowstyle="-|>", color=SUB, lw=1.0),
                        zorder=5)
    t(bg, 33.0, 53.6, "J1 .. J6", 7.8, SUB, ha="center")
    t(bg, 48.4, 53.6, "33 mm,  $R_z(+47^\\circ)$", 7.8, SUB, ha="center")
    t(bg, 64.4, 53.6, "234 mm,  $R_z(+90^\\circ)$", 7.8, SUB, ha="center")

    t(bg, 27.0, 47.6,
      r"$^{F}T_{P} = \mathrm{Trans}(0,0,0.234)\cdot R_z(+90^\circ)$", 9.2, INK)
    t(bg, 45.0, 47.6, "tool.j6_to_probe_*", 8.4, SUB)
    t(bg, 27.0, 44.6,
      r"$^{F}T_{S} = \mathrm{Trans}(0,0,0.033)\cdot R_z(+47^\circ)$", 9.2, INK)
    t(bg, 45.0, 44.6, "ft_sensor.j6_to_sensor_*", 8.4, SUB)
    t(bg, 27.0, 41.6,
      r"$M_{P} = M_{S} + r \times F,\ \ r = 201\ \mathrm{mm}\ \hat z_{P}$",
      9.2, INK)
    t(bg, 45.0, 41.6, "§4.3 wrench reference shift", 8.4, SUB)

    t(bg, 67.0, 47.6, "[!]  +47° is derived (90 − 43), never measured on its own",
      8.4, CRIT)
    t(bg, 67.0, 44.9, "[!]  probe insertion is the measured 49 mm; the solid's",
      8.4, CRIT)
    t(bg, 67.0, 43.1, "      only hard stop would put it at 72 mm, because the",
      8.4, CRIT)
    t(bg, 67.0, 41.3, "      clamp is a smooth channel and the probe slides in it",
      8.4, CRIT)

    # ---- 아래 줄: roll · pitch · yaw ----------------------------------
    rows = [
        ((0.5, 2.0, 32.0, 33.0), "x", 22.0, "(d)", "roll   $\\omega_x$",
         "about the lateral axis → out-of-plane tilt (fanning)",
         "new section · admittance drives $M_x \\rightarrow 0$"),
        ((34.0, 2.0, 65.5, 33.0), "y", 22.0, "(e)", "pitch   $\\omega_y$",
         "about the elevational axis → in-plane rock (heel–toe)",
         "same section · admittance drives $M_y \\rightarrow 0$"),
        ((67.5, 2.0, 99.0, 33.0), "z", 45.0, "(f)", "yaw   $\\omega_z$",
         "about the beam axis → image-plane rotation",
         "new section · image policy holds this axis"),
    ]
    for box, axis, ang, lt, title, sub1, sub2 in rows:
        tag(bg, box[0] + 0.4, 37.2, lt, title, sub1)
        panel_rpy(fig, bg, box, axis, ang, sub2)

    t(bg, 0.5, 0.6,
      "Rotation centre is the tool-frame origin (§4.5). An offset $\\delta$ "
      "between it and the true array face turns a commanded $\\omega$ into "
      "contact slip $v_{slip} = \\omega \\times \\delta$: "
      "$\\omega$ = 0.2 rad/s with $\\delta$ = 20 mm gives 4 mm/s, 40% of the "
      "10 mm/s contact translation budget.", 8.4, SUB)
    t(bg, 99.5, 0.6,
      "teleop mapping:  $A = R_z(\\mathrm{tip\\_roll})\\cdot"
      "\\mathrm{diag}(+1,-1,-1)$,   tip_roll = 0°", 8.4, SUB, ha="right")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        path = f"{OUT}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None,
                    facecolor="white")
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams["font.family"] = [font()]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    build()
