#!/usr/bin/env python3
"""CAD STL 두 개를 콘솔이 그릴 수 있는 형태로 옮긴다 (2026-08-27).

들어오는 것은 조작자가 받은 CAD 그대로다 — ASCII STL, 밀리미터, 부품 자기
좌표계. 나가는 것은 **바이너리 STL · 미터 · 조립 기준점이 원점** 이고, 삼각형
수는 브라우저가 감당할 만큼 줄인 것이다.

왜 변환을 스크립트로 두는가
---------------------------
기준점을 TSX 안의 상수로 적어 두면 CAD 가 바뀔 때 그 숫자가 왜 그 값인지 아무도
모른다. 여기서는 **메시에서 직접 잰다** — 프로브의 배열 면 꼭짓점은 원호를 맞춰
찾고, 마운트의 축은 바닥 디스크 중심에서 찾는다. CAD 가 바뀌면 다시 돌리면 된다.

기준점
------
``probe_4c_rs.stl``
    원점 = **배열 면 중심**(볼록 원호의 꼭짓점). 축은 프로브 규약으로 돌려 둔다
    (DESIGN_NOTES §4.1): ``+z`` 침투, ``+x`` 영상면 내 lateral, ``+y`` elevational.
    CAD 의 ``+Y`` 가 침투 방향이고 ``+X`` 가 배열 장축이므로 ``Rx(+90°)`` 하나다.

``probe_mount.stl``
    원점 = **바닥 디스크 중심의 아랫면**. 즉 PX6D 윗면에 닿는 면이다. 회전은
    건드리지 않는다 — 클램프가 CAD 에서 정확히 45.00° 에 있고, 그 각도 자체가
    표시할 값이라 여기서 지워 버리면 안 된다. 놓는 각도는 TSX 가 정한다.

실측과의 대조 (probe.yaml)
--------------------------
    마운트②  실측 161 mm  ·  **CAD 152.0 mm**  → CAD 를 쓴다
    프로브    실측  49 mm  ·  CAD 로는 정할 수 없다  → 실측을 쓴다

솔리드가 있는 구간은 솔리드를 쓴다는 규칙이고, 그래서 툴 길이가 243 → 234 mm 로
줄었다 (2026-08-27).

프로브 삽입 깊이만 예외인데, **CAD 가 그것을 정하지 못하기 때문** 이다. 클램프는
매끈한 채널이라 프로브가 그 안에서 미끄러진다. 솔리드에 있는 유일한 하드 스토퍼는
케이블 꼬리 대 중앙 보어(Ø11 mm)이고, 간섭이 사라질 때까지 빼면 노출이 72 mm
(총 257 mm)로 나온다 — 실측 49 mm 와 23 mm 차이다. 그 값을 쓸 수는 없다.

⏳ 그 23 mm 는 아직 설명되지 않았다. 본체3.stl 이 지금 팔에 달린 브래킷의 개정판이
맞는지 확인할 것.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

#: 목표 삼각형 수. 브라우저가 STLLoader 로 한 번에 파싱하는 값이라 넉넉히 잡지
#: 않는다. FR5 링크 하나가 ~9 k 이므로 같은 자릿수에 둔다.
TARGET = {"probe": 28_000, "mount": 12_000}


def load_ascii(path: str) -> np.ndarray:
    """ASCII STL 을 ``(T, 3, 3)`` 으로 읽는다."""
    data = open(path, "rb").read()
    parts = data.split(b"vertex")[1:]
    out = np.empty((len(parts), 3), dtype=np.float64)
    for i, piece in enumerate(parts):
        out[i] = [float(x) for x in piece.split(None, 3)[:3]]
    return out.reshape(-1, 3, 3)


def write_binary(path: str, tris: np.ndarray) -> None:
    """바이너리 STL 로 쓴다. 법선은 정점에서 다시 계산한다."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    normal = np.cross(b - a, c - a)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, length, out=np.zeros_like(normal), where=length > 0)
    record = np.zeros(
        len(tris), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    )
    record["n"], record["v"] = normal, tris
    with open(path, "wb") as handle:
        handle.write(b"\0" * 80)
        handle.write(struct.pack("<I", len(tris)))
        handle.write(record.tobytes())


def cluster_decimate(tris: np.ndarray, cell: float) -> np.ndarray:
    """격자 한 칸의 정점을 무게중심 하나로 합친다.

    이 방식을 고른 이유는 **경계상자를 지키기 때문** 이다. 치수가 곧 이 화면의
    내용이므로, 형상을 예쁘게 만드느라 길이가 움직이면 안 된다. 정점은 자기 칸
    (수십 μm) 안에서만 이동한다.
    """
    verts = tris.reshape(-1, 3)
    keys = np.floor(verts / cell).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    reps = np.zeros((len(counts), 3))
    np.add.at(reps, inverse, verts)
    reps /= counts[:, None]
    idx = inverse.reshape(-1, 3)
    keep = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    return reps[idx[keep]]


def decimate_to(tris: np.ndarray, target: int, span: float) -> np.ndarray:
    """목표 삼각형 수에 닿을 때까지 격자를 키운다."""
    if len(tris) <= target:
        return tris
    cell = span / 400.0
    for _ in range(14):
        out = cluster_decimate(tris, cell)
        if len(out) <= target:
            return out
        cell *= 1.35
    return out


def fit_circle(xy: np.ndarray):
    """점들에 원을 맞춘다. 반환 ``(중심, 반지름, 최대잔차)``."""
    design = np.c_[xy, np.ones(len(xy))]
    rhs = (xy ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    centre = sol[:2] / 2.0
    radius = float(np.sqrt(sol[2] + centre @ centre))
    residual = float(np.abs(np.linalg.norm(xy - centre, axis=1) - radius).max())
    return centre, radius, residual


def prepare_probe(tris: np.ndarray) -> np.ndarray:
    """배열 면 꼭짓점을 원점으로, 축을 프로브 규약으로.

    CAD 축: ``+Y`` 가 침투 방향(볼록 면이 그쪽을 본다), ``+X`` 가 배열 장축,
    ``+Z`` 가 elevational. 케이블은 ``-Y`` 로 나간다.
    """
    verts = tris.reshape(-1, 3)

    # 배열 면: 얇은 중앙 띠에서 x 마다 가장 먼 y 를 모아 원을 맞춘다. 볼록 면이
    # 맞다면 잔차가 십분의 몇 mm 로 떨어진다 — 아니면 축 배정이 틀린 것이므로
    # 조용히 넘어가지 않고 멈춘다.
    band = verts[np.abs(verts[:, 2]) < 6.0]
    samples = []
    for lo in np.arange(-40, 40, 4.0):
        mask = (band[:, 0] >= lo) & (band[:, 0] < lo + 4.0)
        if mask.sum() >= 5:
            samples.append([lo + 2.0, band[mask][:, 1].max()])
    centre, radius, residual = fit_circle(np.array(samples))
    if residual > 1.5:
        raise SystemExit(
            f"배열 면이 원호로 안 맞는다 (잔차 {residual:.2f} mm). CAD 축 배정을 "
            "확인하라 — +Y 가 침투 방향이라는 전제가 깨졌다."
        )
    apex_y = float(verts[:, 1].max())
    lateral_c = float(centre[0])
    elevation_c = float((verts[:, 2].min() + verts[:, 2].max()) / 2.0)
    print(f"  배열 면 원호 R = {radius:.1f} mm (잔차 {residual:.2f} mm)")
    print(f"  기준점 = ({lateral_c:.2f}, {apex_y:.2f}, {elevation_c:.2f}) mm")

    moved = tris - np.array([lateral_c, apex_y, elevation_c])
    # Rx(+90°):  x→x,  y→z,  z→−y
    return np.stack([moved[..., 0], -moved[..., 2], moved[..., 1]], axis=-1)


def prepare_mount(tris: np.ndarray) -> np.ndarray:
    """바닥 디스크 중심을 원점으로. 회전은 그대로 둔다."""
    verts = tris.reshape(-1, 3)
    base_z = float(verts[:, 2].min())
    disc = verts[verts[:, 2] < base_z + 5.0]
    centre = np.array(
        [
            (disc[:, 0].min() + disc[:, 0].max()) / 2.0,
            (disc[:, 1].min() + disc[:, 1].max()) / 2.0,
        ]
    )
    radius = float(np.linalg.norm(disc[:, :2] - centre, axis=1).max())

    # 클램프 주축 — TSX 가 이 각도만큼 되돌려 놓는다. 값을 여기서 지우지 않고
    # 찍어 주는 이유는, 실측 장착각(43°)과 대조할 수 있어야 하기 때문이다.
    clamp = verts[verts[:, 2] > base_z + 95.0]
    centred = clamp[:, :2] - clamp[:, :2].mean(0)
    eigenvalues, eigenvectors = np.linalg.eigh(centred.T @ centred / len(centred))
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    print(f"  바닥 디스크 지름 {2 * radius:.1f} mm · 전체 높이 {verts[:, 2].ptp():.1f} mm")
    print(f"  클램프 주축 {angle:.2f}°  ← TSX 의 MOUNT_CLAMP_CAD_DEG 와 같아야 한다")
    return tris - np.array([centre[0], centre[1], base_z])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, help="GE 4C-RS 프로브 ASCII STL")
    parser.add_argument("--mount", required=True, help="프로브 마운트 ASCII STL")
    parser.add_argument("--out", default="public/models/probe", help="출력 디렉터리")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name, path, prepare in (
        ("probe", args.probe, prepare_probe),
        ("mount", args.mount, prepare_mount),
    ):
        print(f"{name}: {path}")
        tris = load_ascii(path)
        print(f"  들어온 삼각형 {len(tris):,}")
        tris = prepare(tris)
        span = float(np.ptp(tris.reshape(-1, 3), axis=0).max())
        tris = decimate_to(tris, TARGET[name], span)
        tris = tris / 1000.0                      # mm → m, 씬 단위
        out = os.path.join(args.out, f"probe_4c_rs.stl" if name == "probe" else "probe_mount.stl")
        write_binary(out, tris)
        low, high = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
        print(f"  나간 삼각형 {len(tris):,} · {os.path.getsize(out) / 1e6:.2f} MB")
        print(f"  경계상자 m  min {np.round(low, 4)}  max {np.round(high, 4)}")
        print(f"  → {out}")


if __name__ == "__main__":
    sys.exit(main())
