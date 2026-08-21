#!/usr/bin/env python3
"""수집 중 화면 — 지금 이 블록이 **쓸 수 있는 것을 모으고 있는지**를 실시간으로 본다.

    python3 monitor.py                 # 가장 최근 블록을 알아서 따라간다
    python3 monitor.py --block align   # 특정 블록만
    python3 monitor.py --text          # 창 없이 터미널에

왜 만들었나
-----------
2026-08-20 과 08-21 두 세션이 같은 방식으로 무너졌다. 데이터가 틀린 것이 아니라
**필요한 동작이 안 들어갔는데 그걸 한 시간 뒤 분석에서야 알았다.** 08-21 은 align
자세를 지시대로 12 개 만들었는데 유지 시간이 평균 1.3 s 라 (문턱 2.5 s) 쓸 수 있는
자세가 0 개였고, 정렬이 안 풀려 회전 지표가 통째로 날아갔다. mag_map 은 자세차
7.2 deg / 위치차 181 mm 로 쓸 수 있는 쌍이 0 개였다.

블록이 끝난 뒤 세는 것(`run_session.census`)은 이미 있다. 그건 "다시 딸까" 를
정해 주지만, **지금 3 초를 세고 있는지**는 못 알려 준다. 조작하는 사람이 보고 있어야
하는 값은 사후 요약이 아니라 지금 이 순간의 값이다.

포트를 안 여는 이유
-------------------
`log_imu` 가 `/dev/ttyACM0` 를 독점한다. 그래서 이 프로그램은 **그 로거가 쓰고 있는
CSV 를 따라 읽는다.** 같은 값을, 한 다리 건너, 지연 0.2 s 안쪽으로 본다. GT 가
필요한 항목(mag_map 의 자세차/위치차)만 ROS 토픽을 따로 구독한다 — 로봇 로거와
같은 토픽이고, 구독자가 둘이어도 서로 방해하지 않는다.

판정을 분석기와 **같은 코드로** 한다
------------------------------------
정지 판정도 문턱도 `qc_common` / `protocol` 에서 그대로 가져다 쓴다. 여기서 따로
세면 화면이 분석기보다 느슨해질 수 있고, 그러면 이 화면은 사람을 안심시키는 일만
하게 된다 — 08-21 의 census 가 정확히 그 실패였다 (0.8 s 로 세어 "정지 9 개").
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import protocol                                                 # noqa: E402
import analyze_track as at                                      # noqa: E402
import qc_common as qc                                          # noqa: E402
from log_imu import COL                                         # noqa: E402


REDRAW_HZ = 4.0
# 이만큼 새 샘플이 없으면 '수집 중이 아니다' 로 본다. log_imu 는 250 Hz 로 쓰고
# CSV 버퍼가 8 KB 라 정상 수집 중에는 0.2 s 를 넘지 않는다.
STALE_S = 3.0
# still_mask 는 샘플당 파이썬 루프라 still 블록(300 s x 250 Hz = 75k)을 매번
# 처음부터 돌리면 4 Hz 를 못 따라온다. 그래서 확정선 뒤만 다시 센다
# (BlockState._extend_mask).


# ------------------------------------------------------------------ CSV 따라읽기
def _is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


class CsvTail:
    """`log_imu` 가 쓰는 CSV 를 이어서 읽는다. 마지막 줄이 잘려 있으면 남겨 둔다."""

    def __init__(self, path):
        self.path = path
        self._fh = open(path, "r")
        self._partial = ""
        self.rows = []

    def poll(self):
        chunk = self._fh.read()
        if not chunk:
            return 0
        data = self._partial + chunk
        # 마지막 개행 뒤는 아직 쓰이는 중일 수 있다.
        cut = data.rfind("\n")
        if cut < 0:
            self._partial = data
            return 0
        self._partial = data[cut + 1:]
        n0 = len(self.rows)
        for line in data[:cut].split("\n"):
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < len(COL):
                continue
            # 헤더는 **여기서** 걸러야 한다. 생성 직후의 파일을 열면 헤더가 아직
            # 버퍼에 있어서 안 보이고, 한 줄 건너뛰는 방식은 그 빈 파일에서 아무것도
            # 못 건너뛴 채 나중에 도착한 헤더를 데이터로 먹는다. 2026-08-21 13:45,
            # align 을 다시 딴 그 순간 모니터가 이걸로 죽었다.
            if not _is_number(parts[COL["dev_us"]]):
                continue
            self.rows.append(parts)
        return len(self.rows) - n0

    def close(self):
        try:
            self._fh.close()
        except Exception:                               # noqa: BLE001
            pass


def _col(rows, name, n0=0):
    i = COL[name]
    out = np.empty(len(rows) - n0)
    for k in range(n0, len(rows)):
        v = rows[k][i]
        out[k - n0] = float(v) if _is_number(v) else np.nan
    return out


# ------------------------------------------------------------------ 블록 상태
class BlockState:
    """한 블록의 실시간 집계. 분석기와 같은 문턱, 같은 정지 판정."""

    def __init__(self, block, csv_path):
        self.block = block
        self.csv = CsvTail(csv_path)
        self.csv_path = csv_path
        self.hold_min_s = _min_still_s(block)
        self.t = np.empty(0)
        self.gyr = np.empty((0, 3))
        self.acc = np.empty((0, 3))
        self.chip_q = np.empty((0, 4))
        self.cal = [None] * 4
        self.mask = np.zeros(0, bool)
        self._final = 0                                 # 앞에서부터 확정된 개수
        self._dt = None                                 # 창 폭을 고정하는 샘플 간격
        self.t0 = None
        self.last_grow = time.time()                    # 마지막으로 행이 늘어난 벽시계

    def poll(self):
        n0 = len(self.csv.rows)
        if not self.csv.poll():
            return
        self.last_grow = time.time()
        rows = self.csv.rows
        t = _col(rows, "dev_us", n0) * 1e-6
        g = np.column_stack([_col(rows, k, n0) for k in ("gx", "gy", "gz")])
        a = np.column_stack([_col(rows, k, n0) for k in ("ax", "ay", "az")])
        q = np.column_stack([_col(rows, k, n0)
                             for k in ("chip_qw", "chip_qx", "chip_qy", "chip_qz")])
        self.t = np.concatenate([self.t, t])
        self.gyr = np.vstack([self.gyr, g])
        self.acc = np.vstack([self.acc, a])
        self.chip_q = np.vstack([self.chip_q, q])
        if self.t0 is None and self.t.size:
            self.t0 = self.t[0]
        last = rows[-1]
        self.cal = [(int(float(last[COL[k]])) if last[COL[k]] else None)
                    for k in ("cal_acc", "cal_gyr", "cal_mag", "cal_rv")]
        self._extend_mask()

    def _extend_mask(self):
        """새로 들어온 만큼만 정지 마스크를 잇는다.

        still_mask 는 i 를 중심으로 앞뒤 반창(k/2)을 본다. 그러므로 어떤 값이
        최종이려면 **양쪽 문맥이 다 있어야** 한다. 꼬리에서 k 만큼은 아직 오른쪽
        문맥이 없으니 확정하지 않고, 다음번에 다시 계산한다. 왼쪽 문맥은 확정선보다
        k 앞에서부터 다시 계산해 채운다. (겹침만 주고 끝을 그대로 얼리면 슬라이스
        머리쪽 k 개가 왼쪽 문맥 없이 굳어 20 k 샘플에 26 개가 어긋났다.)

        창 폭은 블록 처음 2000 샘플의 간격으로 **고정한다.** 안 그러면 조각마다
        창이 75~77 로 흔들려 앞뒤가 다른 창으로 계산된다.
        """
        n = self.t.size
        if n < 8:
            return
        if self._dt is None and n >= 2000:
            self._dt = float(np.median(np.diff(self.t[:2000])))
        dt = self._dt if self._dt else float(np.median(np.diff(self.t)))
        k = max(3, int(round(0.30 / max(dt, 1e-6))))
        start = max(0, self._final - k)
        m = qc.still_mask(self.t[start:], self.gyr[start:], self.acc[start:], dt=dt)
        # 슬라이스 **머리** k 개는 왼쪽 문맥이 잘려 있다. 그 자리에는 이미 양쪽
        # 문맥을 다 보고 계산해 둔 값(mask[:_final])이 있으므로 그것을 남긴다.
        guard = self._final - start
        self.mask = np.concatenate([self.mask[:self._final], m[guard:]])
        self._final = max(0, n - k)

    # -- 집계 ---------------------------------------------------------
    def holds(self):
        return qc.segments_from_mask(self.t, self.mask, self.hold_min_s)

    def moves(self):
        return qc.segments_from_mask(self.t, ~self.mask, 0.2)

    def now(self):
        """(정지인가, 지금 상태가 이어진 시간 [s])."""
        if self.mask.size == 0:
            return False, 0.0
        cur = bool(self.mask[-1])
        i = self.mask.size - 1
        while i > 0 and bool(self.mask[i - 1]) == cur:
            i -= 1
        return cur, float(self.t[-1] - self.t[i])

    def elapsed(self):
        return 0.0 if self.t0 is None or not self.t.size else float(self.t[-1] - self.t0)

    def stale_s(self):
        """마지막으로 새 샘플이 들어온 뒤 흐른 시간 [s].

        **끝난 블록의 마지막 화면을 살아 있는 것처럼 보여주면 안 된다.** 이 QC 가
        반복해서 당한 실패가 정확히 그것이다 (2026-08-20 의 GT 가 한 시간 전
        파일이었는데 표에는 짝비율 100 % 로 멀쩡히 찍혔다).
        """
        return time.time() - self.last_grow

    def rate_hz(self):
        if self.t.size < 50:
            return float("nan")
        tail = self.t[-500:]
        return float((tail.size - 1) / max(tail[-1] - tail[0], 1e-6))


def _min_still_s(block):
    if block == "align":
        return protocol.ALIGN_HOLD_S
    if block == "mag_map":
        return protocol.MAG_MAP_HOLD_S
    return protocol.ZUPT_MIN_STILL_S


def _targets(block):
    """(정지 목표, 이동 목표, 짧은 이동 목표). 없으면 None."""
    if block == "align":
        return protocol.ALIGN_TARGET_POSES, None, None
    if block == "mag_map":
        return protocol.MAG_MAP_POSES * protocol.MAG_MAP_POSITIONS, None, None
    if block.startswith("probe"):
        return None, protocol.PROBE_MIN_SEGMENTS, protocol.PROBE_MIN_SHORT_MOVES
    return None, None, None


# ------------------------------------------------------------------ GT (선택)
class GtListener(threading.Thread):
    """로봇 자세를 따로 구독한다. mag_map 의 자세차/위치차에만 쓴다.

    rclpy 가 없거나 토픽이 없으면 조용히 죽는다 — 모니터의 나머지는 GT 없이도
    전부 동작해야 한다. 이 창 때문에 수집이 멈추는 일은 없어야 한다.
    """

    def __init__(self, robot):
        super().__init__(daemon=True)
        self.robot = robot
        self.ok = False
        self.err = None
        self._lock = threading.Lock()
        self._t, self._p, self._q = [], [], []

    def run(self):
        try:
            import rclpy
            from geometry_msgs.msg import Pose
            rclpy.init(args=None)
            node = rclpy.create_node("qc_monitor")
            topic = qc.topics(self.robot)["pose"]

            def cb(msg):
                with self._lock:
                    self._t.append(time.time())
                    self._p.append((msg.position.x, msg.position.y, msg.position.z))
                    self._q.append((msg.orientation.x, msg.orientation.y,
                                    msg.orientation.z, msg.orientation.w))
                    if len(self._t) > 60000:
                        del self._t[:20000], self._p[:20000], self._q[:20000]

            node.create_subscription(Pose, topic, cb, 50)
            self.ok = True
            rclpy.spin(node)
        except Exception as exc:                        # noqa: BLE001
            self.err = str(exc)
            self.ok = False

    def snapshot(self):
        with self._lock:
            return (np.array(self._t), np.array(self._p), np.array(self._q))


def align_state(state, gt):
    """align 의 **관측도**를 지금 이 순간 낸다. 분석기와 같은 해법(solve_AB)이다.

    유지 시간만 보여주는 것은 절반짜리였다. 2026-08-21 재수집은 시간을 1.3 s ->
    2.2 s 로 고쳤는데도 못 썼다 — 자세들이 서로 충분히 다르지 않아 관측도가
    0.12 였기 때문이다. 정지 자세가 한 축으로만 기울면 B 의 그 축 둘레 성분이
    관측되지 않아 **잔차를 봐도 해가 옳은지 알 수 없다.** 그때 잔차는 어느 쪽으로도
    간다 — 실측 08-21 은 자세 3 개에 1.16 deg 로 작았고(미지수가 더 많아 아무 데나
    맞는다), 합성 예에서는 13 deg 로 컸다(영벡터를 엉뚱하게 고른다). 그래서 잔차가
    아니라 관측도를 봐야 한다.

    같이 내는 '가장 가까운 기존 자세와의 각도' 가 실제로 손을 움직이게 하는 값이다.
    그 값이 작으면 방금 만든 자세는 이미 있는 자세를 한 번 더 만든 것이다.
    """
    tg, pg, qg = gt.snapshot()
    if tg.size < 10 or not state.t.size or state.chip_q.size == 0:
        return None
    shift = time.time() - state.t[-1]
    X, Y = [], []
    for (t0, t1, _i0, _i1) in state.holds():
        span = t1 - t0
        lo, hi = t0 + 0.2 * span, t1 - 0.2 * span
        mg = (tg >= lo + shift) & (tg <= hi + shift)
        mi = (state.t >= lo) & (state.t <= hi)
        if mg.sum() < 5 or mi.sum() < 20:
            continue
        q = state.chip_q[mi]
        q = q[np.isfinite(q).all(1)]
        if q.shape[0] < 20:
            continue
        X.append(qc.project_SO3(qc.quat_xyzw_to_R(qg[mg]).mean(0)))
        Y.append(qc.project_SO3(qc.quat_wxyz_to_R(q).mean(0)))
    out = {"n": len(X), "obs": None, "resid_rms": None, "nearest_deg": None}
    if len(X) >= 2:
        # 프로브 축이 서로 얼마나 갈렸나 — 사람이 바로 고칠 수 있는 값
        u = np.array([x[:, 2] for x in X])
        ang = [float(np.degrees(np.arccos(np.clip(u[i] @ u[j], -1, 1))))
               for i in range(len(u)) for j in range(i + 1, len(u))]
        out["min_pair_deg"] = min(ang)
        out["max_pair_deg"] = max(ang)
    if len(X) >= protocol.ALIGN_MIN_POSES:
        try:
            _A, _B, obs, resid = at.solve_AB(X, Y)
            out["obs"] = float(obs)
            out["resid_rms"] = float(np.sqrt((resid ** 2).mean()))
        except SystemExit:
            pass
    # 지금 자세가 새 방향인가 — 정지 중일 때만 뜻이 있다
    still, _held = state.now()
    if still and X and tg.size:
        mg = tg >= time.time() - 0.5
        if mg.sum() >= 3:
            u_now = qc.project_SO3(qc.quat_xyzw_to_R(qg[mg]).mean(0))[:, 2]
            out["nearest_deg"] = min(
                float(np.degrees(np.arccos(np.clip(u_now @ x[:, 2], -1, 1)))) for x in X)
    return out


def mag_map_pairs(state, gt):
    """지금까지 모인 정지들로 **쓸 수 있는 쌍**을 센다. 분석기와 같은 기준.

    GT 의 시간축은 PC 시계(time.time())이고 IMU 쪽은 장치 시계라 서로 다르다.
    여기서는 정합을 풀지 않고 **정지구간의 벽시계 시각**으로 창을 잡는다 — 몇십
    ms 어긋나도 3 초짜리 정지 안에서는 같은 자세다. 정확한 정합은 분석기가 한다.
    """
    tg, pg, qg = gt.snapshot()
    if tg.size < 10 or not state.t.size:
        return None
    # 장치 시계 -> 벽시계: 마지막 샘플이 방금 도착했다고 보고 평행이동한다.
    shift = time.time() - state.t[-1]
    H = []
    for (t0, t1, _i0, _i1) in state.holds():
        span = t1 - t0
        lo, hi = t0 + 0.2 * span + shift, t1 - 0.2 * span + shift
        m = (tg >= lo) & (tg <= hi)
        if m.sum() < 5:
            continue
        R = qc.quat_xyzw_to_R(qg[m])
        H.append((qc.project_SO3(R.mean(0)), pg[m].mean(0) * 1000.0))
    if len(H) < 2:
        return {"holds": len(H), "pairs": 0, "best_dr": 180.0, "best_dp": 0.0}
    n_ok, best_dr, best_dp = 0, 180.0, 0.0
    for i in range(len(H)):
        for j in range(i + 1, len(H)):
            dR = qc.geodesic_deg(H[i][0], H[j][0])
            dp = float(np.linalg.norm(H[i][1] - H[j][1]))
            best_dr, best_dp = min(best_dr, dR), max(best_dp, dp)
            if dR <= protocol.MAG_MAP_MAX_DR_DEG and dp >= protocol.MAG_MAP_MIN_DP_MM:
                n_ok += 1
    return {"holds": len(H), "pairs": n_ok, "best_dr": best_dr, "best_dp": best_dp}


# ------------------------------------------------------------------ 블록 찾기
def newest_csv(block=None):
    """지금 쓰이고 있는 블록의 CSV. (block, path) 또는 None."""
    if not os.path.isdir(qc.DIR_IMU):
        return None
    best = None
    for name in os.listdir(qc.DIR_IMU):
        if not name.endswith(".csv") or "_raw_" not in name:
            continue
        blk = name.split("_raw_")[0]
        if block and blk != block:
            continue
        path = os.path.join(qc.DIR_IMU, name)
        mt = os.path.getmtime(path)
        if best is None or mt > best[0]:
            best = (mt, blk, path)
    return None if best is None else (best[1], best[2])


# ------------------------------------------------------------------ 그리기
def summary_lines(state, mm, al=None):
    """창과 터미널이 같은 문장을 쓴다 — 두 벌로 갈라지면 하나는 반드시 낡는다."""
    still, held = state.now()
    holds, moves = state.holds(), state.moves()
    md = np.array([b - a for (a, b, *_ ) in moves]) if moves else np.array([])
    lo, hi = protocol.PROBE_SHORT_MOVE_S
    short = int(np.sum((md >= lo) & (md <= hi))) if md.size else 0
    t_hold, t_move, t_short = _targets(state.block)

    stale = state.stale_s()
    live = stale < STALE_S
    lines = []
    lines.append(f"[{state.block}]  {state.elapsed():6.1f} s   {state.rate_hz():5.1f} Hz"
                 f"   샘플 {len(state.csv.rows)}")
    if not live:
        lines.append("")
        lines.append(f"  ** 수집 중이 아니다 — {stale:.0f} s 째 새 샘플이 없다 **")
        lines.append("  아래는 끝난 블록의 마지막 값이다. 지금 값이 아니다.")
    lines.append("")
    if not live:
        pass
    elif still:
        need = state.hold_min_s
        bar = "#" * int(min(held / max(need, 1e-6), 1.0) * 30)
        lines.append(f"  정지  {held:5.2f} s / {need:.1f} s  |{bar:<30}|"
                     + ("  OK" if held >= need else "  더 버텨라"))
    else:
        lines.append(f"  이동  {held:5.2f} s")
    lines.append("")
    lines.append(f"  {'모인' if live else '모였던'} 정지 {len(holds):3d} 개"
                 + (f" / 목표 {t_hold}" if t_hold else "")
                 + f"   (>= {state.hold_min_s:.1f} s)")
    lines.append(f"  이동           {len(moves):3d} 개"
                 + (f" / 목표 {t_move}" if t_move else ""))
    if t_short:
        lines.append(f"  짧은 이동      {short:3d} 개 / 목표 {t_short}"
                     f"   ({lo:.1f}~{hi:.1f} s)")
    if al is not None:
        lines.append("")
        lines.append(f"  정렬 관측도  " + ("%.3f" % al["obs"] if al["obs"] is not None
                                          else f"자세 {al['n']}개 — {protocol.ALIGN_MIN_POSES}개는 있어야 낸다"))
        if al["obs"] is not None:
            lines.append(f"     잔차 rms {al['resid_rms']:.2f} deg"
                         "   (관측도가 낮으면 잔차는 믿을 값이 아니다)")
        if al.get("min_pair_deg") is not None:
            lines.append(f"     자세끼리 벌어진 각  최소 {al['min_pair_deg']:5.1f} deg"
                         f" / 최대 {al['max_pair_deg']:5.1f} deg")
        if al.get("nearest_deg") is not None:
            near = al["nearest_deg"]
            lines.append(f"     지금 자세는 기존과 {near:5.1f} deg 떨어져 있다"
                         + ("   <- 이미 있는 자세다. 더 크게 갈라라" if near < 25.0
                            else "   <- 새 방향이다"))
    if mm is not None:
        lines.append("")
        lines.append(f"  쓸 수 있는 쌍  {mm['pairs']:3d} 개 / 목표 {protocol.MAG_MAP_MIN_PAIRS}")
        lines.append(f"     가장 가까운 자세차 {mm['best_dr']:5.1f} deg"
                     f"  (<= {protocol.MAG_MAP_MAX_DR_DEG:.0f} 이라야)")
        lines.append(f"     가장 먼 위치차     {mm['best_dp']:5.0f} mm"
                     f"  (>= {protocol.MAG_MAP_MIN_DP_MM:.0f} 이라야)")
    lines.append("")
    names = ("acc", "gyr", "mag", "rv")
    gated = set(protocol.CAL_MIN)
    lines.append("  칩 보정  " + "  ".join(
        f"{n}={'?' if v is None else v}" + ("" if n in gated else "(참고)")
        for n, v in zip(names, state.cal)))
    return lines, md, (len(holds), len(moves), short), (t_hold, t_move, t_short)


def _korean_font():
    """이 PC 에 실제로 있는 한글 글꼴 이름. 없으면 기본값.

    이름으로 찍으면 못 찾는다 — 배포판의 한글 글꼴이 대개 `.ttc` 묶음이고
    matplotlib 은 그걸 기본 목록에 안 넣는다. 그리고 묶음의 **첫 얼굴 이름**으로
    등록되므로 (여기서는 'Noto Sans CJK JP') 한국어 이름으로 찾으면 계속 빗나간다.
    CJK 글꼴은 통합이라 그 얼굴에도 한글이 다 들어 있다. 그래서 fontconfig 에게
    파일을 물어보고, 파일로 등록한 뒤, 등록된 이름을 그대로 쓴다.
    """
    import subprocess
    import matplotlib.font_manager as fm
    for query in ("Noto Sans CJK KR", "NanumGothic", ":lang=ko"):
        try:
            r = subprocess.run(["fc-match", "-f", "%{file}", query],
                               capture_output=True, text=True, timeout=5)
        except Exception:                               # noqa: BLE001
            continue
        path = r.stdout.strip()
        if not path or not os.path.exists(path):
            continue
        try:
            fm.fontManager.addfont(path)
            return fm.FontProperties(fname=path).get_name()
        except Exception:                               # noqa: BLE001
            continue
    return "DejaVu Sans"


def run_text(args):
    state, gt, mm, al = None, None, None, None
    try:
        while True:
            found = newest_csv(args.block)
            if found is None:
                sys.stdout.write("\r  수집을 기다린다... (run_session.py 를 띄울 것)   ")
                sys.stdout.flush()
                time.sleep(0.5)
                continue
            blk, path = found
            if state is None or state.csv_path != path:
                if state:
                    state.csv.close()
                state = BlockState(blk, path)
                if blk in ("mag_map", "align") and gt is None and not args.no_gt:
                    gt = GtListener(args.robot)
                    gt.start()
            state.poll()
            if gt is not None and gt.ok:
                mm = mag_map_pairs(state, gt) if state.block == "mag_map" else None
                al = align_state(state, gt) if state.block == "align" else None
            lines, *_ = summary_lines(state, mm, al)
            sys.stdout.write("\033[2J\033[H" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(1.0 / REDRAW_HZ)
    except KeyboardInterrupt:
        print()


def run_gui(args):
    import matplotlib
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        # 에이전트/서비스 셸에는 DISPLAY 가 없다. 이 PC 의 X 소켓을 찾아 붙는다.
        socks = sorted(os.listdir("/tmp/.X11-unix")) if os.path.isdir("/tmp/.X11-unix") else []
        for sock in socks:
            os.environ["DISPLAY"] = ":" + sock.lstrip("X")
            break
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    kfont = _korean_font()
    plt.rcParams["font.family"] = [kfont, "DejaVu Sans"]
    # 본문을 monospace 로 찍으므로 **그 목록에도** 넣어야 한다. 안 넣으면 한글이
    # DejaVu Sans Mono 로 떨어져 네모로 나온다 (한글 글리프가 없다). 이 CJK 글꼴은
    # 진짜 고정폭은 아니라 열이 조금 어긋나지만 읽는 데는 지장이 없다.
    plt.rcParams["font.monospace"] = [kfont, "DejaVu Sans Mono"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(13, 7))
    fig.canvas.manager.set_window_title("qc_track 수집 모니터")
    gsp = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0],
                           hspace=0.28, wspace=0.18,
                           left=0.04, right=0.97, top=0.94, bottom=0.08)
    ax_txt = fig.add_subplot(gsp[:, 0]); ax_txt.axis("off")
    ax_now = fig.add_subplot(gsp[0, 1])
    ax_his = fig.add_subplot(gsp[1, 1])

    box = {"state": None, "gt": None, "mm": None, "al": None}

    def draw(_frame):
        found = newest_csv(args.block)
        ax_txt.clear(); ax_txt.axis("off")
        if found is None:
            ax_txt.text(0.02, 0.5, "수집을 기다린다...\nrun_session.py 를 띄울 것",
                        fontsize=16, va="center", family="monospace")
            return
        blk, path = found
        st = box["state"]
        if st is None or st.csv_path != path:
            if st:
                st.csv.close()
            st = BlockState(blk, path)
            box["state"] = st
            if blk in ("mag_map", "align") and box["gt"] is None and not args.no_gt:
                box["gt"] = GtListener(args.robot)
                box["gt"].start()
        st.poll()
        if box["gt"] is not None and box["gt"].ok:
            box["mm"] = mag_map_pairs(st, box["gt"]) if st.block == "mag_map" else None
            box["al"] = align_state(st, box["gt"]) if st.block == "align" else None
        lines, md, counts, targets = summary_lines(st, box["mm"], box.get("al"))

        ax_txt.text(0.01, 0.99, "\n".join(lines), fontsize=13, va="top",
                    family="monospace")

        # 지금 이 순간 — 이 화면에서 사람이 실제로 보고 반응하는 칸이다.
        still, held = st.now()
        live = st.stale_s() < STALE_S
        ax_now.clear()
        need = st.hold_min_s
        ax_now.barh([0], [held if live else 0.0],
                    color=("tab:green" if (still and held >= need)
                           else "tab:orange" if still else "tab:blue"))
        ax_now.axvline(need, color="crimson", lw=2)
        ax_now.set_xlim(0, max(need * 1.6, held * 1.15, 1.0))
        ax_now.set_yticks([])
        ax_now.set_title("수집 중이 아니다" if not live else
                         (("정지 %.2f s  (문턱 %.1f s)" % (held, need)) if still
                          else "이동 %.2f s" % held),
                         fontsize=15, color=("dimgray" if not live else "black"))
        ax_now.set_xlabel("초")

        ax_his.clear()
        edges = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
        if md.size:
            ax_his.hist(np.clip(md, 0, 8.0), bins=edges, color="tab:blue",
                        edgecolor="white")
        lo, hi = protocol.PROBE_SHORT_MOVE_S
        ax_his.axvspan(lo, hi, color="tab:green", alpha=0.18)
        ax_his.set_xticks(edges)
        ax_his.set_title("이동 길이 분포 (초록 = 짧은 이동)", fontsize=12)
        ax_his.set_xlabel("초"); ax_his.set_ylabel("개수")

    def tick():
        """한 프레임이 터져도 창은 살아 있어야 한다.

        타이머 콜백에서 예외가 나가면 tkinter 가 타이머를 떼어 버려 창이 그대로
        굳는다. 굳은 창은 낡은 값을 계속 보여주므로 **없는 것보다 나쁘다.**
        """
        try:
            draw(None)
        except Exception as exc:                        # noqa: BLE001
            ax_txt.clear(); ax_txt.axis("off")
            ax_txt.text(0.02, 0.5, "그리기 오류 — 계속 시도한다\n%s" % exc,
                        fontsize=12, va="center", color="crimson")
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=int(1000 / REDRAW_HZ))
    timer.add_callback(tick)
    timer.start()
    draw(None)
    plt.show()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default=None, help="이 블록만 본다 (기본: 가장 최근)")
    ap.add_argument("--robot", default=qc.ROBOT)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--text", action="store_true", help="창 없이 터미널에")
    ap.add_argument("--no-gt", action="store_true",
                    help="ROS 를 안 쓴다 (mag_map 의 쌍 개수를 못 낸다)")
    args = ap.parse_args()
    if args.raw:
        qc.use_raw_dir(args.raw)
    (run_text if args.text else run_gui)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
