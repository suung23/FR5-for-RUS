#!/usr/bin/env python3
"""세션 러너 — 블록마다 두 레코더를 띄우고, 조작 지시문을 찍고, 시각을 남긴다.

    python3 run_session.py --dry-run                # 계획만
    python3 run_session.py                          # 기본 블록 전부
    python3 run_session.py --blocks align,probe     # 일부만
    python3 run_session.py --blocks sweep_tilt --excite   # 스크립트 자극

**rclpy 를 import 하지 않는다.** 로봇 로거를 자식 프로세스로 띄우므로, 이 파일은
어느 python 으로 돌려도 된다. 자식은 각자 필요한 인터프리터로 띄운다 —
  · log_robot.py  시스템 python3 (rclpy)
  · log_imu.py    이 벤치가 쓰는 python (pyserial, numpy)

왜 블록마다 프로세스를 새로 띄우나
---------------------------------
한 프로세스가 여러 블록을 도는 구조를 쓰면, 중간 블록에서 죽었을 때 앞의 것까지
같이 잃는다. 블록 하나가 곧 trial 이고 파일 하나이므로, 프로세스도 그 단위로
끝내는 편이 잃는 것이 적다. (Surgilogger QC 의 run_all.sh 와 같은 판단이다.)

시각은 자식이 아니라 **여기서** 찍어 program.json 에 남긴다. 자식의 시작·종료
로그를 나중에 긁어 맞추는 것보다, 시킨 쪽이 적어 두는 편이 틀릴 자리가 적다.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import protocol                                                 # noqa: E402
import qc_common as qc                                          # noqa: E402


def ros_python():
    """rclpy 가 있는 인터프리터. 없으면 None 을 돌려주고 그 사실을 알린다."""
    for cand in (os.environ.get("QC_ROS_PYTHON"), "/usr/bin/python3", sys.executable):
        if not cand or not shutil.which(cand) and not os.path.exists(cand):
            continue
        r = subprocess.run([cand, "-c", "import rclpy"], capture_output=True)
        if r.returncode == 0:
            return cand
    return None


def block_outputs(block):
    """이 블록이 남겨야 하는 파일들. save_table 이 h5py 유무로 확장자를 고른다."""
    return [(name, os.path.join(d, block))
            for name, d in (("robot", qc.DIR_ROBOT), ("imu", qc.DIR_IMU))]


def existing(path_noext):
    for ext in (".h5", ".npz"):
        if os.path.exists(path_noext + ext):
            return path_noext + ext
    return None


def quarantine(path, why):
    """낡은 파일을 분석기 손이 닿지 않는 이름으로 치운다.

    지우지 않는 이유는, 그 파일이 유일한 사본일 수 있어서다. 이름만 바꾸면
    analyze_track 이 못 집고 사람은 여전히 볼 수 있다. 2026-08-20 세션에서
    still 의 GT 가 한 시간 전 파일이었는데 아무도 못 알아챈 것이 이 장치가
    없어서였다.
    """
    stem, ext = os.path.splitext(path)
    dest = f"{stem}.stale{ext}"
    n = 1
    while os.path.exists(dest):
        n += 1
        dest = f"{stem}.stale{n}{ext}"
    os.rename(path, dest)
    print(f"   [격리] {os.path.basename(path)} -> {os.path.basename(dest)}  ({why})")
    return dest


def snapshot(block):
    """블록 시작 전 산출물의 (경로, mtime). 벽시계와 비교하지 않는 이유는,
    파일시스템 시각이 time.time() 보다 ms 단위로 앞설 수 있어서다 — 그러면
    방금 쓴 파일이 '낡았다' 로 몰린다. 전후 비교는 그 문제가 없다."""
    out = {}
    for name, noext in block_outputs(block):
        path = existing(noext)
        out[name] = (path, os.path.getmtime(path) if path else None)
    return out


def check_block(block, entry, before):
    """블록이 실제로 두 파일을 **이번에** 남겼는지 본다. 문제 목록을 돌려준다."""
    problems = []
    for name, noext in block_outputs(block):
        rc = entry["returncode"].get(name)
        if rc is None:
            continue                                    # 애초에 안 띄운 로거
        path = existing(noext)
        old_path, old_mtime = before.get(name, (None, None))
        fresh = path is not None and (old_path != path
                                      or os.path.getmtime(path) != old_mtime)
        if rc != 0:
            problems.append(f"{name} 로거가 {rc} 로 끝났다")
        elif path is None:
            problems.append(f"{name} 파일이 없다 ({os.path.basename(noext)}.h5)")
        elif not fresh:
            problems.append(f"{name} 파일이 이 블록 것이 아니다 (안 바뀌었다)")
        if problems and path is not None and not fresh:
            quarantine(path, "이전 세션 파일")
    return problems


def min_still_s(block):
    """이 블록의 정지가 '쓸 수 있는 정지' 로 인정받는 최소 길이.

    **분석기가 실제로 쓰는 문턱과 같아야 한다.** 전부 ZUPT 문턱(0.8 s)으로 세던
    때 2026-08-21 세션이 무너졌다 — align 에서 러너가 "정지 9 개" 라고 찍었는데
    분석기는 2.5 s 로 세어 0 개였고, 정렬이 안 풀려 회전 지표가 통째로 날아갔다.
    현장 집계가 분석기보다 느슨하면 그 집계는 사람을 안심시키는 일만 한다.
    """
    if block == "align":
        return protocol.ALIGN_HOLD_S
    if block == "mag_map":
        return protocol.MAG_MAP_HOLD_S
    return protocol.ZUPT_MIN_STILL_S


def mag_map_pairs(block):
    """mag_map 의 **쓸 수 있는 쌍** 개수. 없으면 None.

    이 블록의 존재 이유가 '자세는 같고 위치만 다른 쌍' 하나뿐인데, 정지·이동
    개수만 세면 그 쌍이 하나도 없어도 표가 멀쩡해 보인다. 실제로 2026-08-21 에
    ΔR 최소 7.2° / Δp 최대 181 mm 로 쌍이 0 개였고, 90 초를 다 쓰고도 이 블록이
    답할 수 있는 질문이 없었다. 여기서만 로봇 표를 같이 읽는다.
    """
    try:
        import numpy as np
        rb, _ = qc.load_table(os.path.join(qc.DIR_ROBOT, block))
        ib, _ = qc.load_table(os.path.join(qc.DIR_IMU, block))
        if rb is None or ib is None:
            return None
        t = np.asarray(rb["pose/t"], float)
        p = np.asarray(rb["pose/position"], float) * 1000.0
        R = qc.quat_xyzw_to_R(np.asarray(rb["pose/quat_xyzw"], float))
        ti = np.asarray(ib["t"], float)
        still = qc.still_mask(ti, np.asarray(ib["gyr"], float), np.asarray(ib["acc"], float))
        segs = qc.segments_from_mask(ti, still, protocol.MAG_MAP_HOLD_S)
        H = []
        for (t0, t1, _i0, _i1) in segs:
            span = t1 - t0
            m = (t >= t0 + 0.2 * span) & (t <= t1 - 0.2 * span)
            if m.sum() < 5:
                continue
            H.append((qc.project_SO3(R[m].mean(0)), p[m].mean(0)))
        n_ok = best_dr = 0
        best_dr, best_dp = 180.0, 0.0
        for i in range(len(H)):
            for j in range(i + 1, len(H)):
                dR = qc.geodesic_deg(H[i][0], H[j][0])
                dp = float(np.linalg.norm(H[i][1] - H[j][1]))
                best_dr, best_dp = min(best_dr, dR), max(best_dp, dp)
                if dR <= protocol.MAG_MAP_MAX_DR_DEG and dp >= protocol.MAG_MAP_MIN_DP_MM:
                    n_ok += 1
        return {"holds": len(H), "pairs": n_ok,
                "best_dr_deg": best_dr, "best_dp_mm": best_dp}
    except Exception:                                   # noqa: BLE001
        return None


def census(block):
    """블록이 끝나자마자 '무엇이 나왔나' 를 센다. IMU 자이로만으로 세므로 GT 도
    분석 스택도 필요 없다 (mag_map 만 예외 — 아래 mag_map_pairs).

    분석은 나중에 한 번에 돌리는 것이 자연스럽지만, 이 QC 에서 실패하는 방식은
    '데이터가 틀린 것' 이 아니라 '필요한 동작이 안 들어간 것' 이다. 그것을 한
    시간 뒤 분석에서 알면 세션을 다시 잡아야 한다. 90 초짜리 블록은 바로 다시
    딸 수 있으므로, 여기서 세는 것이 값이 싸다.
    """
    try:
        import numpy as np
        cols, _ = qc.load_table(os.path.join(qc.DIR_IMU, block))
    except Exception:                                   # noqa: BLE001
        return None
    if not cols or "gyr" not in cols or "t" not in cols:
        return None
    t, gyr, acc = cols["t"], cols["gyr"], cols.get("acc")
    still = qc.still_mask(t, gyr, acc)
    hold_s = min_still_s(block)
    holds = qc.segments_from_mask(t, still, hold_s)
    moves = qc.segments_from_mask(t, ~still, 0.2)
    md = np.array([t1 - t0 for (t0, t1, *_ ) in moves]) if moves else np.array([])
    hd = np.array([t1 - t0 for (t0, t1, *_ ) in holds]) if holds else np.array([])
    lo, hi = protocol.PROBE_SHORT_MOVE_S
    out = {"holds": len(holds), "hold_min_s": hold_s,
           "hold_longest_s": float(hd.max()) if hd.size else 0.0,
           "moves": len(md),
           "short": int(np.sum((md >= lo) & (md <= hi))) if md.size else 0,
           "bins": [(a, b, int(np.sum((md >= a) & (md < b))))
                    for a, b in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0),
                                 (2.0, 4.0), (4.0, 8.0), (8.0, 1e9))] if md.size else []}
    # 문턱을 못 넘은 짧은 정지도 같이 낸다 — "자세는 만들었는데 짧았다" 와
    # "자세를 안 만들었다" 는 다음에 할 일이 다르다.
    out["holds_short"] = len(qc.segments_from_mask(t, still, 0.5)) - len(holds)
    if block == "mag_map":
        out["mag_map"] = mag_map_pairs(block)
    return out


def print_census(block, c):
    if not c:
        return []
    print(f"   [{block}] 정지 {c['holds']} 개 (>= {c['hold_min_s']:.1f} s, 최장"
          f" {c['hold_longest_s']:.2f} s) / 이동 {c['moves']} 개"
          f"  (짧은 이동 {c['short']} 개)")
    if c.get("holds_short"):
        print(f"      + 문턱을 못 넘은 짧은 정지 {c['holds_short']} 개")
    if c["bins"]:
        print("      이동 길이 분포  " + "  ".join(
            f"{a:g}~{'inf' if b > 1e8 else f'{b:g}'}s:{n}" for a, b, n in c["bins"]))
    mm = c.get("mag_map")
    if mm is not None:
        print(f"      쓸 수 있는 쌍 {mm['pairs']} 개"
              f"  (가장 가까운 자세차 {mm['best_dr_deg']:.1f}°,"
              f" 가장 먼 위치차 {mm['best_dp_mm']:.0f} mm)")
    warn = []
    if block == "align" and c["holds"] < protocol.ALIGN_TARGET_POSES:
        warn.append(f"{protocol.ALIGN_HOLD_S:.1f} s 이상 버틴 자세가 {c['holds']} 개다 —"
                    f" {protocol.ALIGN_TARGET_POSES} 개가 목표이고"
                    f" {protocol.ALIGN_MIN_POSES} 개 미만이면 정렬 자체가 안 풀린다."
                    " 자세를 더 만드는 것보다 **자세마다 더 오래 버티는 것**이"
                    " 먼저다 — 2026-08-21 은 자세 12 개를 만들고도 평균 1.3 s 라"
                    " 쓸 수 있는 것이 0 개였다. 자리마다 셋을 세고 옮길 것")
    if block == "mag_map":
        if mm is None:
            warn.append("로봇 표를 못 읽어 쌍을 못 셌다 — 분석 때까지 이 블록이"
                        " 쓸모 있는지 알 수 없다")
        elif mm["pairs"] < protocol.MAG_MAP_MIN_PAIRS:
            warn.append(f"쓸 수 있는 쌍이 {mm['pairs']} 개다 —"
                        f" {protocol.MAG_MAP_MIN_PAIRS} 개 이상 필요하다."
                        f" 쌍이 되려면 자세차 {protocol.MAG_MAP_MAX_DR_DEG:.0f}° 이하이면서"
                        f" 위치차 {protocol.MAG_MAP_MIN_DP_MM:.0f} mm 이상이라야 한다"
                        f" (지금 최선 {mm['best_dr_deg']:.1f}° / {mm['best_dp_mm']:.0f} mm)."
                        " **손목을 완전히 고정한 채로** 팔만 멀리 옮길 것 —"
                        " 이 쌍이 없으면 이 블록은 아무것도 답하지 못한다")
    if block.startswith("probe"):
        if c["short"] < protocol.PROBE_MIN_SHORT_MOVES:
            warn.append(f"짧은 이동({protocol.PROBE_SHORT_MOVE_S[0]:.1f}~"
                        f"{protocol.PROBE_SHORT_MOVE_S[1]:.1f} s)이 {c['short']} 개다"
                        f" — {protocol.PROBE_MIN_SHORT_MOVES} 개 이상 필요하다."
                        " 이 센서가 실제로 쓸 만한 영역이 그 칸이다")
        if c["moves"] < protocol.PROBE_MIN_SEGMENTS:
            warn.append(f"이동 구간이 {c['moves']} 개다 —"
                        f" {protocol.PROBE_MIN_SEGMENTS} 개 이상이라야 중앙값이"
                        " 통계가 된다")
    for w in warn:
        print(f"      [모자람] {w}")
    return warn


def read_cal(args):
    """자식으로 `log_imu.py --check-cal` 을 띄워 칩 보정 상태를 읽어 온다.

    여기서 직접 포트를 열지 않는 이유는 이 파일의 규약 때문이다 — 러너는
    rclpy 도 pyserial 도 import 하지 않는다 (모듈 docstring). 로거를 띄우는
    것과 같은 방식으로, 필요한 스택을 가진 인터프리터에게 시킨다.
    """
    cmd = [sys.executable, os.path.join(_HERE, "log_imu.py"),
           "--check-cal", "--port", args.port]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as e:               # noqa: BLE001
        return {"error": str(e)}
    for line in reversed(r.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return {"error": (r.stderr.strip().splitlines() or ["출력 없음"])[-1]}


def cal_shortfall(cal, use_mag):
    """문턱에 못 미치는 채널 목록. 못 읽은 채널도 모자란 것으로 친다."""
    need = protocol.CAL_MIN if use_mag else protocol.CAL_MIN_NO_MAG
    cs = (cal or {}).get("cal_status") or {}
    return [(k, cs.get(k), v) for k, v in sorted(need.items())
            if cs.get(k) is None or cs[k] < v]


def cal_gate(block, args):
    """블록을 시작하기 전에 칩 보정 상태를 확인한다. 읽은 값을 돌려준다.

    2026-08-20 세션은 이 확인이 없어서 못 쓰게 됐다. 자력계가 자세를 악화시키는
    것까지는 봤는데, 원인이 '로봇이 만드는 장' 인지 '칩 보정이 애초에 수렴한 적이
    없는 것' 인지 가를 수 없었다. 지금은 값이 기록에 남지만, 기록만으로는 세션이
    끝난 뒤에야 안다 — 90 초를 다 쓰고 rv=0 이었음을 아는 것은 늦다.

    막지 않고 알리기만 하는 선택지를 남겨 둔 이유는, 이 리그(로봇 옆)에서
    자력계 보정이 영영 안 오를 수도 있어서다. 그때는 6 축으로 가는 것이 답이지
    (README §2) 세션을 못 하는 것이 답은 아니다. 다만 그 판단은 사람이 한다.
    """
    if args.no_cal_check:
        return None
    print(f"   [{block}] 칩 보정 상태를 읽는다 ({protocol.CAL_CHECK_S:.0f} s)...")
    cal = read_cal(args)
    if cal.get("error"):
        print(f"   [주의] 칩 보정 상태를 못 읽었다 ({cal['error']}).")
        return cal
    prev = None
    while True:
        short = cal_shortfall(cal, use_mag=not args.no_mag)
        cs = cal.get("cal_status") or {}
        gated = set(protocol.CAL_MIN if not args.no_mag else protocol.CAL_MIN_NO_MAG)
        # gyr 은 판정에서 빠졌지만 계속 보여 준다 — 안 보이면 나중에 '그때 몇이었나'
        # 를 기록으로 답할 수 없다 (protocol.CAL_MIN 주석).
        line = "  ".join(
            f"{k}={'?' if cs.get(k) is None else cs[k]}" + ("" if k in gated else "(참고)")
            for k in ("acc", "gyr", "mag", "rv"))
        changed = "" if prev is None else ("  <- 바뀌었다" if cs != prev else "  <- 그대로다")
        prev = dict(cs)
        print(f"   [{block}] 칩 보정 (0~3)  {line}{changed}")
        if not short:
            return cal
        print("      [모자람] " + ", ".join(
            f"{k} 가 {'없다' if got is None else got} — {need} 이상 필요"
            for k, got, need in short))
        print("      칩 보정은 **움직여야** 오른다. 올리고 싶으면 teleop 으로 팔을"
              " 세 축 둘레로 크게 돌린 뒤 [r] 로 다시 읽는다.")
        print("      rv 가 낮은 채로 딴 블록은 회전 지표가 통째로 의미를 잃고,"
              " mag 이 낮으면 2026-08-20 과 같은 자리에 다시 선다.")
        if args.yes:
            print("      (--yes 라 그대로 진행한다)")
            cal["overridden"] = True
            return cal
        # **Enter 는 진행이다.** 처음에는 Enter 를 '다시 읽기' 로 뒀는데, 막힌
        # 사람이 누르는 키는 언제나 Enter 다 — 두 번의 세션이 여기서 멈췄다.
        # 이 게이트의 목적은 막는 것이 아니라 **알리고 기록에 남기는 것**이다
        # (program.json 의 overridden). 그러니 자연스러운 키가 진행이어야 한다.
        print("      [Enter] 이대로 진행 / [r] 다시 읽기 / [q] 세션 중단")
        try:
            ans = input("      > ").strip().lower()
        except EOFError:
            print("      (stdin 이 없다 — 이대로 진행한다)")
            cal["overridden"] = True
            return cal
        if ans == "q":
            print("      중단한다. 보정을 올린 뒤 이 블록부터 다시 딸 것:")
            print(f"      python3 run_session.py --blocks {block}")
            raise SystemExit(1)
        if ans != "r":                      # Enter 를 포함해 'r' 이 아닌 것은 전부 진행
            cal["overridden"] = True
            return cal
        print(f"      다시 읽는 중 ({protocol.CAL_CHECK_S:.0f} s)...")
        cal = read_cal(args)
        if cal.get("error"):
            print(f"   [주의] 칩 보정 상태를 못 읽었다 ({cal['error']}).")
            return cal


def wait_for_enter(lines, seconds):
    print()
    for ln in lines:
        print("   " + ln)
    print(f"\n   블록 길이 {seconds:.0f} s.  준비되면 Enter (Ctrl-C 로 중단)")
    input()


def run_block(block, seconds, args, py_ros):
    qc.ensure_dirs()
    procs = []
    if py_ros:
        procs.append(("robot", subprocess.Popen(
            [py_ros, os.path.join(_HERE, "log_robot.py"), "--block", block,
             "--seconds", str(seconds), "--robot", args.robot, "--raw", qc.RAW])))
    else:
        print("   [주의] rclpy 를 못 찾았다 — GT 없이 IMU 만 기록한다."
              "  source /opt/ros/jazzy/setup.bash 를 먼저 할 것")
    imu_cmd = [sys.executable, os.path.join(_HERE, "log_imu.py"), "--block", block,
               "--seconds", str(seconds), "--port", args.port, "--raw", qc.RAW,
               "--zero", str(args.zero if block in ("still", "align") else args.zero)]
    if args.no_mag:
        imu_cmd.append("--no-mag")
    procs.append(("imu", subprocess.Popen(imu_cmd)))

    exciter = None
    if args.excite and block == "mag_map" and py_ros:
        # 자세를 붙든 채 옮기는 것은 손으로 두 번 실패했다 (excite_magmap docstring).
        # **여기서 시간으로 기다리지 않는다.** log_imu 가 3 s 영점을 도는 동안
        # 팔이 움직이면 영점이 깨지는데, 스트림 기동이 최대 5 s 라 시간으로는 못
        # 맞춘다. 자극 쪽이 IMU 의 CSV 가 생기는 것을 보고 스스로 시작한다.
        exciter = subprocess.Popen(
            [py_ros, os.path.join(_HERE, "excite_magmap.py"), "--block", block,
             "--robot", args.robot, "--raw", qc.RAW])
    elif args.excite and block.startswith("sweep_") and py_ros:
        # 자극은 로거가 자리를 잡은 뒤에 시작한다 — 앞부분이 잘리면 그 구간의
        # 위상 기준(LEAD_S 정지구간)이 사라진다.
        time.sleep(2.0)
        exciter = subprocess.Popen(
            [py_ros, os.path.join(_HERE, "excite_twist.py"),
             "--axis", block.split("_", 1)[1], "--robot", args.robot, "--raw", qc.RAW])

    t0 = time.time()
    try:
        for _, p in procs:
            p.wait()
    except KeyboardInterrupt:
        for _, p in procs:
            p.send_signal(signal.SIGINT)
        for _, p in procs:
            p.wait()
    finally:
        if exciter and exciter.poll() is None:
            exciter.send_signal(signal.SIGINT)
            exciter.wait()
    return {"block": block, "t_start": t0, "t_stop": time.time(),
            "returncode": {n: p.returncode for n, p in procs}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=",".join(protocol.DEFAULT_BLOCKS))
    ap.add_argument("--robot", default=qc.ROBOT)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--zero", type=float, default=3.0)
    ap.add_argument("--no-mag", action="store_true")
    ap.add_argument("--excite", action="store_true",
                    help="mag_map / sweep_* 에서 desired_twist 로 로봇을 직접 움직인다."
                         " teleop 이 같이 돌면 안 된다 — 발행자가 둘이 된다")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="지시문 확인 없이 바로 시작")
    ap.add_argument("--no-cal-check", action="store_true",
                    help="블록 시작 전 칩 보정 상태 확인을 건너뛴다")
    args = ap.parse_args()

    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    if args.dry_run:
        print(protocol.describe())
        print("\n블록별 지시문")
        for b in blocks:
            print(f"\n[{b}]")
            for ln in protocol.PROMPTS.get(b, ["(지시문 없음)"]):
                print("   " + ln)
        return 0

    if args.raw:
        qc.use_raw_dir(args.raw)
    qc.ensure_dirs()
    py_ros = ros_python()
    print(f"raw_data : {qc.RAW}")
    print(f"rclpy    : {py_ros or '없음'}")
    print(f"블록     : {', '.join(blocks)}")

    program = qc.load_json(qc.PROGRAM_JSON, {}) or {}
    program.setdefault("blocks", [])
    program["robot"] = args.robot
    program["protocol"] = {b: protocol.PROMPTS.get(b, []) for b in blocks}
    for b in blocks:
        seconds = protocol.DURATION.get(b) or protocol.sweep_seconds(b.split("_", 1)[1])
        if b == "mag_map" and args.excite:
            # 스크립트가 도는 시간이 손으로 하는 것보다 길다. 로거가 먼저 끝나면
            # 마지막 자세 묶음이 통째로 안 실린다.
            # 자극이 IMU 영점(최대 8 s)과 상태 수신을 기다린 뒤에 시작하므로,
            # 그 몫까지 얹어 준다. 로거가 먼저 끝나면 마지막 자세 묶음이 안 실린다.
            seconds = protocol.mag_map_seconds() + 40.0
        cal_before = cal_gate(b, args)
        if not args.yes:
            wait_for_enter(protocol.PROMPTS.get(b, [f"블록 {b}"]), seconds)
        if b.startswith("sweep_") and args.excite:
            program.setdefault("sweeps", {})[b] = protocol.plan_sweep(b.split("_", 1)[1])
        before = snapshot(b)
        entry = run_block(b, seconds, args, py_ros)
        entry["cal_before"] = cal_before
        problems = check_block(b, entry, before)
        entry["problems"] = problems
        program["blocks"] = [x for x in program["blocks"] if x.get("block") != b]
        program["blocks"].append(entry)
        qc.dump_json(qc.PROGRAM_JSON, program)
        print(f"   [{b}] 끝 — {entry['returncode']}")
        if problems:
            # 여기서 멈춘다. 남은 블록을 마저 따 봐야, 없는 GT 위에서 정렬을 풀고
            # 그 정렬로 나머지를 전부 평가하게 된다 — 한 블록이 아니라 세션이
            # 통째로 못 쓰게 된다.
            print(f"\n   [중단] {b} 블록이 성하지 않다:")
            for pr in problems:
                print(f"      · {pr}")
            print("      로봇 런치가 떠 있는지 (ros2 topic hz 로 확인),"
                  " 포트가 맞는지 보고 이 블록부터 다시 딸 것:")
            print(f"      python3 run_session.py --blocks {b}")
            return 1

        entry["census"] = census(b)
        short = print_census(b, entry["census"])
        qc.dump_json(qc.PROGRAM_JSON, program)
        if short and not args.yes:
            print("\n   이 블록을 지금 다시 딸 수 있다."
                  "  [Enter] 다음 블록으로 / [r] 이 블록 다시")
            if input("   > ").strip().lower() == "r":
                blocks.append(b)          # 뒤에 다시 붙인다

    tail = f" --run {qc.RAW}" if args.raw else ""
    print(f"\n분석:  python3 analyze_track.py{tail}"
          f"\n그림:  python3 plot_track.py{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
