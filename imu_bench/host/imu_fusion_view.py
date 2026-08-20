#!/usr/bin/env python3
"""터미널 뷰어 — GUI 없이 값·퓨전 상태를 확인하고, 자력계를 보정한다.

    python3 imu_fusion_view.py --calibrate-mag 20   # 자력계 보정 (8자 모션)
    python3 imu_fusion_view.py                      # 값 + 퓨전
    python3 imu_fusion_view.py --raw                # 원시 레코드 덤프
    python3 imu_fusion_view.py --log ../logs        # 로깅하며 보기
    python3 imu_fusion_view.py --no-mag             # 6축 IMU-only 로 비교

그래프로 보려면 imu_gui.py 를 쓴다. 수신·퓨전 로직은 둘 다 imu_stream.ImuStream 이다.
"""

import argparse
import os
import sys
import time

import numpy as np
import serial

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mag_calib                                        # noqa: E402
import zero_ref                                         # noqa: E402
from fusion import quat_to_euler_deg                    # noqa: E402
from imu_log import SessionLogger                       # noqa: E402
from imu_stream import ImuStream                        # noqa: E402
from umi_protocol import (                              # noqa: E402
    FrameParser, decode_payload, read_board_name, TYPE_NAMES,
    REC_ACCEL, REC_GYRO, REC_MAG, REC_RV,
)


def run_mag_calibration(ser, seconds, out_path):
    """seconds 초 동안 MAG 레코드를 모아 하드아이언/스케일을 피팅해 저장한다."""
    print("")
    print("=== 자력계 보정 %.0f초 ===" % seconds)
    print("센서를 공중에서 8자를 그리듯 천천히, 모든 축 방향으로 골고루 돌려주세요.")
    print("(한쪽 방향만 돌리면 커버리지가 낮아 보정이 부정확해집니다)")
    print("")

    parser = FrameParser()
    samples = []
    t_end = time.time() + seconds
    last_note = 0.0
    while time.time() < t_end:
        try:
            data = ser.read(ser.in_waiting or 1)
        except serial.SerialException as e:
            print("시리얼 중단: %s" % e)
            break
        if not data:
            continue
        for _, rtype, _, payload in parser.feed(data, time.time()):
            if rtype != REC_MAG:
                continue
            rec = decode_payload(rtype, payload)
            if rec is not None:
                samples.append([rec.x, rec.y, rec.z])
        now = time.time()
        if now - last_note >= 1.0:
            last_note = now
            sys.stdout.write("\r  남은 시간 %4.1fs   샘플 %5d" % (t_end - now, len(samples)))
            sys.stdout.flush()
    print("")

    if len(samples) < 100:
        print("샘플이 부족합니다 (%d개). 보정 파일을 쓰지 않습니다." % len(samples))
        return

    cal = mag_calib.fit(samples)
    mag_calib.save(cal, out_path)
    print("")
    print("하드아이언 오프셋 : %s uT" % np.round(cal["hard_iron"], 2).tolist())
    print("축 스케일         : %s" % np.round(cal["scale"], 3).tolist())
    print("보정 후 반경      : %.1f uT   (지구 자기장은 보통 25~65)" % cal["radius_uT"])
    print("잔차              : %.1f %%   (낮을수록 좋음, 5%% 이하 권장)" % cal["residual_pct"])
    print("방향 커버리지     : %.0f %%   (높을수록 골고루 돌린 것)" % cal["coverage_deg"])
    print("저장              : %s" % out_path)


def dump_raw(ser, seconds):
    parser = FrameParser()
    t_end = time.time() + seconds if seconds else None
    while t_end is None or time.time() < t_end:
        try:
            data = ser.read(ser.in_waiting or 1)
        except serial.SerialException as e:
            print("시리얼 중단: %s" % e)
            return
        if not data:
            continue
        for _, rtype, seq, payload in parser.feed(data, time.time()):
            rec = decode_payload(rtype, payload)
            if rec is not None:
                print("%-8s seq=%3d %s" % (TYPE_NAMES.get(rtype, hex(rtype)), seq, rec))


def run_zero_calibration(stream, seconds, retries=2, quiet=False):
    """로깅을 시작하기 전에 영점을 잡는다. 정지 판정에 실패하면 다시 시도한다.

    반환한 ZeroReference 를 stream.zero 에 걸면 이후 로그에 상대자세가 함께 남고,
    변위 계산에 필요한 지구프레임 가속도 바이어스가 확정된다.
    """
    for attempt in range(retries + 1):
        if not quiet:
            print("")
            print("=== 영점 캘리브레이션 %.1f초 — 센서를 움직이지 마세요 ===" % seconds)
        stream.start_capture()
        t_end = time.time() + seconds
        while time.time() < t_end:
            time.sleep(0.1)
            if not quiet:
                sys.stdout.write("\r  남은 시간 %4.1fs   샘플 %5d"
                                 % (t_end - time.time(), stream.capture_count()))
                sys.stdout.flush()
        samples = stream.stop_capture()
        if not quiet:
            print("")
        try:
            ref = zero_ref.measure(samples)
        except ValueError as e:
            print("  %s" % e)
            return None
        if not quiet:
            print(ref.report())
            sys.stdout.flush()   # 파일로 리다이렉트되면 블록 버퍼링이라 갇힌다
        if ref.quality["still"]:
            stream.zero = ref
            return ref
        if attempt < retries:
            print("  -> 다시 시도합니다 (%d/%d)" % (attempt + 2, retries + 1))
    print("  ! 정지 상태를 확보하지 못했습니다. 영점 없이 진행합니다 —")
    print("    변위 계산은 이 로그로 못 합니다 (지구프레임 바이어스가 없음).")
    return None


def render(s, use_mag, cal, logger):
    r = s["rates"]
    if not use_mag:
        mode = "6축 IMU-only (yaw 드리프트 있음)"
    elif s["saw_mag"]:
        mode = "9축 MARG" + (" [보정됨]" if cal else " [자력계 미보정 — heading 신뢰 불가]")
    else:
        mode = "6축 IMU-only (MAG 레코드 없음)"

    lines = [
        "[%s]  board=%s" % (mode, s["board"] or "?"),
        "  steps=%d  A=%5.1fHz G=%5.1fHz M=%5.1fHz RV=%5.1fHz  crc_err=%d gaps=%d" % (
            s["n_fuse"], r[REC_ACCEL], r[REC_GYRO], r[REC_MAG], r[REC_RV],
            s["crc_errors"], s["gaps"]),
    ]
    if logger is not None:
        lines.append("  ● 기록 중 %d 샘플 -> %s" % (s["log_n"], s["log_path"]))
    if s["acc"] is not None:
        a = s["acc"]
        lines.append("  accel [m/s^2]  x=%+8.3f y=%+8.3f z=%+8.3f  |a|=%6.3f"
                     % (a[0], a[1], a[2], np.linalg.norm(a)))
    if s["gyr"] is not None:
        g = s["gyr"]
        lines.append("  gyro  [rad/s]  x=%+8.3f y=%+8.3f z=%+8.3f" % (g[0], g[1], g[2]))
    if s["mag_raw"] is not None:
        m = s["mag_raw"]
        lines.append("  mag   [uT] raw x=%+8.3f y=%+8.3f z=%+8.3f  |m|=%6.3f"
                     % (m[0], m[1], m[2], np.linalg.norm(m)))
    if s["mag"] is not None and cal is not None:
        m = s["mag"]
        lines.append("  mag   [uT] cal x=%+8.3f y=%+8.3f z=%+8.3f  |m|=%6.3f"
                     % (m[0], m[1], m[2], np.linalg.norm(m)))

    hq = s["host_q"]
    hr, hp, hy = quat_to_euler_deg(hq)
    lines.append("  host  q=[%+.4f %+.4f %+.4f %+.4f]  rpy=(%+7.2f %+7.2f %+7.2f)"
                 % (hq[0], hq[1], hq[2], hq[3], hr, hp, hy))
    if s["chip_q"] is not None:
        cq = s["chip_q"]
        cr, cp, cy = quat_to_euler_deg(cq)
        lines.append("  chip  q=[%+.4f %+.4f %+.4f %+.4f]  rpy=(%+7.2f %+7.2f %+7.2f)"
                     % (cq[0], cq[1], cq[2], cq[3], cr, cp, cy))
        if s.get("rel_euler") is not None:
            rr, rp, ry = s["rel_euler"]
            lines.append("  영점 대비 상대회전  rpy=(%+7.2f %+7.2f %+7.2f)" % (rr, rp, ry))
        if s["diff_deg"] is None:
            lines.append("  프레임 정렬 대기 중... (호스트 필터 수렴 중)")
        else:
            lines.append("  host(정렬 후) vs chip = %6.2f deg" % s["diff_deg"])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--beta", type=float, default=0.05, help="Madgwick gain")
    ap.add_argument("--hz", type=float, default=10.0, help="화면 갱신 주기")
    ap.add_argument("--raw", action="store_true", help="퓨전 없이 원시 레코드를 그대로 덤프")
    ap.add_argument("--no-mag", action="store_true", help="자력계를 무시하고 6축 IMU-only 퓨전")
    ap.add_argument("--mag-cal", default=os.path.join(_HERE, os.pardir, "mag_cal.json"))
    ap.add_argument("--calibrate-mag", type=float, metavar="SEC",
                    help="SEC 초 동안 샘플을 모아 자력계 보정 파일을 만든다")
    ap.add_argument("--align-after", type=float, default=2.0, metavar="SEC")
    ap.add_argument("--seconds", type=float, default=0.0, help="이 시간 뒤 자동 종료 (0=무한)")
    ap.add_argument("--log", dest="log_dir", nargs="?",
                    const=os.path.join(_HERE, os.pardir, "logs"),
                    help="로깅한다 (경로 생략 시 imu_bench/logs)")
    ap.add_argument("--log-format", choices=("csv", "hdf5"), default="csv")
    ap.add_argument("--zero", type=float, default=3.0, metavar="SEC",
                    help="로깅 시작 전 영점 캘리브레이션 시간 (0 = 건너뜀)")
    args = ap.parse_args()

    # 보정/덤프는 스트림 스레드 없이 시리얼을 직접 쓴다.
    if args.calibrate_mag or args.raw:
        ser = serial.Serial(args.port, args.baud, timeout=0.05)
        time.sleep(0.3)
        print("# port=%s board=%s" % (args.port, read_board_name(ser) or "(no INFO reply)"))
        try:
            if args.calibrate_mag:
                run_mag_calibration(ser, args.calibrate_mag, args.mag_cal)
            else:
                dump_raw(ser, args.seconds)
        except KeyboardInterrupt:
            pass
        finally:
            ser.close()
        return

    cal = None if args.no_mag else mag_calib.load(args.mag_cal)
    if cal:
        print("# mag cal: hard_iron=%s scale=%s radius=%.1fuT residual=%.1f%%" % (
            np.round(cal["hard_iron"], 2).tolist(), np.round(cal["scale"], 3).tolist(),
            cal["radius_uT"], cal["residual_pct"]))
    elif not args.no_mag:
        print("# mag cal: 없음 (%s) — --calibrate-mag 20 으로 먼저 보정하세요" % args.mag_cal)

    stream = ImuStream(port=args.port, baud=args.baud, cal=cal, beta=args.beta,
                       use_mag=not args.no_mag, align_after=args.align_after)
    stream.start()

    # 스트림이 살아나기를 기다린다 (영점은 데이터가 있어야 잡힌다)
    t_wait = time.time() + 5.0
    while time.time() < t_wait and stream.n_fuse < 50:
        if stream.error:
            print(stream.error)
            return
        time.sleep(0.1)

    ref = None
    if args.zero > 0:
        ref = run_zero_calibration(stream, args.zero)

    logger = None
    if args.log_dir:
        logger = SessionLogger(args.log_dir, fmt=args.log_format,
                               meta={"port": args.port, "beta": args.beta,
                                     "use_mag": not args.no_mag, "mag_cal": cal,
                                     "board": stream.board_name,
                                     "zero_ref": ref.to_dict() if ref else None})
        stream.logger = logger
        print("# 로깅: %s" % logger.path)

    t_stop = time.time() + args.seconds if args.seconds else None
    try:
        while stream.is_alive():
            if t_stop and time.time() > t_stop:
                break
            time.sleep(1.0 / args.hz)
            s = stream.snapshot()
            if s["error"]:
                print(s["error"])
                break
            sys.stdout.write("\033[2J\033[H" + render(s, not args.no_mag, cal, logger) + "\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        if logger is not None:
            print("\n로깅 저장: %s (%d 샘플)" % (logger.close(), logger.n))


if __name__ == "__main__":
    main()
