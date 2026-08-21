#!/usr/bin/env python3
"""IMU 레코더 — 한 블록 동안 BNO085 를 받아 raw_data/imu/<block>.h5 에 쓴다.

    python3 log_imu.py --block probe --seconds 90 --zero 3

기존 벤치 스택을 그대로 쓴다 (host/imu_stream.py, host/imu_log.py,
host/zero_ref.py). 여기서 새로 하는 것은 셋뿐이다.

  1. 블록 이름으로 파일을 떨어뜨리고 로봇 로거와 **같은 raw_data 아래** 둔다.
  2. CSV 를 그대로 두면서 분석용 표(.h5/.npz)를 같이 만든다 — 분석기는 열 단위로
     읽고, CSV 파싱을 매번 하면 90 s x 250 Hz 를 여러 번 훑을 때 느리다.
  3. 게이팅 정보(CRC 오류, 시퀀스 누락, 실측 레이트, 영점 품질, 자력계 보정 유무)를
     블록 속성으로 박는다. QC_PLAN §8 — 이걸 기록하지 않으면 나중에 어떤 시행을
     무효로 할지 정할 수 없다.

**raw 를 남기는 것이 핵심이다.** acc/gyr/mag 원시값이 있어야 분석기가 오프라인에서
6 축(자력계 없이) 퓨전을 다시 돌려 9 축과 비교할 수 있다. 로봇 옆의 자기환경이
heading 을 얼마나 망가뜨리는지는 이 비교로만 답이 난다 — 칩 RV 만 남기면 그
질문 자체가 사라진다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "host"))

import qc_common as qc                                          # noqa: E402
import mag_calib                                                # noqa: E402
from imu_fusion_view import run_zero_calibration                # noqa: E402
from imu_log import COLUMNS, SessionLogger                      # noqa: E402
from imu_stream import ImuStream                                # noqa: E402
from umi_protocol import REC_ACCEL, REC_GYRO, REC_MAG, REC_RV   # noqa: E402

COL = {name: i for i, name in enumerate(COLUMNS)}


def _cols(a, *names):
    """CSV 배열에서 열 묶음을 뽑는다. 빈 칸은 nan 으로 온다."""
    return a[:, [COL[n] for n in names]]


def csv_to_table(csv_path, out_noext, attrs):
    """SessionLogger 의 CSV -> 분석용 표. 원본 CSV 는 지우지 않는다."""
    a = np.genfromtxt(csv_path, delimiter=",", names=None, skip_header=1,
                      dtype=float, missing_values="", filling_values=np.nan)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.size == 0:
        return None
    cols = {
        "t_pc": a[:, COL["pc_unix"]],
        "dev_us": a[:, COL["dev_us"]],
        "acc": _cols(a, "ax", "ay", "az"),
        "gyr": _cols(a, "gx", "gy", "gz"),
        "mag": _cols(a, "mx", "my", "mz"),
        "mag_raw": _cols(a, "mx_raw", "my_raw", "mz_raw"),
        "chip_q": _cols(a, "chip_qw", "chip_qx", "chip_qy", "chip_qz"),
        "host_q": _cols(a, "host_qw", "host_qx", "host_qy", "host_qz"),
        "diff_deg": a[:, COL["diff_deg"]],
    }
    # 칩 보정 정확도 (0~3). 옛 CSV 에는 열이 없다 — 그때는 싣지 않는다.
    if "cal_mag" in COL and a.shape[1] > COL["cal_rv"]:
        cols["cal_status"] = _cols(a, "cal_acc", "cal_gyr", "cal_mag", "cal_rv")
    # 장치 시계 -> PC 시계. **이쪽을 분석의 시간축으로 쓴다.**
    # pc_unix 는 시리얼 read 한 번이 여러 프레임을 물고 와 계단으로 양자화돼 있다.
    slope, off, info = qc.fit_clock(cols["dev_us"] * 1e-6, cols["t_pc"])
    cols["t"] = qc.apply_clock(cols["dev_us"] * 1e-6, slope, off)
    attrs = dict(attrs)
    attrs.update({f"clock_{k}": v for k, v in info.items()})
    return qc.save_table(out_noext, cols, attrs), attrs


def check_cal(port, baud, seconds):
    """블록을 시작하기 전에 칩이 스스로 매긴 보정 정확도를 읽어 온다.

    수집은 하지 않는다 — 스트림을 잠깐 열어 acc/gyr/mag/rv 네 레코드가 한 번씩
    올 때까지 기다렸다가 status 를 JSON 으로 찍고 포트를 놓는다. 러너가 이걸
    자식 프로세스로 띄워 읽는다 (러너는 pyserial 을 import 하지 않는다).

    왜 별도 모드인가: 보정 상태는 **수집이 끝난 뒤에 보면 늦다.** 90 초를 다
    쓰고 나서 rv=0 이었음을 알면 그 블록은 회전에 대해 아무 말도 못 한다.
    """
    stream = ImuStream(port=port, baud=baud, cal=None)
    stream.start()
    deadline = time.time() + max(seconds, 1.0) + 2.0
    try:
        while time.time() < deadline and stream.error is None:
            cs = stream.snapshot().get("cal_status") or {}
            if all(cs.get(k) is not None for k in ("acc", "gyr", "mag", "rv")):
                break
            time.sleep(0.1)
        snap = stream.snapshot()
    finally:
        stream.stop()
    if stream.error:
        print(json.dumps({"error": str(stream.error)}))
        return 2
    cs = snap.get("cal_status") or {}
    print(json.dumps({"cal_status": cs,
                      "board": snap.get("board"),
                      "crc_errors": int(snap.get("crc_errors", 0))}))
    return 0 if any(v is not None for v in cs.values()) else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", help="--check-cal 일 때는 필요 없다")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 이면 Ctrl-C 까지")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--zero", type=float, default=3.0,
                    help="블록 시작 전 영점 캘리브레이션 [s]. 0 이면 생략")
    ap.add_argument("--mag-cal", default=os.path.join(os.path.dirname(_HERE), "mag_cal.json"))
    ap.add_argument("--no-mag", action="store_true",
                    help="호스트 퓨전을 6 축으로. 칩 RV 는 그대로 9 축이다")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--check-cal", action="store_true",
                    help="기록하지 않고 칩 보정 상태만 JSON 으로 찍는다")
    args = ap.parse_args()

    if args.check_cal:
        import protocol
        return check_cal(args.port, args.baud, protocol.CAL_CHECK_S)

    if not args.block:
        ap.error("--block 이 필요하다")

    if args.raw:
        qc.use_raw_dir(args.raw)
    qc.ensure_dirs()
    qc.install_shutdown_handlers()

    cal = mag_calib.load(args.mag_cal)
    if cal is None:
        print("  [주의] 자력계 보정이 없다. heading 이 의미를 갖지 못한다 —"
              " ./scripts/run.sh --calibrate-mag 20 을 먼저 돌릴 것")

    stream = ImuStream(port=args.port, baud=args.baud, cal=cal, beta=args.beta,
                       use_mag=not args.no_mag)
    stream.start()

    t_wait = time.time() + 5.0
    while stream.n_fuse < 10 and time.time() < t_wait and stream.error is None:
        time.sleep(0.1)
    if stream.error:
        print(f"  ! 스트림 오류: {stream.error}")
        return 2
    if stream.n_fuse < 10:
        print("  ! 5 초 안에 퓨전 샘플이 안 왔다. 포트/펌웨어 확인")
        return 2
    print(f"  보드 {stream.board_name}   자력계 보정 {'있음' if cal else '없음'}")

    zero = run_zero_calibration(stream, args.zero) if args.zero > 0 else None
    if args.zero > 0 and zero is None:
        # 영점이 없으면 변위 계산의 B/C 단계를 못 한다. 그래도 회전은 살아 있으므로
        # 기록은 계속한다 — 무엇이 없는지를 속성에 남겨 분석기가 알게 한다.
        print("  [주의] 영점 없이 진행한다. 이 블록의 변위 지표는 A 단계만 나온다.")

    logger = SessionLogger(qc.DIR_IMU, fmt="csv",
                           prefix=f"{args.block}_raw",
                           meta={"block": args.block, "port": args.port,
                                 "beta": args.beta, "use_mag": not args.no_mag,
                                 "mag_cal": args.mag_cal if cal else None,
                                 "zero_ref": zero.to_dict() if zero else None})
    t0 = time.time()
    with logger:
        stream.logger = logger
        print(f"[{args.block}] 기록 시작"
              + (f" — {args.seconds:.0f} s" if args.seconds > 0 else " — Ctrl-C 까지"))
        try:
            deadline = t0 + args.seconds if args.seconds > 0 else None
            while stream.is_alive():
                time.sleep(0.25)
                if deadline and time.time() >= deadline:
                    break
                if stream.error:
                    print(f"  ! 스트림 오류: {stream.error}")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            stream.logger = None
    stream.stop()

    snap = stream.snapshot()
    attrs = {
        "block": args.block, "started_unix": t0, "stopped_unix": time.time(),
        "board": snap.get("board") or "?", "n": logger.n,
        "crc_errors": int(snap.get("crc_errors", 0)),
        "seq_gaps": int(snap.get("gaps", 0)),
        "use_mag": bool(not args.no_mag),
        "mag_cal": bool(cal is not None),
        "zero_still": bool(zero.quality["still"]) if zero else False,
        "csv": os.path.basename(logger.path),
    }
    for k, label in ((REC_ACCEL, "accel"), (REC_GYRO, "gyro"),
                     (REC_MAG, "mag"), (REC_RV, "rv")):
        attrs[f"rate_{label}_hz"] = float(snap["rates"].get(k, float("nan")))
    if zero:
        attrs["zero_gravity_ms2"] = float(np.linalg.norm(zero.b_E))
        attrs["zero_scale_err_pct"] = float(
            100.0 * (np.linalg.norm(zero.b_E) / qc.G0 - 1.0))

    out = csv_to_table(logger.path, os.path.join(qc.DIR_IMU, args.block), attrs)
    if out is None:
        print("  ! 표를 만들 수 없다 (샘플 없음)")
        return 2
    path, attrs = out
    if zero:
        qc.dump_json(os.path.join(qc.DIR_META, f"zero_{args.block}.json"), zero.to_dict())
    print(f"[{args.block}] {logger.n} 샘플, CRC {attrs['crc_errors']} / 누락 "
          f"{attrs['seq_gaps']}, skew {attrs.get('clock_skew_ppm', float('nan')):.0f} ppm"
          f" -> {os.path.basename(path)}")
    qc.append_trial({"block": args.block, "source": "imu", **attrs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
