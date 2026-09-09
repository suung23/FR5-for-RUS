#!/usr/bin/env python3
"""초음파 candidate 프레임 + BNO085 IMU 를 하나의 시간축에서 동시 수집한다.

두 소스는 물리적으로 다른 경로로 온다.

    IMU : USB 시리얼 /dev/ttyACM0, BNO085 9축, 가속·자이로 ~200Hz / RV·자력계 ~100Hz
    US  : Wi-Fi TCP 192.168.1.1:5002, 256x256 candidate 프레임 ~8fps

같은 시간축에 올리는 방법: **둘 다 호스트 수신 순간의 `time.time()`(PC unix 시계)로
찍는다.** IMU 스트림(`imu_stream.ImuStream`)이 이미 이 시계로 `pc_unix` 를 남기므로,
US 프레임도 마지막 블록이 도착한 순간 같은 시계로 찍으면 두 로그가 `pc_unix` 로
직접 조인된다.

정직하게 밝혀둘 것:
  * 이건 **수신 시각** 정렬이다. IMU 는 USB(마이크로초급 지연), US 는 Wi-Fi TCP +
    프로브 내부 획득 지연을 거친다. 두 클럭의 skew 는 dev_us 로 사후 보정할 수 있지만,
    **US 경로의 고정 엔드투엔드 지연은 여기서 지워지지 않는다** — 실측 대상이다
    (DESIGN_NOTES: US 프레임그래버 엔드투엔드 지연 ⏳).
  * US 프레임은 방향·scan conversion 이 검증되기 전까지 candidate 다. 원시 바이트를
    변환 없이 그대로 저장한다.

산출물 (out_dir/us_imu_<stamp>/):
  imu_<stamp>.csv        IMU 전 레이트 로그 (SessionLogger, analyze_log.py 호환)
  imu_<stamp>.meta.json  IMU 메타
  us_frames.bin          256x256 uint8 candidate 프레임을 그대로 이어붙인 원시 스트림
  us_index.csv           pc_unix, us_seq, frame_id, byte_offset  (프레임당 한 줄)
  session.meta.json      두 로그를 묶는 세션 메타 (조인 키 = pc_unix)

    python3 host/us_imu_collect.py                       # 기본값으로 수집
    python3 host/us_imu_collect.py --host 192.168.1.1 --port /dev/ttyACM0
    python3 host/us_imu_collect.py --show                # 본체 화면에 US 미리보기
    python3 host/us_imu_collect.py --duration 60         # 60초 후 자동 종료

US 프레임을 되읽는 법:
    import numpy as np
    frames = np.fromfile("us_frames.bin", dtype=np.uint8).reshape(-1, 256, 256)
    # us_index.csv 의 us_seq 행이 frames[us_seq] 에 대응한다
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time

import numpy as np

# --- 두 리포의 모듈 경로 ------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_IMU_HOST = _THIS
# 패키지 디렉터리 이름이 두 철자로 존재했다 (fr5_control / fr5_contorl). 있는 쪽을 쓴다.
_FR5_VISION = next((d for d in (
    os.path.abspath(os.path.join(_THIS, "..", "..", "fr5_control", "fr5_vision")),
    os.path.abspath(os.path.join(_THIS, "..", "..", "fr5_contorl", "fr5_vision")),
) if os.path.isdir(d)), os.path.abspath(os.path.join(_THIS, "..", "..", "fr5_control", "fr5_vision")))
for _p in (_IMU_HOST, _FR5_VISION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from imu_log import SessionLogger              # noqa: E402  imu_bench/host
from imu_stream import ImuStream               # noqa: E402  imu_bench/host
import mag_calib                               # noqa: E402  imu_bench/host
from fr5_vision.us_protocol import (           # noqa: E402  fr5_control/fr5_vision
    CANDIDATE_FRAME_SHAPE,
    UsScannerSession,
    PROFILES,
    SL2C,
)

FRAME_BYTES = CANDIDATE_FRAME_SHAPE[0] * CANDIDATE_FRAME_SHAPE[1]


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def collect(host, port, baud, out_dir, duration, mag_cal_path, use_mag, show, profile=SL2C, max_frames=0):
    session_dir = os.path.join(out_dir, "us_imu_" + _stamp())
    os.makedirs(session_dir, exist_ok=True)

    cal = mag_calib.load(mag_cal_path) if mag_cal_path else None

    # --- IMU: 전 레이트 로그를 SessionLogger 에 남기며 백그라운드 수신 ---
    imu_logger = SessionLogger(
        session_dir, fmt="csv",
        meta={"source": "BNO085 via XIAO nRF52840 Sense", "port": port, "use_mag": use_mag},
        prefix="imu",
    )
    imu = ImuStream(port=port, baud=baud, cal=cal, use_mag=use_mag, logger=imu_logger)
    imu.start()

    # --- US: 이 프로세스가 유일한 TCP 클라이언트 ---
    us = UsScannerSession(host, profile=profile)
    scan_requested = False
    us_bin = open(os.path.join(session_dir, "us_frames.bin"), "wb")
    us_index = open(os.path.join(session_dir, "us_index.csv"), "w", newline="")
    us_writer = csv.writer(us_index)
    us_writer.writerow(["pc_unix", "us_seq", "frame_id", "byte_offset"])

    renderer = None
    if show:
        try:
            import cv2  # noqa: E402
            from viewer.direct_display import render_candidate_frame  # tracer, 있으면 사용
            renderer = (cv2, render_candidate_frame)
            cv2.namedWindow("US+IMU collect", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("US+IMU collect", 900, 700)
        except Exception as exc:  # noqa: BLE001
            print(f"--show 비활성 (렌더러 로드 실패: {exc})")
            renderer = None

    stop = {"flag": False}

    def _handle(_signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    us_seq = 0
    byte_offset = 0
    started = time.monotonic()
    last_report = started
    connected = False
    last_us_stamp = None
    latest_image = None

    print(f"수집 시작 -> {session_dir}")
    print(f"  IMU {port} @ {baud}   US {host}:5002/5003   (공통 축 = pc_unix / time.time())")
    if imu.error:
        print(f"  경고: IMU 오류 = {imu.error}")

    try:
        while not stop["flag"]:
            if duration is not None and time.monotonic() - started >= duration:
                break

            if not us.is_open:
                try:
                    us.open()
                    connected = True
                except OSError as exc:
                    if not stop["flag"]:
                        print(f"  US 연결 대기 ({type(exc).__name__}); 2초 후 재시도", flush=True)
                    time.sleep(2.0)
                    continue

            if us.can_command_scan and not scan_requested and time.monotonic() - started >= 1.6:
                us.start_scan(); scan_requested = True          # C10UR: 클라이언트가 스캔을 시작한다
            try:
                frames = us.poll(0.05)
            except (ConnectionError, OSError) as exc:
                print(f"  US 세션 끊김: {type(exc).__name__}: {exc}; 재연결 시도")
                us.close()
                connected = False
                continue

            for frame in frames:
                # 마지막 블록이 도착한 직후 = IMU 와 같은 time.time() 시계로 찍는다.
                pc_unix = time.time()
                raw = frame.raw_pixels
                us_bin.write(raw)
                us_writer.writerow([f"{pc_unix:.6f}", us_seq, frame.frame_id, byte_offset])
                byte_offset += len(raw)
                us_seq += 1
                last_us_stamp = pc_unix
                if renderer is not None:
                    latest_image = frame.as_uint8_image()
            if max_frames and us_seq >= max_frames:
                print(f"  US {max_frames} 프레임 도달 — 종료"); break

            if renderer is not None:
                cv2, render = renderer
                canvas = render(
                    latest_image, zoom_percent=200, gain_percent=100, orientation=1,
                    canvas_size=(900, 700), scanner_active=us.scanner_active, frame_count=us_seq,
                )
                cv2.imshow("US+IMU collect", canvas)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                    break

            now = time.monotonic()
            if now - last_report >= 5.0:
                elapsed = now - last_report
                snap = imu.snapshot()
                rates = snap.get("rates", {}) or {}
                imu_hz = ", ".join(f"{v:.0f}" for v in rates.values()) if rates else "?"
                us_hz = us_seq / (now - started) if now > started else 0.0
                active = us.scanner_active
                stale = "" if last_us_stamp and time.time() - last_us_stamp < 2.0 else "  [US 정지]"
                print(f"  t={now - started:5.0f}s  US {us_seq} frames ({us_hz:.1f} fps, "
                      f"active={active})  IMU {imu_logger.n} rows [{imu_hz} Hz]{stale}",
                      flush=True)
                last_report = now
    finally:
        if us.is_open and us.can_command_scan:
            try:
                us.stop_scan()
                for _ in range(6):
                    us.poll(0.2)                             # 정지 바이트를 ~1 s 보낸다
            except Exception:  # noqa: BLE001
                pass
        us.close()
        us_bin.close()
        us_index.close()
        imu.stop()
        # ImuStream 이 Thread 내부 _stop 을 Event 로 덮어써 join() 이 깨진다.
        # 데몬 스레드이므로 잠깐 물러나 마지막 append 를 흘려보내고 로거를 닫는다.
        time.sleep(0.2)
        imu_path = imu_logger.close()
        if renderer is not None:
            renderer[0].destroyAllWindows()

        session_meta = {
            "created": _stamp(),
            "time_axis": "pc_unix (host time.time() at reception, shared by both streams)",
            "join_key": "pc_unix",
            "caveat": (
                "수신 시각 정렬임. IMU=USB(µs급), US=Wi-Fi TCP + 프로브 내부 획득 지연. "
                "skew 는 dev_us 로 보정 가능하나 US 고정 엔드투엔드 지연은 미측정."
            ),
            "us": {
                "host": host, "frames": us_seq, "probe": profile.name, "probe_note": profile.note,
                "frame_shape": list(profile.frame_shape), "dtype": "uint8",
                "bin": "us_frames.bin", "index": "us_index.csv",
                "note": "candidate 프레임 — 방향/scan conversion 미검증. 원시 바이트 무변환 저장.",
            },
            "imu": {
                "port": port, "rows": imu_logger.n,
                "csv": os.path.basename(imu_path),
                "meta": os.path.basename(imu_path).replace(".csv", ".meta.json"),
                "error": imu.error,
            },
        }
        with open(os.path.join(session_dir, "session.meta.json"), "w", encoding="utf-8") as fh:
            json.dump(session_meta, fh, indent=2, ensure_ascii=False)

        print(f"\n수집 종료: {session_dir}")
        print(f"  US  {us_seq} frames  -> us_frames.bin ({byte_offset} bytes) + us_index.csv")
        print(f"  IMU {imu_logger.n} rows -> {os.path.basename(imu_path)}")
        if imu.error:
            print(f"  IMU 오류: {imu.error}")
        if us_seq == 0:
            print("  주의: US 프레임 0개 — 프로브가 스캔/active 상태였는지 확인.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="192.168.1.1", help="프로브 AP 주소")
    parser.add_argument("--probe", choices=sorted(PROFILES), default="sl2c", help="프로브 프로파일")
    parser.add_argument("--max-frames", type=int, default=0, help="이 수의 US 프레임 뒤 자동 종료 (0 = 없음)")
    parser.add_argument("--port", default=os.environ.get("IMU_PORT", "/dev/ttyACM0"))
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--out-dir", default=os.path.join(_THIS, "..", "logs"))
    parser.add_argument("--duration", type=float, default=None, help="초 (기본: Ctrl+C 까지)")
    parser.add_argument("--mag-cal", default=None, help="자력계 보정 파일 (없으면 원시)")
    parser.add_argument("--no-mag", action="store_true", help="6축 퓨전 (자력계 미사용)")
    parser.add_argument("--show", action="store_true", help="US 미리보기 창 (DISPLAY 필요)")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration 은 0보다 커야 합니다")

    collect(
        host=args.host, port=args.port, baud=args.baud,
        out_dir=os.path.abspath(args.out_dir), duration=args.duration,
        mag_cal_path=args.mag_cal, use_mag=not args.no_mag, show=args.show,
        profile=PROFILES[args.probe], max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
