"""PX6D 센서를 시리얼로 직접 읽어 보는 점검 도구.

ROS 없이 도는 단독 스크립트다. 센서가 열거되는 순간 배선·프로토콜·축 배정을
가장 먼저 확인하는 용도이며, 제어 경로에는 쓰지 않는다.

    ros2 run fr5_control px6d_probe --port /dev/ttyACM1
    python3 -m fr5_control.px6d_probe --port /dev/ttyACM1 --tare --rate 100

센서는 **CDC-ACM 장치**다 (2026-08-21 커널 로그: GigaDevice GD32-CDC_ACM,
28e9:018a). 따라서 ``/dev/ttyUSB*`` 가 아니라 ``/dev/ttyACM*`` 로 뜬다. 이 장비의
``ttyACM0`` 은 IMU 벤치의 XIAO 보드가 이미 쓰고 있어 보통 ``ttyACM1`` 이 된다.
장치별 고정 경로가 필요하면 ``/dev/serial/by-id/`` 를 쓴다.

``/dev/ttyACM*`` 는 ``dialout`` 그룹이라 세션에 그룹이 안 붙어 있으면 열리지
않는다. 그럴 때는 ``sg dialout -c "python3 -m fr5_control.px6d_probe ..."`` 로
감싼다.

주의: 여기서 찍히는 축 순서와 부호는 :data:`~fr5_control.px6d_protocol.AXIS_ORDER`
의 잠정 배정을 그대로 따른다. 매뉴얼 §5.3 과 §5.4 가 서로 어긋나므로, 각 축을
손으로 눌러 보며 실측으로 확인하기 전까지는 참고값이다.
"""

from __future__ import annotations

import argparse
import sys
import time

from fr5_control.px6d_protocol import (
    AXIS_ORDER,
    build_command,
    build_set_rate,
    CMD_GET_VERSION,
    CMD_STREAM_START,
    CMD_STREAM_STOP,
    CMD_TARE,
    FrameParser,
    SERIAL_BAUD,
    STREAM_START_DATA,
)

#: 명령을 보내고 응답을 기다리는 시간. 1 kHz 스트림 기준으로 넉넉하다.
_REPLY_TIMEOUT_S = 1.0


def _parse_args(argv):
    """명령행 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="PX6D 6축 F/T 센서 시리얼 점검")
    parser.add_argument("--port", default="/dev/ttyACM1", help="시리얼 포트 (기본 %(default)s)")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD, help="보레이트 (기본 %(default)s)")
    parser.add_argument("--rate", type=int, default=100, help="회신 주기 Hz, 4 의 배수 (기본 %(default)s)")
    parser.add_argument("--duration", type=float, default=5.0, help="스트림 수집 시간 초 (기본 %(default)s)")
    parser.add_argument("--tare", action="store_true", help="시작 전 영점 보정. 반드시 무부하 상태에서")
    return parser.parse_args(argv)


def _drain(port, parser, seconds):
    """일정 시간 동안 읽어 프레임 목록을 모은다."""
    frames = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            frames.extend(parser.feed(chunk))
    return frames


def main(argv=None) -> int:
    """센서에 붙어 버전을 읽고 wrench 를 잠시 스트리밍한다.

    Args:
        argv: 명령행 인자. ``None`` 이면 :data:`sys.argv` 를 쓴다.

    Returns:
        종료 코드. 0 은 프레임을 하나 이상 받았다는 뜻이다.
    """
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    try:
        import serial
    except ImportError:
        print("pyserial 이 필요하다: pip install pyserial", file=sys.stderr)
        return 2

    try:
        port = serial.Serial(args.port, args.baud, timeout=0.05)
    except serial.SerialException as exc:
        print(f"{args.port} 를 열 수 없다: {exc}", file=sys.stderr)
        return 2

    parser = FrameParser()
    with port:
        # 이전 세션이 자동 회신을 켜둔 채 죽었을 수 있다. 조용한 상태에서 시작한다.
        port.write(build_command(CMD_STREAM_STOP, 0x00))
        time.sleep(0.1)
        port.reset_input_buffer()

        port.write(build_command(CMD_GET_VERSION))
        for frame in _drain(port, parser, _REPLY_TIMEOUT_S):
            if frame.cmd == CMD_GET_VERSION:
                text = frame.payload.rstrip(b"\x00").decode("ascii", "replace")
                print(f"펌웨어 버전: {text!r}  (raw {frame.payload.hex(' ')})")
                break
        else:
            print("버전 응답이 없다 — 보레이트나 배선을 의심한다", file=sys.stderr)

        if args.tare:
            print("영점 보정 — 무부하 상태여야 한다")
            port.write(build_command(CMD_TARE))
            _drain(port, parser, 0.5)

        port.write(build_set_rate(args.rate))
        time.sleep(0.05)
        port.write(build_command(CMD_STREAM_START, STREAM_START_DATA))

        print(f"{args.duration:.1f} 초 수집 — {'  '.join(f'{a:>9}' for a in AXIS_ORDER)}")
        started = time.monotonic()
        frames = _drain(port, parser, args.duration)
        elapsed = time.monotonic() - started

        port.write(build_command(CMD_STREAM_STOP, 0x00))

    wrenches = [f.wrench() for f in frames if f.is_wrench]
    for values in wrenches[:: max(1, len(wrenches) // 10)][:10]:
        print("  " + "  ".join(f"{v:>9.4f}" for v in values))

    print(
        f"\n프레임 {len(wrenches)} 개 / {elapsed:.2f} 초 = {len(wrenches) / elapsed:.1f} Hz"
        f"  (요청 {args.rate} Hz, CRC 불일치 {parser.crc_errors} 회)"
    )
    return 0 if wrenches else 1


if __name__ == "__main__":
    sys.exit(main())
