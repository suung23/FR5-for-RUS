"""PX6D 센서 값을 터미널에서 실시간으로 보는 도구.

``px6d_probe`` 가 "붙는가"를, ``px6d_viz`` 가 "무엇이 찍히는가"를 창으로 보는
도구라면, 이쪽은 같은 것을 **화면 없이** 본다. SSH 로 붙었거나 로봇 옆 콘솔만
있을 때 쓴다. matplotlib 을 요구하지 않고 pyserial 만 있으면 돈다.

    ros2 run fr5_control px6d_monitor --port /dev/ttyACM0
    python3 -m fr5_control.px6d_monitor --port /dev/ttyACM0 --zero
    python3 -m fr5_control.px6d_monitor --once            # 한 줄 찍고 끝
    python3 -m fr5_control.px6d_monitor --plain > log.txt # 파이프·기록용

``/dev/ttyACM*`` 는 ``dialout`` 그룹이라 세션에 그룹이 안 붙어 있으면 열리지
않는다. 그럴 때는 ``sg dialout -c "python3 -m fr5_control.px6d_monitor ..."`` 로
감싼다.

**막대는 축 배정 실측용이다.** :data:`~fr5_control.px6d_protocol.AXIS_ORDER` 는
매뉴얼 §5.3 과 §5.4 가 어긋나 잠정 상태다. 센서를 고정하고 한 축씩 눌러 어느
막대가 움직이는지 보면 배정과 부호가 바로 확인된다. 오른쪽 피크 유지값
(``peak``) 은 시작 이후 최대·최소라 손을 뗀 뒤에도 남으므로, 누른 축을 놓치지
않는다. 그 전까지 축 이름은 참고값이다.

센서 영점(``--tare``) 과 소프트 영점(``--zero``) 은 다르다. 앞쪽은 센서 내부
기준을 바꾸고 뒤쪽은 화면에만 적용된다. 드리프트를 보려면 ``--zero`` 만 쓰고
``--tare`` 는 건드리지 않는다 — 센서 기준이 바뀌면 관측할 대상이 사라진다.

끝낼 때는 ``Ctrl-C``. 스트림 정지 명령을 보내고 나가므로 다음 세션이 조용한
포트에서 시작한다.
"""

from __future__ import annotations

import argparse
import sys
import time
import unicodedata

from fr5_control.contact_state import ContactDetector
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

#: 막대 한쪽 길이. 전체 폭은 이 값의 두 배 + 가운데 눈금 한 칸이다.
_BAR_HALF = 16

#: 힘 축(앞 3 개)과 모멘트 축(뒤 3 개)의 경계. AXIS_ORDER 가 Fx,Fy,Fz,Mx,My,Mz 다.
_N_FORCE = 3

#: 실효 주기를 재는 창. 짧으면 튀고 길면 굼뜨다.
_RATE_WINDOW_S = 1.0


def _parse_args(argv):
    """명령행 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="PX6D 6축 F/T 센서 터미널 실시간 표시")
    parser.add_argument("--port", default="/dev/ttyACM0", help="시리얼 포트 (기본 %(default)s)")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD, help="보레이트 (기본 %(default)s)")
    parser.add_argument("--rate", type=int, default=1000, help="센서 회신 주기 Hz, 4 의 배수 (기본 %(default)s)")
    parser.add_argument("--fps", type=float, default=10.0, help="화면 갱신 주기 (기본 %(default)s)")
    parser.add_argument("--force-span", type=float, default=10.0, help="힘 막대가 끝까지 차는 값 N (기본 %(default)s)")
    parser.add_argument("--torque-span", type=float, default=0.5, help="모멘트 막대가 끝까지 차는 값 N·m (기본 %(default)s)")
    parser.add_argument("--duration", type=float, default=0.0, help="수집 시간 초. 0 이면 Ctrl-C 까지 (기본 %(default)s)")
    parser.add_argument("--tare", action="store_true", help="시작 전 센서 영점 보정. 반드시 무부하 상태에서")
    parser.add_argument("--zero", action="store_true", help="소프트 영점. 처음 받은 값들의 평균을 화면에서만 뺀다")
    parser.add_argument(
        "--zero-samples", type=int, default=200, help="소프트 영점에 쓸 표본 수 (기본 %(default)s)"
    )
    parser.add_argument("--once", action="store_true", help="한 줄만 찍고 끝낸다. 스크립트에서 쓰기 좋다")
    parser.add_argument("--plain", action="store_true", help="제자리 갱신 없이 한 줄씩 덧붙인다. 파이프·기록용")
    parser.add_argument(
        "--contact", action="store_true",
        help="접근/접촉 판정을 함께 띄운다 (DESIGN_NOTES §4.4 · contact_state)",
    )
    parser.add_argument("--enter-n", type=float, default=7.0,
                        help="접촉 진입 문턱 N, F_n 기준 (기본 %(default)s)")
    parser.add_argument("--release-n", type=float, default=0.2,
                        help="접촉 이탈 문턱 N (기본 %(default)s)")
    parser.add_argument("--normal-force-sign", type=float, default=-1.0,
                        help="F_n = sign x F_z. probe.yaml 과 맞춘다 (기본 %(default)s)")
    return parser.parse_args(argv)


def _spans(force_span: float, torque_span: float) -> tuple[float, ...]:
    """축마다 막대가 끝까지 차는 기준값을 만든다.

    Args:
        force_span: 힘 축의 기준값 N.
        torque_span: 모멘트 축의 기준값 N·m.

    Returns:
        :data:`AXIS_ORDER` 순서의 기준값 6 개.
    """
    return tuple(force_span if i < _N_FORCE else torque_span for i in range(len(AXIS_ORDER)))


def _units() -> tuple[str, ...]:
    """축마다 붙일 단위 문자열."""
    return tuple("N  " if i < _N_FORCE else "N·m" for i in range(len(AXIS_ORDER)))


def _bar(value: float, span: float) -> str:
    """0 을 가운데 둔 좌우 대칭 막대를 그린다.

    Args:
        value: 표시할 값.
        span: 막대가 한쪽 끝까지 차는 값. 0 이하면 눈금만 돌려준다.

    Returns:
        폭이 ``2 * _BAR_HALF + 1`` 로 고정된 문자열. 범위를 넘으면 해당 끝이
        ``»`` 또는 ``«`` 로 바뀌어 잘렸음을 알린다.
    """
    cells = ["·"] * (2 * _BAR_HALF + 1)
    cells[_BAR_HALF] = "┼"
    if span <= 0:
        return "".join(cells)

    clipped = max(-1.0, min(1.0, value / span))
    n = int(round(clipped * _BAR_HALF))
    lo, hi = (_BAR_HALF + n, _BAR_HALF) if n < 0 else (_BAR_HALF, _BAR_HALF + n)
    for i in range(lo, hi + 1):
        cells[i] = "█"
    if n == 0:
        cells[_BAR_HALF] = "┼"
    if value > span:
        cells[-1] = "»"
    elif value < -span:
        cells[0] = "«"
    return "".join(cells)


def _display_width(text: str) -> int:
    """터미널에서 차지하는 칸 수. 한글·기호는 두 칸을 먹는다.

    Args:
        text: 잴 문자열.

    Returns:
        칸 수. :func:`len` 과 달리 동아시아 문자를 2 로 센다.
    """
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    """표시 폭 기준으로 채운다. 한글이 섞인 머리글 정렬에 쓴다."""
    fill = " " * max(0, width - _display_width(text))
    return fill + text if right else text + fill


def _rate(stamps) -> str:
    """도착 시각 목록에서 실효 주기를 만든다.

    창이 다 차기 전에도 맞는 값이 나오도록 벽시계가 아니라 표본끼리의 간격으로
    잰다. 그래서 시작 직후에도 주기가 0 에서 차오르거나 튀지 않는다.

    Args:
        stamps: 최근 프레임의 도착 시각. 오름차순이어야 한다.

    Returns:
        ``"1001.2"`` 처럼 폭이 고정된 문자열. 아직 못 잴 때는 ``"    —"``.
    """
    if len(stamps) < 2:
        return "     —"
    span = stamps[-1] - stamps[0]
    if span <= 0:
        return "     —"
    return f"{(len(stamps) - 1) / span:6.1f}"


def _render(values, peaks, spans, units, status: str) -> list[str]:
    """한 번 그릴 화면을 줄 목록으로 만든다.

    Args:
        values: 현재 6 축 값. ``None`` 이면 아직 받은 프레임이 없다는 뜻이다.
        peaks: 축마다 ``(최소, 최대)`` 쌍.
        spans: 축마다 막대 기준값.
        units: 축마다 단위 문자열.
        status: 맨 아래에 붙일 상태 한 줄.

    Returns:
        화면에 그대로 찍을 줄 목록.
    """
    bar_width = 2 * _BAR_HALF + 1
    lines = [
        "  " + _pad("축", 4) + _pad("값", 12, right=True) + " " * 6
        + _pad("막대 (0 이 가운데)", bar_width) + "  peak (min … max)"
    ]
    for i, name in enumerate(AXIS_ORDER):
        if values is None:
            lines.append(f"  {name:<4}" + _pad("—", 12, right=True) + "      (대기)")
            continue
        lo, hi = peaks[i]
        lines.append(
            f"  {name:<4}{values[i]:>12.4f} {units[i]}  {_bar(values[i], spans[i])}"
            f"  {lo:>9.4f} … {hi:>9.4f}"
        )
    lines.append(f"  {status}")
    return [line.rstrip() for line in lines]


def _drain(port, parser, seconds):
    """일정 시간 동안 읽어 프레임 목록을 모은다."""
    frames = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            frames.extend(parser.feed(chunk))
    return frames


def _handshake(port, parser, args) -> None:
    """조용한 포트에서 시작해 버전을 읽고 스트림을 켠다."""
    # 이전 세션이 자동 회신을 켜둔 채 죽었을 수 있다.
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


def _loop(port, parser, args):
    """프레임을 받아 화면을 갱신한다.

    ``Ctrl-C`` 는 여기서 받는다. 밖에서 받으면 지금까지 센 프레임 수를 잃어
    정상 종료인데도 "0 개 수신" 으로 보고하게 된다.

    Returns:
        ``(wrench 프레임 개수, 스트림 중 CRC 불일치 횟수)``. CRC 는 핸드셰이크와
        첫 동기에서 생긴 재동기를 빼고 세므로, 회선 품질만 남는다.
    """
    spans, units = _spans(args.force_span, args.torque_span), _units()
    n_axes = len(AXIS_ORDER)
    offsets = [0.0] * n_axes
    peaks = [(float("inf"), float("-inf"))] * n_axes
    zero_acc, zero_n = [0.0] * n_axes, 0

    detector = ContactDetector(
        enter_n=args.enter_n,
        release_n=args.release_n,
        normal_force_sign=args.normal_force_sign,
    ) if args.contact else None
    prev_t = None

    latest = None
    total = 0
    stamps: list[float] = []          # 최근 프레임 도착 시각, 실효 주기 계산용
    drawn = 0                         # 지난 번에 찍은 줄 수. 제자리 갱신에 쓴다
    # 스트림을 켠 뒤 여기까지 오는 사이에 쌓인 프레임을 버린다. 그대로 읽으면
    # 첫 화면의 실효 주기가 실제의 수십 배로 뜨고, 중간에 잘린 프레임 하나가
    # CRC 불일치로 잡힌다.
    port.reset_input_buffer()
    parser.reset()
    crc_base = parser.crc_errors     # 핸드셰이크 재동기는 회선 품질이 아니다
    synced = False

    started = time.monotonic()
    next_draw = started
    interval = 1.0 / args.fps if args.fps > 0 else 0.0

    try:
        while True:
            now = time.monotonic()
            if args.duration > 0 and now - started >= args.duration:
                break

            chunk = port.read(port.in_waiting or 1)
            for frame in parser.feed(chunk) if chunk else ():
                if not frame.is_wrench:
                    continue
                raw = frame.wrench()
                if not synced:
                    # 버퍼를 버린 직후에는 프레임 한가운데부터 읽히므로 파서가
                    # 한 번 재동기한다. 첫 프레임이 선 시점을 기준으로 잡아야 그
                    # 한 번이 회선 오류로 오해되지 않는다.
                    crc_base, synced = parser.crc_errors, True
                total += 1
                stamps.append(time.monotonic())

                if args.zero and zero_n < args.zero_samples:
                    zero_acc = [a + v for a, v in zip(zero_acc, raw)]
                    zero_n += 1
                    if zero_n == args.zero_samples:
                        offsets = [a / zero_n for a in zero_acc]
                        peaks = [(float("inf"), float("-inf"))] * n_axes

                latest = tuple(v - o for v, o in zip(raw, offsets))
                peaks = [(min(lo, v), max(hi, v)) for (lo, hi), v in zip(peaks, latest)]

                if detector is not None:
                    # 판정은 **보정 전 원값**으로 한다. 소프트 영점은 화면을
                    # 읽기 좋게 만드는 것이지 물리량을 바꾸지 않는다. 판정을 화면
                    # 설정에 딸려 보내면 --zero 를 준 순간 문턱이 옮겨진다.
                    now_s = stamps[-1]
                    detector.update(raw[2], (now_s - prev_t) if prev_t else 0.0)
                    prev_t = now_s

            cutoff = time.monotonic() - _RATE_WINDOW_S
            while stamps and stamps[0] < cutoff:
                stamps.pop(0)

            now = time.monotonic()
            if now < next_draw:
                continue
            next_draw = now + interval

            if latest is None:
                if args.once:
                    continue          # 첫 프레임을 받을 때까지 기다린다
            elif args.once:
                print("  ".join(f"{v:.4f}" for v in latest))
                return total, parser.crc_errors - crc_base

            if args.zero and zero_n < args.zero_samples:
                zeroing = f" · 소프트 영점 {zero_n}/{args.zero_samples}"
            else:
                zeroing = " · 소프트 영점 적용" if args.zero else ""

            contact = f"  |  {detector.describe()}" if detector is not None else ""
            status = (
                f"{_rate(stamps)} Hz  프레임 {total}  "
                f"CRC 불일치 {parser.crc_errors - crc_base}{zeroing}{contact}"
            )

            if args.plain:
                if latest is not None:
                    print("  ".join(f"{v:9.4f}" for v in latest) + f"   # {status}", flush=True)
                continue

            lines = _render(latest, peaks, spans, units, status)
            out = "\x1b[F" * drawn + "".join(f"\x1b[2K{line}\n" for line in lines)
            sys.stdout.write(out)
            sys.stdout.flush()
            drawn = len(lines)

    except KeyboardInterrupt:
        pass

    return total, parser.crc_errors - crc_base


def main(argv=None) -> int:
    """센서에 붙어 wrench 를 터미널에 계속 찍는다.

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
        port = serial.Serial(args.port, args.baud, timeout=0.01)
    except serial.SerialException as exc:
        print(f"{args.port} 를 열 수 없다: {exc}", file=sys.stderr)
        return 2

    parser = FrameParser()
    total, crc_errors = 0, 0
    with port:
        _handshake(port, parser, args)
        if not args.once and not args.plain:
            print(f"Ctrl-C 로 종료. 축 이름은 잠정 배정이다 (AXIS_ORDER {'/'.join(AXIS_ORDER)}).")
        try:
            total, crc_errors = _loop(port, parser, args)
        finally:
            # 다음 세션이 조용한 포트에서 시작하도록 항상 스트림을 끈다.
            port.write(build_command(CMD_STREAM_STOP, 0x00))

    if not args.once:
        print(f"\n프레임 {total} 개 수신, CRC 불일치 {crc_errors} 회")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
