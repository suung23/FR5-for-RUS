"""PX6D 축 배정을 손으로 눌러 확정하는 절차 도구.

    ros2 run fr5_control px6d_axis_id --port /dev/ttyACM0
    python3 -m fr5_control.px6d_axis_id --port /dev/ttyACM0 --out axis_id.json

**왜 필요한가.** :data:`~fr5_control.px6d_protocol.AXIS_ORDER` 는 매뉴얼 §5.3 과
§5.4 가 서로 어긋나 잠정 상태다. 그리고 DESIGN_NOTES §4.4 의 부호 규약
``F_n = -F_z^probe`` 도 미검증이다 (반대면 admittance 가 발산한다). 둘 다 눌러
보는 것 말고는 확인할 방법이 없다.

``px6d_viz`` 의 막대 화면으로도 눈으로 볼 수 있지만, 눈으로 본 것은 기록이 남지
않고 나중에 "그때 어느 막대였더라" 가 된다. 이 도구는 **같은 판단을 숫자로 남긴다.**

**절차.** 시행마다 세 구간을 준다 — 준비 / 가함 / 복귀. 가하기 직전 구간의 평균을
기준선으로 잡고, 가하는 동안의 최대 이탈을 채널마다 잰다. 기준선을 시행마다 새로
잡으므로 자세를 바꿔 가며 눌러도 되고, 자중이 서서히 변해도 시행 안에서는 상수다.

**누르는 방향은 프로브 프레임 기준이다** (DESIGN_NOTES §4.1):

    +z = 조직 침투 방향(법선)   +x = 트랜스듀서 배열 방향(영상면 내)
    +y = elevational (영상면 밖)

가장 중요한 시행은 첫 번째 ``+z 압축`` 이다. 여기서 어느 채널이 어느 부호로
움직이는지가 곧 ``normal_force_sign`` 이다.

결과는 표로 찍고 ``--out`` 으로 JSON 을 남긴다. 교차 성분도 함께 남기므로, 축이
깨끗이 갈리지 않으면 (예: 최대와 차순위가 비슷하면) 그것도 보인다 — 그 경우는
센서가 프로브 축과 틀어져 장착됐다는 뜻이고, ``j6_to_sensor_rpy`` 로 풀 문제다.
"""

from __future__ import annotations

import argparse
import json
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

_REPLY_TIMEOUT_S = 1.0

#: 힘 축(앞 3 개)과 모멘트 축(뒤 3 개)의 경계.
_N_FORCE = 3

#: 시행 목록. (키, 화면에 띄울 지시, 힘인가 모멘트인가)
#:
#: 순서에 뜻이 있다. 첫 시행이 부호 규약을 정하는 자리라 맨 앞에 둔다. 힘 셋을
#: 먼저 끝내고 모멘트로 넘어가는 것은, 손이 같은 종류의 동작을 이어서 하는 편이
#: 헷갈리지 않기 때문이다.
TRIALS = (
    ("Fz+", "프로브 끝을 책상에 대고 **축 방향으로 눌러라** (조직을 누르듯, +z 압축)", "force"),
    ("Fx+", "프로브를 **배열 방향(영상면 안)으로 밀어라** — +x", "force"),
    ("Fy+", "프로브를 **elevational 방향(영상면 밖)으로 밀어라** — +y", "force"),
    ("Mx+", "프로브를 **+x 축 둘레로 비틀어라** (영상면이 앞뒤로 기울게)", "moment"),
    ("My+", "프로브를 **+y 축 둘레로 비틀어라** (영상면이 좌우로 기울게)", "moment"),
    ("Mz+", "프로브를 **침투축 둘레로 비틀어라** (영상면이 회전하게) — +z", "moment"),
)


def _parse_args(argv):
    """명령행 인자를 읽는다."""
    p = argparse.ArgumentParser(description="PX6D 축 배정·부호 실측 절차")
    p.add_argument("--port", default="/dev/ttyACM0", help="시리얼 포트 (기본 %(default)s)")
    p.add_argument("--baud", type=int, default=SERIAL_BAUD, help="보레이트 (기본 %(default)s)")
    p.add_argument("--rate", type=int, default=1000, help="센서 회신 주기 Hz (기본 %(default)s)")
    p.add_argument("--settle", type=float, default=3.0, help="준비 구간 초 (기본 %(default)s)")
    p.add_argument("--hold", type=float, default=4.0, help="가함 구간 초 (기본 %(default)s)")
    p.add_argument("--baseline", type=float, default=1.0,
                   help="가하기 직전 기준선 구간 초 (기본 %(default)s)")
    p.add_argument("--tare", action="store_true",
                   help="시작 전 센서 영점. 무부하 상태에서만")
    p.add_argument("--only", default=None,
                   help="특정 시행만 (쉼표 구분, 예: Fz+,Mz+). 기본은 전부")
    p.add_argument("--out", default=None, help="결과 JSON 경로")
    return p.parse_args(argv)


def summarise(baseline, samples):
    """한 시행의 기준선과 표본에서 채널별 최대 이탈을 낸다.

    Args:
        baseline: 기준선 구간의 표본. 각 원소가 6 축 값.
        samples: 가함 구간의 표본. 같은 형식.

    Returns:
        ``(deltas, winner, ratio)``. ``deltas`` 는 채널별 부호 있는 최대 이탈,
        ``winner`` 는 그 크기가 가장 큰 채널 이름, ``ratio`` 는 최대와 차순위의
        비다. 비가 1 에 가까우면 축이 갈리지 않은 것이고 장착 틀어짐을 의심한다.
        표본이 없으면 ``(None, None, None)``.

    누적이 아니라 **부호 있는 최대 이탈**을 쓴다. 절대값 최대만 보면 눌렀다 놓는
    반동이 반대 부호로 더 크게 잡히는 일이 있는데, 그러면 부호 규약이 뒤집힌다.
    그래서 채널마다 최대·최소를 모두 보고 **크기가 큰 쪽의 부호**를 남긴다.
    """
    if not baseline or not samples:
        return None, None, None

    n = len(AXIS_ORDER)
    base = [sum(s[i] for s in baseline) / len(baseline) for i in range(n)]

    deltas = []
    for i in range(n):
        devs = [s[i] - base[i] for s in samples]
        lo, hi = min(devs), max(devs)
        deltas.append(hi if abs(hi) >= abs(lo) else lo)

    # 힘과 모멘트는 단위가 달라 한 줄로 비교하면 안 된다. 같은 종류 안에서만 겨룬다.
    order = sorted(range(n), key=lambda i: abs(deltas[i]), reverse=True)
    winner = AXIS_ORDER[order[0]]

    same_kind = [i for i in order if (i < _N_FORCE) == (order[0] < _N_FORCE)]
    ratio = None
    if len(same_kind) > 1:
        second = abs(deltas[same_kind[1]])
        ratio = float("inf") if second == 0 else abs(deltas[same_kind[0]]) / second
    return deltas, winner, ratio


def _drain(port, parser, seconds, sink=None):
    """일정 시간 읽는다. ``sink`` 가 있으면 wrench 를 거기에 모은다."""
    deadline = time.monotonic() + seconds
    frames = []
    while time.monotonic() < deadline:
        chunk = port.read(port.in_waiting or 1)
        if not chunk:
            continue
        for frame in parser.feed(chunk):
            if sink is not None and frame.is_wrench:
                sink.append(frame.wrench())
            else:
                frames.append(frame)
    return frames


def _countdown(label: str, seconds: float) -> None:
    """남은 시간을 한 줄에 덮어쓰며 센다."""
    end = time.monotonic() + seconds
    while True:
        left = end - time.monotonic()
        if left <= 0:
            break
        sys.stdout.write(f"\r\x1b[2K  {label} {left:4.1f} s")
        sys.stdout.flush()
        time.sleep(min(0.1, max(0.0, left)))
    sys.stdout.write("\r\x1b[2K")
    sys.stdout.flush()


def _ensure_stream(port, parser, args, attempts: int = 3) -> bool:
    """스트림이 실제로 흐르는지 확인하고, 아니면 다시 켠다.

    시행은 사람이 손으로 누르는 것이라 한 번 실패하면 그 수고가 통째로 버려진다.
    켜라고 보냈다고 켜진 것으로 믿지 않고, 프레임이 오는 것을 보고 넘어간다.

    Args:
        attempts: 다시 켜 볼 횟수.

    Returns:
        프레임을 하나라도 받았으면 ``True``.
    """
    for i in range(attempts):
        port.write(build_set_rate(args.rate))
        time.sleep(0.05)
        port.write(build_command(CMD_STREAM_START, STREAM_START_DATA))
        time.sleep(0.05)
        port.reset_input_buffer()
        parser.reset()

        probe: list = []
        _drain(port, parser, 0.3, probe)
        if probe:
            return True
        print(f"  스트림이 안 온다 — 다시 켠다 ({i + 1}/{attempts})", file=sys.stderr)
        port.write(build_command(CMD_STREAM_STOP, 0x00))
        time.sleep(0.2)
    return False


def run_trial(port, parser, key, instruction, args) -> dict:
    """시행 하나. 지시 → 준비 → 기준선 → 가함 → 요약."""
    print(f"\n[{key}] {instruction}")
    if not _ensure_stream(port, parser, args):
        print(f"  [{key}] 스트림이 없어 건너뛴다", file=sys.stderr)
        return {"trial": key, "instruction": instruction, "n_baseline": 0,
                "n_hold": 0, "deltas": None, "winner": None, "separation": None}
    _countdown("준비 — 아직 누르지 마라", args.settle)

    baseline: list = []
    print("  기준선 측정 중 — 손을 대지 마라")
    _drain(port, parser, args.baseline, baseline)

    samples: list = []
    print("  ** 지금 눌러라 **")
    deadline = time.monotonic() + args.hold
    while time.monotonic() < deadline:
        chunk = port.read(port.in_waiting or 1)
        if not chunk:
            continue
        for frame in parser.feed(chunk):
            if frame.is_wrench:
                samples.append(frame.wrench())
    print("  놓아라")

    deltas, winner, ratio = summarise(baseline, samples)
    if deltas is None:
        print(f"  ⚠️ [{key}] 표본이 없다 — 이 시행은 다시 하라 "
              f"(--only {key})", file=sys.stderr)
    return {"trial": key, "instruction": instruction,
            "n_baseline": len(baseline), "n_hold": len(samples),
            "deltas": deltas, "winner": winner, "separation": ratio}


def format_report(results) -> str:
    """시행 결과를 표로. 사람이 읽고 바로 판단할 수 있게."""
    head = "  " + f"{'시행':<6}" + "".join(f"{a:>10}" for a in AXIS_ORDER) + f"{'판정':>10}{'분리비':>9}"
    lines = [head, "  " + "-" * (len(head) - 2)]
    for r in results:
        if r["deltas"] is None:
            lines.append(f"  {r['trial']:<6}" + "  (표본 없음)")
            continue
        cells = "".join(f"{d:>10.4f}" for d in r["deltas"])
        sep = "   —" if r["separation"] is None else (
            "   ∞" if r["separation"] == float("inf") else f"{r['separation']:>9.1f}")
        lines.append(f"  {r['trial']:<6}{cells}{r['winner']:>10}{sep}")
    return "\n".join(lines)


def interpret(results) -> list[str]:
    """표에서 바로 읽히는 결론을 문장으로. 특히 부호 규약."""
    out = []
    by_key = {r["trial"]: r for r in results}

    fz = by_key.get("Fz+")
    if fz and fz["deltas"] is not None:
        idx = AXIS_ORDER.index(fz["winner"])
        val = fz["deltas"][idx]
        out.append(
            f"침투축(+z 압축)에 반응한 채널은 {fz['winner']} 이고 부호는 "
            f"{'양수' if val > 0 else '음수'} ({val:+.4f})."
        )
        if fz["winner"] == "Fz":
            sign = -1.0 if val < 0 else 1.0
            out.append(
                f"→ probe.yaml 의 ft_sensor.normal_force_sign = {sign:+.1f} "
                f"(F_n = sign x F_z, 양수 = 압축). 현재 설정값은 -1.0 이다."
            )
        else:
            out.append(
                f"⚠️ 침투축이 Fz 가 아니라 {fz['winner']} 로 나왔다. AXIS_ORDER 가 "
                "실제와 다르거나 센서가 틀어져 장착됐다는 뜻이다 — 둘을 가르려면 "
                "나머지 시행의 판정을 함께 봐야 한다."
            )

    weak = [r["trial"] for r in results
            if r["separation"] is not None and r["separation"] != float("inf")
            and r["separation"] < 3.0]
    if weak:
        out.append(
            f"⚠️ 분리비가 3 미만인 시행: {', '.join(weak)}. 해당 축은 단독으로 "
            "가해지지 않았거나(자세·지렛대), 센서 축이 프로브 축과 틀어져 있다."
        )

    mapping = {r["trial"]: r["winner"] for r in results if r["winner"]}
    if len(set(mapping.values())) < len(mapping):
        out.append(
            f"⚠️ 서로 다른 시행이 같은 채널을 지목했다: {mapping}. 배정을 확정할 수 "
            "없으니 해당 시행을 다시 하라 (--only 로 골라 다시 할 수 있다)."
        )
    return out


def main(argv=None) -> int:
    """센서에 붙어 시행을 차례로 돌리고 표와 JSON 을 낸다."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    try:
        import serial
    except ImportError:
        print("pyserial 이 필요하다: pip install pyserial", file=sys.stderr)
        return 2

    trials = TRIALS
    if args.only:
        want = {t.strip() for t in args.only.split(",") if t.strip()}
        trials = tuple(t for t in TRIALS if t[0] in want)
        unknown = want - {t[0] for t in TRIALS}
        if unknown:
            print(f"모르는 시행 이름: {sorted(unknown)}", file=sys.stderr)
            return 2
        if not trials:
            print("고른 시행이 없다", file=sys.stderr)
            return 2

    try:
        port = serial.Serial(args.port, args.baud, timeout=0.01)
    except serial.SerialException as exc:
        print(f"{args.port} 를 열 수 없다: {exc}", file=sys.stderr)
        return 2

    parser = FrameParser()
    results = []
    with port:
        port.write(build_command(CMD_STREAM_STOP, 0x00))
        time.sleep(0.1)
        port.reset_input_buffer()

        port.write(build_command(CMD_GET_VERSION))
        for frame in _drain(port, parser, _REPLY_TIMEOUT_S):
            if frame.cmd == CMD_GET_VERSION:
                text = frame.payload.rstrip(b"\x00").decode("ascii", "replace")
                print(f"펌웨어 버전: {text!r}")
                break
        else:
            print("버전 응답이 없다 — 보레이트나 배선을 의심한다", file=sys.stderr)

        if args.tare:
            print("영점 보정 — 무부하 상태여야 한다")
            port.write(build_command(CMD_TARE))
            _drain(port, parser, 0.5)

        if not _ensure_stream(port, parser, args):
            print("스트림을 켜지 못했다 — 케이블과 포트를 확인하라", file=sys.stderr)
            port.write(build_command(CMD_STREAM_STOP, 0x00))
            return 2

        print(f"\n시행 {len(trials)} 개. 프로브 프레임 기준이다 "
              "(+z 침투 · +x 배열 · +y elevational).")
        print("각 시행은 준비 → 기준선 → 가함 순이고, 화면 지시에 맞춰 움직이면 된다.")
        try:
            for key, instruction, _kind in trials:
                results.append(run_trial(port, parser, key, instruction, args))
        except KeyboardInterrupt:
            print("\n중단됨 — 여기까지의 결과만 낸다")
        finally:
            port.write(build_command(CMD_STREAM_STOP, 0x00))

    if not results:
        print("결과가 없다", file=sys.stderr)
        return 1

    print("\n" + format_report(results))
    print("\n읽는 법: 각 행에서 크기가 가장 큰 칸이 그 방향에 대응하는 채널이다.")
    print("분리비 = 최대 / 차순위 (같은 종류 안에서). 3 미만이면 축이 갈리지 않은 것이다.\n")
    for line in interpret(results):
        print("  " + line)

    if args.out:
        payload = {"axis_order": list(AXIS_ORDER), "port": args.port,
                   "settle_s": args.settle, "hold_s": args.hold,
                   "baseline_s": args.baseline, "tared": bool(args.tare),
                   "trials": results}
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print(f"\n{args.out} 에 저장했다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
