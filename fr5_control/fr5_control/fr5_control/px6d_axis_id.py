"""PX6D 축 배정을 손으로 눌러 확정하는 절차 도구.

    ros2 run fr5_control px6d_axis_id --port /dev/ttyACM0
    python3 -m fr5_control.px6d_axis_id --port /dev/ttyACM0 --out axis_id.json

**왜 필요한가.** :data:`~fr5_control.px6d_protocol.AXIS_ORDER` 는 매뉴얼 §5.3 과
§5.4 가 서로 어긋나 잠정 상태다. 그리고 DESIGN_NOTES §4.4 의 부호 규약
``F_n = -F_z^probe`` 도 미검증이다 (반대면 admittance 가 발산한다). 둘 다 눌러
보는 것 말고는 확인할 방법이 없다.

``px6d_viz`` 의 막대 화면으로도 눈으로 볼 수 있지만, 눈으로 본 것은 기록이 남지
않고 나중에 "그때 어느 막대였더라" 가 된다. 이 도구는 **같은 판단을 숫자로 남긴다.**

**절차는 두 단계다.**

1 단계 — **센서 기준 확정.** PX6D 케이스에 인쇄된 축 표시를 따라 민다. 프로브
방향으로 미는 것은 "채널 배정" 과 "마운트 회전" 이 곱해진 결과를 보는 것이라,
어긋나 있어도 어느 쪽이 어긋났는지 가릴 수 없다. 케이스 표시를 따르면 마운트
회전이 식에서 빠지고 채널 배정과 부호만 남는다. 축방향(z)은 적층이 동축이라
센서와 프로브가 공유하므로, 여기서 얻는 z 결과가 곧 ``normal_force_sign`` 이고
마운트 회전과 무관하게 확정된다.

2 단계 — **그 기준 위에서 프로브 각도.** 배열 장축으로 민 반응을 센서 가로면
(Fx, Fy)에서 보면 각도가 하나 나오고, 1 단계 ``Sx+`` 의 각도에서 빼면 그것이
마운트 회전각이다. **사분면이니 기준이니 하는 말이 필요 없다** — 두 번 밀어서
각도 차를 재는 것이 전부다.

    ros2 run fr5_control px6d_axis_id --stage sensor    # 1 단계만
    ros2 run fr5_control px6d_axis_id --stage probe     # 2 단계만
    ros2 run fr5_control px6d_axis_id                   # 둘 다 (기본)

**시행마다 세 구간을 준다** — 준비 / 가함 / 복귀. 가하기 직전 구간의 평균을
기준선으로 잡고, 가하는 동안의 최대 이탈을 채널마다 잰다. 기준선을 시행마다 새로
잡으므로 자세를 바꿔 가며 눌러도 되고, 자중이 서서히 변해도 시행 안에서는 상수다.

2 단계의 방향은 **프로브 면의 생김새로 정의된다.** convex 프로브의 면은 한쪽이
길고 한쪽이 짧은 길쭉한 모양이다. 그 세 방향이 그대로 프로브 프레임이다
(DESIGN_NOTES §4.1):

    +z  면을 뚫고 나가는 방향. 조직으로 파고드는 쪽 = **누르는** 방향.
    +x  면의 **긴 쪽**. 배열 소자가 늘어선 방향이고, 초음파 영상이 퍼지는 쪽이다.
    +y  면의 **짧은 쪽**. 영상에는 안 나오는 두께 방향이다.

**z 만 누르고 x·y 는 민다.** 셋 다 "힘을 준다" 는 점은 같지만 방향이 다르다 —
z 는 면에 **수직으로** 파고드는 힘이고, x·y 는 면과 **나란하게** 미끄러뜨리는
힘이다. x 를 잰다고 긴 쪽 끝을 눌러 버리면 그것은 x 힘이 아니라 y 축 둘레
모멘트가 되고, 각도가 엉뚱하게 나온다.

"영상면" 은 +x 와 +z 가 만드는 평면 — 화면에 보이는 부채꼴이 그 안에 있다.
"영상면 안" 은 +x, "영상면 밖" 은 +y 를 가리키지만, **밀 때는 그냥 면의 긴 쪽 /
짧은 쪽으로 생각하면 된다.**

가장 중요한 시행은 1 단계 첫 번째 ``Sz+`` 압축이다. 여기서 어느 채널이 어느
부호로 움직이는지가 곧 ``normal_force_sign`` 이다.

결과는 표로 찍고 ``--out`` 으로 JSON 을 남긴다. 교차 성분도 함께 남기므로, 축이
깨끗이 갈리지 않으면 (예: 최대와 차순위가 비슷하면) 그것도 보인다 — 그 경우는
센서가 프로브 축과 틀어져 장착됐다는 뜻이고, ``j6_to_sensor_rpy`` 로 풀 문제다.
"""

from __future__ import annotations

import argparse
import json
import math
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
# 1 단계 — **센서 자신의 기준**. 방향은 PX6D 케이스에 인쇄된 축 표시를 따른다.
#
# 이 단계가 먼저인 이유. 프로브 방향으로 눌러서 얻는 것은 "채널 배정" 과 "마운트
# 회전" 이 **곱해진 결과** 라, 어긋나 있어도 어느 쪽이 어긋났는지 가릴 수 없다.
# 케이스 표시를 따라 누르면 마운트 회전이 식에서 빠지고 채널 배정만 남는다.
#
# 축방향(z)은 적층이 동축이라 센서와 프로브가 **공유한다.** 그래서 이 단계에서
# 얻는 z 결과가 곧 normal_force_sign 이고, 마운트 회전과 무관하게 확정된다.
SENSOR_TRIALS = (
    ("Sz+", "프로브 끝을 책상에 대고 **축 방향으로 눌러라** (조직을 누르듯 압축). "
            "적층이 동축이라 이 축은 센서와 프로브가 공유한다", "force"),
    ("Sx+", "**센서 케이스에 인쇄된 +x 화살표 방향으로 옆으로 밀어라.** 프로브를 옆으로 미는 것이며, 면을 파고들지 않게 한다", "force"),
    ("Sy+", "**센서 케이스에 인쇄된 +y 화살표 방향으로** 밀어라. Sx+ 와 직각이며, 같은 세기로 미는 것이 중요하다", "force"),
)

# 2 단계 — 1 단계에서 잡은 센서 기준 위에서 **프로브가 어디에 놓였는지**.
#
# 배열 장축으로 민 반응을 센서의 가로면(Fx, Fy)에서 보면 각도가 하나 나오고,
# 그것을 1 단계의 Sx+ 각도에서 빼면 곧 마운트 회전각이다. 사분면이니 기준이니
# 하는 말이 필요 없다 — 두 번 밀어서 각도 차를 재는 것이 전부다.
PROBE_TRIALS = (
    ("Px+", "프로브 머리를 잡고 **옆으로 밀어라** — 면의 긴 쪽이 가리키는 방향으로. "
            "누르는 것이 아니다 (그건 Sz+ 였다). 면을 따라 미끄러뜨리는 힘이다", "force"),
    ("Py+", "같은 방식으로 **옆으로 밀되**, 이번엔 면의 짧은 쪽 방향으로. "
            "Px+ 와 직각이고, 역시 누르는 것이 아니다", "force"),
    ("Mz+", "프로브를 **제자리에서 비틀어라** — 면을 책상에 댄 채 손잡이를 돌리듯. "
            "프로브가 옆으로 밀리지 않게 한다", "moment"),
)

#: 옛 이름. 전체를 한 번에 돌리던 시절의 순서다.
TRIALS = SENSOR_TRIALS + PROBE_TRIALS


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
    p.add_argument("--stage", default="both", choices=("sensor", "probe", "both"),
                   help="sensor=센서 기준 확정, probe=그 위에서 프로브 각도, "
                        "both=차례로 (기본 %(default)s)")
    p.add_argument("--only", default=None,
                   help="특정 시행만 (쉼표 구분, 예: Sz+,Px+). --stage 보다 우선한다")
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


def lateral_angle_deg(deltas):
    """가로면(Fx, Fy) 반응이 가리키는 각도와 그 품질.

    Args:
        deltas: :func:`summarise` 가 낸 채널별 부호 있는 최대 이탈.

    Returns:
        ``(각도[도], 가로 크기, 축방향 누출비)``. 각도는 센서 +x 에서 잰
        ``atan2(Fy, Fx)`` 다. **누출비** 는 ``|Fz| / 가로크기`` 로, 가로로 민다고
        했는데 축방향이 같이 밀린 정도다. 이것이 크면 각도를 믿을 수 없다 —
        비스듬히 밀었거나 프로브가 책상에 눌린 것이다.

        ``deltas`` 가 없거나 가로 반응이 사실상 0 이면 ``(None, 0.0, None)``.
    """
    if deltas is None:
        return None, 0.0, None
    fx, fy, fz = deltas[0], deltas[1], deltas[2]
    magnitude = math.hypot(fx, fy)
    if magnitude < 1e-6:
        return None, 0.0, None
    leak = abs(fz) / magnitude
    return math.degrees(math.atan2(fy, fx)), magnitude, leak


def pair_orthogonality(results, x_key: str, y_key: str):
    """한 프레임의 x·y 시행이 실제로 직교했는지 본다.

    x 와 y 는 정의상 90° 떨어져 있다. 잰 각이 그렇지 않다면 **둘 중 하나를 엉뚱한
    방향으로 밀었다는 뜻** 이고, 그 시행에서 나온 각도를 쓰면 안 된다. 각도 차만
    보고 있으면 이 오류가 드러나지 않으므로 (차는 언제나 계산되니까) 따로 본다.

    Returns:
        ``(사이각[도], 문장 또는 None)``. 각을 낼 수 없으면 ``(None, 문장)``.
    """
    by_key = {r["trial"]: r for r in results}
    x_trial, y_trial = by_key.get(x_key), by_key.get(y_key)
    if not x_trial or not y_trial:
        return None, None

    x_angle, x_mag, _ = lateral_angle_deg(x_trial["deltas"])
    y_angle, y_mag, _ = lateral_angle_deg(y_trial["deltas"])
    if x_angle is None or y_angle is None:
        return None, f"{x_key}/{y_key} 중 하나가 밀리지 않았다 — 직교성을 볼 수 없다."

    delta = (y_angle - x_angle + 180.0) % 360.0 - 180.0
    error = abs(abs(delta) - 90.0)
    if error > 20.0:
        weakest = x_key if x_mag < y_mag else y_key
        return delta, (
            f"⚠️ {x_key} 와 {y_key} 의 사이각이 {delta:+.1f}° 다 — 90° 여야 한다 "
            f"({error:.0f}° 어긋남). 둘 중 하나를 엉뚱한 방향으로 밀었다는 뜻이고, "
            f"약하게 밀린 {weakest} 부터 다시 하라 "
            f"({x_key} {x_mag:.1f} N · {y_key} {y_mag:.1f} N)."
        )
    return delta, (
        f"{x_key}/{y_key} 사이각 {delta:+.1f}° — 직교한다 ({error:.0f}° 오차). "
        f"두 축이 **회전** 관계이며 반사가 아니라는 뜻이다."
    )


def mounting_angle_deg(results):
    """마운트 회전각. **채널 프레임에서 본 프로브 긴 쪽의 각도** 다.

    이것이 곧 ``registration.mounting_angle_deg`` 다. 보상 코드의 사슬은
    ``{P}R{S} = Rz(-θ)`` 이고 여기서 ``{S}`` 는 **센서가 보고하는 채널 프레임**
    이므로, 필요한 것은 그 프레임에서 프로브 +x 가 놓인 각도다.

    센서는 **민 방향의 반대로 보고한다** (``Sz+`` 압축에서 Fz 가 음수로 나온 것이
    그 증거다). 그래서 반응각에서 180° 를 빼야 실제 방향이 된다.

    ``Py+`` 가 있으면 90° 떨어져 있는지로 교차 확인한다 — 값 하나만 믿지 않는다.

    Returns:
        ``(각도[도], 진단 문장들)``. 각도를 낼 수 없으면 ``(None, 문장들)``.
    """
    by_key = {r["trial"]: r for r in results}
    notes: list[str] = []

    probe = by_key.get("Px+")
    if not probe:
        notes.append("Px+ 를 해야 마운트 회전각이 나온다.")
        return None, notes

    response, magnitude, leak = lateral_angle_deg(probe["deltas"])
    if response is None:
        notes.append("Px+ 가 밀리지 않았다 — 더 세게, 면과 나란하게 밀어라.")
        return None, notes

    angle = (response - 180.0 + 180.0) % 360.0 - 180.0
    notes.append(
        f"Px+ 반응 {response:+.1f}° ({magnitude:.1f} N) → 실제 방향 {angle:+.1f}° "
        f"(센서는 민 방향의 반대로 보고한다)"
    )
    if leak > 0.5:
        notes.append(
            f"⚠️ Px+ 의 축방향 누출이 크다 ({leak:.2f}). 밀면서 눌린 것이다 — "
            "직교 교차확인이 맞으면 각도는 써도 되지만, 아니면 다시 하라."
        )

    short = by_key.get("Py+")
    if short:
        short_angle, _, _ = lateral_angle_deg(short["deltas"])
        if short_angle is not None:
            gap = (short_angle - response + 180.0) % 360.0 - 180.0
            error = abs(abs(gap) - 90.0)
            if error <= 10.0:
                notes.append(
                    f"교차확인: Py+ 와 {gap:+.1f}° 떨어져 있다 — 90° 에서 {error:.1f}° "
                    "오차. 두 방향이 서로를 확인한다."
                )
            else:
                notes.append(
                    f"⚠️ 교차확인 실패: Py+ 와 {gap:+.1f}° 다 (90° 여야 한다). "
                    "둘 중 하나를 엉뚱한 방향으로 밀었다."
                )

    sensor = by_key.get("Sx+")
    if sensor:
        s_angle, _, _ = lateral_angle_deg(sensor["deltas"])
        if s_angle is not None:
            relative = (response - s_angle + 180.0) % 360.0 - 180.0
            notes.append(
                f"참고: 케이스 인쇄 +x 로부터는 {relative:+.1f}° 다. "
                "이 값은 설정에 넣지 않는다 — 데이터가 사는 프레임은 인쇄가 아니라 채널이다."
            )

    notes.append(
        "  180° 더한 값도 물리적으로 구분되지 않는다 (둘 다 정상 회전이다). "
        "그 차이는 x·y 의 **부호** 만 바꾸고 어느 축이 영상면인지는 바꾸지 않으므로, "
        "접촉 제어와 힘 한계에는 영향이 없다. 영상의 좌우가 뒤집히는 문제이며 "
        "실제 영상을 봐야 정해진다."
    )
    return angle, notes


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

    fz = by_key.get("Sz+") or by_key.get("Fz+")
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

    for label, x_key, y_key in (("센서", "Sx+", "Sy+"), ("프로브", "Px+", "Py+")):
        _, note = pair_orthogonality(results, x_key, y_key)
        if note:
            out.append(f"[{label}] {note}")

    angle, notes = mounting_angle_deg(results)
    out.extend(notes)
    if angle is not None:
        out.append(
            f"→ 마운트 회전각 = {angle:+.1f}° ({{P}}R{{S}} = Rz(-θ)). "
            "이 값을 직접 적는 자리는 없다 — probe.yaml 의 tool.j6_to_probe_rpy 와 "
            "ft_sensor.j6_to_sensor_rpy 의 yaw 차이로 **유도된다** "
            "(FrameRegistration.from_stack). 아래 관계를 쓴다."
        )
        broken = [
            label for label, x_key, y_key in
            (("센서", "Sx+", "Sy+"), ("프로브", "Px+", "Py+"))
            if (pair_orthogonality(results, x_key, y_key)[1] or "").startswith("⚠️")
        ]
        if broken:
            out.append(
                f"  ⚠️ 다만 {', '.join(broken)} 프레임의 직교성이 깨져 있다. "
                "위 각도는 **믿지 마라** — 어긋난 시행을 다시 한 뒤 다시 계산하라."
            )
        out.append(
            "  주의: 이 각은 **센서 기준 프로브 각도** 다. probe.yaml 에는 플랜지 "
            "기준 값만 들어가며, 관계는 다음과 같다:\n"
            f"    ft_sensor.j6_to_sensor_rpy.yaw = tool.j6_to_probe_rpy.yaw - ({angle:+.1f}°)\n"
            "  ✅ 2026-08-27 현재 플랜지→프로브 = +90° 이므로 플랜지→센서 = +47° 다."
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

    trials = {"sensor": SENSOR_TRIALS, "probe": PROBE_TRIALS,
              "both": SENSOR_TRIALS + PROBE_TRIALS}[args.stage]
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
            print("⚠️ 센서 쪽 영점을 다시 쓴다. 저장된 교정 프로파일의 전자영점\n"
                  "   (~/.ros/fr5_px6d_calibration.json 의 bias) 이 그 순간 무효가 되고,\n"
                  "   보상은 있지도 않은 오프셋을 계속 빼게 된다 — 자세와 무관한 수 N 의\n"
                  "   잔차로 나타난다. 이걸 보냈으면 --calib 로 교정을 다시 하라.",
                  file=sys.stderr)
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
