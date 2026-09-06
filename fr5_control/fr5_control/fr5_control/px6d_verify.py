"""무접촉 검증 — 보상된 렌치가 자세를 바꿔도 0 에 머무는지 잰다.

    ./scripts/start_session.sh --calib      # 터미널 1 — 브리지가 떠 있어야 한다
    ros2 run fr5_control px6d_verify        # 터미널 2

**브리지가 필요하다.** 이 도구는 시리얼을 직접 열지 않는다 — 재는 것이 원값이
아니라 **보상된** 렌치이고, 보상은 브리지만 한다 (자세 × 교정 프로파일). 그래서
``ws://localhost:8765`` 에 붙는다.

자세도 필요하므로 ``--calib`` 로 띄운다. 그 모드는 브리지가 컨트롤러에서 관절각을
읽기로만 가져오고 제어 스택은 안 띄우므로, 조작자가 드래그 모드로 팔을 옮기는
동안 ``us_servo`` 가 맞서지 않는다.

교정이 **유효하다고 판정된 것** 과 **실제로 맞는 것** 은 다르다. 적합은 잡은
자세들 위에서 잔차를 최소화하므로, 그 자세들에서 작은 것은 당연하다. 이 도구는
**적합에 쓰이지 않은 자세** 에서 보상 뒤 힘이 얼마나 남는지를 본다.

**절차는 조작자가 이끈다.** 터미널이 자세 하나를 지시하고 기다린다. 드래그 모드로
옮긴 뒤 스페이스바를 누르면 그때부터 표본을 모은다. 한 자세가 끝나면 다음 자세를
지시한다.

수집 시점을 도구가 판정하지 않는 것이 요점이다. 관절이 멈춘 것과 조작자가 손을 뗀
것은 다르고, 그 차이는 화면 밖에 있어서 도구가 알 수 없다. 아는 사람이 누르게 하면
그 문제가 사라진다.

판정은 남은 힘의 **최대값** 으로 한다. 평균은 오해를 부른다 — 자세 대부분에서
작고 한 자세에서만 큰 것이 전형적인 실패이고, 평균은 그것을 감춘다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import termios
import tty

import numpy as np

#: 검증할 자세, 지시 문구와 함께. 순서대로 진행한다.
POSES = (
    ("아래 수직", "프로브가 **똑바로 아래**를 보게 하라"),
    ("위 수직", "프로브가 **똑바로 위**를 보게 하라 — 보상은 여기서 가장 크게 틀린다"),
    ("옆", "프로브를 **옆으로 눕혀라** (축이 수평)"),
)


def wait_for_space(prompt: str) -> str:
    """스페이스바를 기다린다.

    Returns:
        ``"go"`` 스페이스, ``"skip"`` s, ``"quit"`` q 또는 Ctrl-C.
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()
    if not sys.stdin.isatty():
        # 파이프로 돌리는 경우(시험 등)는 기다리지 않는다.
        sys.stdout.write("  (tty 아님 — 바로 진행)\n")
        return "go"

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            key = sys.stdin.read(1)
            if key == " ":
                return "go"
            if key in ("s", "S"):
                return "skip"
            if key in ("q", "Q", "\x03"):
                return "quit"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


async def collect_one(url: str, count: int, quiet_n: float):
    """한 자세에서 보상된 힘을 ``count`` 표본 모은다.

    Returns:
        ``(크기 목록, 창 진폭 목록, 보상 전 최대)``. 창 진폭은 그 1 초 안에서
        힘이 얼마나 흔들렸는가로, 조작자가 그 자세를 다시 잡을지 판단하는 근거다.
    """
    import websockets

    magnitudes: list = []
    swings: list = []
    raw_max = 0.0
    async with websockets.connect(url, open_timeout=5) as socket:
        while len(magnitudes) < count:
            message = json.loads(await socket.recv())
            if message.get("type") != "wrench":
                continue
            stages = message.get("compensated")
            if not stages or "contactProbe" not in stages:
                raise RuntimeError(
                    "보상 단계가 오지 않는다 — 교정 프로파일이 안 실렸다는 뜻이다"
                )

            force = np.asarray(stages["contactProbe"][:3], dtype=float)
            magnitude = float(np.linalg.norm(force))
            magnitudes.append(magnitude)
            raw_max = max(raw_max, float(np.linalg.norm(message["force"])))

            extremes = message.get("forceExtremes")
            swing = 0.0
            if extremes:
                swing = float(
                    np.abs(np.asarray(extremes[1]) - np.asarray(extremes[0])).max()
                )
            swings.append(swing)

            bar = "#" * min(30, int(magnitude / quiet_n * 15))
            sys.stdout.write(
                f"\r\x1b[2K    {len(magnitudes)}/{count}  {magnitude:6.3f} N  "
                f"(창 진폭 {swing:.2f})  {bar}"
            )
            sys.stdout.flush()
    sys.stdout.write("\r\x1b[2K")
    return magnitudes, swings, raw_max


def report_pose(name: str, magnitudes, swings, quiet_n: float) -> dict:
    """한 자세의 결과를 찍고 요약을 돌려준다."""
    worst = float(np.max(magnitudes))
    mean = float(np.mean(magnitudes))
    swing = float(np.max(swings)) if swings else 0.0
    mark = "✅" if worst < quiet_n else "⚠️"
    print(
        f"    {mark} {name}: 평균 {mean:.3f} N · 최대 {worst:.3f} N "
        f"· 창 진폭 최대 {swing:.2f} N"
    )
    if swing > 0.3:
        print(
            "       창 진폭이 크다 — 그 사이 힘이 흔들렸다는 뜻이다. "
            "손이 닿았거나 조립이 아직 안정되지 않았으면 다시 잡아라."
        )
    return {
        "pose": name,
        "samples": len(magnitudes),
        "mean_n": mean,
        "max_n": worst,
        "max_swing_n": swing,
        "passed": bool(worst < quiet_n),
    }


def main(argv=None) -> int:
    """자세를 하나씩 지시하고, 조작자가 누르면 모은다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="ws://localhost:8765", help="브리지 주소")
    parser.add_argument("--count", type=int, default=6, help="자세마다 모을 표본 수")
    parser.add_argument(
        "--quiet-n", type=float, default=0.5,
        help="합격선 [N]. 남은 힘의 최대가 이보다 크면 불합격",
    )
    parser.add_argument(
        "--out", default=os.path.expanduser("~/.ros/fr5_px6d_verify.json"),
        help="결과 기록 경로. 화면에만 찍으면 그 창을 닫는 순간 사라진다",
    )
    parser.add_argument(
        "--poses", default="",
        help="검증할 자세만 쉼표로. 비우면 전부. 예: '옆'",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    wanted = [name.strip() for name in args.poses.split(",") if name.strip()]
    known = {name for name, _ in POSES}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        print(f"모르는 자세: {unknown}", file=sys.stderr)
        print(f"  쓸 수 있는 이름: {[name for name, _ in POSES]}", file=sys.stderr)
        return 2
    plan = [item for item in POSES if not wanted or item[0] in wanted]

    print()
    print("  무접촉 검증")
    print("  프로브가 아무것도 닿지 않게 하라 — 팬텀도, 책상도, 손도.")
    print(f"  자세마다 {args.count} 표본(약 {args.count} 초)을 모은다.")
    print("  [스페이스] 수집 시작   [s] 이 자세 건너뛰기   [q] 종료")

    results: list = []
    raw_max = 0.0
    for index, (name, instruction) in enumerate(plan, start=1):
        print()
        print(f"  [{index}/{len(plan)}] {name}")
        print(f"      {instruction}")
        while True:
            key = wait_for_space("      옮긴 뒤 손을 떼고 스페이스바 ▸ ")
            if key == "quit":
                print("\n  중단.")
                return 130
            if key == "skip":
                print("      건너뜀")
                break

            print()
            try:
                magnitudes, swings, pose_raw = asyncio.run(
                    collect_one(args.url, args.count, args.quiet_n)
                )
            except KeyboardInterrupt:
                print("\n  중단.")
                return 130
            except (ConnectionRefusedError, OSError) as exc:
                # 이 도구를 처음 쓰면 거의 항상 여기로 온다. 이름이
                # `px6d_verify` 라 시리얼을 직접 여는 것처럼 보이기 때문이다.
                # "Connect call failed" 만 찍고 끝내면 무엇을 안 띄웠는지가
                # 어디에도 없다.
                print(f"\n  브리지에 닿지 못했다 ({args.url}): {exc}", file=sys.stderr)
                print("  이 도구는 시리얼이 아니라 **브리지의 보상된 렌치**를 읽는다.",
                      file=sys.stderr)
                print("  다른 터미널에서 먼저:  ./scripts/start_session.sh --calib",
                      file=sys.stderr)
                return 2
            except Exception as exc:  # noqa: BLE001 - 사람이 읽을 문장으로
                print(f"\n  수집 실패: {exc}", file=sys.stderr)
                return 2

            raw_max = max(raw_max, pose_raw)
            results.append(report_pose(name, magnitudes, swings, args.quiet_n))

            again = wait_for_space("      [스페이스] 다음   [s] 이 자세 다시 ▸ ")
            print()
            if again == "quit":
                print("  중단.")
                return 130
            if again == "skip":
                results.pop()
                continue
            break

    if not results:
        print("\n  잰 자세가 없다.", file=sys.stderr)
        return 2

    worst = max(item["max_n"] for item in results)
    print()
    print(f"  {'자세':<12}{'표본':>7}{'평균 N':>10}{'최대 N':>10}")
    print("  " + "-" * 39)
    for item in results:
        flag = "  ←" if not item["passed"] else ""
        print(
            f"  {item['pose']:<12}{item['samples']:>7}"
            f"{item['mean_n']:>10.3f}{item['max_n']:>10.3f}{flag}"
        )
    print()
    print(f"  보상 전 최대 {raw_max:.2f} N  →  보상 후 최대 {worst:.3f} N")
    print(f"  합격선 {args.quiet_n:.2f} N")
    passed = worst < args.quiet_n
    print("  ✅ 합격" if passed else "  ⚠️ 불합격")

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump({
                    "count": args.count,
                    "quiet_n": args.quiet_n,
                    "raw_max_n": raw_max,
                    "worst_n": worst,
                    "passed": passed,
                    "poses": results,
                }, handle, ensure_ascii=False, indent=2)
            print(f"  기록: {args.out}")
        except OSError as exc:
            print(f"  ⚠️ 결과를 남기지 못했다: {exc}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
