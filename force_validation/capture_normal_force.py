#!/usr/bin/env python3
"""전자저울 검증용 capture mode — 스페이스바로 잰다.

    python3 capture_normal_force.py --out-dir captures

실험 흐름:

1. probe 를 전자저울 위 접촉면에 댄다.
2. **전자저울 표시값이 안정된 것을 눈으로 확인한다.**
3. 스페이스바를 누른다 — 그 순간의 corrected normal force 가 저장된다.
4. 전자저울 질량은 따로 적어 둔다. 끝난 뒤 template CSV 에 넣는다.

여기서 재는 값은 **그 순간의 값** 이다. 평활도 이동평균도 그것을 대신하지
않는다. 다만 그 순간이 흔들린 순간이었는지 알 수 있도록 앞뒤 250 ms 의 median,
mean, SD 를 함께 남긴다 — 판정이 아니라 근거로.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from fv.capture import DEBOUNCE_S, MAX_DATA_AGE_MS, ForceValidationCapture, KeyReader


def status_line(capture, sign_note: str) -> str:
    """한 줄 상태. 사양이 화면에 요구하는 네 가지를 담는다."""
    age = capture.data_age_ms
    force = capture.latest
    calib = "OK" if capture.calibration_valid else "BAD"
    frame = "OK" if capture.probe_frame_ready() else "BAD"
    fresh = "OK" if age <= MAX_DATA_AGE_MS else f"{age:.0f}ms"
    value = "  ——  " if force is None else f"{force:+7.3f}"
    return (
        f"  calib {calib:>3} · gravity {calib:>3} · probe-frame {frame:>3} · "
        f"stream {fresh:>6} │ F_n {value} N │ n={len(capture.rows)} │ {sign_note}"
    )


def main(argv=None) -> int:
    """Capture 루프를 돈다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--namespace", default="/fr5_right", help="로봇 네임스페이스")
    parser.add_argument("--out-dir", default="captures", help="CSV 를 쓸 폴더")
    parser.add_argument(
        "--sign", type=float, default=-1.0,
        help="probe frame Fz 에 곱할 부호. 누르는 방향이 양수가 되게 한다",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="이미 있는 robot_force_captures.csv 에 이어서 잰다. sample_id 가 "
             "그 뒤부터 이어진다 — 안 주면 파일을 새로 쓰므로 앞서 잰 것이 사라진다",
    )
    parser.add_argument(
        "--allow-invalid", action="store_true",
        help="교정이 유효하지 않아도 시작한다. 기록에는 무효로 남는다",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.sign not in (1.0, -1.0):
        print("--sign 은 +1 또는 -1 이어야 한다", file=sys.stderr)
        return 2
    sign_note = (
        "as-published (probe +z positive)" if args.sign > 0
        else "negated at capture (compression positive)"
    )

    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("force_validation_capture")
    capture = ForceValidationCapture(node, args.namespace, args.sign, args.out_dir)

    existing = capture.load_existing() if args.append else 0

    print()
    print("  전자저울 normal-force 검증 — capture mode")
    print(f"  파이프라인 최종 출력만 읽는다: {args.namespace}/wrench_px6d")
    print(f"  부호 규약: {sign_note}")
    print("  [스페이스] capture   [u] 직전 취소   [q] 저장하고 종료")
    if existing:
        print(f"  기존 {existing} 점을 읽었다 — sample {capture.next_id} 부터 이어 잰다.")
    elif os.path.exists(capture.capture_path):
        print(f"  ⚠️ {capture.capture_path} 가 이미 있다. 저장할 때 **덮어쓴다** — "
              "이어 재려면 --append 를 주고 다시 시작하라.")
    print()

    # 시작 조건. 사양이 요구하는 대로, 하나라도 유효하지 않으면 시작하지 않는다.
    deadline = time.time() + 5.0
    while time.time() < deadline and capture.blocking_reason():
        rclpy.spin_once(node, timeout_sec=0.05)
    blocked = capture.blocking_reason()
    if blocked and not args.allow_invalid:
        print(f"  시작할 수 없다 — {blocked}", file=sys.stderr)
        print("  교정을 끝내고 브리지가 보상된 값을 내는지 확인하라.", file=sys.stderr)
        print("  그래도 기록만 남기려면 --allow-invalid.", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 2
    if blocked:
        print(f"  ⚠️ {blocked} — --allow-invalid 로 계속한다. 기록은 무효로 남는다.")

    last_draw = 0.0
    try:
        with KeyReader() as keys:
            while True:
                rclpy.spin_once(node, timeout_sec=0.01)
                now = time.time()
                if now - last_draw > 0.1:
                    last_draw = now
                    sys.stdout.write("\r\x1b[2K" + status_line(capture, sign_note))
                    sys.stdout.flush()

                key = keys.poll()
                if key is None:
                    continue
                if key in ("q", "Q", "\x03"):
                    break
                if key in ("u", "U"):
                    removed = capture.drop_last()
                    sys.stdout.write("\r\x1b[2K")
                    print(f"  ↩ sample {removed['sample_id']} 취소" if removed
                          else "  취소할 capture 가 없다")
                    continue
                if key != " ":
                    continue

                row = capture.capture(sign_note)
                sys.stdout.write("\r\x1b[2K")
                if row is None:
                    print(f"  … debounce ({DEBOUNCE_S * 1000:.0f} ms) — 무시")
                    continue
                mark = "✓" if row["capture_valid"] == "TRUE" else "✗"
                print(
                    f"  {mark} sample {row['sample_id']:>3}  "
                    f"F_n {float(row['robot_normal_force_instant_N']):+7.3f} N  "
                    f"(250 ms SD {row['robot_normal_force_std_250ms_N'] or '—'})"
                    + (f"  ⚠️ {row['notes']}" if row["notes"] else "")
                )
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\r\x1b[2K")
        node.destroy_node()
        rclpy.shutdown()

    if not capture.rows:
        print("  잰 것이 없다 — 파일을 쓰지 않는다.")
        return 1

    capture_path, template_path = capture.write()
    valid = sum(1 for row in capture.rows if row["capture_valid"] == "TRUE")
    print()
    print(f"  capture {len(capture.rows)} 개 (유효 {valid})")
    print(f"    {capture_path}")
    print(f"    {template_path}   ← 전자저울 질량[g]을 여기에 적는다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
