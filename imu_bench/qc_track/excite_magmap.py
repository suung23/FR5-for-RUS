#!/usr/bin/env python3
"""스크립트 mag_map — **자세를 붙든 채** 팔만 옮긴다.

    python3 excite_magmap.py --dry-run       # 계획만 (ROS 없이도 돈다)
    python3 run_session.py --blocks mag_map --excite

왜 스크립트인가
---------------
이 블록의 존재 이유는 '자세는 같고 위치만 다른 정지 쌍' 하나뿐이다. 그 쌍이 있으면
센서 고정 자기왜곡(`m = K R h + b`)의 예측이 완전히 같아야 하므로, 측정이 다르면
**남는 설명은 위치에 따른 장 변화뿐이다** — 적합도, 해의 유일성 걱정이 필요 없는
유일한 판정이다.

손으로 두 번 했고 두 번 다 쌍이 0 개였다. 2026-08-21 두 번째 시도는 정지 18 개,
153 쌍을 전부 뒤져도 자세차 3 deg 이하인 쌍이 2 개뿐이었고 그 둘은 1 mm / 39 mm
떨어져 있었다. **실제로 움직인 쌍은 예외 없이 자세가 8 deg 넘게 돌았다.** Touch
스타일러스로 직교 병진만 손으로 내는 것은 노력의 문제가 아니다.

각속도 0 인 twist 를 보내면 미분 IK 가 자세를 붙든 채 옮긴다. 자세차가 **설계상**
0 이 된다. 그래서 이 블록만 로봇이 스스로 돈다.

안전
----
  · `freespace` 로 띄운 상태에서, 프로브가 아무것도 안 닿는 자세에서만 돌린다.
  · **시작 자세 둘레로 각 축 ±STEP 만큼 빈 공간이 있어야 한다.** 별 모양으로
    ±160 mm 씩 나간다 — 시작 전에 그 공간을 눈으로 확인하는 것은 사람 몫이다.
    `--dry-run` 이 얼마나 나갈지 먼저 찍는다.
  · 속도는 protocol 예산으로 묶는다 (60 mm/s, 12 deg/s). freespace 상한의 40 % / 14 %.
  · `desired_twist` 의 **유일한 발행자**여야 한다. teleop 이 같이 돌면 두 발행자가
    번갈아 나가 팔이 덜컹거린다 — 스타일러스에서 손을 떼고 시작할 것.
  · 후퇴 신호, 추종오차 폭주, 상태 두절, 다리 시간초과 — 넷 중 하나라도 걸리면
    twist 0 을 보내고 **그 자리에서 멈춘다.** 자동 복구는 안 한다.
  · Ctrl-C 하면 0 twist 를 몇 번 보내고 끝낸다 (워치독이 잡기 전에 세운다).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol                                                 # noqa: E402
import qc_common as qc                                          # noqa: E402

# 목표까지 남은 거리에 곱하는 이득 [1/s]. 큰 값이 필요 없다 — 어차피 속도를
# 예산으로 자르므로, 이 이득이 하는 일은 **마지막 몇 cm 에서 부드럽게 서는 것**뿐이다.
KP = 1.5
# 상태가 이만큼 안 오면 두절로 본다. 발행이 150~190 Hz 라 넉넉한 값이다.
POSE_TIMEOUT_S = 0.5


# 이 둘은 ROS 를 만지지 않는다 (`_spin` 만 부른다). 모듈 수준에 두어 rclpy 없이도
# 시험할 수 있게 한다 — 아래 두 사고가 로봇을 세워 놓고서야 드러났기 때문이다.
def joint_limits_deg():
    """관절 한계 [(lower, upper), ...]. **probe.yaml 한 곳에서 읽는다.**

    여기에 숫자를 베껴 두면 언젠가 두 벌이 갈라지고, 갈라진 쪽이 안전 판정을
    한다. 설치된 fr5_control 의 config 를 그대로 읽고, 못 찾으면 None 을
    돌려주어 **감시를 껐다고 소리내어 알린다** (조용히 통과시키지 않는다).
    """
    try:
        import yaml
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory("fr5_control"),
                            "config", "probe.yaml")
        with open(path) as fh:
            doc = yaml.safe_load(fh)
        node = doc["/**"]["ros__parameters"] if "/**" in doc else next(iter(doc.values()))
        lim = node["safety"]["joint_limits_deg"]
        return list(zip(lim["lower"], lim["upper"])), path
    except Exception:                                   # noqa: BLE001
        return None, None


def wait_for_pose(node, seconds=8.0):
    """상태가 들어올 때까지 기다린다.

    **`node.t_pose` 로 대기 시간을 재면 안 된다.** 초기값이 0.0(=1970 년)이라
    `time.time() - node.t_pose` 가 17 억이 되어 루프가 한 번도 안 돌고 곧장
    포기한다 — 2026-08-21 첫 실행이 정확히 이걸로 죽었다. 같은 시각 로봇 로거는
    같은 토픽을 멀쩡히 받고 있었는데 이쪽만 '못 받았다' 로 찍혔다.
    """
    deadline = time.time() + seconds
    while node.p is None and time.time() < deadline:
        node._spin()
        time.sleep(0.02)
    return node.p is not None


def wait_for_imu(node, block, seconds=30.0):
    """IMU 로거가 **영점을 끝내고 기록을 시작할 때까지** 기다린다.

    log_imu 는 블록 시작에 3 s 영점 캘리브레이션을 도는데 그동안 센서가 정지해
    있어야 한다. 그 사이에 팔을 움직이면 영점이 깨지고, 영점이 깨지면 변위의
    B/C 단계가 통째로 날아간다.

    기다리는 신호로 **CSV 파일의 등장**을 쓴다 — log_imu 가 SessionLogger 를 영점
    뒤에 여는 덕에, 그 파일이 생겼다는 것이 곧 '영점 끝, 기록 시작' 이다.
    monitor.py 도 같은 신호를 쓴다. 시간으로 어림잡으면 스트림 기동이 들쭉날쭉해서
    (최대 5 s) 맞출 수가 없다.
    """
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            names = [n for n in os.listdir(qc.DIR_IMU)
                     if n.startswith(f"{block}_raw_") and n.endswith(".csv")]
        except OSError:
            names = []
        for n in names:
            path = os.path.join(qc.DIR_IMU, n)
            if os.path.getmtime(path) > node.t_launch and os.path.getsize(path) > 0:
                return True
        node._spin()
        time.sleep(0.1)
    return False


def make_node(robot, rate_hz):
    """rclpy 를 여기서 import 한다 — --dry-run 은 ROS 없는 기계에서도 돌아야 한다."""
    from geometry_msgs.msg import Pose, Twist
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, Float32MultiArray

    class MagMapDriver(Node):
        def __init__(self):
            super().__init__("qc_excite_magmap")
            t = qc.topics(robot)
            self.pub = self.create_publisher(Twist, t["twist"], 10)
            self.create_subscription(Pose, t["pose"], self._on_pose, 50)
            self.create_subscription(JointState, t["joints"], self._on_joints, 20)
            self.create_subscription(Bool, t["retreat"], self._on_retreat, 10)
            self.create_subscription(Float32MultiArray, t["track_err"], self._on_err, 10)
            self.rate = rate_hz
            self.p = None                       # 현재 위치 [m], base
            self.R = None                       # 현재 자세, base<-tool
            self.t_pose = 0.0
            self.retreating = False
            self.track_err = 0.0
            self.holds = []                     # 정지 구간 (분석기가 읽는다)
            self.abort = None
            self.t_launch = time.time()         # 이 시각 뒤에 생긴 CSV 만 이 블록 것이다
            self.q_deg = None                   # 관절 [deg]
            self.limits, lim_path = joint_limits_deg()
            self.margin_hits = []
            if self.limits is None:
                self.get_logger().warn(
                    "관절 한계를 못 읽었다 (fr5_control/config/probe.yaml)."
                    " **관절 여유 감시 없이 돈다** — 눈으로 지켜볼 것.")
            else:
                self.get_logger().info(
                    f"관절 여유 감시 켬 (여유 {protocol.MAG_MAP_JOINT_MARGIN_DEG:.0f} deg,"
                    f" 한계 출처 {lim_path})")

        # -- 구독 ------------------------------------------------------
        def _on_pose(self, msg):
            self.p = np.array([msg.position.x, msg.position.y, msg.position.z])
            self.R = qc.quat_xyzw_to_R(np.array([[msg.orientation.x, msg.orientation.y,
                                                  msg.orientation.z, msg.orientation.w]]))[0]
            self.t_pose = time.time()

        def _on_joints(self, msg):
            if msg.position:
                self.q_deg = np.degrees(np.asarray(msg.position, float))

        def joint_margin(self):
            """한계까지 남은 여유 중 **가장 작은 것** [deg]과 그 관절 번호.

            감시를 못 하면 (한계를 못 읽었거나 상태가 아직 없으면) None 을 준다 —
            부르는 쪽이 '여유가 넉넉하다' 로 오해하면 안 되므로 큰 수가 아니라
            None 이다.
            """
            if self.limits is None or self.q_deg is None:
                return None
            n = min(len(self.limits), len(self.q_deg))
            best, idx = None, -1
            for i in range(n):
                lo, hi = self.limits[i]
                m = min(self.q_deg[i] - lo, hi - self.q_deg[i])
                if best is None or m < best:
                    best, idx = float(m), i
            return best, idx

        def _on_retreat(self, msg):
            if msg.data:
                self.retreating = True

        def _on_err(self, msg):
            if msg.data:
                self.track_err = float(max(abs(v) for v in msg.data))

        # -- 발행 ------------------------------------------------------
        def send(self, lin=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0)):
            m = Twist()
            m.linear.x, m.linear.y, m.linear.z = (float(v) for v in lin)
            m.angular.x, m.angular.y, m.angular.z = (float(v) for v in ang)
            self.pub.publish(m)

        def stop(self, n=5):
            for _ in range(n):
                self.send()
                time.sleep(1.0 / self.rate)

        # -- 감시 ------------------------------------------------------
        def guard(self):
            """멈춰야 할 이유가 있으면 그 문장을 돌려준다. 자동 복구는 안 한다."""
            if self.retreating:
                return "후퇴 신호가 떴다 (접촉 또는 워치독)"
            if self.p is None:
                return "로봇 자세를 한 번도 못 받았다 — 런치가 떠 있는지 볼 것"
            if time.time() - self.t_pose > POSE_TIMEOUT_S:
                return f"로봇 상태가 {time.time() - self.t_pose:.1f} s 째 안 온다"
            if self.track_err > 0.5:
                return f"추종오차가 {self.track_err:.2f} 로 크다"
            return None

        def _spin(self):
            import rclpy
            rclpy.spin_once(self, timeout_sec=0.0)

        # -- 다리 ------------------------------------------------------
        def go_to(self, target_m, label):
            """각속도 0 으로 목표 위치까지. 자세는 미분 IK 가 붙든다."""
            v_max = protocol.MAG_MAP_V_MMS / 1000.0
            t0 = time.time()
            while True:
                self._spin()
                why = self.guard()
                if why:
                    self.abort = why
                    return False
                err = target_m - self.p
                dist = float(np.linalg.norm(err))
                if dist <= protocol.MAG_MAP_TOL_MM / 1000.0:
                    self.stop(2)
                    return True
                jm = self.joint_margin()
                if jm is not None and jm[0] < protocol.MAG_MAP_JOINT_MARGIN_DEG:
                    # **블록을 버리지 않는다.** 쌍은 목표가 아니라 실제로 선 자리로
                    # 계산되므로, 덜 간 자리도 그 자리대로 쓸 수 있다.
                    self.get_logger().warn(
                        f"{label}: J{jm[1] + 1} 이 한계에서 {jm[0]:.1f} deg 까지 왔다"
                        f" — 여기서 멈춘다 (남은 거리 {dist * 1000:.0f} mm)")
                    self.margin_hits.append({"leg": label, "joint": jm[1] + 1,
                                             "margin_deg": jm[0],
                                             "short_mm": dist * 1000})
                    self.stop(2)
                    return True
                if time.time() - t0 > protocol.MAG_MAP_LEG_TIMEOUT_S:
                    self.abort = (f"{label}: {protocol.MAG_MAP_LEG_TIMEOUT_S:.0f} s 안에"
                                  f" 목표에 못 갔다 (남은 거리 {dist * 1000:.0f} mm)."
                                  " 관절 한계나 특이점일 수 있다")
                    self.stop()
                    return False
                v = err * KP
                speed = float(np.linalg.norm(v))
                if speed > v_max:
                    v = v * (v_max / speed)
                self.send(lin=v)                # 각속도는 0 — 이것이 이 블록의 전부다
                time.sleep(1.0 / self.rate)

        def turn(self, axis, deg, label):
            """base 축 둘레로 deg 만큼. 병진 0."""
            w = math.radians(protocol.MAG_MAP_W_DEG_S)
            axis = np.array(axis, float)
            axis = axis / np.linalg.norm(axis)
            R0 = self.R.copy()
            t0 = time.time()
            while True:
                self._spin()
                why = self.guard()
                if why:
                    self.abort = why
                    return False
                done = qc.geodesic_deg(R0, self.R)
                if done >= deg:
                    self.stop(2)
                    return True
                if time.time() - t0 > protocol.MAG_MAP_LEG_TIMEOUT_S:
                    self.abort = f"{label}: 회전이 {done:.1f}/{deg:.0f} deg 에서 멈췄다"
                    self.stop()
                    return False
                self.send(ang=axis * w)         # 병진은 0
                time.sleep(1.0 / self.rate)

        def hold(self, seconds):
            """세워 둔다. 이 창이 곧 분석기가 쓰는 정지 구간이다."""
            t0 = time.time()
            while time.time() - t0 < seconds:
                self._spin()
                self.send()                     # 0 twist 를 계속 보낸다 (워치독)
                time.sleep(1.0 / self.rate)
            self.holds.append({"t_start": t0, "t_stop": time.time(),
                               "p_mm": (self.p * 1000.0).tolist()})
            return True

        # -- 전체 ------------------------------------------------------
        def run(self, block="mag_map"):
            if not wait_for_pose(self):
                self.abort = "로봇 자세를 못 받았다 — 런치가 떠 있는지 볼 것"
                return False
            if not wait_for_imu(self, block):
                self.abort = ("IMU 로거가 기록을 시작하지 않았다 — 영점 중에 팔을"
                              " 움직이면 영점이 깨지므로 여기서 멈춘다")
                return False
            p0 = self.p.copy()
            self.get_logger().info(
                "시작 위치 [%.0f %.0f %.0f] mm — 이 자리 둘레로 +-%.0f mm 를 쓴다"
                % (*(p0 * 1000), protocol.MAG_MAP_STEP_MM))
            legs = protocol.plan_mag_map()
            n_hold = 0
            for i, (kind, payload) in enumerate(legs):
                if kind == "move":
                    tgt = p0 + np.array(payload, float) / 1000.0
                    lab = f"[{i + 1}/{len(legs)}] 이동 {tuple(payload)} mm"
                    self.get_logger().info(lab)
                    if not self.go_to(tgt, lab):
                        return False
                elif kind == "turn":
                    axis, deg = payload
                    lab = f"[{i + 1}/{len(legs)}] 회전 {deg:.0f} deg (축 {axis})"
                    self.get_logger().info(lab)
                    if not self.turn(axis, deg, lab):
                        return False
                else:
                    n_hold += 1
                    self.get_logger().info(
                        f"[{i + 1}/{len(legs)}] 정지 {payload:.1f} s  ({n_hold}/15)")
                    self.hold(payload)
            return True

    return MagMapDriver()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=qc.ROBOT)
    ap.add_argument("--rate", type=float, default=100.0, help="발행 주기 [Hz]")
    ap.add_argument("--step-mm", type=float, default=protocol.MAG_MAP_STEP_MM,
                    help="별 모양의 보폭 [mm]. 작업공간이 좁으면 줄인다")
    ap.add_argument("--block", default="mag_map",
                    help="이 블록의 IMU 로거가 기록을 시작할 때까지 기다린다")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.step_mm < protocol.MAG_MAP_MIN_DP_MM:
        ap.error(f"보폭이 {args.step_mm:.0f} mm 면 쌍이 하나도 안 선다 —"
                 f" 쌍의 위치차 문턱이 {protocol.MAG_MAP_MIN_DP_MM:.0f} mm 다")
    protocol.MAG_MAP_STEP_MM = args.step_mm

    if args.dry_run:
        legs = protocol.plan_mag_map()
        print(f"스크립트 mag_map — 다리 {len(legs)} 개, 약 {protocol.mag_map_seconds():.0f} s")
        print(f"  이동속도 {protocol.MAG_MAP_V_MMS:.0f} mm/s   회전속도"
              f" {protocol.MAG_MAP_W_DEG_S:.0f} deg/s   보폭"
              f" {protocol.MAG_MAP_STEP_MM:.0f} mm")
        print(f"  시작 자세 둘레로 각 축 +-{protocol.MAG_MAP_STEP_MM:.0f} mm 의"
              " 빈 공간이 있어야 한다. 먼저 눈으로 확인할 것.")
        cur = (0.0, 0.0, 0.0)
        for kind, payload in legs:
            if kind == "move":
                print(f"    이동  {tuple(payload)} mm"
                      f"   ({math.dist(cur, payload):.0f} mm)")
                cur = payload
            elif kind == "turn":
                print(f"    회전  {payload[1]:.0f} deg  축 {payload[0]}")
            else:
                print(f"    정지  {payload:.1f} s")
        return 0

    if args.raw:
        qc.use_raw_dir(args.raw)
    import rclpy
    rclpy.init()
    qc.install_shutdown_handlers()
    node = make_node(args.robot, args.rate)
    ok = False
    try:
        ok = node.run(args.block)
    except KeyboardInterrupt:
        node.get_logger().warn("중단 — twist 0 을 보내고 끝낸다")
    finally:
        node.stop()
        if node.abort:
            node.get_logger().error(f"[중단] {node.abort}")
            node.get_logger().error(
                "자동 복구는 안 한다. 팔이 어디에 섰는지 눈으로 보고,"
                " 안전하면 teleop 으로 시작 자리 근처로 되돌린 뒤 다시 딸 것.")
        prog = qc.load_json(qc.PROGRAM_JSON, {}) or {}
        blocks = [b for b in prog.get("blocks", []) if b.get("block") != "mag_map_script"]
        blocks.append({"block": "mag_map_script", "holds": node.holds,
                       "completed": bool(ok), "abort": node.abort,
                       "step_mm": protocol.MAG_MAP_STEP_MM,
                       "margin_hits": node.margin_hits})
        prog["blocks"] = blocks
        qc.dump_json(qc.PROGRAM_JSON, prog)
        node.get_logger().info(f"정지 자리 {len(node.holds)} 개를 남겼다")
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
