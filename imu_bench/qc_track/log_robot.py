#!/usr/bin/env python3
"""GT 레코더 — FR5 상태를 한 블록 동안 받아 raw_data/robot/<block>.h5 에 쓴다.

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    python3 log_robot.py --block probe --seconds 90

**시스템 python3 로 돌린다** (rclpy 가 거기에만 있다). 분석은 h5py 가 있는 쪽에서
한다. 두 쪽이 갈리므로 qc_common.save_table 이 h5/npz 양쪽을 쓴다.

무엇을 GT 로 삼나
-----------------
``/<robot>/ee_wrt_base`` (geometry_msgs/Pose). us_servo_node 가 컨트롤러 상태
패키지의 ``tl_cur_pos`` 를 그대로 낸 것이다. probe.yaml 이 ``allow_missing_tool:
true`` 인 동안 tool 은 J6 플랜지이므로, 이 자세가 곧 **IMU 가 붙은 강체의 자세**다.

시각 — Pose 에는 헤더가 없다
----------------------------
그래서 두 가지를 같이 남긴다.

  pose/pc_ts   구독 콜백에서 찍은 벽시계. 스케줄링 지터가 섞여 있다.
  pose/t       같은 발행 틱의 joint_states 헤더 시각. **이쪽을 쓴다.**

us_servo_node._publish_status() 가 pose 와 joint_states 를 한 콜백에서 같은
stamp 로 내므로, pc_ts 가 가장 가까운 joint_states 의 헤더 시각을 그 pose 의
시각으로 삼을 수 있다. 짝을 못 찾으면(허용 20 ms) pc_ts 를 쓰고 pose/paired 에
0 을 남긴다 — 분석기가 그 비율을 보고 시간축을 믿을지 정한다.

레이트
------
``rates.status_publish_hz`` 기본값은 **30 Hz** 다. 레이턴시를 ms 단위로 재려면
너무 성기다 (한 샘플이 33 ms). QC 때는 올려서 띄울 것:

    ros2 launch fr5_launch us_phase0.launch.py backend:=fairino freespace:=true \
        --ros-args -p rates.status_publish_hz:=200.0

이 로거는 실측 레이트를 그대로 기록하고, 분석기가 30 Hz 대면 경고한다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qc_common as qc                                          # noqa: E402

import rclpy                                                    # noqa: E402
from geometry_msgs.msg import Pose, Twist, WrenchStamped        # noqa: E402
from rclpy.node import Node                                     # noqa: E402
from rclpy.qos import qos_profile_sensor_data                   # noqa: E402
from sensor_msgs.msg import JointState                          # noqa: E402
from std_msgs.msg import Bool, Float32MultiArray                # noqa: E402


def _stamp_s(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


class RobotLogger(Node):
    def __init__(self, robot: str, block: str):
        super().__init__(f"qc_log_robot_{block}")
        self.block = block
        self.pose_topic = qc.topics(robot)["pose"]
        t = qc.topics(robot)

        self.pose_pc, self.pose_p, self.pose_q = [], [], []
        self.jt_t, self.jt_pc, self.jt_pos, self.jt_vel, self.jt_eff = [], [], [], [], []
        self.wr_t, self.wr = [], []
        self.tw_pc, self.tw = [], []
        self.err_pc, self.err = [], []
        self.retreat_pc, self.retreat = [], []

        # 센서 QoS 로 받는 것과 기본 QoS 로 받는 것을 섞지 않는다 — 발행자가 기본
        # (RELIABLE, depth 10) 이므로 같은 프로파일로 받아야 붙는다.
        self.create_subscription(Pose, t["pose"], self._on_pose, 50)
        self.create_subscription(JointState, t["joints"], self._on_joints, 50)
        self.create_subscription(WrenchStamped, t["wrench"], self._on_wrench, 50)
        self.create_subscription(Twist, t["twist"], self._on_twist, 50)
        self.create_subscription(Float32MultiArray, t["track_err"], self._on_err, 10)
        self.create_subscription(Bool, t["retreat"], self._on_retreat, 10)
        _ = qos_profile_sensor_data          # 쓰지 않지만 의도를 남긴다

        self.t0 = time.time()
        self.get_logger().info(f"[{block}] 기록 시작 — {t['pose']}")

    # -- 콜백 ------------------------------------------------------------
    def _on_pose(self, msg: Pose):
        self.pose_pc.append(time.time())
        self.pose_p.append((msg.position.x, msg.position.y, msg.position.z))
        self.pose_q.append((msg.orientation.x, msg.orientation.y,
                            msg.orientation.z, msg.orientation.w))

    def _on_joints(self, msg: JointState):
        self.jt_t.append(_stamp_s(msg.header))
        self.jt_pc.append(time.time())
        self.jt_pos.append(list(msg.position[:6]) + [np.nan] * max(0, 6 - len(msg.position)))
        self.jt_vel.append(list(msg.velocity[:6]) + [np.nan] * max(0, 6 - len(msg.velocity)))
        self.jt_eff.append(list(msg.effort[:6]) + [np.nan] * max(0, 6 - len(msg.effort)))

    def _on_wrench(self, msg: WrenchStamped):
        w = msg.wrench
        self.wr_t.append(_stamp_s(msg.header))
        self.wr.append((w.force.x, w.force.y, w.force.z,
                        w.torque.x, w.torque.y, w.torque.z))

    def _on_twist(self, msg: Twist):
        self.tw_pc.append(time.time())
        self.tw.append((msg.linear.x, msg.linear.y, msg.linear.z,
                        msg.angular.x, msg.angular.y, msg.angular.z))

    def _on_err(self, msg: Float32MultiArray):
        self.err_pc.append(time.time())
        d = list(msg.data[:2]) + [np.nan] * max(0, 2 - len(msg.data))
        self.err.append(d)

    def _on_retreat(self, msg: Bool):
        self.retreat_pc.append(time.time())
        self.retreat.append(1.0 if msg.data else 0.0)

    # -- 저장 ------------------------------------------------------------
    def pair_pose_times(self):
        """pose 의 pc_ts 를 같은 발행 틱의 joint_states 헤더 시각으로 바꾼다.

        구독자 쪽 스케줄링 지터를 GT 시간축에서 빼는 것이 목적이다. 짝은
        joint_states 의 **pc_ts** 로 찾고(같은 콜백에서 나왔으므로 도착도 같이
        한다), 넘겨 주는 값은 그 메시지의 **헤더 시각**이다.
        """
        pose_pc = np.asarray(self.pose_pc, float)
        jt_pc = np.asarray(self.jt_pc, float)
        jt_t = np.asarray(self.jt_t, float)
        if pose_pc.size == 0 or jt_pc.size == 0:
            return pose_pc.copy(), np.zeros(pose_pc.size, float)
        idx = np.searchsorted(jt_pc, pose_pc)
        idx = np.clip(idx, 1, jt_pc.size - 1)
        left, right = idx - 1, idx
        take = np.where(np.abs(jt_pc[left] - pose_pc) <= np.abs(jt_pc[right] - pose_pc),
                        left, right)
        d = np.abs(jt_pc[take] - pose_pc)
        ok = d <= qc.PAIR_TOL_S
        return np.where(ok, jt_t[take], pose_pc), ok.astype(float)

    def save(self):
        if not self.pose_pc:
            # 저장을 안 했다는 사실은 **exit code 로** 나가야 한다. 로그로만 알리면
            # 러너가 0 을 보고 다음 블록으로 넘어가고, 그 자리에 이전 세션의 파일이
            # 남아 있으면 분석기가 조용히 그것을 쓴다. 2026-08-20 세션에서 still
            # 블록이 정확히 그렇게 날아갔다 (GT 는 한 시간 전 것, 겹침 -3783 s).
            self.get_logger().error(
                "pose 를 한 건도 못 받았다 — 저장하지 않는다."
                f"  {self.pose_topic} 가 발행 중인지 확인할 것"
                " (ros2 topic hz). 로봇 런치가 안 떠 있으면 이렇게 된다")
            return None
        t_paired, paired = self.pair_pose_times()
        cols = {
            "pose/t": t_paired,
            "pose/pc_ts": np.asarray(self.pose_pc, float),
            "pose/paired": paired,
            "pose/position": np.asarray(self.pose_p, float),
            "pose/quat_xyzw": np.asarray(self.pose_q, float),
            "joints/t": np.asarray(self.jt_t, float),
            "joints/pc_ts": np.asarray(self.jt_pc, float),
            "joints/position": np.asarray(self.jt_pos, float),
            "joints/velocity": np.asarray(self.jt_vel, float),
            "joints/effort": np.asarray(self.jt_eff, float),
        }
        for name, ts, vals, width in (("wrench", self.wr_t, self.wr, 6),
                                      ("twist", self.tw_pc, self.tw, 6),
                                      ("track_err", self.err_pc, self.err, 2),
                                      ("retreat", self.retreat_pc, self.retreat, 1)):
            cols[f"{name}/t"] = np.asarray(ts, float)
            cols[f"{name}/data"] = (np.asarray(vals, float).reshape(-1, width)
                                    if vals else np.empty((0, width)))
        audit = qc.rate_audit(t_paired, "pose")
        attrs = {"block": self.block, "started_unix": self.t0,
                 "stopped_unix": time.time(),
                 "pose_rate_hz": audit.get("rate_hz", float("nan")),
                 "paired_frac": float(paired.mean()),
                 "retreat_events": int(np.sum(np.asarray(self.retreat, float) > 0.5)),
                 "note": qc.ROBOT_STATE_LAG_NOTE}
        path = qc.save_table(os.path.join(qc.DIR_ROBOT, self.block), cols, attrs)
        self.get_logger().info(
            f"[{self.block}] pose {len(self.pose_pc)} @ {attrs['pose_rate_hz']:.1f} Hz "
            f"(짝 {100 * attrs['paired_frac']:.0f} %) -> {os.path.basename(path)}")
        if attrs["pose_rate_hz"] < 60:
            self.get_logger().warn(
                f"pose 레이트가 {attrs['pose_rate_hz']:.0f} Hz 다. 지연을 ms 로 재려면"
                " rates.status_publish_hz 를 올려서 띄울 것 (권장 200)")
        qc.append_trial({"block": self.block, "source": "robot", **attrs})
        return path


def main() -> int:
    ap = argparse.ArgumentParser()
    # choices 를 protocol.BLOCKS 로 묶지 않는다 — 임시 블록(재촬영, 실험)을
    # 막아 봐야 얻는 것이 없고, 파일명은 어차피 이 문자열이다.
    ap.add_argument("--block", required=True)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="0 이면 Ctrl-C / SIGTERM 까지")
    ap.add_argument("--robot", default=qc.ROBOT)
    ap.add_argument("--raw", default=None, help="raw_data 위치 (기본: 패키지 안)")
    args = ap.parse_args()

    if args.raw:
        qc.use_raw_dir(args.raw)
    qc.ensure_dirs()

    rclpy.init()
    qc.install_shutdown_handlers()          # rclpy.init() 뒤에 걸어야 먹는다
    node = RobotLogger(args.robot, args.block)
    deadline = time.time() + args.seconds if args.seconds > 0 else None
    saved = None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if deadline and time.time() >= deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        saved = node.save()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if saved else 2


if __name__ == "__main__":
    raise SystemExit(main())
