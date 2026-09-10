#!/usr/bin/env python3
"""면외(elevational) 스텝 스윕 — 관측 모델 y(d) 를 재는 **교정용** 수집.

    source ~/FR5-for-RUS/install/setup.bash
    python3 capture_sweep.py --label dry --dry-run          # 지령 없이 계획만 출력
    python3 capture_sweep.py --label c1_2N --range-mm 30 --step-mm 0.2

이 데이터는 **학습에 쓰지 않는다.** 이 연구의 수집 프로토콜은 프리핸드이고 로봇을 쓰지
않는 것이 전제다. 여기서 로봇을 쓰는 이유는 하나뿐이다 — 프리핸드 IMU 로는 면외 변위 d 를
믿을 만하게 잴 수 없기 때문이다 (2026-09-10 실측: sweep_y 65 구간에서 ZUPT 가 제거한
누적 드리프트가 순변위의 **2.75 배**, 순변위 중앙 14 mm 에 드리프트 중앙 60 mm).
로봇 FK 는 d 를 직접 주고, admittance 가 접촉을 붙잡아 준다. 즉 **센서 드리프트 보정과
모델 의미 확인**이 목적이고, 정책은 이 데이터를 보지 않는다.

무엇을 확정하는가 (docs/IMAGE_SERVOING_MATH.md 의 ⏳ 항목)
------------------------------------------------------
1. **관측 모델의 함수 형태.** offset_filter (§3.9) 는 y = A₀·√(1−(d/R)²) (타원체 단면) 을
   가정한다. 프리핸드 sweep_y 19 개로는 이 가정이 가우시안보다 나빴지만(3/18), d 가
   부정확해 **판정이 불가능**했다. FK 를 쓰면 결정된다.
2. **σ_q.** 필터의 사각지대 폭 ∝ σ_q. config 기본값 0.03, 프리핸드 실측 0.09–0.13.
3. **이방성 b/a.** lateral 대비 elevational 단면 비. b/a < 1.2 이면 rz 는 미세 서보가
   아니라 초기 자세 탐색 축으로 재분류해야 한다.
4. **접촉 결합 g(ρ).** 프리핸드에서 면적과 A-line 접촉비율의 상관이 r = 0.77 이었다.
   접촉을 붙잡은 채로 재야 h(d) 와 g(ρ) 가 분리된다.

기록하는 것
-----------
스텝마다: 지령 d [mm], **FK 실측 자세**, 렌치(제어가 쓰는 그 값 그대로), 정착 후 US 프레임 N 장.
프레임은 `us_frames.bin` + `us_index.csv` 로 수집기와 **같은 레이아웃**이라 BmodeConverter·U-Net
경로가 그대로 돈다. 여기서 재보상하거나 거르지 않는다.

안전
----
힘 한계·후퇴는 `us_diff_ik_node` 가 관리한다 (이 스크립트는 문턱을 만들지 않는다).
`/diag/retreating` 이 뜨거나 접촉이 끊기면 **즉시 지령을 멈추고 저장**한다.
빔축(z) 은 절대 지령하지 않는다 — 접촉은 admittance 가 잡는다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from geometry_msgs.msg import Pose, Twist, WrenchStamped
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, String
except ImportError:  # --dry-run 은 ROS 없이도 계획을 볼 수 있어야 한다
    rclpy = None
    Node = object       # 클래스 정의만 통과시키고, 실행은 main() 에서 막는다
    qos_profile_sensor_data = None


def _latched():
    """전환할 때만 발행되는 토픽용 QoS (TRANSIENT_LOCAL).

    `probing_mode` 는 모드가 **바뀔 때만** 나간다. 기본 QoS(VOLATILE) 로 구독하면
    늦게 붙은 이 스크립트는 래치된 현재 모드를 못 받아 `self.mode` 가 빈 문자열로
    남고, 접촉 확인이 실제로는 접촉 중인데도 실패한다 (`fh/capture.py` 와 같은 규약).
    """
    return QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


#: 프로브 body 프레임 (docs/FRAMES_AND_SE2.md §1): x lateral, y elevational, z beam.
#: ⏳ desired_twist 가 이 규약대로 프로브 body 에 실리는지는 --dry-run 으로 먼저 눈으로 확인할 것.
AXIS_ELEVATIONAL = 1
AXIS_LATERAL = 0


@dataclass
class SweepPlan:
    range_mm: float = 30.0
    step_mm: float = 0.2
    speed_mm_s: float = 2.0
    settle_s: float = 0.4
    frames_per_step: int = 3
    repeats: int = 2
    lateral_offsets_mm: tuple[float, ...] = (0.0,)

    @property
    def steps_per_pass(self) -> int:
        return int(round(2 * self.range_mm / self.step_mm)) + 1

    def offsets(self) -> np.ndarray:
        """−range → +range 를 훑는 지령 오프셋 [mm]."""
        return np.linspace(-self.range_mm, self.range_mm, self.steps_per_pass)

    def duration_estimate_s(self) -> float:
        per_step = self.step_mm / max(self.speed_mm_s, 1e-6) + self.settle_s
        passes = self.repeats * len(self.lateral_offsets_mm)
        return per_step * self.steps_per_pass * passes

    def describe(self) -> str:
        return (
            "스윕 계획\n"
            f"  범위        ±{self.range_mm:.1f} mm,  스텝 {self.step_mm:.2f} mm"
            f"  →  스텝 {self.steps_per_pass} 개/패스\n"
            f"  이동 속도   {self.speed_mm_s:.1f} mm/s  (스텝당 {1000*self.step_mm/self.speed_mm_s:.0f} ms)\n"
            f"  정착        {self.settle_s:.2f} s,  프레임 {self.frames_per_step} 장/스텝\n"
            f"  반복        {self.repeats} 회 × 횡위치 {list(self.lateral_offsets_mm)} mm\n"
            f"  예상 시간   {self.duration_estimate_s()/60:.1f} 분"
        )


@dataclass
class StepRecord:
    """스텝 하나. 지령과 실측을 **따로** 남긴다 — 둘이 갈라지는 것 자체가 관찰 대상이다."""
    index: int
    pass_index: int
    lateral_mm: float
    commanded_d_mm: float
    t_pc_start: float
    t_pc_end: float
    pose_xyz: list = field(default_factory=list)      # FK, base 프레임 [m]
    pose_quat: list = field(default_factory=list)
    wrench: list = field(default_factory=list)        # fx fy fz mx my mz
    frame_rows: list = field(default_factory=list)    # us_index.csv 의 행 번호
    probing_mode: str = ""
    retreating: bool = False


class SweepCapture(Node):
    """지령·되먹임·프레임을 한 세션 폴더로 모은다."""

    def __init__(self, args, plan: SweepPlan):
        super().__init__("elevational_sweep_capture")
        self.args, self.plan = args, plan
        ns = args.namespace
        self.twist_pub = self.create_publisher(Twist, f"{ns}/desired_twist", 10)
        self.create_subscription(Pose, f"{ns}/ee_wrt_base", self._on_pose, 10)
        self.create_subscription(WrenchStamped, f"{ns}/wrench_px6d", self._on_wrench, 10)
        self.create_subscription(String, f"{ns}/probing_mode", self._on_mode, _latched())
        self.create_subscription(Bool, "/diag/retreating", self._on_retreat, 10)
        # `us_frame_node` 는 /us/image 를 qos_profile_sensor_data(BEST_EFFORT) 로 낸다.
        # 기본 QoS(RELIABLE) 로 구독하면 DDS 가 아예 짝을 맺지 않아 **프레임이 하나도
        # 오지 않는다** — 에러 없이 조용히 빈 세션이 저장된다. 발행측에 맞춘다.
        self.create_subscription(Image, args.image_topic, self._on_image, qos_profile_sensor_data)

        self.pose = None
        self.wrench = None
        self.mode = ""
        self.retreating = False
        self.frames: list[np.ndarray] = []
        self.frame_t: list[float] = []
        self.steps: list[StepRecord] = []
        self._warned_shape = False

    # -- 되먹임 -----------------------------------------------------------
    def _on_pose(self, msg: Pose) -> None:
        self.pose = ([msg.position.x, msg.position.y, msg.position.z],
                     [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])

    def _on_wrench(self, msg: WrenchStamped) -> None:
        f, t = msg.wrench.force, msg.wrench.torque
        self.wrench = [f.x, f.y, f.z, t.x, t.y, t.z]

    def _on_mode(self, msg: String) -> None:
        self.mode = msg.data

    def _on_retreat(self, msg: Bool) -> None:
        if msg.data and not self.retreating:
            self.get_logger().warn("후퇴 신호 — 지령을 멈춘다")
        self.retreating = bool(msg.data)

    def _on_image(self, msg: Image) -> None:
        # us_frame_node 는 candidate 극좌표 프레임을 그대로 낸다 (160 × 512, mono8, step = width).
        a = np.frombuffer(msg.data, np.uint8)
        if a.size != msg.height * msg.width:
            # 조용히 버리면 "프레임이 안 온다" 와 구별되지 않는다. 한 번은 말한다.
            if not self._warned_shape:
                self._warned_shape = True
                self.get_logger().error(
                    f"영상 형식이 예상과 다르다: {msg.width}×{msg.height} "
                    f"encoding={msg.encoding!r} step={msg.step} bytes={a.size} — 프레임을 버린다")
            return
        self.frames.append(a.reshape(msg.height, msg.width).copy())
        self.frame_t.append(time.time())

    # -- 지령 -------------------------------------------------------------
    def _publish_velocity(self, axis: int, mm_s: float) -> None:
        tw = Twist()
        v = [0.0, 0.0, 0.0]
        v[axis] = mm_s / 1000.0
        tw.linear.x, tw.linear.y, tw.linear.z = v
        self.twist_pub.publish(tw)

    def _stop(self) -> None:
        self.twist_pub.publish(Twist())

    def _move(self, axis: int, delta_mm: float) -> None:
        """속도 지령을 계산된 시간만큼 낸다. 위치 지령이 아니므로 FK 로 실측을 남긴다."""
        if abs(delta_mm) < 1e-9:
            return
        speed = self.plan.speed_mm_s * (1.0 if delta_mm > 0 else -1.0)
        dur = abs(delta_mm) / self.plan.speed_mm_s
        t0 = time.time()
        while time.time() - t0 < dur:
            if self.retreating:
                break
            self._publish_velocity(axis, speed)
            rclpy.spin_once(self, timeout_sec=0.005)
        self._stop()

    def _settle_and_record(self, rec: StepRecord) -> None:
        n0 = len(self.frames)
        t0 = time.time()
        while time.time() - t0 < self.plan.settle_s or len(self.frames) - n0 < self.plan.frames_per_step:
            rclpy.spin_once(self, timeout_sec=0.01)
            if time.time() - t0 > self.plan.settle_s + 2.0:
                self.get_logger().warn("프레임이 오지 않는다 — 스텝 %d" % rec.index)
                break
        rec.frame_rows = list(range(n0, len(self.frames)))
        if self.pose:
            rec.pose_xyz, rec.pose_quat = self.pose
        rec.wrench = list(self.wrench or [])
        rec.probing_mode = self.mode
        rec.retreating = self.retreating
        rec.t_pc_end = time.time()

    # -- 실행 -------------------------------------------------------------
    def run(self) -> int:
        plan = self.plan
        self.get_logger().info("접촉·보정 확인 중 …")
        t0 = time.time()
        while time.time() - t0 < 5.0 and (self.pose is None or self.wrench is None):
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.pose is None or self.wrench is None:
            self.get_logger().error("자세/렌치 토픽이 오지 않는다. 제어 스택이 떠 있는지 확인하십시오.")
            return 1
        if not self.args.skip_contact_check and "contact" not in self.mode.lower():
            self.get_logger().error(f"probing_mode = {self.mode!r} — 접촉 상태가 아니다. "
                                    "접촉 후 다시 실행하거나 --skip-contact-check 로 무시하십시오.")
            return 1

        idx = 0
        cur_d = 0.0        # 현재 면외 위치 [mm] — 지령 누적. 실측은 FK 로 따로 남긴다.
        for lat in plan.lateral_offsets_mm:
            if lat:
                self.get_logger().info(f"횡위치 {lat:+.1f} mm 로 이동")
                self._move(AXIS_LATERAL, lat)
            for p in range(plan.repeats):
                offsets = plan.offsets() if p % 2 == 0 else plan.offsets()[::-1]
                self.get_logger().info(f"패스 {p+1}/{plan.repeats}  (횡 {lat:+.1f} mm, 시작 d={offsets[0]:+.1f} mm)")
                for d in offsets:
                    if self.retreating:
                        self.get_logger().error("후퇴 중 — 중단하고 저장한다")
                        self._stop()
                        return self._save(aborted=True)
                    self._move(AXIS_ELEVATIONAL, d - cur_d)
                    cur_d = d
                    rec = StepRecord(index=idx, pass_index=p, lateral_mm=lat,
                                     commanded_d_mm=float(d), t_pc_start=time.time(), t_pc_end=0.0)
                    self._settle_and_record(rec)
                    self.steps.append(rec)
                    idx += 1
                    if idx % 25 == 0:
                        fn = np.linalg.norm(rec.wrench[:3]) if rec.wrench else float("nan")
                        total = plan.steps_per_pass * plan.repeats * len(plan.lateral_offsets_mm)
                        self.get_logger().info(f"  스텝 {idx}/{total}  d={d:+.1f} mm  |F|={fn:.2f} N")
            self._move(AXIS_ELEVATIONAL, -cur_d)      # 면외 원점으로
            cur_d = 0.0
            if lat:
                self._move(AXIS_LATERAL, -lat)
        self._stop()
        return self._save(aborted=False)

    def _save(self, aborted: bool) -> int:
        out = os.path.join(self.args.out_dir, f"sweep_{self.args.label}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out, exist_ok=True)
        if self.frames:
            shape = self.frames[0].shape
            with open(os.path.join(out, "us_frames.bin"), "wb") as f:
                for fr in self.frames:
                    f.write(fr.tobytes())
            with open(os.path.join(out, "us_index.csv"), "w") as f:
                f.write("pc_unix,us_seq\n")
                for i, t in enumerate(self.frame_t):
                    f.write(f"{t:.6f},{i}\n")
        else:
            shape = (0, 0)
        with open(os.path.join(out, "steps.csv"), "w") as f:
            f.write("index,pass,lateral_mm,commanded_d_mm,t_start,t_end,"
                    "x,y,z,qw,qx,qy,qz,fx,fy,fz,mx,my,mz,frame_first,frame_last,mode,retreating\n")
            for r in self.steps:
                p = r.pose_xyz + r.pose_quat if r.pose_xyz else [float("nan")] * 7
                w = r.wrench if r.wrench else [float("nan")] * 6
                a = r.frame_rows[0] if r.frame_rows else -1
                b = r.frame_rows[-1] if r.frame_rows else -1
                f.write(f"{r.index},{r.pass_index},{r.lateral_mm:.3f},{r.commanded_d_mm:.3f},"
                        f"{r.t_pc_start:.6f},{r.t_pc_end:.6f},"
                        + ",".join(f"{v:.6f}" for v in p) + "," + ",".join(f"{v:.6f}" for v in w)
                        + f",{a},{b},{r.probing_mode},{int(r.retreating)}\n")
        meta = {
            "purpose": "elevational observation-model calibration (NOT training data)",
            "aborted": aborted,
            "plan": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(self.plan).items()},
            "namespace": self.args.namespace,
            "image_topic": self.args.image_topic,
            "us": {"frames": len(self.frames), "frame_shape": list(shape), "dtype": "uint8",
                   "note": "candidate 극좌표 원본. us_frame_node 출력 그대로."},
            "axes": {"elevational": AXIS_ELEVATIONAL, "lateral": AXIS_LATERAL,
                     "convention": "probe body: x lateral, y elevational, z beam",
                     "caveat": "⏳ desired_twist 축 매핑은 dry-run 으로 확인할 것"},
            "steps": len(self.steps),
        }
        with open(os.path.join(out, "session.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self.get_logger().info(f"저장: {out}  (스텝 {len(self.steps)}, 프레임 {len(self.frames)})")
        return 1 if aborted else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="세션 이름 (예: c1_2N)")
    ap.add_argument("--namespace", default="/fr5_right")
    ap.add_argument("--image-topic", default="/us/image")
    ap.add_argument("--range-mm", type=float, default=30.0)
    ap.add_argument("--step-mm", type=float, default=0.2)
    ap.add_argument("--speed-mm-s", type=float, default=2.0)
    ap.add_argument("--settle-s", type=float, default=0.4)
    ap.add_argument("--frames-per-step", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--lateral-offsets-mm", type=float, nargs="*", default=[0.0])
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--skip-contact-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="지령을 내지 않고 계획만 출력")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])

    plan = SweepPlan(range_mm=a.range_mm, step_mm=a.step_mm, speed_mm_s=a.speed_mm_s,
                     settle_s=a.settle_s, frames_per_step=a.frames_per_step, repeats=a.repeats,
                     lateral_offsets_mm=tuple(a.lateral_offsets_mm))
    print(plan.describe())
    if a.dry_run:
        print("\n--dry-run: 지령을 내지 않았다. 실행 전 확인할 것 —")
        print("  1) 프로브가 팬텀에 접촉해 목표 힘을 유지 중인가 (probing_mode)")
        print("  2) desired_twist 의 linear.y 가 정말 면외 축인가"
              " (작은 지령 하나를 손으로 확인)")
        print("  3) 스윕 범위가 팬텀 밖으로 나가지 않는가")
        return 0
    if rclpy is None:
        print("rclpy 를 불러올 수 없다. source install/setup.bash 후 다시 실행하십시오.")
        return 1

    rclpy.init()
    node = SweepCapture(a, plan)
    try:
        return node.run()
    except KeyboardInterrupt:
        node._stop()
        return node._save(aborted=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
