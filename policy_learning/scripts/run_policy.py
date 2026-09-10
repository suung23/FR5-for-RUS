#!/usr/bin/env python3
"""학습된 policy 를 실시간으로 돌린다 — 접촉이 잡힌 뒤 영상 품질을 유지하는 면내/회전 서보.

    # 1) 지령 없이 결정만 본다 (기본값). 축 규약·중력·지각 속도를 여기서 확인한다.
    python3 scripts/run_policy.py runs/qres2/last.pt

    # 2) 확인이 끝나면 실제로 움직인다
    python3 scripts/run_policy.py runs/qres2/last.pt --execute --axes rot

시작·정지
---------
사용자가 시점을 지정한다. ``{ns}/policy_enable`` (Bool) 에 true 를 쏘면 시작하고 false 로 멈춘다.

    ros2 topic pub --once /us/policy_enable std_msgs/Bool "{data: true}"

접촉이 없으면(``--start-force`` 미만) enable 이 켜져 있어도 지령하지 않는다. ``/diag/retreating``,
접촉 소실, 프레임 끊김에서 즉시 정지한다. 빔축(z) 은 **어떤 경우에도 지령하지 않는다** — 접촉은
admittance 가 잡는다 (capture_sweep.py 와 같은 규약).

왜 기본이 회전만인가 (--axes rot)
---------------------------------
2026-09-10 측정: 프리핸드 IMU 의 병진 라벨은 드리프트가 순변위의 2.75 배라 학습이 되지 않았다
(mae_x 12 mm, 라벨 산포 14.5 mm — 상수 예측 수준). 회전은 AHRS 직접이라 mae 0.36~0.66°로 라벨
산포보다 작다. 즉 **모델이 실제로 배운 것은 회전뿐**이다. --axes all 은 병진까지 지령하지만
근거가 없으므로 진단용으로만 쓴다.

기록
----
<out>/decisions.csv (tick 마다 지령·Q̂·접촉력·모드), <out>/meta.json. 프레임은 --save-frames 로.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import Pose, Twist, WrenchStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, String
except ImportError:                                   # ROS 없이 --help 는 되게 한다
    rclpy = None
    Node = object

from _common import setup_logging

from rus_policy.bmode import BmodeConverter
from rus_policy.dataset import OBS_VEC_DIM, _resize_frames
from rus_policy.model import ACTION_DIM
from rus_policy.perception import STATE_DIM, apply_frame_transform, build_backend
from rus_policy.train import load_policy

AX_LIN, AX_ANG = slice(0, 3), slice(3, 6)


def quat_to_R(w: float, x: float, y: float, z: float) -> np.ndarray:
    """단위 쿼터니언 → 회전행렬 (base ← probe)."""
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def build_observation(cfg, items, t_now, quat, prev_net, prev_t, device):
    """링버퍼 → 모델 배치 하나. ROS 없이 시험할 수 있도록 순수 함수로 둔다.

    items: (t, bmode(H,W) uint8, state(STATE_DIM), quality) 의 리스트, 오래된 것부터.
    학습 시 dataset.build_sample 과 같은 규약이다 — 모자라면 가장 오래된 것으로 앞을 채우고
    그 자리는 invalid, obs_frame_max_age_s 를 넘긴 프레임도 invalid.
    """
    import torch

    m = cfg.timing.obs_frames
    if not items:
        return None
    items = items[-m:]
    pad = m - len(items)
    idx = [0] * pad + list(range(len(items)))
    frames = _resize_frames(np.stack([items[i][1] for i in idx]), tuple(cfg.perception.frame_size))
    frame_dt = np.array([items[i][0] - t_now for i in idx], np.float32)
    valid = np.ones(m, bool)
    valid[:pad] = False
    valid &= (-frame_dt) <= cfg.timing.obs_frame_max_age_s
    if not valid.any():
        return None
    state = np.nan_to_num(np.stack([items[i][2] for i in idx])).astype(np.float32)

    vec = np.zeros(OBS_VEC_DIM, np.float32)
    vec[0:3] = prev_net
    vec[3] = 1.0 if prev_t is not None else 0.0
    vec[4] = min(t_now - prev_t, 10.0) if prev_t is not None else 10.0
    if quat is not None:                              # 지구 +z 를 프로브 프레임에서 (§1.7)
        vec[5:8] = quat_to_R(*quat).T @ np.array([0.0, 0.0, 1.0])
    # 힘·arbiter 채널은 프리핸드 학습과 같게 0 으로 둔다 (관측 분포를 맞춘다)
    t = lambda a, d: torch.as_tensor(a, dtype=d, device=device)[None]
    return {"frames": t(frames.astype(np.float32) / 255.0, torch.float32),
            "frame_dt": t(frame_dt, torch.float32), "frame_valid": t(valid, torch.bool),
            "state": t(state, torch.float32), "vec": t(vec, torch.float32),
            "quality_now": items[-1][3]}


class PolicyRunner(Node):
    def __init__(self, args, model, cfg, backend):
        super().__init__("rus_policy_runner")
        self.args, self.model, self.cfg, self.backend = args, model, cfg, backend
        self.m = cfg.timing.obs_frames
        self.k = cfg.timing.chunk_steps
        self.dt = cfg.timing.chunk_dt
        self.dev = next(model.parameters()).device

        # 로봇 토픽과 정책 토픽은 **다른 네임스페이스**다. 예전에는 하나(`--namespace`,
        # 기본 `/us`)로 둘 다 잡아서 자세·렌치·twist 가 `/us/...` 를 보고 있었는데
        # 그것을 내는 노드가 없다 — 렌치가 0 이면 `start_force` 게이트가 영영 안 열려
        # `policy_enable` 을 켜도 아무 일이 없다 (2026-09-10 확인).
        ns = args.robot_namespace
        self.twist_pub = self.create_publisher(Twist, f"{ns}/desired_twist", 10)
        self.create_subscription(Pose, f"{ns}/ee_wrt_base", self._on_pose, 10)
        self.create_subscription(WrenchStamped, f"{ns}/wrench_px6d", self._on_wrench, 10)
        self.create_subscription(Bool, args.enable_topic, self._on_enable, 10)
        self.create_subscription(Bool, "/diag/retreating", self._on_retreat, 10)
        if args.start_on == "probing":
            # `probing_mode` 는 **전환할 때만** 발행되고 TRANSIENT_LOCAL 로 래치된다.
            # 기본 QoS(VOLATILE) 로 구독하면 늦게 붙은 이 노드는 현재 모드를 못 받고
            # 다음 전환까지 조용히 기다린다.
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, f"{ns}/probing_mode", self._on_mode, latched)
        # us_frame_node 는 BEST_EFFORT 로 낸다. 기본 QoS 로 구독하면 프레임이 하나도 오지 않는다.
        self.create_subscription(Image, args.image_topic, self._on_image, qos_profile_sensor_data)

        self.buf: deque = deque(maxlen=self.m)          # (t, bmode, state, quality)
        self.pose = None
        self.wrench = None
        self.enabled = False
        self.retreating = False
        self.conv: Optional[BmodeConverter] = None
        self.prev_net = np.zeros(3, np.float32)         # 직전 chunk 의 실현 변위 (x, y, θz)
        self.prev_t = None
        self.rows: list[dict] = []
        self.n_img = self.n_drop = 0
        self._warned = False
        self.create_timer(1.0 / cfg.timing.policy_hz, self._tick)
        self.get_logger().info(
            f"준비됨 — {'DRY-RUN (지령 없음)' if not args.execute else '실행 모드'}, 축={args.axes}, "
            f"m={self.m} k={self.k} @{cfg.timing.policy_hz:.0f}Hz, "
            f"enable 토픽: {args.enable_topic}" + (" · 접촉 프로빙에 맞춰 자동 시작" if args.start_on == "probing" else ""))

    # -- 되먹임 -----------------------------------------------------------
    def _on_pose(self, msg: Pose):
        self.pose = (msg.position, msg.orientation)

    def _on_wrench(self, msg: WrenchStamped):
        f = msg.wrench.force
        self.wrench = np.array([f.x, f.y, f.z], np.float32)

    def _on_enable(self, msg: Bool):
        if bool(msg.data) != self.enabled:
            self.get_logger().warn(f"정책 {'시작' if msg.data else '정지'}")
        self.enabled = bool(msg.data)
        if not self.enabled:
            self._stop()

    def _on_mode(self, msg) -> None:
        """접촉 프로빙에 들어가면 켜고, 나오면 끈다 (`--start-on probing`).

        런북 §6 은 인퍼런싱 시작을 "접촉 ~1 N, 영상이 보이기 시작하는 시점" 으로 잡는다.
        접촉 프로빙 진입이 바로 그 시점이다 — 제어 스택이 접촉을 판정해 z 를 가져간
        순간이라, 사람이 초를 재는 것보다 재현성이 좋다.

        **켜는 것은 계산을 시작한다는 뜻일 뿐이다.** 지령은 `--execute` 가 있어야 나가고,
        그때도 `--start-force` 미만이면 나가지 않는다. 모드에서 나오면 즉시 멈춘다.
        """
        mode = str(msg.data)
        want = mode.startswith("contact_probing")
        if want == self.enabled:
            return
        self.enabled = want
        self.get_logger().warn(
            f"정책 {'시작' if want else '정지'} — probing_mode={mode!r} (자동)"
            # desired_twist 는 발행자가 하나여야 한다 (imu_bench/qc_track/excite_magmap.py §29).
            # 모드 전환이 곧 인계 시점이므로, 조작자는 여기서 Touch 지령을 놓아야 한다.
            + ("   ⚠️ teleop 이 desired_twist 를 같이 내면 두 발행자가 싸운다 — 인계 확인" if want else ""))
        if not want:
            self._stop()

    def _on_retreat(self, msg: Bool):
        if msg.data and not self.retreating:
            self.get_logger().error("후퇴 신호 — 지령 중단")
            self._stop()
        self.retreating = bool(msg.data)

    def _on_image(self, msg: Image):
        a = np.frombuffer(msg.data, np.uint8)
        if a.size != msg.height * msg.width:
            if not self._warned:
                self._warned = True
                self.get_logger().error(f"영상 형식 예상 밖: {msg.width}×{msg.height} {msg.encoding!r} — 버린다")
            return
        raw = a.reshape(msg.height, msg.width)
        now = time.time()
        if self.buf and (now - self.buf[-1][0]) > self.cfg.timing.obs_frame_max_age_s:
            # 끊겼다 돌아왔다. 지각의 시간 상태를 이어 붙이면 끊긴 구간을 건너뛴 것이 된다.
            self.get_logger().warn("영상이 끊겼다 돌아왔다 — 지각 스트림을 다시 시작한다")
            self.buf.clear()
            if self.backend is not None:
                self.backend.reset()
        if self.conv is None:                            # 첫 프레임에서 변환기 확정
            self.conv = BmodeConverter({}, raw.shape, out_size=max(self.cfg.perception.frame_size))
            self.get_logger().info(f"B-mode 변환: {json.dumps(self.conv.describe(), ensure_ascii=False)}")
        t0 = time.time()
        bm = apply_frame_transform(self.conv(raw)[None], self.cfg.perception.frame_transform)[0]
        if self.backend is None:
            state, q = np.zeros(STATE_DIM, np.float32), float("nan")
        else:
            # step() 이라야 직전 프레임 상태를 이어받는다. run(frame[None]) 을 매번 부르면
            # 모든 프레임이 "첫 프레임" 이 되어 quality 를 포함한 시간 의존 특징이 학습 때와
            # 달라진다 (perception.UnetPerception.step 주석 참조).
            vec, q, _tok, _e = self.backend.step(bm)
            state, q = np.nan_to_num(vec).astype(np.float32), float(q)
        dt_ms = (time.time() - t0) * 1000
        if dt_ms > 1000.0 / max(self.args.min_perception_fps, 1e-6):
            self.n_drop += 1
        self.buf.append((time.time(), bm, state, q))
        self.n_img += 1
        if self.n_img % 50 == 0:
            self.get_logger().info(f"프레임 {self.n_img} 장, 지각 {dt_ms:.0f} ms/장, 느림 {self.n_drop} 회")

    # -- 관측 조립 ---------------------------------------------------------
    def _observation(self):
        quat = None
        if self.pose is not None:
            o = self.pose[1]
            quat = (o.w, o.x, o.y, o.z)
        return build_observation(self.cfg, list(self.buf), time.time(), quat,
                                 self.prev_net, self.prev_t, self.dev)

    # -- 지령 -------------------------------------------------------------
    def _stop(self):
        if self.args.execute:
            self.twist_pub.publish(Twist())

    def _publish(self, vel6: np.ndarray):
        """vel6 = [vx, vy, vz mm/s, wx, wy, wz deg/s] (프로브 프레임). z 병진은 항상 버린다."""
        tw = Twist()
        lin, ang = vel6[AX_LIN].copy(), vel6[AX_ANG].copy()
        lin[2] = 0.0                                        # 빔축은 admittance 몫
        if self.args.axes == "rot":
            lin[:] = 0.0
        lin = np.clip(lin, -self.args.max_mm_s, self.args.max_mm_s)
        ang = np.clip(ang, -self.args.max_deg_s, self.args.max_deg_s)
        tw.linear.x, tw.linear.y, tw.linear.z = (lin / 1000.0).tolist()
        tw.angular.x, tw.angular.y, tw.angular.z = np.radians(ang).tolist()
        if self.args.execute:
            self.twist_pub.publish(tw)
        return lin, ang

    # -- 주기 --------------------------------------------------------------
    def _tick(self):
        import torch
        fn = float(np.linalg.norm(self.wrench)) if self.wrench is not None else 0.0
        gate = (self.enabled and not self.retreating and fn >= self.args.start_force
                and len(self.buf) > 0)
        if not gate:
            self._stop()
            return
        obs = self._observation()
        if obs is None:
            self._stop()
            return
        qn = obs.pop("quality_now")
        with torch.no_grad():
            sel = self.model.select_action(obs, n_samples=self.args.z_samples,
                                           gamma=self.cfg.train.gamma_mode_consistency,
                                           prev_dy=torch.as_tensor([self.prev_net[1]], device=self.dev))
        a = sel["a"][0, 0].cpu().numpy()                    # 첫 스텝 속도 (B,k,6) → (6,)
        net = sel["net"][0].cpu().numpy()
        qhat = float(sel["Q_hat"][0].mean())
        lin, ang = self._publish(a.astype(np.float64))
        self.prev_net = np.array([net[0], net[1], net[5]], np.float32)   # 레거시 3축 규약
        self.prev_t = time.time()
        self.rows.append({
            "t": self.prev_t, "F_n": fn, "Q_now": qn, "Q_hat": qhat,
            **{f"cmd_{n}": v for n, v in zip("vx vy vz".split(), lin)},
            **{f"cmd_{n}": v for n, v in zip("wx wy wz".split(), ang)},
            **{f"net_{n}": float(v) for n, v in zip("x y z thx thy thz".split(), net)},
            "margin": float(sel["mode_margin"][0].mean()) if sel["mode_margin"].ndim else float("nan"),
            "n_frames": len(self.buf), "enabled": int(self.enabled),
        })
        if len(self.rows) % 10 == 0:
            self.get_logger().info(
                f"F={fn:4.1f}N  Q={qn:.3f} Q̂={qhat:.3f}  "
                f"ω=({ang[0]:+5.2f},{ang[1]:+5.2f},{ang[2]:+5.2f})°/s  v=({lin[0]:+5.2f},{lin[1]:+5.2f})mm/s")

    def save(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        if self.rows:
            with open(out / "decisions.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.rows[0].keys()))
                w.writeheader(); w.writerows(self.rows)
        (out / "meta.json").write_text(json.dumps({
            "checkpoint": self.args.checkpoint, "axes": self.args.axes, "execute": self.args.execute,
            "z_samples": self.args.z_samples, "start_force_N": self.args.start_force,
            "max_mm_s": self.args.max_mm_s, "max_deg_s": self.args.max_deg_s,
            "n_ticks": len(self.rows), "n_frames": self.n_img, "n_slow_perception": self.n_drop,
            "bmode": self.conv.describe() if self.conv else None,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n기록: {out}  (tick {len(self.rows)}, 프레임 {self.n_img})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--out", default=None, help="기록 폴더 (기본 runs/live_<시각>)")
    p.add_argument("--robot-namespace", default="/fr5_right",
                   help="제어 스택 토픽의 네임스페이스 — desired_twist · ee_wrt_base · "
                        "wrench_px6d · probing_mode 가 여기 있다")
    p.add_argument("--enable-topic", default="/us/policy_enable",
                   help="시작/정지 Bool 토픽 (런북 §4)")
    p.add_argument("--start-on", default="topic", choices=["topic", "probing"],
                   help="topic=사용자가 enable 토픽으로 시작 (기본, 런북 §6). "
                        "probing=접촉 프로빙 진입에 맞춰 자동 시작")
    p.add_argument("--image-topic", default="/us/image")
    p.add_argument("--axes", default="rot", choices=["rot", "all"], help="rot=회전만 (기본), all=병진까지")
    p.add_argument("--execute", action="store_true", help="실제로 지령한다 (기본은 계산만)")
    p.add_argument("--z-samples", type=int, default=64, help="후보 수 M — Q̂ 순위가 약하므로 넉넉히")
    p.add_argument("--start-force", type=float, default=1.0, help="이 접촉력[N] 미만이면 지령하지 않는다")
    p.add_argument("--max-mm-s", type=float, default=3.0)
    p.add_argument("--max-deg-s", type=float, default=5.0)
    p.add_argument("--min-perception-fps", type=float, default=8.0, help="이보다 느린 지각은 '느림' 으로 센다")
    p.add_argument("--device", default="auto")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)
    if rclpy is None:
        raise SystemExit("ROS 2 (rclpy) 를 찾을 수 없습니다. `source ~/FR5-for-RUS/install/setup.bash` 후 실행하십시오.")

    model, cfg = load_policy(args.checkpoint, device=args.device)
    backend = build_backend(cfg)
    print(f"모델 {args.checkpoint}  헤드={cfg.model.head}  행동 {ACTION_DIM} 축  지각={cfg.perception.backend}")
    out = Path(args.out) if args.out else Path("runs") / time.strftime("live_%Y%m%d_%H%M%S")

    rclpy.init()
    node = PolicyRunner(args, model, cfg, backend)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.save(out)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
