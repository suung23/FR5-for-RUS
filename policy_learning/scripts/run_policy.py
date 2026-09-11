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
    from std_msgs.msg import Bool, Float32, String
except ImportError:                                   # ROS 없이 --help 는 되게 한다
    rclpy = None
    Node = object

from _common import setup_logging

from rus_policy.bmode import BmodeConverter
from rus_policy.dataset import OBS_VEC_DIM, _resize_frames
from rus_policy.episode import (CONDITIONS, PlaceboBuffer, StartGate, Thresholds, judge,
                                motion_check, randomize_direction)
from rus_policy.model import ACTION_DIM
from rus_policy.view_quality import view_quality
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
        # 실시간 지각이 도는 곳이 여기뿐이라 Q_raw 의 유일한 출처다. force_search 노드가
        # 이걸 받아 힘 설정값을 품질 경사 방향으로 옮긴다 (DESIGN_NOTES §8.4).
        # 지령과 무관하게 항상 낸다 — 정책이 멈춰 있어도 힘 탐색은 품질을 봐야 한다.
        # Q_seg 와 Q_raw 는 **다른 신호**다 (DESIGN_NOTES §6.2 · §5.3).
        #   Q_seg  분할 기반. "방광을 제대로 보이게" — 영상 축, 성공 판정의 재료.
        #   Q_raw  고전 영상처리, 세그 비의존. "일단 제대로 닿게" — **힘 축**.
        # 힘 탐색은 Q_raw 를 최대화하는 최소 F_n* 를 찾는다. 둘을 한 토픽에 담으면
        # 어느 쪽으로 힘이 움직였는지 되짚을 수 없다.
        self.q_seg_pub = self.create_publisher(Float32, f"{ns}/image_quality_seg", 10)
        self.q_raw_pub = self.create_publisher(Float32, f"{ns}/image_quality_raw", 10)
        # 힘 설정값이 실제로 Q_raw 를 따라 움직이는가. force_search 가 없으면 설정값이
        # 출발값에 **고정**되는데, 그것은 설계가 아니라 사고다.
        self.create_subscription(Float32, f"{ns}/force_setpoint_bar", self._on_f_bar, 10)
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
        # 에피소드 틀 (실험용). duration 0 이면 예전처럼 계속 돈다.
        self.condition = args.condition
        self.gate = StartGate(area_max=args.gate_area_max, quality_min=args.gate_quality_min,
                              confirm_s=args.gate_confirm_s)
        # 사전 등록된 위약(EVAL_PLAN §5)은 "같은 속도·같은 지속시간의 무작위 회전" 이다.
        # stale-obs 는 더 강한 통제지만 계획서에 없으므로 기본이 아니다.
        self.placebo = (PlaceboBuffer(args.placebo_delay_s)
                        if (self.condition == "placebo" and args.placebo_mode == "stale-obs")
                        else None)
        self.rng = np.random.default_rng(args.placebo_seed)
        self.t_start: Optional[float] = None      # 게이트가 열린 시각 = 에피소드 시작
        self.states: list = []                    # 판정에 쓸 상태 이력
        self.state_t: list = []
        self.finished = False
        self.stopped_by_operator_s: Optional[float] = None   # 계획보다 먼저 Stop 으로 닫혔으면 그 시각
        self.attempt = 0            # --loop: 시도 번호. enable 이 올라갈 때마다 는다
        self.attempts: list = []    # 시도별 판정 (한 줄씩 화면에 쌓는다)
        self.announced = False      # 이번 시도에서 성공을 이미 알렸는가
        # 시작 조건은 **실험 장치**다 (계획서 §3 의 'teleop 실패 자세' 판정). 운용에서는
        # 버튼이 조작자의 결정이므로 막지 않는다 — 다만 조건 충족 여부는 기록한다.
        self.gate_required = args.start_gate == 'on'
        self.gate_met_at_start = None
        self._warned = False
        self._uncompensated = ""    # 보상 안 된 렌치가 오고 있으면 그 frame_id
        self.q_raw = float("nan")   # 최근 Q_raw (힘 축)
        self.q_view = float("nan")  # 최근 Q_seg — 방광 뷰 품질 (화면·기록)
        self.q_train = float("nan")  # 최근 옛 Q — 정책 입력에 들어가는 값 (학습 때 정의)
        self.f_bar = None           # 힘 탐색이 알리는 설정값
        self.f_bar_t = 0.0
        self._warned_no_search = False
        self._idle_reason = ""      # 왜 지령하지 않는가. 바뀔 때와 5 s 마다 알린다.
        self._idle_logged = 0.0
        # 방향 관성. 체크포인트 값(0.2)은 보너스 차 1.60 으로 후보 사이 Q̂ 차(중앙 0.021)의
        # 75 배다 — 테스트 관측 496 개 전부에서 방향을 Q̂ 가 아니라 **직전 방향이** 정했다
        # (2026-09-11). 한 번 정한 방향을 영원히 유지하는 것이 그 결과다. --gamma 로 바꾼다.
        self.gamma = (float(args.gamma) if args.gamma is not None
                      else float(cfg.train.gamma_mode_consistency))
        self.get_logger().info(
            f"방향 관성 gamma {self.gamma:g} (보너스 차 {self.gamma * cfg.timing.chunk_steps:.3f}"
            + (", 체크포인트 값" if args.gamma is None else ", --gamma 로 지정") + ")")
        self.create_timer(1.0 / cfg.timing.policy_hz, self._tick)
        # 추론 사이를 메우는 재발행. 워치독이 기대하는 발행 주기(50 Hz)에 맞춘다 —
        # _republish 의 주석에 이유가 있다. 0 이면 옛 거동(추론할 때만 발행).
        self._held = None
        self._held_t = 0.0
        self._held_max_age_s = 3.0 / float(cfg.timing.policy_hz)
        if self.args.republish_hz > 0.0:
            self.create_timer(1.0 / self.args.republish_hz, self._republish)
            self.get_logger().info(
                f"지령 재발행 {self.args.republish_hz:.0f} Hz "
                f"(추론 {cfg.timing.policy_hz:.0f} Hz · 최대 유지 {self._held_max_age_s:.1f} s)")
        self.get_logger().info(
            f"준비됨 — {'DRY-RUN (지령 없음)' if not args.execute else '실행 모드'}, 축={args.axes}, "
            f"m={self.m} k={self.k} @{cfg.timing.policy_hz:.0f}Hz, "
            f"enable 토픽: {args.enable_topic}" + (" · 접촉 프로빙에 맞춰 자동 시작" if args.start_on == "probing" else ""))

    # -- 되먹임 -----------------------------------------------------------
    def _on_pose(self, msg: Pose):
        self.pose = (msg.position, msg.orientation)

    #: 중력 보상된 렌치임을 알리는 frame_id 접미사. us_diff_ik_node 와 같은 규약이다.
    COMPENSATED_FRAME_SUFFIX = "_probe"

    def _on_wrench(self, msg: WrenchStamped):
        # 보상 전 렌치에는 마운트·프로브 자중이 자세에 따라 10 N 가량 실려 있다. 그것을
        # 접촉력으로 믿으면 start_force 게이트가 **닿지 않았는데도** 열린다. 브리지는 교정이
        # 유효할 때만 보상값을 내고 frame_id 접미사로 그것을 알린다 (telemetry_bridge §1243).
        if not msg.header.frame_id.endswith(self.COMPENSATED_FRAME_SUFFIX):
            self.wrench = None
            self._uncompensated = msg.header.frame_id
            return
        self._uncompensated = ""
        f = msg.wrench.force
        self.wrench = np.array([f.x, f.y, f.z], np.float32)

    def _on_enable(self, msg: Bool):
        want = bool(msg.data)
        if want == self.enabled:
            return
        self.get_logger().warn(f"정책 {'시작' if want else '정지'}")
        self.enabled = want
        if not want:
            self._stop()
            if self.args.loop:
                self._close_attempt()          # 놓는 순간이 한 시도의 끝이다
            elif self.args.duration > 0:
                self._stop_episode()

    def _stop_episode(self) -> None:
        """에피소드 모드(run_experiment)에서 Stop 은 그 에피소드를 닫는다.

        예전에는 enable 만 내리고 계속 떠 있었다. _tick 은 꺼져 있으면 시간 판정까지 가지
        않으므로 에피소드가 영영 끝나지 않고, 드라이버는 subprocess 에 묶여 다음 자세를
        묻지 못한다 — teleop 은 돌아왔는데 실험은 멈춘 채로 남는다.

        시작 전(접촉 부족 등으로 아직 t_start 가 없다)이면 닫지 않는다. 조작자가 teleop 으로
        다시 닿고 Start 를 누르면 그 자리에서 이어진다.
        """
        if self.t_start is None:
            self.get_logger().warn(
                "Stop — 에피소드는 아직 시작 전이다. teleop 으로 다시 닿고 Start 하면 이어서 한다")
            return
        self.stopped_by_operator_s = time.time() - self.t_start
        self.finished = True
        self.get_logger().warn(
            f"Stop — 에피소드를 {self.stopped_by_operator_s:.0f} s 에서 닫는다 "
            f"(계획 {self.args.duration:.0f} s). meta.json 의 stopped_by_operator_s 에 남는다")

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

    def _on_f_bar(self, msg: Float32):
        self.f_bar, self.f_bar_t = float(msg.data), time.time()

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
        now_t = time.time()
        self.buf.append((now_t, bm, state, q))
        self.state_t.append(now_t)
        self.states.append(state)
        q_raw = self.backend.raw_quality(bm) if self.backend is not None else float("nan")
        # 화면·기록의 Q_seg 는 **방광 뷰 품질**이다 (rus_policy.view_quality) — 면적과 좌우
        # 중심이 가중치의 63 %. 정책의 입력(state 의 quality 특징)은 학습 때 값 그대로 둔다:
        # 그것을 바꾸면 정책이 처음 보는 입력을 받는다. q 는 기록용으로만 남긴다.
        q_view = view_quality(state)[0] if self.backend is not None else float("nan")
        self.q_view, self.q_train = q_view, float(q)
        self.q_seg_pub.publish(Float32(data=float(q_view)))  # NaN 도 그대로 — "측정 안 됨"
        self.q_raw_pub.publish(Float32(data=float(q_raw)))
        self.q_raw = q_raw
        self.n_img += 1
        if self.n_img % 50 == 0:
            self.get_logger().info(f"프레임 {self.n_img} 장, 지각 {dt_ms:.0f} ms/장, 느림 {self.n_drop} 회")

    def _pose_columns(self) -> dict:
        """ee_wrt_base 의 최신값을 기록용 열로. 없으면 NaN."""
        nan = float("nan")
        if self.pose is None:
            return {k: nan for k in ("ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz")}
        p, o = self.pose
        return {"ee_x": float(p.x), "ee_y": float(p.y), "ee_z": float(p.z),
                "ee_qw": float(o.w), "ee_qx": float(o.x), "ee_qy": float(o.y), "ee_qz": float(o.z)}

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
        self._held = None                      # 재발행을 멈춘다
        if self.args.execute:
            self.twist_pub.publish(Twist())

    def _republish(self):
        """직전 지령을 **발행 주기에 맞춰** 다시 낸다 (ZOH).

        정책은 policy_hz(5 Hz, 200 ms)로 추론하는데, 제어 스택의 워치독은 **50 Hz
        발행자**를 기준으로 맞춰져 있다 (probe.yaml: twist_hold_s 0.04 = "발행 주기
        20 ms 의 2배", twist_timeout_s 0.1). 200 ms 마다 한 번만 내면 매 주기가 이렇게
        된다:

            0–40 ms    그대로
            40–100 ms  1.0 → 0 으로 선형 감쇠 (_hold_fade)
            100–200 ms **두절 판정** — 접촉 프로빙 분기가 zeros(6) 을 넣어
                       정책의 회전 세 축이 통째로 사라진다

        평균 0.35 배에 매 주기 절반은 정확히 0 이다. 정책이 3 °/s 를 내도 실효는
        1 °/s 에 50 % 끊김이고, 로봇은 "거의 안 움직이는" 것처럼 보인다.

        워치독을 늦추는 것이 아니라 **발행을 빠르게 한다.** 워치독은 정책이 죽었을 때
        프로브를 물리는 유일한 장치이므로 그대로 둔다 — 재발행이 멈추면 100 ms 안에
        후퇴가 걸리고, 그것이 바로 우리가 원하는 거동이다.
        """
        if self._held is None or not self.args.execute:
            return
        # **오래된 지령은 재발행하지 않는다.** 재발행은 추론 사이를 메우는 것이지 추론을
        # 대신하는 것이 아니다. _tick 이 멈추면(추론 예외·교착·GIL 기아) 이 타이머만
        # 살아남아 마지막 지령을 영원히 내보내게 되는데, 그러면 워치독이 영영 안 걸리고
        # 로봇은 아무도 판단하지 않는 속도로 계속 돈다.
        #
        # 세 주기(5 Hz → 0.6 s)까지만 믿는다. 느린 추론 한두 번은 넘기고, 정말 멈춘
        # 경우에는 재발행이 끊겨 100 ms 뒤 워치독이 프로브를 물린다.
        if (time.time() - self._held_t) > self._held_max_age_s:
            if self._held is not None:
                self.get_logger().warn(
                    f"추론이 {self._held_max_age_s:.1f} s 넘게 멎었다 — 재발행을 멈춘다 "
                    "(제어 스택 워치독이 후퇴시킨다)")
            self._held = None
            return
        self.twist_pub.publish(self._held)

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
        self._held, self._held_t = tw, time.time()   # 다음 추론까지 이 값을 유지한다
        if self.args.execute:
            self.twist_pub.publish(tw)
        return lin, ang

    # -- 주기 --------------------------------------------------------------
    def _idle(self, reason: str, *, publish: bool = True) -> None:
        """지령하지 않는 이유를 알린다.

        조용히 멈춰 있으면 조작자는 "눌렀는데 아무 일도 없다" 만 본다 — 접촉이 없는 것인지,
        시작 조건이 안 열린 것인지, 렌치가 아예 안 오는 것인지 구별할 수 없다. 이유가 바뀔
        때와 5 s 마다 한 번 찍는다 (매 tick 찍으면 5 Hz 로 로그가 흐른다).

        ``publish=False`` 는 **desired_twist 가 이 러너의 것이 아닐 때**다 — 0 조차 내지 않는다.
        """
        if publish:
            self._stop()
        else:
            self._held = None                  # 재발행만 멈추고 채널은 건드리지 않는다
        now = time.time()
        if reason != self._idle_reason or (now - self._idle_logged) > 5.0:
            self.get_logger().warn(f"대기 — {reason}")
            self._idle_reason, self._idle_logged = reason, now

    def _tick(self):
        import torch
        t = time.time()          # 이 tick 의 시각. 게이트·에피소드·위약이 모두 이 값을 쓴다.
        fn = float(np.linalg.norm(self.wrench)) if self.wrench is not None else 0.0
        # 꺼져 있거나 후퇴 중이면 desired_twist 에 **아무것도 내지 않는다.**
        #
        # Stop 뒤 그 채널은 Touch 의 것이다. 예전에는 여기서도 매 tick(5 Hz) 0 을 냈고,
        # 그러면 teleop 이 "처음" 으로 돌아오지 않는다: 쥐고 있으면 200 ms 마다 조작자 지령이
        # 한 번씩 0 으로 덮이고, 놓으면 그 0 이 twist 를 신선하게 만들어 워치독 후퇴를 매번
        # 푼다 (us_diff_ik_node: "twist 복귀 — 후퇴 해제"). 후퇴 중에 0 을 내는 것도 같은
        # 이유로 후퇴를 무른다. 정지의 0 은 _on_enable 이 전환 순간에 한 번만 낸다.
        if not self.enabled:
            return self._idle(f"정책이 꺼져 있다 ({self.args.enable_topic} 에 true)",
                              publish=False)
        if self.retreating:
            return self._idle("/diag/retreating — 제어 스택이 후퇴 중", publish=False)
        if self.wrench is None:
            if self._uncompensated:
                return self._idle(
                    f"렌치가 **중력 보상되지 않았다** (frame_id={self._uncompensated!r}) — "
                    "교정을 마쳐야 접촉력을 믿을 수 있다. 보상 전 값에는 자중 10 N 이 실려 있다")
            return self._idle(f"렌치가 오지 않는다 ({self.args.robot_namespace}/wrench_px6d)")
        if fn < self.args.start_force:
            return self._idle(f"접촉 부족 ‖F‖={fn:.2f} N < {self.args.start_force:.2f} N")
        if not self.buf:
            return self._idle("영상 프레임이 아직 없다")
        obs = self._observation()
        if obs is None:
            return self._idle("유효한 관측 프레임이 없다 (전부 오래됐거나 무효)")
        qn = obs.pop("quality_now")

        # 시작 조건. 열리기 전에는 아무것도 지령하지 않는다 — 잘 보이는 자세에서 출발하면
        # 아무것도 안 해도 성공이라 찾는 능력을 못 잰다.
        if self.t_start is None:
            st = self.states[-1] if self.states else None
            if st is None:
                return self._idle("지각 상태가 아직 없다")
            from rus_policy.episode import AREA
            met = self.gate.update(t, st, self.q_raw)
            if self.gate_required and not met:
                return self._idle(
                    f"시작 조건 미충족 — 면적비 {st[AREA]:.3f} (< {self.args.gate_area_max}) · "
                    f"Q_raw {self.q_raw:.2f} (≥ {self.args.gate_quality_min}) · "
                    f"유지 {self.gate.held_s:.1f}/{self.args.gate_confirm_s:.1f} s")
            self.gate_met_at_start = bool(met)
            if not met:
                self.get_logger().warn(
                    f"시작 조건 **미충족인 채로** 시작한다 (--start-gate off) — "
                    f"면적비 {st[AREA]:.3f} · Q_raw {self.q_raw:.2f}. 기록에 남는다")
            self.t_start = t
            self.get_logger().warn(
                f"시작 조건 충족 — 에피소드 시작 (조건 {self.condition}, "
                + (f"{self.args.duration:.0f} s)" if self.args.duration > 0 else "무제한)"))

        if self.args.duration > 0 and (t - self.t_start) >= self.args.duration:
            self._stop()
            self.finished = True
            self.get_logger().warn("에피소드 종료 — 지령을 멈춘다")
            return

        # 위약: 같은 정책·같은 후보 분포로, **지금이 아닌 관측**에서 지령을 만든다.
        if self.placebo is not None:
            self.placebo.push(t, obs)
            past = self.placebo.take(t)
            if past is None:
                # 아직 지연만큼 안 쌓였다 — 위약이 성립하지 않는다
                return self._idle(f"위약 관측 버퍼 채우는 중 ({self.placebo.n_buffered} 장)")
            obs = past
        if not self._warned_no_search and (t - self.f_bar_t) > 5.0:
            self._warned_no_search = True
            self.get_logger().warn(
                f"⚠️ 힘 탐색이 보이지 않는다 ({self.args.robot_namespace}/force_setpoint_bar 무음) — "
                "힘 설정값이 출발값에 고정된 채 돈다. "
                "`ros2 run fr5_control force_search --ros-args -p execute:=true` 를 띄웠는가")
        if not self.announced and self.t_start is not None:
            v = self._judge_now()
            if v.get("success"):
                self.announced = True
                self.get_logger().warn(
                    f"○ 진단 가능 뷰 도달 — 시작 후 {t - self.t_start:.0f} s. "
                    "계속 두면 유지되는지 본다. Stop 으로 이 시도를 닫는다")
        if self._idle_reason:
            # 여기까지 왔다는 것은 실제로 지령을 낸다는 뜻이다. 관측을 얻은 자리에서 지우면
            # 시작 조건에서 막히는 동안 매 tick "해제 → 대기" 가 번갈아 찍힌다.
            self.get_logger().warn(f"대기 해제 — {self._idle_reason}")
            self._idle_reason = ""
        with torch.no_grad():
            sel = self.model.select_action(obs, n_samples=self.args.z_samples,
                                           gamma=self.gamma,
                                           prev_dy=torch.as_tensor([self.prev_net[1]], device=self.dev))
        a = sel["a"][0, 0].cpu().numpy()                    # 첫 스텝 속도 (B,k,6) → (6,)
        net = sel["net"][0].cpu().numpy()
        qhat = float(sel["Q_hat"][0].mean())
        # 조건을 지나기 **전**의 정책 의도. 위약은 아래에서 방향을 섞고 hold 는 0 으로
        # 막으므로, 그 뒤의 값만 찍으면 hold 에피소드 내내 "정책이 0 을 낸다" 로 보인다
        # (2026-09-11 에 실제로 그렇게 읽혔다). 모델이 무엇을 원했는지는 지령과 따로 봐야 한다.
        a_policy = a.copy()
        if self.condition == "placebo" and self.placebo is None:
            # 크기는 정책이 고른 그대로, 방향만 무의미하게. 크기를 다시 뽑으면 조건 사이에서
            # "움직임의 양" 이 어긋나고, 그것이 이 대조군이 통제하려던 변수다.
            a = randomize_direction(a, self.rng)
        if self.condition in ("hold", "expert"):
            # hold 는 바닥선, expert 는 사람이 지령한다. 둘 다 러너는 **기록만** 한다.
            self._stop()
            lin, ang = np.zeros(3), np.zeros(3)
        else:
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
            "condition": self.condition, "t_episode": t - self.t_start,
            "Q_raw": self.q_raw,
            "Q_view": self.q_view,       # 새 Q_seg (면적·중심 63 %). Q_now 는 옛 정의 그대로
            "f_bar": self.f_bar if self.f_bar is not None else float("nan"),
            # 로봇이 **실제로** 어디를 향했는가. 지령(cmd_*)만 적으면 "정책대로 움직였나" 에
            # 답할 수 없다 — save() 의 motion_check 가 이 열과 cmd_w* 를 비교한다.
            **self._pose_columns(),
        })
        if len(self.rows) % 10 == 0:
            # 두 줄로 나눈다: 첫 줄은 **정책이 원한 것**, 둘째 줄은 **로봇에 간 것**.
            # 조건(hold·placebo)이 둘을 갈라놓으므로 한 줄에 섞으면 어느 쪽인지 모른다.
            # 정책 줄은 상한으로 자르기 전 값이다 — 3 °/s 를 넘겨 원하면 그대로 보인다.
            pw = a_policy[AX_ANG]
            tag = {"hold": "hold — 지령 안 함", "expert": "expert — 사람이 지령",
                   "placebo": "placebo — 방향 무작위", "policy": "policy"}.get(
                       self.condition, self.condition)
            self.get_logger().info(
                f"F={fn:4.1f}N  Q_seg={self.q_view:.3f} Q_raw={self.q_raw:.3f} "
                f"Q̂={qhat:.3f} (정책 입력 Q {qn:.3f})  [{tag}]\n"
                f"    정책  ω=({pw[0]:+5.2f},{pw[1]:+5.2f},{pw[2]:+5.2f})°/s   "
                f"chunk 순변위 θ=({net[3]:+5.1f},{net[4]:+5.1f},{net[5]:+5.1f})°\n"
                f"    지령  ω=({ang[0]:+5.2f},{ang[1]:+5.2f},{ang[2]:+5.2f})°/s   "
                f"v=({lin[0]:+5.2f},{lin[1]:+5.2f})mm/s")

    def _judge_now(self) -> dict:
        """지금까지 쌓인 상태로 판정한다. 시도가 시작되지 않았으면 빈 dict."""
        if self.t_start is None or not self.states:
            return {}
        m = [i for i, tt in enumerate(self.state_t) if tt >= self.t_start]
        if not m:
            return {}
        thr = Thresholds(area_min=self.args.success_area_min,
                         component_min=self.args.success_component_min,
                         centroid_max=self.args.success_centroid_max,
                         hold_s=self.args.success_hold_s)
        return judge([self.state_t[i] for i in m],
                     np.stack([self.states[i] for i in m]), thr)

    def _close_attempt(self) -> None:
        """한 시도를 닫고 기록한 뒤 다음을 위해 비운다 (--loop).

        프로세스를 다시 띄우지 않는 이유는 모델 로딩과 지각 워밍업이 시도마다 반복되면
        조작자가 그 사이를 기다려야 하기 때문이다. 평가 루프는 "접근 → 전환 → 찾기 →
        다시" 이고, 그 사이에 터미널을 만질 일이 없어야 한다.
        """
        v = self._judge_now()
        if not v and self.t_start is None:
            self.get_logger().warn("시도가 시작되지 않았다 (시작 조건 미충족) — 기록하지 않는다")
        else:
            self.attempt += 1
            out = Path(self.args.out or ".") / f"attempt_{self.attempt:03d}"
            self.save(out, quiet=True)
            row = {"n": self.attempt, "success": bool(v.get("success")),
                   "best_run_s": v.get("best_run_s", float("nan")),
                   "t_total_s": (self.state_t[-1] - self.t_start) if self.t_start else float("nan"),
                   "q_seg": v.get("q_mean", float("nan")), "dir": out.name}
            self.attempts.append(row)
            mark = "○ 성공" if row["success"] else "✗ 실패"
            self.get_logger().warn(
                f"시도 {row['n']}  {mark}  최장 연속 {row['best_run_s']:.1f} s  "
                f"소요 {row['t_total_s']:.0f} s  → {out.name}")
            ok = sum(a["success"] for a in self.attempts)
            self.get_logger().warn(f"  누적 {ok}/{len(self.attempts)}")
        # 다음 시도를 위해 비운다. 관측 버퍼와 지각 스트림은 그대로 둔다 — 프로브는
        # 그 자리에 있고, 끊긴 적이 없다.
        self.t_start = None
        self.states, self.state_t, self.rows = [], [], []
        self.gate.reset()
        self.announced = False

    def save(self, out: Path, quiet: bool = False):
        out.mkdir(parents=True, exist_ok=True)
        if self.rows:
            with open(out / "decisions.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(self.rows[0].keys()))
                w.writeheader(); w.writerows(self.rows)
        verdict = {}
        if self.t_start is not None and self.states:
            m = [i for i, tt in enumerate(self.state_t) if tt >= self.t_start]
            if m:
                thr = Thresholds(area_min=self.args.success_area_min,
                                 component_min=self.args.success_component_min,
                                 centroid_max=self.args.success_centroid_max,
                                 hold_s=self.args.success_hold_s)
                verdict = judge([self.state_t[i] for i in m],
                                np.stack([self.states[i] for i in m]), thr)
                print("\n판정: " + "  ".join(f"{k}={v}" for k, v in verdict.items()))
        motion = self._motion_check()
        if motion:
            print(f"움직임: 지령 {motion['commanded_deg']:.1f}° · 실제 {motion['realized_deg']:.1f}° "
                  f"(처음↔끝 {motion['realized_net_deg']:.1f}°) — {motion['verdict']}")
        if self.states:
            # 판정 임계를 나중에 다시 훑으려면 원자료가 있어야 한다 — 파일럿의 목적이
            # "예비 촬영으로 임계를 확정한다" 이므로 판정 결과만 남기면 되돌아갈 수 없다.
            np.savez_compressed(out / "states.npz", t=np.asarray(self.state_t, np.float64),
                                state=np.stack(self.states).astype(np.float32),
                                t_start=np.float64(self.t_start if self.t_start else np.nan))
        (out / "meta.json").write_text(json.dumps({
            "checkpoint": self.args.checkpoint, "axes": self.args.axes, "execute": self.args.execute,
            "z_samples": self.args.z_samples, "start_force_N": self.args.start_force,
            "gamma_mode_consistency": self.gamma,
            "max_mm_s": self.args.max_mm_s, "max_deg_s": self.args.max_deg_s,
            "condition": self.condition, "duration_s": self.args.duration,
            "episode_started": self.t_start is not None,
            "stopped_by_operator_s": self.stopped_by_operator_s,
            "start_gate": self.args.start_gate, "gate_met_at_start": self.gate_met_at_start,
            "motion": motion,
            "gate": {"area_max": self.args.gate_area_max, "quality_min": self.args.gate_quality_min,
                     "confirm_s": self.args.gate_confirm_s, "held_s": self.gate.held_s},
            "thresholds": {"area_min": self.args.success_area_min,
                           "component_min": self.args.success_component_min,
                           "centroid_max": self.args.success_centroid_max,
                           "hold_s": self.args.success_hold_s},
            "placebo_mode": self.args.placebo_mode, "placebo_seed": self.args.placebo_seed,
            "verdict": verdict,
            "n_ticks": len(self.rows), "n_frames": self.n_img, "n_slow_perception": self.n_drop,
            "bmode": self.conv.describe() if self.conv else None,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if not quiet:
            print(f"\n기록: {out}  (tick {len(self.rows)}, 프레임 {self.n_img})")

    def _motion_check(self) -> dict:
        """에피소드 구간의 지령 회전 대 실제 회전 (rus_policy.episode.motion_check)."""
        if not self.rows:
            return {}
        rows = [r for r in self.rows if self.t_start is None or r["t"] >= self.t_start]
        if len(rows) < 2:
            return {}
        return motion_check(
            [r["t"] for r in rows],
            np.array([[r["cmd_wx"], r["cmd_wy"], r["cmd_wz"]] for r in rows], float),
            np.array([[r["ee_qw"], r["ee_qx"], r["ee_qy"], r["ee_qz"]] for r in rows], float))

    def summary(self) -> None:
        if not self.attempts:
            return
        ok = sum(a["success"] for a in self.attempts)
        print(f"\n시도 {len(self.attempts)} · 성공 {ok} ({ok/len(self.attempts):.0%})")
        print(f"  {'#':>3} {'결과':>5} {'최장연속':>9} {'소요':>7}  기록")
        for a in self.attempts:
            print(f"  {a['n']:>3} {'성공' if a['success'] else '실패':>5} "
                  f"{a['best_run_s']:>8.1f}s {a['t_total_s']:>6.0f}s  {a['dir']}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--out", default=None, help="기록 폴더 (기본 runs/live_<시각>)")
    p.add_argument("--robot-namespace", default="/fr5_right",
                   help="제어 스택 토픽의 네임스페이스 — desired_twist · ee_wrt_base · "
                        "wrench_px6d · probing_mode 가 여기 있다")
    # 브리지(telemetry_bridge)가 {robot_ns}/policy_enable 로 낸다. us_diff_ik_node 도 같은
    # 이름을 구독해 régime 을 함께 연다 — 셋이 같은 이름을 봐야 버튼 하나가 다 움직인다.
    p.add_argument("--enable-topic", default="/fr5_right/policy_enable",
                   help="시작/정지 Bool 토픽. 브리지가 내는 {robot_ns}/policy_enable 과 같아야 한다")
    p.add_argument("--start-on", default="topic", choices=["topic", "probing"],
                   help="topic=사용자가 enable 토픽으로 시작 (기본, 런북 §6). "
                        "probing=접촉 프로빙 진입에 맞춰 자동 시작")
    p.add_argument("--image-topic", default="/us/image")
    p.add_argument("--condition", default="policy", choices=list(CONDITIONS),
                   help="hold=정지(바닥선) · placebo=지연 관측으로 같은 분포의 움직임 · "
                        "policy=학습된 정책 · expert=숙련자(러너는 기록만)")
    p.add_argument("--loop", action="store_true",
                   help="평가 루프. Start/Stop 한 번이 한 시도이고, 프로세스는 계속 떠 있는다 — "
                        "접근 → 전환 → 찾기 → 다시 사이에 터미널을 만질 일이 없다")
    p.add_argument("--duration", type=float, default=0.0,
                   help="에피소드 길이 [s]. 0 이면 무제한 (예전 거동)")
    p.add_argument("--placebo-mode", default="random-direction",
                   choices=["random-direction", "stale-obs"],
                   help="random-direction=사전 등록된 위약 (같은 크기, 무작위 방향) · "
                        "stale-obs=지연 관측 (더 강한 통제이나 계획서 밖)")
    p.add_argument("--placebo-seed", type=int, default=0, help="위약 방향 난수 — 세션을 재현한다")
    p.add_argument("--placebo-delay-s", type=float, default=30.0,
                   help="stale-obs 일 때의 관측 지연 [s]")
    p.add_argument("--gamma", type=float, default=None,
                   help="방향 관성 (모드 일관성 보너스). 없으면 체크포인트 값(0.2). 0.2 는 후보 사이 "
                        "Q̂ 차의 75 배라 방향이 절대 안 바뀐다. 0.005 면 Q̂ 가 약 20 %% 를 정한다")
    p.add_argument("--republish-hz", type=float, default=50.0,
                   help="추론 사이에 직전 지령을 다시 내는 주기 [Hz]. 제어 스택의 워치독은 "
                        "50 Hz 발행자 기준이라 policy_hz(5 Hz)만으로는 매 주기 절반이 "
                        "두절로 잡힌다. 0 이면 재발행하지 않는다 (옛 거동)")
    p.add_argument("--start-gate", default="off", choices=["off", "on"],
                   help="on=시작 조건을 만족할 때까지 기다린다 (실험). off=버튼이 곧 시작이고 "
                        "조건 충족 여부는 기록만 한다 (운용·평가 루프, 기본)")
    p.add_argument("--gate-area-max", type=float, default=0.02, help="시작: 면적비 < 이 값")
    p.add_argument("--gate-quality-min", type=float, default=0.6, help="시작: Q_raw ≥ 이 값")
    p.add_argument("--gate-confirm-s", type=float, default=2.0,
                   help="시작 조건 판정 창 [s]. 계획서 §3 은 10 s 정지 후 마지막 2 s 를 쓴다 — "
                        "10 s 정지는 조작자의 절차이고 여기 값은 그 판정 창이다")
    p.add_argument("--success-area-min", type=float, default=0.08, help="성공: 면적비 ≥ (파일럿이 확정)")
    p.add_argument("--success-component-min", type=float, default=0.80, help="성공: 연결성분 ≥")
    p.add_argument("--success-centroid-max", type=float, default=0.30,
                   help="성공: |중심 − 0.5| ≤ (0.30 = 중앙 60 % 폭). 가장자리 뷰를 배제한다")
    p.add_argument("--success-hold-s", type=float, default=3.0, help="성공: 연속 유지 [s]")
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
    args.out = str(out)
    node = PolicyRunner(args, model, cfg, backend)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        if args.loop:
            node.summary()
        else:
            node.save(out)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
