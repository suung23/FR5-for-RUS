"""스페이스바 capture mode.

**기존 force pipeline 의 최종 출력만 읽는다.** 원시 wrench 를 다시 보정하지
않고, 좌표변환을 다시 하지 않고, 중력 모델을 다시 적용하지 않는다. 브리지가
``{ns}/wrench_px6d`` 로 내는 값이 곧 corrected probe-frame wrench 이고, 그것을
그대로 받는다 — 파이프라인을 두 벌 두면 어느 쪽이 로봇을 움직이는 값인지
알 수 없게 된다.

토픽을 쓰는 이유가 하나 더 있다. 화면용 websocket 은 초당 한 장이고 각 장은 그
1 초의 평균이라, 사양이 요구하는 "100 ms 보다 오래된 자료면 거부" 를 만족할 수
없다. 토픽은 센서 속도 그대로 온다.
"""

from __future__ import annotations

import csv
import math
import os
import select
import sys
import termios
import time
import tty
from collections import deque
from datetime import datetime, timezone

import numpy as np

CAPTURE_COLUMNS = (
    "sample_id",
    "timestamp_local",
    "timestamp_utc",
    "robot_normal_force_instant_N",
    "robot_normal_force_median_250ms_N",
    "robot_normal_force_mean_250ms_N",
    "robot_normal_force_std_250ms_N",
    "force_data_age_ms",
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "tcp_rx_rad",
    "tcp_ry_rad",
    "tcp_rz_rad",
    "sensor_calibration_valid",
    "gravity_compensation_valid",
    "probe_frame_transform_valid",
    "force_sign_convention",
    "capture_valid",
    "notes",
)

TEMPLATE_COLUMNS = ("sample_id", "scale_mass_g", "valid_trial", "notes")

#: 자료가 이보다 오래되면 capture 를 거부한다 [ms].
MAX_DATA_AGE_MS = 100.0

#: 스페이스바 auto-repeat 을 막는다 [s].
DEBOUNCE_S = 0.3

#: 보조 통계를 내는 창 [s]. 앞뒤로 각각 이만큼.
BUFFER_S = 0.25


class KeyReader:
    """터미널을 raw 로 두고 한 글자씩 논블로킹으로 읽는다."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.saved = None

    def __enter__(self):
        if self.fd is not None:
            self.saved = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
        return self

    def __exit__(self, *exc):
        if self.fd is not None and self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        return False

    def poll(self):
        """눌린 글자 하나 또는 ``None``."""
        if self.fd is None:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if ready else None


def rotation_to_rpy(rotation) -> tuple:
    """3x3 → 고정축 XYZ ``(rx, ry, rz)`` [rad]. 제어 스택과 같은 규약."""
    r = np.asarray(rotation, dtype=float)
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:
        return 0.0, pitch, math.atan2(-r[0, 1], r[1, 1])
    return math.atan2(r[2, 1], r[2, 2]), pitch, math.atan2(r[1, 0], r[0, 0])


class ForceValidationCapture:
    """기존 파이프라인 출력을 듣고, 스페이스바 시점의 값을 남긴다."""

    def load_existing(self) -> int:
        """이미 있는 capture 를 읽어 이어서 잰다.

        도구는 끝날 때 파일을 통째로 다시 쓴다. 이어붙이려면 먼저 읽어 두어야
        하고, 읽지 않으면 앞서 잰 것이 지워진다 — 90 초짜리 절차를 두 번 하게
        만드는 종류의 손실이다.

        Returns:
            읽어들인 줄 수.
        """
        if not os.path.exists(self.capture_path):
            return 0
        with open(self.capture_path, newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        if not rows:
            return 0
        self.rows = rows
        self.next_id = max(int(float(row["sample_id"])) for row in rows) + 1
        return len(rows)

    def __init__(self, node, namespace: str, sign: float, out_dir: str) -> None:
        self.node = node
        self.sign = sign
        self.out_dir = out_dir
        self.capture_path = os.path.join(out_dir, "robot_force_captures.csv")
        self.template_path = os.path.join(out_dir, "ground_truth_scale_template.csv")

        self.samples: deque = deque(maxlen=4000)
        self.latest = None
        self.latest_at = 0.0
        self.latest_frame = ""
        self.joints = None
        self.calibration_valid = None
        self.rows: list = []
        self.next_id = 1
        self.last_capture_at = 0.0

        from geometry_msgs.msg import WrenchStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool

        node.create_subscription(
            WrenchStamped, f"{namespace}/wrench_px6d", self._on_wrench, 50)
        node.create_subscription(
            JointState, f"{namespace}/joint_states", self._on_joints, 10)
        node.create_subscription(
            Bool, f"{namespace}/calibration_valid", self._on_calibration, 10)

    # -- 구독 -------------------------------------------------------------

    def _on_wrench(self, message) -> None:
        force = message.wrench.force
        normal = self.sign * float(force.z)
        now = time.time()
        self.samples.append((now, normal))
        self.latest = normal
        self.latest_at = now
        self.latest_frame = message.header.frame_id

    def _on_joints(self, message) -> None:
        if len(message.position) >= 6:
            self.joints = list(message.position)[:6]

    def _on_calibration(self, message) -> None:
        self.calibration_valid = bool(message.data)

    # -- 상태 -------------------------------------------------------------

    @property
    def data_age_ms(self) -> float:
        """최신 자료가 몇 ms 전 것인가. 받은 적이 없으면 무한대."""
        if self.latest is None:
            return float("inf")
        return (time.time() - self.latest_at) * 1000.0

    def probe_frame_ready(self) -> bool:
        """브리지가 probe frame 으로 내고 있는가.

        보상이 걸리면 브리지가 frame_id 를 ``*_probe`` 로 바꾼다. 센서 프레임
        이름이 오면 좌표변환이 안 걸린 원시값이라는 뜻이다.
        """
        return self.latest_frame.endswith("_probe")

    def window(self, seconds: float = BUFFER_S) -> np.ndarray:
        """최근 ``seconds`` 의 표본. 보조 통계에만 쓴다."""
        cutoff = time.time() - seconds
        return np.array([value for stamp, value in self.samples if stamp >= cutoff],
                        dtype=float)

    def tcp_pose(self):
        """``(x, y, z, rx, ry, rz)`` 또는 ``None``.

        관절각에서 순기구학으로 만든다 — 컨트롤러의 TCP 설정과 무관하게 정의가
        하나뿐이기 때문이다.
        """
        if self.joints is None:
            return None
        from fr5_control.robot_backend import fr5_forward_kinematics

        transform = fr5_forward_kinematics(self.joints)
        rx, ry, rz = rotation_to_rpy(transform[:3, :3])
        return (float(transform[0, 3]), float(transform[1, 3]), float(transform[2, 3]),
                rx, ry, rz)

    # -- capture ----------------------------------------------------------

    def blocking_reason(self) -> str:
        """지금 capture 할 수 없는 이유. 없으면 빈 문자열."""
        if self.calibration_valid is not True:
            return "sensor calibration 이 유효하지 않다"
        if self.latest is None:
            return "force stream 이 아직 오지 않았다"
        if not self.probe_frame_ready():
            return (
                f"probe frame 이 아니다 (frame_id={self.latest_frame!r}) — "
                "중력보상·좌표변환이 걸리지 않은 원시값이다"
            )
        return ""

    def capture(self, sign_note: str) -> dict | None:
        """지금 값을 한 줄로 남긴다. debounce 에 걸리면 ``None``."""
        now = time.time()
        if now - self.last_capture_at < DEBOUNCE_S:
            return None
        self.last_capture_at = now

        age = self.data_age_ms
        instant = self.latest
        window = self.window()
        pose = self.tcp_pose()
        blocked = self.blocking_reason()
        stale = age > MAX_DATA_AGE_MS

        notes = []
        if blocked:
            notes.append(blocked)
        if stale:
            notes.append(f"force data {age:.0f} ms 로 오래됐다 (한계 {MAX_DATA_AGE_MS:.0f})")
        if pose is None:
            notes.append("joint_states 가 없어 TCP 자세를 남기지 못했다")

        row = {
            "sample_id": self.next_id,
            "timestamp_local": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "robot_normal_force_instant_N": "" if instant is None else f"{instant:.6f}",
            "robot_normal_force_median_250ms_N": (
                f"{float(np.median(window)):.6f}" if window.size else ""),
            "robot_normal_force_mean_250ms_N": (
                f"{float(window.mean()):.6f}" if window.size else ""),
            "robot_normal_force_std_250ms_N": (
                f"{float(window.std(ddof=1)):.6f}" if window.size > 1 else ""),
            "force_data_age_ms": f"{age:.1f}" if math.isfinite(age) else "",
            "tcp_x_m": f"{pose[0]:.6f}" if pose else "",
            "tcp_y_m": f"{pose[1]:.6f}" if pose else "",
            "tcp_z_m": f"{pose[2]:.6f}" if pose else "",
            "tcp_rx_rad": f"{pose[3]:.6f}" if pose else "",
            "tcp_ry_rad": f"{pose[4]:.6f}" if pose else "",
            "tcp_rz_rad": f"{pose[5]:.6f}" if pose else "",
            "sensor_calibration_valid": "TRUE" if self.calibration_valid else "FALSE",
            "gravity_compensation_valid": "TRUE" if self.calibration_valid else "FALSE",
            "probe_frame_transform_valid": "TRUE" if self.probe_frame_ready() else "FALSE",
            "force_sign_convention": sign_note,
            "capture_valid": "FALSE" if (blocked or stale or instant is None) else "TRUE",
            "notes": "; ".join(notes),
        }
        self.rows.append(row)
        self.next_id += 1
        return row

    def drop_last(self):
        """직전 capture 를 지운다. sample id 도 되돌린다."""
        if not self.rows:
            return None
        removed = self.rows.pop()
        self.next_id -= 1
        return removed

    # -- 저장 -------------------------------------------------------------

    def write(self) -> tuple:
        """Capture CSV 와 전자저울 template 을 쓴다.

        template 은 capture 의 sample id 를 그대로 복사한다 — 손으로 옮겨 적으면
        어긋나고, 어긋난 id 는 분석에서 조용히 다른 trial 을 짝지운다.
        """
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.capture_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CAPTURE_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)

        with open(self.template_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TEMPLATE_COLUMNS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({
                    "sample_id": row["sample_id"],
                    "scale_mass_g": "",
                    "valid_trial": "TRUE",
                    "notes": "",
                })
        return self.capture_path, self.template_path
