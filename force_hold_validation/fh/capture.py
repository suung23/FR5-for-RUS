"""캡처 — 제어 스택이 실제로 쓰는 값만 기록한다.

렌치를 다시 유도하지 않는다. ``wrench_px6d`` 는 브리지가 보상해 낸 값이고 그것이
곧 ``us_diff_ik`` 가 문턱에 대는 값이므로, 여기서 한 벌 더 계산하면 언젠가 로봇이
보는 것과 갈라진다 (force_validation 이 같은 이유로 같은 규약을 쓴다).
"""
from __future__ import annotations

import csv
import json
import os
import select
import sys
import termios
import time
import tty

from geometry_msgs.msg import Pose, WrenchStamped
from rcl_interfaces.srv import GetParameters
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

#: ``probing_mode`` 는 **전환할 때만** 발행되고 TRANSIENT_LOCAL 로 래치된다.
#: 기본 QoS(VOLATILE) 로 구독하면 늦게 붙은 캡처는 래치된 현재 값을 못 받고,
#: 실행 중에 전환이 없으면 모드가 영영 빈 문자열로 남는다 — 2026-09-02 실험에서
#: 15 개 중 14 개가 그렇게 기록됐다. 발행자와 같은 내구성을 요구해야 한다.
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)

#: 기록할 키와 뜻. README 의 표와 같아야 한다.
KEYS = {"i": "syringe in", "w": "syringe withdraw", "s": "settled", " ": "mark"}


class KeyReader:
    """블로킹 없이 키 하나를 읽는다. 터미널 설정은 반드시 되돌린다."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def get(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class ForceHoldCapture:
    """한 번의 실행을 파일 두 개로 남긴다: 표본 CSV 와 지령 메타 JSON."""

    def __init__(self, node, namespace: str, sign: float, label: str,
                 run_type: str, out_dir: str, step_ml=None) -> None:
        self.node = node
        self.sign = float(sign)
        self.label = label
        self.run_type = run_type
        self.out_dir = out_dir
        self.step_ml = step_ml

        self.rows = []
        self.mode = ""
        self.calibration_valid = None
        self.t0 = None
        self.pending_event = ""
        #: 마지막으로 받은 플랜지 자세. 힘이 일정한데 **팔이 물러났다** 는 것이
        #: 조절이 일어났다는 직접 증거이고, 그 변위 없이는 "힘이 안 올랐다" 밖에
        #: 말할 수 없다 — 안 올랐는지 애초에 오르지 않을 상황이었는지 못 가른다.
        self.pose = None

        node.create_subscription(WrenchStamped, f"{namespace}/wrench_px6d",
                                 self._on_wrench, 50)
        node.create_subscription(String, f"{namespace}/probing_mode",
                                 self._on_mode, LATCHED)
        node.create_subscription(Bool, f"{namespace}/calibration_valid",
                                 self._on_calibration, 10)
        node.create_subscription(Pose, f"{namespace}/ee_wrt_base",
                                 self._on_pose, 10)

    # -- 수집 --------------------------------------------------------------

    def _on_mode(self, msg: String) -> None:
        self.mode = msg.data

    def _on_calibration(self, msg: Bool) -> None:
        self.calibration_valid = bool(msg.data)

    def _on_pose(self, msg: Pose) -> None:
        q = msg.orientation
        self.pose = (msg.position.x, msg.position.y, msg.position.z,
                     q.x, q.y, q.z, q.w)

    def _on_wrench(self, msg: WrenchStamped) -> None:
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        f = msg.wrench.force
        normal = self.sign * f.z
        magnitude = (f.x ** 2 + f.y ** 2 + f.z ** 2) ** 0.5
        # 부호는 법선 성분이 준다 — us_diff_ik._control_force 와 같은 규약이다.
        signed = magnitude if normal >= 0.0 else -magnitude
        # 자세는 20 Hz 로 오고 렌치는 1 kHz 다. 마지막 값을 들고 간다 — 보간하면
        # 측정하지 않은 중간값을 측정한 것처럼 적게 된다.
        self.rows.append((now - self.t0, f.x, f.y, f.z, normal, signed,
                          self.mode, self.calibration_valid, self.pending_event,
                          self.pose))
        self.pending_event = ""

    def mark(self, key: str) -> str:
        """다음 표본에 사건을 붙인다. 표본 시각이 곧 사건 시각이 된다."""
        self.pending_event = key
        return KEYS.get(key, key)

    def undo(self) -> bool:
        """마지막 사건을 지운다. 표본은 남긴다 — 지우면 시계열에 구멍이 난다."""
        for index in range(len(self.rows) - 1, -1, -1):
            if self.rows[index][-1]:
                self.rows[index] = self.rows[index][:-1] + ("",)
                return True
        return False

    # -- 지령 ---------------------------------------------------------------

    def read_commanded(self, node, target_node: str, timeout_s: float = 3.0) -> dict:
        """``us_diff_ik`` 가 **지금** 들고 있는 값을 읽는다.

        설정 파일을 읽지 않는 이유는, 몇 시간 전 파일이 아니라 이 실행에서 로봇이
        실제로 무엇을 하라고 지시받았는지가 필요하기 때문이다. 실행 중에
        ``ros2 param set`` 으로 바꿨다면 그쪽이 진실이다.
        """
        names = ["contact_control.target_force_n", "contact_control.deadband_n",
                 "contact_control.admittance_b_z", "teleop.contact_probing_force_n",
                 "teleop.contact_probing_release_n", "safety.warn_contact_force_n",
                 "safety.max_contact_force_n", "ft_sensor.contact_force_mode",
                 # 대조군인지 아닌지. 이것이 메타에 없으면 두 팔이 파일에서 구별되지
                 # 않는다 — 힘 궤적만 보고 "제어가 나빴다" 와 "제어가 없었다" 를
                 # 가르는 것은 사후에 불가능하다.
                 "contact_control.force_hold_enabled"]
        client = node.create_client(GetParameters, f"{target_node}/get_parameters")
        if not client.wait_for_service(timeout_sec=timeout_s):
            return {"parameters_read": False,
                    "reason": f"{target_node} 파라미터 서비스에 닿지 못했다"}
        request = GetParameters.Request(names=names)
        future = client.call_async(request)
        import rclpy
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
        if future.result() is None:
            return {"parameters_read": False, "reason": "파라미터 응답이 없다"}
        out = {"parameters_read": True}
        for name, value in zip(names, future.result().values):
            key = name.split(".")[-1]
            out[key] = value.string_value if value.type == 4 else (
                value.double_value if value.type == 3 else
                value.integer_value if value.type == 2 else value.bool_value)
        return out

    # -- 저장 ---------------------------------------------------------------

    def save(self, meta_extra: dict) -> tuple:
        os.makedirs(self.out_dir, exist_ok=True)
        samples = os.path.join(self.out_dir, f"{self.label}_samples.csv")
        with open(samples, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t_s", "fx_n", "fy_n", "fz_n", "fn_n", "mag_n",
                             "probing_mode", "calibration_valid", "event",
                             "tip_x_m", "tip_y_m", "tip_z_m",
                             "q_x", "q_y", "q_z", "q_w"])
            for row in self.rows:
                t, fx, fy, fz, fn, mag, mode, valid, event, pose = row
                pose_cells = ([f"{v:.6f}" for v in pose] if pose
                              else ["", "", "", "", "", "", ""])
                writer.writerow([f"{t:.4f}", f"{fx:.4f}", f"{fy:.4f}", f"{fz:.4f}",
                                 f"{fn:.4f}", f"{mag:.4f}", mode,
                                 "" if valid is None else str(valid).upper(), event]
                                + pose_cells)
        meta = {"label": self.label, "run_type": self.run_type,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "samples": len(self.rows), "step_ml": self.step_ml,
                "normal_force_sign": self.sign, **meta_extra}
        meta_path = os.path.join(self.out_dir, f"{self.label}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=1)
        return samples, meta_path
