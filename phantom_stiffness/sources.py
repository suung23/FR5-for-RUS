#!/usr/bin/env python3
"""세 갈래 입력을 각각 독립된 스레드로 읽는다 — 힘(PX6D), 초음파(Wi-Fi TCP), 자세(ROS).

셋은 물리 경로가 전부 다르다 (USB 시리얼 / Wi-Fi TCP / DDS). 그래서 **하나가 없어도
나머지는 돈다.** 프로브 전원이 꺼져 있거나 제어 스택이 안 떠 있는 상태에서도 힘만으로
GUI 를 띄울 수 있어야 팬텀을 올려놓고 눌러 보는 일이 가능하다.

시간축은 수집기들과 같은 규약이다 — **호스트 수신 순간의 `time.time()` (pc_unix)**.
US 프레임에는 장비 쪽 지연이 따로 있고 (2026-09-10 실측 +200 ms, `imu_bench` README),
여기서 보정하지 않는다. 기록에 원시 수신 시각을 남기고 해석에서 뺀다.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "fr5_control", "fr5_control"),
           os.path.join(_ROOT, "fr5_control", "fr5_vision"),
           os.path.join(_ROOT, "imu_bench", "host")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


#: `rclpy.init` 은 프로세스에 한 번이다. 자세와 렌치를 각각 스레드로 띄우면 둘이
#: 동시에 초기화를 시도할 수 있어 잠근다.
_RCLPY_LOCK = threading.Lock()


def ensure_rclpy():
    """rclpy 를 한 번만 초기화하고 모듈을 준다."""
    import rclpy
    with _RCLPY_LOCK:
        if not rclpy.ok():
            rclpy.init(args=[])
    return rclpy


def dedicated_executor(node):
    """노드 하나만 도는 executor.

    ``rclpy.spin_once(node)`` 는 **전역 기본 executor** 를 쓴다. 자세와 렌치를 각각
    스레드로 구독하면 둘이 같은 executor 를 돌리려다 `RuntimeError: Executor is already
    spinning` 으로 한쪽이 통째로 죽는다 — 그것도 조용히, 그 스레드만. 그래서 스레드마다
    자기 executor 를 만든다. (2026-09-10, --force-ros 를 붙이며 실제로 밟았다.)
    """
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    return ex


# --------------------------------------------------------------------------- 힘


class Px6dReader(threading.Thread):
    """PX6D 를 시리얼로 스트리밍해 최신 wrench 와 최근 이력을 들고 있다.

    Args:
        port: 시리얼 장치. 이 셀에서는 `/dev/ttyACM0` 이다 (IMU 가 ttyACM1).
        rate_hz: 회신 주기 요청값. **센서가 이 명령을 무시한다** — 아래 Note.
        history_s: 그래프와 정착 평균에 쓸 이력 길이 [s].

    Note:
        **`CMD_SET_RATE` 는 이 펌웨어(v1.0.2) 에서 듣지 않는다** (2026-09-10 실측:
        100 Hz 를 요청해도 1001.7 Hz, 1000 Hz 를 요청해도 1002.7 Hz). 프로토콜상
        4..1020 Hz 가 유효하고 명령 바이트도 정상이지만 스트림은 늘 ~1 kHz 다.
        `px6d_probe` · `px6d_monitor` 의 ``--rate`` 도 같은 이유로 실효가 없다.
        받는 쪽에서 이력을 시간으로 잘라 쓰는 편이 안전하다.

        **하드웨어 영점(`CMD_TARE`) 은 보내지 않는다.** 센서 내부 기준을 바꾸면
        저장된 교정 프로파일의 bias 가 그 순간 무효가 되고, 보상이 있지도 않은
        오프셋을 계속 뺀다. 화면 영점은 :meth:`soft_zero` 로 충분하다.
    """

    def __init__(self, port: str = "/dev/ttyACM0", baud: int | None = None,
                 rate_hz: int = 1000, history_s: float = 30.0) -> None:
        super().__init__(daemon=True, name="px6d")
        from fr5_control.px6d_protocol import SERIAL_BAUD

        self.port_name = port
        self.baud = int(baud or SERIAL_BAUD)
        self.rate_hz = int(rate_hz)
        self.history_s = float(history_s)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t: deque[float] = deque()
        self._w: deque[tuple] = deque()
        self.latest: tuple | None = None        # 원시 6 축 (Fx..Mz)
        self.zero = np.zeros(6)                 # 소프트 영점 (화면·기록 모두에 적용)
        self.frames = 0
        self.crc_errors = 0
        self.version = ""
        self.error: str | None = None
        self.connected = False
        self._rate_est = 0.0

    # -- 이력 ---------------------------------------------------------------
    def _push(self, t: float, w: tuple) -> None:
        with self._lock:
            self._t.append(t)
            self._w.append(w)
            cutoff = t - self.history_s
            while self._t and self._t[0] < cutoff:
                self._t.popleft()
                self._w.popleft()
            self.latest = w
            self.frames += 1

    def history(self) -> tuple[np.ndarray, np.ndarray]:
        """(시각 [s], wrench [N, N·m]) — 소프트 영점을 뺀 값."""
        with self._lock:
            if not self._t:
                return np.empty(0), np.empty((0, 6))
            return np.asarray(self._t), np.asarray(self._w) - self.zero

    def window(self, seconds: float) -> np.ndarray:
        """최근 `seconds` 구간의 wrench 표본 (영점 적용). 정착 평균에 쓴다."""
        t, w = self.history()
        if t.size == 0:
            return np.empty((0, 6))
        return w[t >= t[-1] - seconds]

    def current(self) -> np.ndarray | None:
        """최신 wrench (영점 적용). 아직 하나도 못 받았으면 None."""
        with self._lock:
            return None if self.latest is None else np.asarray(self.latest) - self.zero

    def soft_zero(self, seconds: float = 0.5) -> bool:
        """최근 구간 평균을 화면 영점으로 잡는다. 센서 기준은 건드리지 않는다."""
        w = self.window(seconds)
        if w.size == 0:
            return False
        with self._lock:
            self.zero = self.zero + w.mean(axis=0)
        return True

    def clear_zero(self) -> None:
        with self._lock:
            self.zero = np.zeros(6)

    @property
    def rate(self) -> float:
        return self._rate_est

    # -- 스레드 -------------------------------------------------------------
    def run(self) -> None:
        try:
            import serial
        except ImportError:
            self.error = "pyserial 없음 (pip install pyserial)"
            return
        from fr5_control.px6d_protocol import (
            build_command, build_set_rate, CMD_GET_VERSION, CMD_STREAM_START,
            CMD_STREAM_STOP, FrameParser, STREAM_START_DATA,
        )

        try:
            port = serial.Serial(self.port_name, self.baud, timeout=0.01)
        except Exception as exc:  # noqa: BLE001  (serial.SerialException 포함)
            self.error = f"{self.port_name} 열기 실패: {exc}"
            return

        parser = FrameParser()
        try:
            with port:
                # 이전 세션이 자동 회신을 켜둔 채 죽었을 수 있다.
                port.write(build_command(CMD_STREAM_STOP, 0x00))
                time.sleep(0.1)
                port.reset_input_buffer()

                port.write(build_command(CMD_GET_VERSION))
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not self.version:
                    for f in parser.feed(port.read(port.in_waiting or 1)):
                        if f.cmd == CMD_GET_VERSION:
                            self.version = f.payload.rstrip(b"\x00").decode("ascii", "replace").strip("\x00")

                port.write(build_set_rate(self.rate_hz))
                time.sleep(0.05)
                port.write(build_command(CMD_STREAM_START, STREAM_START_DATA))
                self.connected = True

                n0, t0 = 0, time.monotonic()
                while not self._stop.is_set():
                    chunk = port.read(port.in_waiting or 1)
                    if not chunk:
                        continue
                    now = time.time()
                    for f in parser.feed(chunk):
                        if f.is_wrench:
                            self._push(now, f.wrench())
                    self.crc_errors = parser.crc_errors
                    dt = time.monotonic() - t0
                    if dt >= 1.0:                       # 실효 레이트 추정
                        self._rate_est = (self.frames - n0) / dt
                        n0, t0 = self.frames, time.monotonic()
                port.write(build_command(CMD_STREAM_STOP, 0x00))
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.connected = False

    def stop(self) -> None:
        self._stop.set()


# ----------------------------------------------------------------------- 초음파


class UsReader(threading.Thread):
    """프로브 AP 에 TCP 로 붙어 candidate 극좌표 프레임을 받는다.

    저장은 언제나 **극좌표 원본 무변환**이다 (`us_frames.bin` 레이아웃 동일).
    부채꼴 변환은 화면에서만 한다 — 기하가 아직 candidate 이기 때문이다.
    """

    def __init__(self, host: str = "192.168.1.1", probe: str = "c10ur",
                 autostart_scan: bool = True) -> None:
        super().__init__(daemon=True, name="us")
        from fr5_vision.us_protocol import PROFILES

        if probe not in PROFILES:
            raise ValueError(f"알 수 없는 프로브 {probe!r} (있는 것: {sorted(PROFILES)})")
        self.host = host
        self.profile = PROFILES[probe]
        self.autostart_scan = bool(autostart_scan)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._want_scan = bool(autostart_scan)
        self._session = None
        self.latest: np.ndarray | None = None       # 극좌표 (n_lines, n_samples)
        self.latest_t = 0.0
        self.frames = 0
        self.error: str | None = None
        self.connected = False
        self._rate_est = 0.0

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.profile.frame_shape

    @property
    def rate(self) -> float:
        return self._rate_est

    @property
    def scanning(self) -> bool:
        s = self._session
        return bool(s is not None and s.scanner_active)

    def current(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return (None if self.latest is None else self.latest.copy()), self.latest_t

    def toggle_scan(self) -> bool:
        """스캔 시작/정지. C10UR 은 클라이언트가 명령한다."""
        self._want_scan = not self._want_scan
        return self._want_scan

    def run(self) -> None:
        from fr5_vision.us_protocol import UsScannerSession

        sess = UsScannerSession(self.host, profile=self.profile)
        self._session = sess
        try:
            sess.open()
            self.connected = True
        except Exception as exc:  # noqa: BLE001
            self.error = f"{self.host} 접속 실패: {exc}"
            return

        started_at = time.monotonic()
        commanded = False
        n0, t0 = 0, time.monotonic()
        try:
            while not self._stop.is_set():
                try:
                    frames = sess.poll(0.05)
                except (ConnectionError, OSError) as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    break
                now = time.time()
                for fr in frames:
                    with self._lock:
                        self.latest = fr.as_uint8_image()
                        self.latest_t = now
                        self.frames += 1
                # setup 바이트가 다 나간 뒤에만 스캔을 명령한다.
                if sess.can_command_scan and time.monotonic() - started_at > self.profile.ready_after_s:
                    if self._want_scan and not commanded:
                        sess.start_scan()
                        commanded = True
                    elif not self._want_scan and commanded:
                        sess.stop_scan()
                        commanded = False
                dt = time.monotonic() - t0
                if dt >= 1.0:
                    self._rate_est = (self.frames - n0) / dt
                    n0, t0 = self.frames, time.monotonic()
        finally:
            self.connected = False
            try:
                sess.close()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        self._stop.set()


# ------------------------------------------------------------------------ 자세


def probe_axis(quat_wxyz) -> np.ndarray:
    """쿼터니언에서 **툴 z 축이 베이스에서 보는 방향**을 뽑는다 (회전행렬 3 열).

    강성의 깊이는 이 축에 투영한 이동이다. 베이스 z 변위를 그냥 쓰면 프로브가
    기울어 있을 때 그만큼 틀린다 (`fh/analysis.axial_travel` 과 같은 규약).
    """
    w, x, y, z = (float(v) for v in quat_wxyz)
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


class PoseReader(threading.Thread):
    """제어 스택의 FK 자세를 구독한다 (`{ns}/ee_wrt_base`). 스택이 없으면 조용히 논다.

    깊이 기준은 **접촉 시점 한 번**으로 고정한다 (:meth:`set_reference`). 표본마다
    축을 다시 잡으면 프로브가 회전하는 구간에서 회전이 이동으로 섞여 든다.
    """

    def __init__(self, namespace: str = "/fr5_right", wrench_topic: str | None = None) -> None:
        super().__init__(daemon=True, name="pose")
        self.namespace = namespace
        self.wrench_topic = wrench_topic
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._node = None
        self._exec = None
        self.pos: np.ndarray | None = None
        self.quat: np.ndarray | None = None
        self.stack_wrench: np.ndarray | None = None     # 제어가 쓰는 렌치 (보상 후)
        self.ref_pos: np.ndarray | None = None
        self.ref_axis: np.ndarray | None = None
        self.error: str | None = None
        self.available = False
        self.messages = 0

    def set_reference(self) -> bool:
        """지금 자세를 깊이 0 으로 잡는다. 접촉 순간에 부른다."""
        with self._lock:
            if self.pos is None or self.quat is None:
                return False
            self.ref_pos = self.pos.copy()
            self.ref_axis = probe_axis(self.quat)
            return True

    def clear_reference(self) -> None:
        with self._lock:
            self.ref_pos = self.ref_axis = None

    def depth_mm(self) -> float | None:
        """기준에서 프로브 축 방향으로 파고든 깊이 [mm]. 양수 = 조직 쪽."""
        with self._lock:
            if self.pos is None or self.ref_pos is None or self.ref_axis is None:
                return None
            return float((self.pos - self.ref_pos) @ self.ref_axis) * 1000.0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pos": None if self.pos is None else self.pos.tolist(),
                "quat": None if self.quat is None else self.quat.tolist(),
                "stack_wrench": None if self.stack_wrench is None else self.stack_wrench.tolist(),
            }

    def run(self) -> None:
        try:
            from rclpy.node import Node
            from geometry_msgs.msg import Pose, WrenchStamped
        except ImportError as exc:
            self.error = f"rclpy 없음 — 자세 없이 돈다 ({exc})"
            return

        try:
            rclpy = ensure_rclpy()
            node = Node("stiffness_gui_pose")
            self._node = node

            def on_pose(msg) -> None:
                with self._lock:
                    self.pos = np.array([msg.position.x, msg.position.y, msg.position.z])
                    self.quat = np.array([msg.orientation.w, msg.orientation.x,
                                          msg.orientation.y, msg.orientation.z])
                    self.messages += 1

            def on_wrench(msg) -> None:
                f, t = msg.wrench.force, msg.wrench.torque
                with self._lock:
                    self.stack_wrench = np.array([f.x, f.y, f.z, t.x, t.y, t.z])

            node.create_subscription(Pose, f"{self.namespace}/ee_wrt_base", on_pose, 10)
            if self.wrench_topic:
                node.create_subscription(WrenchStamped, self.wrench_topic, on_wrench, 10)
            self.available = True
            executor = dedicated_executor(node)
            self._exec = executor
            while not self._stop.is_set() and rclpy.ok():
                executor.spin_once(timeout_sec=0.05)
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.available = False
            try:
                if self._node is not None:
                    self._node.destroy_node()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        self._stop.set()


def rotation_from_euler_xyz_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """FR5 의 `tl_cur_pos[3:6]` (**외재 xyz 오일러, 도**) 를 회전행렬로.

    `fr5_status_node` 가 `scipy` 의 ``Rotation.from_euler("xyz", …, degrees=True)`` 로
    하는 것과 같다 — 소문자 xyz 는 **고정축 기준 외재** 회전이라 ``R = Rz·Ry·Rx`` 다.
    이 venv 에는 scipy 가 없어 직접 쓴다.
    """
    a, b, c = (np.radians(v) for v in (rx_deg, ry_deg, rz_deg))
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return np.array([
        [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa],
        [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa],
        [-sb,     cb * sa,                cb * ca],
    ])


def quat_from_rotation(R: np.ndarray) -> np.ndarray:
    """회전행렬 → 쿼터니언 `[w, x, y, z]`. 수치적으로 가장 큰 성분을 기준으로 푼다."""
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])


class RpcPoseReader(threading.Thread):
    """FR5 에서 TCP 자세를 **직접** 읽는다 — ROS 없이.

    강성만 재는 데에는 제어 스택이 필요 없다. 필요한 것은 깊이 하나이고, 그것은
    로봇이 이미 UDP 실시간 스트림으로 뿌리는 `tl_cur_pos` 에 있다. `GetActualTCPPose`
    는 XML-RPC 왕복 없이 그 state package 를 바로 읽으므로 (`DESIGN_NOTES` §2.1)
    지연도 거의 없다.

    **읽기만 한다.** 이 클래스는 어떤 동작 명령도 내지 않는다. 로봇은 조작자가
    펜던트 조그나 드래그로 움직인다.

    깊이 규약은 :class:`PoseReader` 와 같다 — 접촉 시점에 고정한 프로브 축(툴 z)에
    투영한 이동. 두 경로가 같은 값을 내야 나중에 자료를 섞을 수 있다.
    """

    def __init__(self, robot_ip: str = "192.168.58.3", rate_hz: float = 30.0) -> None:
        super().__init__(daemon=True, name="rpc-pose")
        self.robot_ip = robot_ip
        self.period = 1.0 / max(1e-3, float(rate_hz))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.pos: np.ndarray | None = None
        self.quat: np.ndarray | None = None
        self.joints: list | None = None
        self.ref_pos: np.ndarray | None = None
        self.ref_axis: np.ndarray | None = None
        self.error: str | None = None
        self.available = False
        self.messages = 0
        self.stack_wrench = None            # RPC 경로에는 보상된 렌치가 없다

    # -- PoseReader 와 같은 얼굴 -------------------------------------------
    def set_reference(self) -> bool:
        with self._lock:
            if self.pos is None or self.quat is None:
                return False
            self.ref_pos = self.pos.copy()
            self.ref_axis = probe_axis(self.quat)
            return True

    def clear_reference(self) -> None:
        with self._lock:
            self.ref_pos = self.ref_axis = None

    def depth_mm(self) -> float | None:
        with self._lock:
            if self.pos is None or self.ref_pos is None or self.ref_axis is None:
                return None
            return float((self.pos - self.ref_pos) @ self.ref_axis) * 1000.0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pos": None if self.pos is None else self.pos.tolist(),
                "quat": None if self.quat is None else self.quat.tolist(),
                "joints": list(self.joints) if self.joints else None,
                "stack_wrench": None,
            }

    def run(self) -> None:
        try:
            from fr5_control.fairino import Robot
        except ImportError as exc:
            self.error = f"fairino SDK 를 못 찾는다 ({exc})"
            return
        try:
            robot = Robot.RPC(self.robot_ip)
        except Exception as exc:  # noqa: BLE001
            self.error = f"{self.robot_ip} 접속 실패: {exc}"
            return

        try:
            while not self._stop.is_set():
                state = getattr(robot, "robot_state_pkg", None)
                if state is None:
                    self.error = "state package 가 비어 있다 — 로봇 전원·네트워크 확인"
                    time.sleep(0.5)
                    continue
                p = list(state.tl_cur_pos)
                R = rotation_from_euler_xyz_deg(p[3], p[4], p[5])
                with self._lock:
                    self.pos = np.array(p[0:3]) / 1000.0      # mm → m
                    self.quat = quat_from_rotation(R)
                    self.joints = list(state.jt_cur_pos)
                    self.messages += 1
                    self.available = True
                    self.error = None
                time.sleep(self.period)
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.available = False

    def stop(self) -> None:
        self._stop.set()


class RosWrenchReader(threading.Thread):
    """제어 스택이 내는 렌치를 구독한다 — 시리얼 대신.

    **teleop 중에는 이쪽을 써야 한다.** `telemetry_bridge` 가 `/dev/ttyACM0` 을 이미
    쥐고 있어 같은 포트를 두 번 열 수 없다. 게다가 이 토픽의 값은 **중력·레버암이
    보상된, 제어가 실제로 쓰는 그 값**이라 강성을 그것으로 재는 편이 맞다
    (`force_hold_validation` 도 같은 이유로 스택의 렌치를 그대로 기록한다).

    :class:`Px6dReader` 와 같은 얼굴을 하고 있어 GUI 는 둘을 바꿔 끼우기만 하면 된다.
    """

    def __init__(self, topic: str = "/fr5_right/wrench_px6d", history_s: float = 30.0) -> None:
        super().__init__(daemon=True, name="ros-wrench")
        self.topic = topic
        self.history_s = float(history_s)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t: deque[float] = deque()
        self._w: deque[tuple] = deque()
        self.latest: tuple | None = None
        self.zero = np.zeros(6)
        self.frames = 0
        self.crc_errors = 0
        self.version = ""              # 시리얼이 아니라 펌웨어 버전을 모른다
        self.error: str | None = None
        self.connected = False
        self._rate_est = 0.0

    # -- Px6dReader 와 같은 얼굴 -------------------------------------------
    def _push(self, t: float, w: tuple) -> None:
        with self._lock:
            self._t.append(t)
            self._w.append(w)
            cutoff = t - self.history_s
            while self._t and self._t[0] < cutoff:
                self._t.popleft()
                self._w.popleft()
            self.latest = w
            self.frames += 1

    def history(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if not self._t:
                return np.empty(0), np.empty((0, 6))
            return np.asarray(self._t), np.asarray(self._w) - self.zero

    def window(self, seconds: float) -> np.ndarray:
        t, w = self.history()
        if t.size == 0:
            return np.empty((0, 6))
        return w[t >= t[-1] - seconds]

    def current(self) -> np.ndarray | None:
        with self._lock:
            return None if self.latest is None else np.asarray(self.latest) - self.zero

    def soft_zero(self, seconds: float = 0.5) -> bool:
        w = self.window(seconds)
        if w.size == 0:
            return False
        with self._lock:
            self.zero = self.zero + w.mean(axis=0)
        return True

    def clear_zero(self) -> None:
        with self._lock:
            self.zero = np.zeros(6)

    @property
    def rate(self) -> float:
        return self._rate_est

    def run(self) -> None:
        try:
            from rclpy.node import Node
            from geometry_msgs.msg import WrenchStamped
        except ImportError as exc:
            self.error = f"rclpy 없음 ({exc})"
            return
        try:
            rclpy = ensure_rclpy()
            node = Node("stiffness_gui_wrench")

            def on_wrench(msg) -> None:
                f, t = msg.wrench.force, msg.wrench.torque
                self._push(time.time(), (f.x, f.y, f.z, t.x, t.y, t.z))

            # 브리지는 기본 QoS(RELIABLE, depth 10) 로 낸다.
            node.create_subscription(WrenchStamped, self.topic, on_wrench, 10)
            self.connected = True
            executor = dedicated_executor(node)
            n0, t0 = 0, time.monotonic()
            while not self._stop.is_set() and rclpy.ok():
                executor.spin_once(timeout_sec=0.05)
                dt = time.monotonic() - t0
                if dt >= 1.0:
                    self._rate_est = (self.frames - n0) / dt
                    n0, t0 = self.frames, time.monotonic()
            node.destroy_node()
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.connected = False

    def stop(self) -> None:
        self._stop.set()
