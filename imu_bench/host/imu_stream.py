"""시리얼 스트림 -> 파싱 -> 퓨전을 백그라운드 스레드에서 돌리고, 최신 상태와
플롯용 이력을 스레드 안전하게 넘겨준다.

터미널 뷰어(imu_fusion_view.py)와 GUI(imu_gui.py)가 같은 것을 쓰도록 여기 모았다.
GUI 는 그리는 데만 집중하고, 시리얼이 잠깐 밀려도 프레임을 놓치지 않는다.

이력은 두 갈래로 나뉜다:
  * 로깅  — 모든 퓨전 스텝(약 250Hz)을 그대로 남긴다. 데이터가 목적이므로 솎지 않는다.
  * 플롯  — decimate 해서 보관한다. 화면은 픽셀 수만큼만 보여줄 수 있고, 250Hz 전체를
            매 프레임 다시 그리면 GUI 가 시리얼보다 느려진다.
"""

import collections
import threading
import time

import numpy as np
import serial

import mag_calib
from fusion import (
    Madgwick, quat_angle_deg, quat_conj, quat_mul, quat_to_euler_deg, seed_quat,
)
from umi_protocol import (
    FrameParser, U32Unwrapper, decode_payload, read_board_name, read_fw_tag,
    REC_ACCEL, REC_GYRO, REC_MAG, REC_RV, REC_FUSED, REC_TOF, REC_HALL, REC_INFO,
)


class RateMeter:
    """최근 window 초 동안의 실제 수신율을 센다."""

    def __init__(self, window=2.0):
        self.window = window
        self.stamps = {}

    def tick(self, key, now):
        buf = self.stamps.setdefault(key, [])
        buf.append(now)
        cut = now - self.window
        while buf and buf[0] < cut:
            buf.pop(0)

    def hz(self, key):
        buf = self.stamps.get(key, [])
        if len(buf) < 2:
            return 0.0
        span = buf[-1] - buf[0]
        return (len(buf) - 1) / span if span > 0 else 0.0


class ImuStream(threading.Thread):
    def __init__(self, port="/dev/ttyACM0", baud=115200, cal=None, beta=0.05,
                 use_mag=True, seed=True, align_after=2.0,
                 history_s=30.0, plot_hz=50.0, logger=None):
        super().__init__(daemon=True)
        self.port_name = port
        self.baud = baud
        self.cal = cal
        self.use_mag = use_mag
        self.do_seed = seed and use_mag
        self.align_after = align_after
        self.logger = logger

        self.filt = Madgwick(beta=beta)
        self.parser = FrameParser()
        self.rate = RateMeter()
        self._unwrap = {t: U32Unwrapper() for t in
                        (REC_ACCEL, REC_GYRO, REC_MAG, REC_RV, REC_FUSED, REC_TOF, REC_HALL)}

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._capture = None          # 영점 캘리브레이션용 임시 버퍼
        self.zero = None              # ZeroReference (있으면 상대자세를 로그에 남긴다)
        self.error = None
        self.board_name = None
        # 펌웨어 태그 ('V' 응답 "FW,<tag>"). None 이면 구 펌웨어 — 자이로 보정이 꺼져 있어 cal_gyr 0 / cal_rv 1 에 머문다.
        self.fw_tag = None
        # 호스트 명령 ('S' DCD 저장, 'C' 보정 재적용) 에 대한 마지막 INFO 응답: (문자열, 수신 시각)
        self.last_info = None
        self.dcd_saved_at = None
        self._ser = None
        self._wlock = threading.Lock()

        # 최신 값
        self.acc = self.gyr = self.mag = self.mag_raw = None
        self.chip_q = None
        # BNO085 가 레코드마다 실어 보내는 보정 정확도 (0=미보정 ~ 3=완전).
        # 지금까지 파싱만 하고 버렸다 — 그래서 2026-08-20 QC 에서 '자력계가
        # 교란된 것인가, 애초에 보정이 안 된 것인가' 를 가릴 수 없었다.
        # 둘은 대처가 정반대라 (환경을 바꾸느냐 / 보정을 다시 뜨느냐) 반드시
        # 갈라야 한다.
        self.cal_status = {"acc": None, "gyr": None, "mag": None, "rv": None}
        self.align = None
        self.saw_mag = False
        self.seeded = not self.do_seed
        self.n_fuse = 0
        self.t_start = None

        # 플롯용 이력 (decimate)
        n = int(history_s * plot_hz)
        self._decim = max(1, int(round(250.0 / plot_hz)))
        self.hist = {k: collections.deque(maxlen=n) for k in
                     ("t", "ax", "ay", "az", "gx", "gy", "gz",
                      "mx", "my", "mz", "hr", "hp", "hy", "cr", "cp", "cy", "diff")}

    # ------------------------------------------------------------------ 제어
    def stop(self):
        self._stop.set()

    def reset_align(self):
        """정렬 오프셋과 seed 를 다시 잡는다 (GUI 의 [C] 키)."""
        with self._lock:
            self.align = None
            self.seeded = not self.do_seed
            self.t_start = time.time()

    def send(self, data: bytes) -> bool:
        """펌웨어에 명령 바이트를 보낸다 ('S' = DCD 를 BNO085 플래시에 저장, 'C' = 동적 보정 재적용, 'V' = 태그).
        응답은 INFO 레코드로 돌아와 last_info 에 남는다. 포트가 아직 안 열렸으면 False."""
        with self._wlock:
            ser = self._ser
            if ser is None:
                return False
            try:
                ser.write(data)
                return True
            except (serial.SerialException, OSError):
                return False

    def start_capture(self):
        """영점 캘리브레이션용 raw 샘플 수집을 시작한다."""
        with self._lock:
            self._capture = []

    def capture_count(self):
        with self._lock:
            return 0 if self._capture is None else len(self._capture)

    def stop_capture(self):
        with self._lock:
            out = self._capture
            self._capture = None
        return out or []

    def snapshot(self):
        """GUI/터미널이 그릴 최신 상태 한 벌. 이력은 numpy 배열로 복사해 넘긴다."""
        with self._lock:
            snap = {
                "acc": None if self.acc is None else self.acc.copy(),
                "gyr": None if self.gyr is None else self.gyr.copy(),
                "mag": None if self.mag is None else self.mag.copy(),
                "mag_raw": None if self.mag_raw is None else self.mag_raw.copy(),
                "host_q": self.filt.q.copy(),
                "chip_q": None if self.chip_q is None else self.chip_q.copy(),
                "aligned_q": (None if (self.align is None or self.chip_q is None)
                              else quat_mul(self.align, self.filt.q)),
                "align_ready": self.align is not None,
                "held_s": 0.0 if self.t_start is None else time.time() - self.t_start,
                "n_fuse": self.n_fuse,
                "saw_mag": self.saw_mag,
                "cal_status": dict(self.cal_status),
                "rates": {k: self.rate.hz(k) for k in
                          (REC_ACCEL, REC_GYRO, REC_MAG, REC_RV)},
                "crc_errors": self.parser.crc_errors,
                "gaps": sum(self.parser.seq_gaps.values()),
                "error": self.error,
                "board": self.board_name,
                "fw_tag": self.fw_tag,
                "last_info": self.last_info,
                "dcd_saved_at": self.dcd_saved_at,
                "hist": {k: np.asarray(v) for k, v in self.hist.items()},
                "log_n": 0 if self.logger is None else self.logger.n,
                "log_path": None if self.logger is None else self.logger.path,
                "zero": self.zero,
            }
        if self.zero is not None and snap["chip_q"] is not None:
            snap["rel_euler"] = self.zero.relative_euler_deg(snap["chip_q"])
        else:
            snap["rel_euler"] = None
        if snap["aligned_q"] is not None:
            snap["diff_deg"] = quat_angle_deg(snap["aligned_q"], snap["chip_q"])
        else:
            snap["diff_deg"] = None
        return snap

    # -------------------------------------------------------------------- 본체
    def run(self):
        # 포트가 다른 프로세스에 잡혀 있거나(직전 GUI 가 아직 닫는 중, PermissionError) 보드가 잠깐
        # 빠졌다 들어오면 열기가 실패한다. 한 번 실패로 스레드가 죽으면 GUI 를 다시 켜야 하므로,
        # stop() 전까지 2 s 마다 다시 연다. 그동안 error 에 이유를 남겨 상태줄에 보이게 한다.
        ser = None
        while not self._stop.is_set():
            try:
                # timeout 은 짧아야 한다. 50 ms 면 read() 한 번이 50 ms 분량(약 500 B)을
                # 통째로 물고 오고, 그 안의 모든 레코드가 **같은 pc_ts** 를 받는다 —
                # 타임스탬프 해상도가 통째로 50 ms 로 뭉개진다. 5 ms 면 레코드 몇 개
                # 수준이라 수신 시각이 실제 도착 시각을 따라간다.
                ser = serial.Serial(self.port_name, self.baud, timeout=0.005)
                with self._lock:
                    self.error = None
                break
            except (serial.SerialException, OSError) as e:
                with self._lock:
                    self.error = "포트를 열 수 없습니다 (2 s 후 재시도): %s" % e
                self._stop.wait(2.0)
        if ser is None:
            return

        time.sleep(0.3)
        name = read_board_name(ser)
        fw = read_fw_tag(ser)
        with self._lock:
            self.board_name = name
            self.fw_tag = fw
            self.t_start = time.time()
        with self._wlock:
            self._ser = ser

        last_fuse_us = None
        try:
            while not self._stop.is_set():
                try:
                    data = ser.read(ser.in_waiting or 1)
                except serial.SerialException as e:
                    with self._lock:
                        self.error = "시리얼 중단: %s" % e
                    break
                if not data:
                    continue
                now = time.time()
                frames = self.parser.feed(data, now)
                # 한 read() 가 여러 레코드를 물고 왔으면 전부 같은 now 를 받는다.
                # 바이트는 115200 bps 로 균일하게 도착했으므로, 청크를 전송시간만큼
                # 거슬러 올라가 프레임 순서대로 시각을 펼친다.
                span = len(data) * 10.0 / self.baud     # 8N1 -> 비트 10개/바이트
                n = len(frames)
                for i, (_, rtype, _, payload) in enumerate(frames):
                    rec = decode_payload(rtype, payload)
                    if rec is None:
                        continue
                    ts = now - span * (n - 1 - i) / max(n, 1)
                    self.rate.tick(rtype, ts)
                    last_fuse_us = self._on_record(rtype, rec, ts, last_fuse_us)
        finally:
            with self._wlock:
                self._ser = None
            ser.close()

    def _on_record(self, rtype, rec, now, last_fuse_us):
        with self._lock:
            if rtype == REC_INFO:
                text = str(rec)
                if text.startswith("DBG"):
                    return last_fuse_us
                self.last_info = (text, now)
                if text == "DCD,OK":
                    self.dcd_saved_at = now
                elif text.startswith("FW,"):
                    self.fw_tag = text[3:]
                return last_fuse_us
            if rtype == REC_GYRO:
                self.gyr = np.array([rec.x, rec.y, rec.z])
                self.cal_status["gyr"] = rec.status
            elif rtype == REC_MAG:
                self.mag_raw = np.array([rec.x, rec.y, rec.z])
                self.mag = mag_calib.apply(self.cal, self.mag_raw)
                self.saw_mag = True
                self.cal_status["mag"] = rec.status
            elif rtype == REC_RV:
                self.chip_q = np.array([rec.qw, rec.qx, rec.qy, rec.qz])
                self.cal_status["rv"] = rec.status
            elif rtype == REC_ACCEL:
                self.acc = np.array([rec.x, rec.y, rec.z])
                self.cal_status["acc"] = rec.status
                t_us = self._unwrap[rtype].unwrap(rec.evt_us)

                if not self.seeded and self.mag is not None:
                    q0 = seed_quat(self.acc, self.mag)
                    if q0 is not None:
                        self.filt.q = q0
                    self.seeded = True

                if self.gyr is not None:
                    dt = 0.005 if last_fuse_us is None else (t_us - last_fuse_us) * 1e-6
                    if 0.0 < dt < 0.5:
                        m = self.mag if (self.saw_mag and self.use_mag) else None
                        self.filt.update(self.gyr, self.acc, m, dt)
                        self.n_fuse += 1
                        self._after_fusion(now, t_us)
                    last_fuse_us = t_us
        return last_fuse_us

    def _after_fusion(self, now, dev_us):
        """락을 쥔 채 호출된다. 영점 캡처 + 정렬 갱신 + 로깅 + 플롯 이력 적재."""
        if self._capture is not None and self.chip_q is not None and self.gyr is not None:
            self._capture.append((now, self.acc.copy(), self.gyr.copy(), self.chip_q.copy()))

        host_q = self.filt.q
        hr, hp, hy = quat_to_euler_deg(host_q)

        chip_q = self.chip_q
        cr = cp = cy = diff = None
        if chip_q is not None:
            cr, cp, cy = quat_to_euler_deg(chip_q)
            # 호스트가 수렴할 시간을 준 뒤 프레임 오프셋을 한 번 고정한다.
            if self.align is None and (now - self.t_start) >= self.align_after:
                self.align = quat_mul(chip_q, quat_conj(host_q))
            if self.align is not None:
                diff = quat_angle_deg(quat_mul(self.align, host_q), chip_q)

        rel = (None, None, None)
        if self.zero is not None and chip_q is not None:
            rel = self.zero.relative_euler_deg(chip_q)

        if self.logger is not None:
            mr = self.mag_raw if self.mag_raw is not None else (None,) * 3
            mc = self.mag if self.mag is not None else (None,) * 3
            g = self.gyr if self.gyr is not None else (None,) * 3
            self.logger.append([
                now, dev_us,
                self.acc[0], self.acc[1], self.acc[2],
                g[0], g[1], g[2],
                mr[0], mr[1], mr[2],
                mc[0], mc[1], mc[2],
                host_q[0], host_q[1], host_q[2], host_q[3], hr, hp, hy,
                None if chip_q is None else chip_q[0], None if chip_q is None else chip_q[1],
                None if chip_q is None else chip_q[2], None if chip_q is None else chip_q[3],
                cr, cp, cy, diff,
                rel[0], rel[1], rel[2],
                self.cal_status["acc"], self.cal_status["gyr"],
                self.cal_status["mag"], self.cal_status["rv"],
            ])

        if self.n_fuse % self._decim:
            return
        h = self.hist
        h["t"].append(now)
        h["ax"].append(self.acc[0]); h["ay"].append(self.acc[1]); h["az"].append(self.acc[2])
        g = self.gyr if self.gyr is not None else np.zeros(3)
        h["gx"].append(g[0]); h["gy"].append(g[1]); h["gz"].append(g[2])
        m = self.mag if self.mag is not None else np.full(3, np.nan)
        h["mx"].append(m[0]); h["my"].append(m[1]); h["mz"].append(m[2])
        h["hr"].append(hr); h["hp"].append(hp); h["hy"].append(hy)
        h["cr"].append(np.nan if cr is None else cr)
        h["cp"].append(np.nan if cp is None else cp)
        h["cy"].append(np.nan if cy is None else cy)
        h["diff"].append(np.nan if diff is None else diff)
