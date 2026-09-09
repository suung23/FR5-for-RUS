"""UMI 디바이스 바이너리 프로토콜 파서 (펌웨어 v2, 이중 스트림).

프레임: [0xAA][0x55][type][len][seq][payload][crc8]
  - crc8: type..payload, 다항식 0x07 (CRC-8/CCITT), init 0x00
  - 리틀엔디언, 타입별 8-bit sequence number (손실 검출용)

사용처: visualize3d.py(표시), dataset_recorder.py(무손실 기록), verify_firmware.py(검증).

타임스탬프 원칙 (TIMESTAMP_SYNC_REPORT.md):
  파서는 시각을 만들지 않는다 — 호출자가 read() 직후 찍은 PC 절대시간을 feed() 에
  넘기면, 그 청크에서 파싱된 모든 프레임에 그 수신 시각이 부여된다.
"""

import struct
from collections import namedtuple

SYNC0, SYNC1 = 0xAA, 0x55

REC_FUSED, REC_ACCEL, REC_GYRO, REC_MAG, REC_RV = 0x01, 0x10, 0x11, 0x12, 0x13
REC_TOF, REC_HALL, REC_TIMESYNC, REC_STATS, REC_INFO = 0x20, 0x21, 0x30, 0x31, 0x7F

TYPE_NAMES = {
    REC_FUSED: "FUSED", REC_ACCEL: "ACCEL", REC_GYRO: "GYRO", REC_MAG: "MAG",
    REC_RV: "RV", REC_TOF: "TOF", REC_HALL: "HALL",
    REC_TIMESYNC: "TIMESYNC", REC_STATS: "STATS", REC_INFO: "INFO",
}

Fused = namedtuple("Fused", "mcu_us qw qx qy qz dist_mm grip_pct")
ImuVec = namedtuple("ImuVec", "evt_us x y z status")
Rv = namedtuple("Rv", "evt_us qw qx qy qz acc_rad status")
Tof = namedtuple("Tof", "mcu_us range_mm range_status")
Hall = namedtuple("Hall", "mcu_us adc")
Timesync = namedtuple("Timesync", "mcu_now_us evt_last_us mcu_at_rx_us")
Stats = namedtuple("Stats", "serial_drop i2c_err imu_reset gen")

_S_FUSED = struct.Struct("<IfffffB")
_S_IMUVEC = struct.Struct("<IfffB")
_S_RV = struct.Struct("<IfffffB")
_S_TOF = struct.Struct("<IHB")
_S_HALL = struct.Struct("<IH")
_S_TSYNC = struct.Struct("<III")
_S_STATS = struct.Struct("<HHB7H")


def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def decode_payload(rec_type, payload):
    """페이로드를 타입별 namedtuple 로 해석한다. 모르는 타입/길이 불일치는 None."""
    try:
        if rec_type == REC_FUSED:
            return Fused(*_S_FUSED.unpack(payload))
        if rec_type in (REC_ACCEL, REC_GYRO, REC_MAG):
            return ImuVec(*_S_IMUVEC.unpack(payload))
        if rec_type == REC_RV:
            return Rv(*_S_RV.unpack(payload))
        if rec_type == REC_TOF:
            return Tof(*_S_TOF.unpack(payload))
        if rec_type == REC_HALL:
            return Hall(*_S_HALL.unpack(payload))
        if rec_type == REC_TIMESYNC:
            return Timesync(*_S_TSYNC.unpack(payload))
        if rec_type == REC_STATS:
            v = _S_STATS.unpack(payload)
            return Stats(v[0], v[1], v[2], v[3:])
        if rec_type == REC_INFO:
            return payload.decode("ascii", "replace")
    except struct.error:
        return None
    return None


class FrameParser:
    """스트리밍 프레임 파서. CRC 불일치 시 해당 sync 이후부터 재동기화한다.

    feed(data, pc_ts) -> [(pc_ts, type, seq, payload_bytes), ...]
    타입별 seq 를 추적해 self.seq_gaps[type] 에 누락 개수를 누적한다.
    """

    def __init__(self):
        self.buf = bytearray()
        self.crc_errors = 0
        self.junk_bytes = 0
        self.seq_gaps = {}     # type -> lost record count (from seq deltas)
        self.counts = {}       # type -> received count
        self._last_seq = {}

    def feed(self, data, pc_ts):
        out = []
        self.buf.extend(data)
        buf = self.buf
        while True:
            idx = buf.find(b"\xaa\x55")
            if idx < 0:
                self.junk_bytes += max(0, len(buf) - 1)
                del buf[:-1]           # 쪼개진 0xAA 대비 마지막 1바이트 유지
                return out
            if idx > 0:
                self.junk_bytes += idx
                del buf[:idx]
            if len(buf) < 5:
                return out
            rec_type, length, seq = buf[2], buf[3], buf[4]
            end = 5 + length + 1
            if len(buf) < end:
                return out
            if crc8(buf[2:5 + length]) == buf[5 + length]:
                payload = bytes(buf[5:5 + length])
                del buf[:end]
                last = self._last_seq.get(rec_type)
                if last is not None:
                    gap = (seq - last - 1) % 256
                    if gap:
                        self.seq_gaps[rec_type] = self.seq_gaps.get(rec_type, 0) + gap
                self._last_seq[rec_type] = seq
                self.counts[rec_type] = self.counts.get(rec_type, 0) + 1
                out.append((pc_ts, rec_type, seq, payload))
            else:
                self.crc_errors += 1
                del buf[:2]            # 이 sync 는 포기하고 다음 후보부터 재탐색


class U32Unwrapper:
    """uint32 µs 카운터(약 71.6분 주기 wrap)를 단조증가 float µs 로 펼친다."""

    def __init__(self):
        self.offset = 0
        self.last = None

    def unwrap(self, v):
        if self.last is not None and v < self.last and self.last - v > 2**31:
            self.offset += 2**32
        self.last = v
        return float(v + self.offset)


def read_board_name(ser, timeout=1.0):
    """'?' 를 보내 보드 이름을 얻는다. 바이너리 INFO(0x7F) 레코드가 정상 경로이며,
    구(CSV) 펌웨어의 'NAME,<board>' ASCII 응답도 인식한다. 실패 시 None."""
    import time
    try:
        ser.reset_input_buffer()
        ser.write(b"?")
        parser = FrameParser()
        text = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = ser.read(ser.in_waiting or 1)
            if not data:
                continue
            for _, rec_type, _, payload in parser.feed(data, 0.0):
                if rec_type == REC_INFO:
                    name = payload.decode("ascii", "replace")
                    if not name.startswith("DBG"):
                        return name
            text += data
            for line in text.split(b"\n"):
                if line.startswith(b"NAME,"):
                    return line.split(b",", 1)[1].strip().decode("ascii", "replace")
    except Exception:
        pass
    return None


def read_fw_tag(ser, timeout=1.0):
    """'V' 를 보내 펌웨어 태그를 얻는다 (INFO "FW,<tag>" 또는 구 CSV 의 "FW,<tag>" 행). 2026-09-10 이전 펌웨어는
    'V' 에 답하지 않으므로 None — 그 빌드는 자이로 동적 보정이 꺼져 있어 cal_gyr 0 / cal_rv 1 에 머문다."""
    import time
    try:
        ser.reset_input_buffer()
        ser.write(b"V")
        parser = FrameParser()
        text = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = ser.read(ser.in_waiting or 1)
            if not data:
                continue
            for _, rec_type, _, payload in parser.feed(data, 0.0):
                if rec_type == REC_INFO:
                    msg = payload.decode("ascii", "replace")
                    if msg.startswith("FW,"):
                        return msg[3:]
            text += data
            for line in text.split(b"\n"):
                if line.startswith(b"FW,"):
                    return line[3:].strip().decode("ascii", "replace")
    except Exception:
        pass
    return None
