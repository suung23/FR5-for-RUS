"""PX6D 6축 F/T 센서 프레임 프로토콜 (매뉴얼 §5.1, §5.2).

출처는 PaXini 가 2026-08-21 메일로 보낸 제품 자료 패키지의
``PX6D六维力传感器说明书.pdf`` §5 통신 프로토콜이다. RS485 와 USB 는 프레임
형식이 같고 물리 계층만 다르므로 한 모듈로 다룬다.

**CRC8 다항식은 매뉴얼에 적혀 있지 않다.** 문서의 예제 프레임에서 역산했다 —
poly 0x07, init 0x00, 반전 없음, xorout 0x00 (표준 CRC-8) 이며 헤더 ``AA 55``
를 포함한 프레임 전체가 대상이다. 6 바이트 명령 8 개와 13 바이트 응답 2 개,
총 10 개 예제에서 일치를 확인했다.

미해결로 남은 것 두 가지는 실장비 응답으로 확인해야 한다.

1. 매뉴얼의 24 바이트 데이터 응답 예제는 PDF 가 이미지라 바이트를 옮기는
   과정에서 오독이 섞였고, CRC 로는 어느 바이트가 틀렸는지 특정되지 않는다.
2. 축 순서는 CANFD 절(§5.3)의 ``(Fx, Fy, Fz, Mx, My, Mz)`` 를 따랐다. CAN
   절(§5.4)의 바이트 설명은 첫 축을 Fz 로 읽히게 쓰여 있어 서로 어긋난다.
   :data:`AXIS_ORDER` 를 실측으로 검증하기 전까지 부호·축 배정을 신뢰하지 않는다.

보유 개체(2026-08 입고)는 **USB 변형, 5V 버스파워**다. 내부 MCU 는 GigaDevice
GD32 이고 USB CDC-ACM 으로 붙는다 (``28e9:018a``, Product ``GD32-CDC_ACM``).
별도 드라이버 없이 커널 ``cdc_acm`` 이 잡으며 ``/dev/ttyACM*`` 로 뜬다.

따라서 FR5 컨트롤러의 ``ft_sensor_data`` 경로로는 값이 오지 않는다. PC 가 직접
시리얼로 읽어야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

# --- 물리 계층 ---------------------------------------------------------------

#: 매뉴얼 §5.1.1. USB 변형도 같은 설정으로 열거된다 (벤더 스크린샷의 COM42).
#: USB 변형은 CDC-ACM 이라 보레이트가 실제로는 무시될 수 있으나, 그대로 넘긴다.
SERIAL_BAUD = 921600
SERIAL_BYTESIZE = 8
SERIAL_STOPBITS = 1
SERIAL_PARITY = "N"

# --- 프레임 구조 -------------------------------------------------------------

HEADER = b"\xaa\x55"
DEFAULT_DEVICE_ID = 0x7F

#: ``AA 55`` + 설비 ID + CMD. 페이로드와 CRC 는 이 뒤에 온다.
PREFIX_SIZE = 4
CRC_SIZE = 1

CMD_SET_ID = 0x01
CMD_SET_RATE = 0x02
CMD_STREAM_START = 0x03
CMD_STREAM_STOP = 0x04
CMD_READ_ONCE = 0x05
CMD_GET_VERSION = 0x07
CMD_TARE = 0x10

#: 응답 페이로드 길이. 센서 데이터(6축 x float32)만 24 바이트이고 나머지는 8.
_RESPONSE_PAYLOAD_SIZE = {CMD_STREAM_START: 24}
_DEFAULT_PAYLOAD_SIZE = 8

WRENCH_PAYLOAD_SIZE = 24

#: §5.3 기준. §5.4 와 어긋나므로 실측 검증 전까지 잠정이다.
AXIS_ORDER = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

#: 회신 주기 설정의 데이터 바이트는 ``Freq / 4`` 다 (§5.2.2 예: 100 Hz -> 0x19).
RATE_DIVISOR = 4
MIN_RATE_HZ = RATE_DIVISOR
MAX_RATE_HZ = RATE_DIVISOR * 0xFF

#: ``CMD_STREAM_START`` 의 데이터 바이트. 매뉴얼은 1 kHz 자동 회신에 0x04 를 쓴다.
STREAM_START_DATA = 0x04


class ProtocolError(ValueError):
    """프레임이 규격을 벗어났다."""


def crc8(data: bytes) -> int:
    """헤더를 포함한 프레임 바이트열의 CRC-8 (poly 0x07, init 0x00).

    Args:
        data: CRC 바이트를 뺀 프레임 전체.

    Returns:
        0..255 범위의 검사값.
    """
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def build_command(cmd: int, data: int = 0x01, device_id: int = DEFAULT_DEVICE_ID) -> bytes:
    """단일 데이터 바이트를 갖는 명령 프레임을 만든다 (§5.2.1 (a)).

    Args:
        cmd: 명령 코드. ``CMD_*`` 상수 중 하나.
        data: 데이터 바이트. 명령마다 의미가 다르다.
        device_id: 설비 ID. 공장 기본값은 0x7F.

    Returns:
        CRC 까지 붙인 6 바이트 프레임.

    Raises:
        ProtocolError: 인자가 한 바이트에 담기지 않는다.
    """
    for name, value in (("cmd", cmd), ("data", data), ("device_id", device_id)):
        if not 0 <= value <= 0xFF:
            raise ProtocolError(f"{name} 는 0..255 여야 한다: {value}")
    body = HEADER + bytes([device_id, cmd, data])
    return body + bytes([crc8(body)])


def build_set_rate(rate_hz: int, device_id: int = DEFAULT_DEVICE_ID) -> bytes:
    """자동 회신 주기를 설정하는 프레임을 만든다.

    Args:
        rate_hz: 회신 주기. :data:`RATE_DIVISOR` 의 배수여야 한다.
        device_id: 설비 ID.

    Returns:
        6 바이트 명령 프레임.

    Raises:
        ProtocolError: 주기가 범위를 벗어나거나 나누어떨어지지 않는다.
    """
    if not MIN_RATE_HZ <= rate_hz <= MAX_RATE_HZ:
        raise ProtocolError(f"주기는 {MIN_RATE_HZ}..{MAX_RATE_HZ} Hz 여야 한다: {rate_hz}")
    if rate_hz % RATE_DIVISOR:
        raise ProtocolError(f"주기는 {RATE_DIVISOR} 의 배수여야 한다: {rate_hz}")
    return build_command(CMD_SET_RATE, rate_hz // RATE_DIVISOR, device_id)


def decode_wrench(payload: bytes) -> tuple[float, ...]:
    """24 바이트 페이로드를 6 축 wrench 로 푼다.

    Args:
        payload: 센서 데이터 응답의 페이로드. 32 비트 부동소수 6 개, 리틀엔디안.

    Returns:
        :data:`AXIS_ORDER` 순서의 값 6 개. 힘은 N, 모멘트는 N·m.

    Raises:
        ProtocolError: 길이가 24 바이트가 아니다.
    """
    if len(payload) != WRENCH_PAYLOAD_SIZE:
        raise ProtocolError(f"wrench 페이로드는 {WRENCH_PAYLOAD_SIZE} 바이트여야 한다: {len(payload)}")
    return struct.unpack("<6f", payload)


@dataclass(frozen=True)
class Frame:
    """검증을 통과한 수신 프레임 하나."""

    device_id: int
    cmd: int
    payload: bytes

    @property
    def is_wrench(self) -> bool:
        """센서 데이터 프레임인지."""
        return self.cmd == CMD_STREAM_START and len(self.payload) == WRENCH_PAYLOAD_SIZE

    def wrench(self) -> tuple[float, ...]:
        """6 축 값으로 푼다.

        Returns:
            :data:`AXIS_ORDER` 순서의 값 6 개.

        Raises:
            ProtocolError: 센서 데이터 프레임이 아니다.
        """
        if not self.is_wrench:
            raise ProtocolError(f"CMD 0x{self.cmd:02X} 는 센서 데이터 프레임이 아니다")
        return decode_wrench(self.payload)


class FrameParser:
    """바이트 스트림에서 프레임을 뽑는다.

    921600 bps 에서 1 kHz 로 흘러드는 스트림은 read() 경계가 프레임과 맞지
    않는다. 그래서 누적 버퍼를 두고, CRC 가 틀리면 헤더를 한 바이트 밀어
    재동기한다 — 잡음 한 번에 스트림 전체를 잃지 않기 위해서다.
    """

    def __init__(self, max_buffer: int = 4096) -> None:
        """파서를 만든다.

        Args:
            max_buffer: 버퍼 상한. 넘으면 오래된 바이트를 버린다. 헤더가 영영
                오지 않는 회선에서 메모리가 무한히 늘지 않게 한다.
        """
        self._buffer = bytearray()
        self._max_buffer = max_buffer
        self.crc_errors = 0

    def reset(self) -> None:
        """누적 버퍼를 비운다. 포트 입력을 버린 직후 재동기하는 용도다.

        ``crc_errors`` 는 그대로 둔다 — 지금까지의 회선 품질 기록이라 버퍼를
        비웠다고 없던 일이 되지는 않는다.
        """
        self._buffer.clear()

    def feed(self, chunk: bytes) -> list[Frame]:
        """바이트를 넣고 완성된 프레임을 모두 돌려준다.

        Args:
            chunk: 방금 읽은 바이트열.

        Returns:
            이번 호출로 완성된 프레임 목록. 없으면 빈 목록.
        """
        self._buffer.extend(chunk)
        if len(self._buffer) > self._max_buffer:
            del self._buffer[: len(self._buffer) - self._max_buffer]

        frames: list[Frame] = []
        while True:
            frame = self._take_one()
            if frame is None:
                return frames
            frames.append(frame)

    def _take_one(self) -> Frame | None:
        """버퍼 앞에서 프레임 하나를 떼어낸다. 아직 못 만들면 ``None``."""
        while True:
            start = self._buffer.find(HEADER)
            if start < 0:
                # 헤더 첫 바이트가 걸쳐 들어올 수 있으므로 1 바이트는 남긴다.
                del self._buffer[: max(0, len(self._buffer) - 1)]
                return None
            if start:
                del self._buffer[:start]
            if len(self._buffer) < PREFIX_SIZE:
                return None

            cmd = self._buffer[3]
            size = _RESPONSE_PAYLOAD_SIZE.get(cmd, _DEFAULT_PAYLOAD_SIZE)
            total = PREFIX_SIZE + size + CRC_SIZE
            if len(self._buffer) < total:
                return None

            candidate = bytes(self._buffer[:total])
            if crc8(candidate[:-1]) != candidate[-1]:
                # 헤더처럼 보였을 뿐이다. 한 바이트 밀어 다시 찾는다.
                self.crc_errors += 1
                del self._buffer[:1]
                continue

            del self._buffer[:total]
            return Frame(device_id=candidate[2], cmd=cmd, payload=candidate[PREFIX_SIZE:-1])
