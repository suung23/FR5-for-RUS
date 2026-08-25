"""px6d_viz 의 비 GUI 부분 회귀 테스트 — 링 버퍼와 수신 스레드.

그리기는 화면이 있어야 하니 여기서 다루지 않는다. 대신 값이 화면에 닿기 전까지
거치는 두 단계를 고정한다: 원형 버퍼의 순서와, 1 kHz 를 표시용으로 줄이는
블록 평균이다. 이 둘이 틀리면 그래프는 멀쩡해 보이면서 값이 거짓이 된다.
"""

import struct
import time

from fr5_control.px6d_protocol import (
    build_command,
    CMD_STREAM_START,
    CMD_TARE,
    crc8,
    HEADER,
)
from fr5_control.px6d_viz import Ring, SensorReader
import numpy as np
import pytest


def _wrench_frame(values) -> bytes:
    """센서 데이터 프레임 한 개를 만든다 (매뉴얼 §5.2.1 (b))."""
    body = HEADER + bytes([0x7F, CMD_STREAM_START]) + struct.pack('<6f', *values)
    return body + bytes([crc8(body)])


class _FakePort:
    """미리 정한 바이트열을 한 번에 내주는 시리얼 포트 대역.

    ``read`` 가 두 번째 호출부터 빈 바이트를 돌려주므로, 리더 스레드는 바쁘게
    돌되 데이터는 더 받지 않는다.
    """

    def __init__(self, payload: bytes = b'') -> None:
        self._payload = payload
        self.writes: list[bytes] = []

    @property
    def in_waiting(self) -> int:
        return len(self._payload)

    def read(self, _n: int) -> bytes:
        chunk, self._payload = self._payload, b''
        return chunk

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)


def _drain(reader: SensorReader, expected: int, timeout: float = 2.0) -> None:
    """리더가 프레임을 다 삼킬 때까지 기다린다."""
    deadline = time.monotonic() + timeout
    while reader.total_frames < expected and time.monotonic() < deadline:
        time.sleep(0.005)


# --- Ring -------------------------------------------------------------------

def test_ring_keeps_insertion_order():
    """넣은 순서대로 나온다."""
    ring = Ring(capacity=4, channels=2)
    for i in range(3):
        ring.append(float(i), [i, -i])
    times, data = ring.snapshot()
    assert list(times) == [0.0, 1.0, 2.0]
    assert data.tolist() == [[0, 0], [1, -1], [2, -2]]


def test_ring_overwrites_oldest_when_full():
    """가득 차면 가장 오래된 것부터 덮어쓴다."""
    ring = Ring(capacity=3, channels=1)
    for i in range(5):
        ring.append(float(i), [i])
    times, data = ring.snapshot()
    # 가장 오래된 두 개(0, 1)가 밀려나고 순서는 유지된다.
    assert list(times) == [2.0, 3.0, 4.0]
    assert data.ravel().tolist() == [2, 3, 4]


def test_ring_snapshot_is_a_copy():
    """사본이라 그리는 쪽이 만져도 버퍼가 안 변한다."""
    ring = Ring(capacity=2, channels=1)
    ring.append(0.0, [1.0])
    _, data = ring.snapshot()
    data[0, 0] = 99.0
    assert ring.snapshot()[1][0, 0] == 1.0


# --- SensorReader -----------------------------------------------------------

def test_reader_decimates_by_block_mean():
    """표본 4 개마다 평균 한 점. 솎아내기가 아니라 평균이어야 한다."""
    samples = [[float(i)] * 6 for i in range(8)]
    port = _FakePort(b''.join(_wrench_frame(s) for s in samples))
    ring = Ring(capacity=8)
    reader = SensorReader(port, ring, decimation=4)
    reader.start()
    try:
        _drain(reader, len(samples))
    finally:
        reader.stop()
        reader.join(timeout=1.0)

    assert reader.total_frames == 8
    _, data = ring.snapshot()
    # (0+1+2+3)/4 = 1.5 와 (4+5+6+7)/4 = 5.5. 솎아냈다면 3.0 과 7.0 이 나온다.
    assert data.shape == (2, 6)
    assert np.allclose(data[:, 0], [1.5, 5.5])


def test_reader_holds_back_an_incomplete_block():
    """블록이 안 차면 내보내지 않는다. 반쪽 평균은 값을 왜곡한다."""
    port = _FakePort(b''.join(_wrench_frame([1.0] * 6) for _ in range(3)))
    ring = Ring(capacity=4)
    reader = SensorReader(port, ring, decimation=4)
    reader.start()
    try:
        _drain(reader, 3)
    finally:
        reader.stop()
        reader.join(timeout=1.0)

    assert reader.total_frames == 3
    assert ring.snapshot()[0].size == 0


def test_reader_exposes_latest_sample_unaveraged():
    """숫자 표시는 최신 원본이어야 한다 — 평균이 끝나기를 기다리면 안 된다."""
    port = _FakePort(_wrench_frame([1, 2, 3, 4, 5, 6]))
    reader = SensorReader(port, Ring(capacity=4), decimation=100)
    reader.start()
    try:
        _drain(reader, 1)
    finally:
        reader.stop()
        reader.join(timeout=1.0)

    assert np.allclose(reader.latest, [1, 2, 3, 4, 5, 6])


def test_reader_sends_queued_commands_from_its_own_thread():
    """키 입력은 큐에만 넣고, 포트 쓰기는 리더 스레드가 한다."""
    port = _FakePort()
    reader = SensorReader(port, Ring(capacity=4), decimation=1)
    reader.send(CMD_TARE)
    assert port.writes == []  # 아직 스레드가 안 돌았다.

    reader.start()
    try:
        deadline = time.monotonic() + 2.0
        while build_command(CMD_TARE) not in port.writes and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        reader.stop()
        reader.join(timeout=1.0)

    assert build_command(CMD_TARE) in port.writes


def test_reader_stops_the_stream_on_exit():
    """뷰어가 죽어도 센서가 계속 떠들지 않게 한다."""
    port = _FakePort()
    reader = SensorReader(port, Ring(capacity=4), decimation=1)
    reader.start()
    reader.stop()
    reader.join(timeout=1.0)
    assert not reader.is_alive()
    assert port.writes[-1] == build_command(0x04, 0x00)  # CMD_STREAM_STOP


def test_reader_survives_a_torn_frame():
    """CRC 가 깨진 프레임 하나가 뒤따르는 프레임을 잡아먹으면 안 된다."""
    good = _wrench_frame([1.0] * 6)
    torn = bytearray(_wrench_frame([9.0] * 6))
    torn[-1] ^= 0xFF                       # CRC 만 망가뜨린다.
    port = _FakePort(bytes(torn) + good)
    ring = Ring(capacity=4)
    reader = SensorReader(port, ring, decimation=1)
    reader.start()
    try:
        _drain(reader, 1)
    finally:
        reader.stop()
        reader.join(timeout=1.0)

    assert reader.crc_errors > 0
    _, data = ring.snapshot()
    assert np.allclose(data[-1], [1.0] * 6)


if __name__ == '__main__':
    pytest.main([__file__])
