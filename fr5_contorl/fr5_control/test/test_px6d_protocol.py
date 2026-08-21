"""PX6D 프레임 프로토콜 회귀 테스트 — 매뉴얼 예제를 골든 벡터로 쓴다.

CRC8 규격이 매뉴얼에 없어 예제에서 역산했으므로, 예제와의 일치가 이 모듈의
유일한 근거다. 규격을 손대면 여기가 먼저 깨져야 한다.
"""

import struct

from fr5_control.px6d_protocol import (
    AXIS_ORDER,
    build_command,
    build_set_rate,
    CMD_GET_VERSION,
    CMD_READ_ONCE,
    CMD_SET_ID,
    CMD_STREAM_START,
    CMD_STREAM_STOP,
    CMD_TARE,
    crc8,
    decode_wrench,
    Frame,
    FrameParser,
    ProtocolError,
    STREAM_START_DATA,
    WRENCH_PAYLOAD_SIZE,
)
import pytest


def _hex(text: str) -> bytes:
    """공백이 섞인 16 진 문자열을 바이트로."""
    return bytes.fromhex(text.replace(" ", ""))


#: 매뉴얼 §5.1.3 / §5.2.2 의 명령 예제. CRC 까지 포함한 원문 그대로다.
MANUAL_COMMANDS = {
    "set_id": "AA 55 7F 01 01 AE",
    "rate_4hz": "AA 55 7F 02 01 91",
    "rate_100hz": "AA 55 7F 02 19 D9",
    "stream_start": "AA 55 7F 03 04 9F",
    "stream_stop": "AA 55 7F 04 00 E8",
    "read_once": "AA 55 7F 05 01 FA",
    "get_version": "AA 55 7F 07 01 D0",
    "tare": "AA 55 7F 10 01 EC",
}

#: 바이트를 그대로 옮길 수 있었던 응답 예제. 길이가 명령과 달라 CRC 를 교차 검증한다.
MANUAL_RESPONSES = {
    "tare_ack": "AA 55 7F 10 00 00 00 00 00 00 00 00 E2",
    "version": "AA 55 7F 07 00 76 30 2E 31 2E 31 00 5B",
}


@pytest.mark.parametrize("name", sorted(MANUAL_COMMANDS))
def test_crc_matches_manual_commands(name):
    """역산한 CRC8 이 매뉴얼 명령 예제와 일치한다."""
    frame = _hex(MANUAL_COMMANDS[name])
    assert crc8(frame[:-1]) == frame[-1]


@pytest.mark.parametrize("name", sorted(MANUAL_RESPONSES))
def test_crc_matches_manual_responses(name):
    """길이가 다른 응답 프레임에서도 같은 CRC 규격이 성립한다."""
    frame = _hex(MANUAL_RESPONSES[name])
    assert crc8(frame[:-1]) == frame[-1]


@pytest.mark.parametrize(
    "cmd,data,expected",
    [
        (CMD_SET_ID, 0x01, MANUAL_COMMANDS["set_id"]),
        (CMD_STREAM_START, STREAM_START_DATA, MANUAL_COMMANDS["stream_start"]),
        (CMD_STREAM_STOP, 0x00, MANUAL_COMMANDS["stream_stop"]),
        (CMD_READ_ONCE, 0x01, MANUAL_COMMANDS["read_once"]),
        (CMD_GET_VERSION, 0x01, MANUAL_COMMANDS["get_version"]),
        (CMD_TARE, 0x01, MANUAL_COMMANDS["tare"]),
    ],
)
def test_build_command_reproduces_manual(cmd, data, expected):
    """조립한 명령이 매뉴얼 원문 바이트와 정확히 같다."""
    assert build_command(cmd, data) == _hex(expected)


@pytest.mark.parametrize(
    "rate_hz,expected",
    [(4, MANUAL_COMMANDS["rate_4hz"]), (100, MANUAL_COMMANDS["rate_100hz"])],
)
def test_build_set_rate_reproduces_manual(rate_hz, expected):
    """주기 설정의 데이터 바이트가 Freq/4 규칙과 맞는다."""
    assert build_set_rate(rate_hz) == _hex(expected)


@pytest.mark.parametrize("rate_hz", [0, 3, 6, 1024])
def test_build_set_rate_rejects_bad_rate(rate_hz):
    """나누어떨어지지 않거나 범위를 벗어난 주기는 조용히 반올림하지 않는다."""
    with pytest.raises(ProtocolError):
        build_set_rate(rate_hz)


@pytest.mark.parametrize("bad", [-1, 256])
def test_build_command_rejects_out_of_range(bad):
    """한 바이트에 안 담기는 인자는 잘라내지 않고 거부한다."""
    with pytest.raises(ProtocolError):
        build_command(bad)


def test_decode_wrench_roundtrip():
    """리틀엔디안 float32 6 개를 축 순서대로 푼다."""
    values = (1.5, -2.25, 3.125, -0.5, 0.75, -0.0625)
    assert decode_wrench(struct.pack("<6f", *values)) == values
    assert len(AXIS_ORDER) == len(values)


@pytest.mark.parametrize("size", [0, WRENCH_PAYLOAD_SIZE - 1, WRENCH_PAYLOAD_SIZE + 1])
def test_decode_wrench_rejects_wrong_length(size):
    """길이가 어긋나면 조용히 자르지 않고 던진다."""
    with pytest.raises(ProtocolError):
        decode_wrench(bytes(size))


def _wrench_frame(values, device_id=0x7F):
    """센서 데이터 응답 프레임을 만든다 — 매뉴얼 24 바이트 예제의 대역이다."""
    body = b"\xaa\x55" + bytes([device_id, CMD_STREAM_START]) + struct.pack("<6f", *values)
    return body + bytes([crc8(body)])


def test_parser_extracts_wrench_frame():
    """온전한 프레임 하나를 그대로 되돌린다."""
    values = (0.1, 0.2, 0.3, 0.01, 0.02, 0.03)
    frames = FrameParser().feed(_wrench_frame(values))
    assert len(frames) == 1
    assert frames[0].is_wrench
    assert frames[0].wrench() == pytest.approx(values, abs=1e-6)


def test_parser_reassembles_across_chunk_boundaries():
    """read() 경계가 프레임을 가르더라도 잃지 않는다.

    921600 bps / 1 kHz 스트림에서는 경계가 프레임과 맞는 쪽이 오히려 드물다.
    """
    values = (1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
    stream = _wrench_frame(values) * 3
    parser = FrameParser()
    collected = []
    for i in range(0, len(stream), 7):
        collected.extend(parser.feed(stream[i:i + 7]))
    assert len(collected) == 3
    for frame in collected:
        assert frame.wrench() == pytest.approx(values, abs=1e-6)


def test_parser_resyncs_after_garbage():
    """잡음이나 헤더를 닮은 바이트가 섞여도 이후 프레임을 되찾는다."""
    values = (5.0, -5.0, 0.0, 0.5, -0.5, 0.0)
    parser = FrameParser()
    frames = parser.feed(b"\xaa\x55\x7f\x03rubbish" + _wrench_frame(values))
    assert len(frames) == 1
    assert frames[0].wrench() == pytest.approx(values, abs=1e-6)
    assert parser.crc_errors > 0


def test_parser_drops_frame_with_corrupted_crc():
    """CRC 가 틀린 프레임을 통과시키지 않는다."""
    corrupted = bytearray(_wrench_frame((1.0,) * 6))
    corrupted[-1] ^= 0xFF
    parser = FrameParser()
    assert parser.feed(bytes(corrupted)) == []
    assert parser.crc_errors > 0


def test_parser_handles_short_ack_frames():
    """8 바이트 페이로드 응답도 길이 표에 따라 끊어 읽는다."""
    frames = FrameParser().feed(_hex(MANUAL_RESPONSES["version"]))
    assert len(frames) == 1
    assert frames[0].cmd == CMD_GET_VERSION
    assert frames[0].payload == _hex("00 76 30 2E 31 2E 31 00")
    assert not frames[0].is_wrench


def test_parser_buffer_is_bounded():
    """헤더가 영영 오지 않아도 버퍼가 무한히 자라지 않는다."""
    parser = FrameParser(max_buffer=64)
    for _ in range(100):
        parser.feed(b"\x00" * 64)
    assert len(parser._buffer) <= 64


def test_wrench_on_non_data_frame_raises():
    """ACK 프레임을 wrench 로 읽으려 하면 막는다."""
    with pytest.raises(ProtocolError):
        Frame(device_id=0x7F, cmd=CMD_TARE, payload=bytes(8)).wrench()
