"""US 수신 경로 회귀 테스트 — 장비 없이 mock 스캐너로 검증한다."""

from pathlib import Path
import socket
import subprocess
import sys
import time

from fr5_vision.us_mock_scanner import frame_to_blocks, make_synthetic_frame
from fr5_vision.us_protocol import (
    BLOCK_SIZE,
    BLOCKS_PER_FRAME,
    build_block,
    CANDIDATE_FRAME_SHAPE,
    CandidateFrameAssembler,
    Tcp5002BlockParser,
    UsScannerSession,
)
import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _assemble(blocks, chunk_size=None):
    """블록 바이트열을 파서/조립기에 흘려 완성된 프레임 목록을 얻는다."""
    parser, assembler = Tcp5002BlockParser(), CandidateFrameAssembler()
    stream = b"".join(blocks)
    chunks = (
        [stream]
        if chunk_size is None
        else [stream[i:i + chunk_size] for i in range(0, len(stream), chunk_size)]
    )
    frames = []
    for chunk in chunks:
        for block in parser.feed(chunk):
            frame = assembler.add(block)
            if frame is not None:
                frames.append(frame)
    return frames


def test_block_layout_matches_observed_size():
    """관찰된 525바이트(13 헤더 + 512 페이로드) 구조를 유지한다."""
    block = build_block(0x62, 3, bytes(512))
    assert len(block) == BLOCK_SIZE == 525


def test_frame_roundtrip_is_lossless():
    """256x256 프레임이 블록 왕복 후 바이트 단위로 같아야 한다."""
    original = make_synthetic_frame(0, np.random.default_rng(0))
    frames = _assemble(frame_to_blocks(0x62, original))
    assert len(frames) == 1
    assert frames[0].frame_id == 0x62
    assert np.array_equal(frames[0].as_uint8_image(), original)


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 525, 1024, 4096])
def test_parser_handles_split_reads(chunk_size):
    """TCP 는 블록 경계를 지켜주지 않는다. 어떤 분할이어도 결과가 같아야 한다."""
    original = make_synthetic_frame(1, np.random.default_rng(1))
    frames = _assemble(frame_to_blocks(0x07, original), chunk_size=chunk_size)
    assert len(frames) == 1
    assert np.array_equal(frames[0].as_uint8_image(), original)


def test_incomplete_run_yields_no_frame():
    """128블록이 다 모이기 전에는 프레임을 내면 안 된다."""
    original = make_synthetic_frame(2, np.random.default_rng(2))
    blocks = frame_to_blocks(0x40, original)
    assert len(blocks) == BLOCKS_PER_FRAME
    assert _assemble(blocks[:-1]) == []


def _wait_port(port, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(('127.0.0.1', port), 0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def test_live_session_against_mock_scanner():
    """세션 핸드셰이크 -> active -> 프레임 수신까지 loopback 으로 확인한다.

    UsScannerSession 은 관찰된 타이밍(setup 20B @0.5s, 68B @1.8s)을 그대로 쓰므로
    이 테스트는 구조상 2초 이상 걸린다.
    """
    mock = subprocess.Popen(
        [sys.executable, '-m', 'fr5_vision.us_mock_scanner', '--bind', '127.0.0.1',
         '--active-delay', '0.0'],
        cwd=PACKAGE_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        assert _wait_port(5002) and _wait_port(5003), 'mock scanner did not listen'
        session = UsScannerSession('127.0.0.1')
        session.open()
        try:
            frames, started = [], time.monotonic()
            while time.monotonic() - started < 8.0 and len(frames) < 3:
                frames.extend(session.poll(0.05))
            assert session.scanner_active, 'scanner never reported active state'
            assert len(frames) >= 3, f'expected >=3 frames, got {len(frames)}'
            assert frames[0].as_uint8_image().shape == CANDIDATE_FRAME_SHAPE
        finally:
            session.close()
    finally:
        mock.terminate()
        mock.wait(timeout=5)
