"""TCP 5002/5003 초음파 스캐너 프로토콜 (관찰 기반).

`tracer/ubuntu_22_04/protocol/tcp5002_offline.py` 와
`tracer/ubuntu_22_04/receiver/direct_tcp_receiver.py` 에서 벤더링했다.
`tracer/` 는 별도 리포이고 이 워크스페이스의 `.gitignore` 대상이므로,
ROS 패키지가 자립하려면 복사본이 필요하다.

여기 있는 바이트열은 2026-08-06 Windows direct-AP PCAP 에서 **관찰된 것만**
담는다. 추정한 장비 제어 명령은 만들지 않는다.

수신 프레임은 방향/scan conversion 이 FrameBridge 출력과 대조 검증되기
전까지 candidate(후보) 이미지이며 임상 판독이나 측정에 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import select
import socket
import time

import numpy as np


# --- 관찰된 TCP 5002 블록 구조 -------------------------------------------------

BLOCK_PREFIX = b"\x00\x00\x00\x5a\xa5\xff\x00\x5a\xa5\xff\x00"
HEADER_SIZE = len(BLOCK_PREFIX) + 2
PAYLOAD_SIZE = 512
BLOCK_SIZE = HEADER_SIZE + PAYLOAD_SIZE
BLOCKS_PER_FRAME = 128
CANDIDATE_FRAME_SHAPE = (256, 256)

# --- 관찰된 세션 바이트열 ------------------------------------------------------

VIDEO_PORT = 5002
CONTROL_PORT = 5003
KEEPALIVE_INTERVAL_SECONDS = 0.2

VIDEO_KEEPALIVE = b"\x00\x00\x00\x00"
CONTROL_INITIAL = bytes.fromhex("5aa50373")
CONTROL_READY = bytes.fromhex("5aa52373")
CONTROL_ACTIVE = bytes.fromhex("5aa5a373")
SCANNER_IDLE = bytes.fromhex("5aa50773")
SCANNER_ACTIVE = bytes.fromhex("5aa58773")
SETUP_20 = bytes.fromhex("5aa503735ff5000000003c000000000000000070")
SETUP_68 = bytes.fromhex(
    "5aa523735ee500007f7f7f7f7f7f7f7f00000000588500005b59010200a00000000000"
    "cc272b2f33383c4044484c5054595d616559950000400219403c008c00000000af"
)

SETUP_20_DELAY_SECONDS = 0.5
SETUP_68_DELAY_SECONDS = 1.8


@dataclass(frozen=True)
class Tcp5002Block:
    """관찰된 525바이트 애플리케이션 블록 하나."""

    counter: int
    payload: bytes

    @property
    def frame_id(self) -> int:
        return self.counter >> 8

    @property
    def block_index(self) -> int:
        return self.counter & 0xFF


@dataclass(frozen=True)
class CandidateFrame:
    """완성된 65,536바이트 candidate 프레임."""

    frame_id: int
    raw_pixels: bytes

    def as_uint8_image(self) -> np.ndarray:
        """변환 없이 256x256 uint8 로만 해석해 반환한다."""
        return np.frombuffer(self.raw_pixels, dtype=np.uint8).reshape(CANDIDATE_FRAME_SHAPE)


def build_block(frame_id: int, block_index: int, payload: bytes) -> bytes:
    """관찰된 형식대로 525바이트 블록을 만든다 (mock 스캐너 및 테스트용)."""
    if len(payload) != PAYLOAD_SIZE:
        raise ValueError(f"payload must be {PAYLOAD_SIZE} bytes, got {len(payload)}")
    counter = ((frame_id & 0xFF) << 8) | (block_index & 0xFF)
    return BLOCK_PREFIX + counter.to_bytes(2, "big") + payload


class Tcp5002BlockParser:
    """임의 크기 TCP read 를 관찰된 블록 단위로 잘라낸다."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, tcp_bytes: bytes) -> list[Tcp5002Block]:
        self._buffer.extend(tcp_bytes)
        blocks: list[Tcp5002Block] = []

        while True:
            prefix_index = self._buffer.find(BLOCK_PREFIX)
            if prefix_index < 0:
                # read 경계에 걸친 prefix 를 인식할 만큼만 뒤를 남긴다.
                keep = len(BLOCK_PREFIX) - 1
                if len(self._buffer) > keep:
                    del self._buffer[:-keep]
                break
            if prefix_index:
                del self._buffer[:prefix_index]
            if len(self._buffer) < BLOCK_SIZE:
                break

            counter = int.from_bytes(self._buffer[len(BLOCK_PREFIX):HEADER_SIZE], "big")
            payload = bytes(self._buffer[HEADER_SIZE:BLOCK_SIZE])
            blocks.append(Tcp5002Block(counter=counter, payload=payload))
            del self._buffer[:BLOCK_SIZE]

        return blocks


class CandidateFrameAssembler:
    """128개 블록을 candidate 프레임 하나로 묶는다."""

    def __init__(self) -> None:
        self._frame_id: int | None = None
        self._blocks: dict[int, bytes] = {}

    def add(self, block: Tcp5002Block) -> CandidateFrame | None:
        if block.block_index >= BLOCKS_PER_FRAME:
            return None

        if self._frame_id != block.frame_id:
            self._frame_id = block.frame_id
            self._blocks = {}

        self._blocks[block.block_index] = block.payload
        if len(self._blocks) != BLOCKS_PER_FRAME:
            return None

        raw_pixels = b"".join(self._blocks[index] for index in range(BLOCKS_PER_FRAME))
        frame = CandidateFrame(frame_id=block.frame_id, raw_pixels=raw_pixels)
        self._frame_id = None
        self._blocks = {}
        return frame


class UsScannerSession:
    """PCAP 에서 관찰된 TCP 교환만 수행하는 수신 세션.

    scan/gain 등 장비 설정 명령은 노출하지 않으며 보내지도 않는다.
    """

    def __init__(
        self,
        scanner_host: str,
        video_port: int = VIDEO_PORT,
        control_port: int = CONTROL_PORT,
        connect_timeout: float = 3.0,
    ) -> None:
        self.scanner_host = scanner_host
        self.video_port = video_port
        self.control_port = control_port
        self.connect_timeout = connect_timeout
        self.video_socket: socket.socket | None = None
        self.control_socket: socket.socket | None = None
        self._started_at = 0.0
        self._next_keepalive = 0.0
        self._sent_setup_20 = False
        self._sent_setup_68 = False
        self._scanner_active = False
        self._block_parser = Tcp5002BlockParser()
        self._frame_assembler = CandidateFrameAssembler()

    @property
    def scanner_active(self) -> bool:
        return self._scanner_active

    @property
    def is_open(self) -> bool:
        return self.video_socket is not None and self.control_socket is not None

    def open(self) -> None:
        self.video_socket = socket.create_connection(
            (self.scanner_host, self.video_port), self.connect_timeout
        )
        try:
            self.control_socket = socket.create_connection(
                (self.scanner_host, self.control_port), self.connect_timeout
            )
        except Exception:
            self.video_socket.close()
            self.video_socket = None
            raise

        self.video_socket.setblocking(False)
        self.control_socket.setblocking(False)
        self._started_at = time.monotonic()
        self._next_keepalive = self._started_at
        self._sent_setup_20 = False
        self._sent_setup_68 = False
        self._scanner_active = False
        self._block_parser = Tcp5002BlockParser()
        self._frame_assembler = CandidateFrameAssembler()

    def close(self) -> None:
        for scanner_socket in (self.video_socket, self.control_socket):
            if scanner_socket is not None:
                try:
                    scanner_socket.close()
                except OSError:
                    pass
        self.video_socket = None
        self.control_socket = None

    def _send_observed_session_bytes(self, now: float) -> None:
        assert self.video_socket is not None
        assert self.control_socket is not None
        elapsed = now - self._started_at

        if not self._sent_setup_20 and elapsed >= SETUP_20_DELAY_SECONDS:
            self.control_socket.sendall(SETUP_20)
            self._sent_setup_20 = True
        if not self._sent_setup_68 and elapsed >= SETUP_68_DELAY_SECONDS:
            self.control_socket.sendall(SETUP_68)
            self._sent_setup_68 = True

        if now < self._next_keepalive:
            return
        self.video_socket.sendall(VIDEO_KEEPALIVE)
        if self._scanner_active:
            self.control_socket.sendall(CONTROL_ACTIVE)
        elif self._sent_setup_68:
            self.control_socket.sendall(CONTROL_READY)
        else:
            self.control_socket.sendall(CONTROL_INITIAL)
        self._next_keepalive = now + KEEPALIVE_INTERVAL_SECONDS

    def poll(self, timeout_seconds: float = 0.05) -> list[CandidateFrame]:
        """세션을 한 번 돌리고 이번에 완성된 candidate 프레임들을 반환한다."""
        if self.video_socket is None or self.control_socket is None:
            raise RuntimeError("scanner session is not open")

        self._send_observed_session_bytes(time.monotonic())
        readable, _, _ = select.select(
            [self.video_socket, self.control_socket], [], [], timeout_seconds
        )
        frames: list[CandidateFrame] = []
        for scanner_socket in readable:
            data = scanner_socket.recv(65536)
            if not data:
                raise ConnectionError("scanner closed the TCP session")
            if scanner_socket is self.control_socket:
                if SCANNER_ACTIVE in data and not self._scanner_active:
                    self._scanner_active = True
                continue
            for block in self._block_parser.feed(data):
                frame = self._frame_assembler.add(block)
                if frame is not None:
                    frames.append(frame)
        return frames
