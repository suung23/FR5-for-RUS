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

# --- 프로브 프로파일 (2026-09-09) ---------------------------------------------------------
#
# 두 번째 프로브(Konted C10UR, Wi-Fi AP "US-1C …", 192.168.1.1:5002/5003) 를 pktmon 으로 캡처해 보니
# 같은 골격이다: 5aa5 헤더의 4바이트 keepalive, 20/52(68)바이트 setup, 그리고 영상 블록의 8바이트 동기
# 5aa5ff005aa5ff00 + [frame_id][block_index] + 512 payload. 다른 것은 상수와 두 가지 거동뿐이다:
#   * C10UR 은 스캔 시작을 **클라이언트가 명령**한다: 5aa58250 을 보내면 프로브가 5aa58250 으로 답하고,
#     그 뒤 5aa5a250 keepalive 로 유지한다. 정지는 5aa50250. (SL-2C 는 프로브 버튼으로 시작 → 5aa58773 수신.)
#   * 프레임이 160 블록 = 81,920 바이트 = 320 라인 × 256 깊이 표본 — scan conversion **이전**의 극좌표 데이터다
#     (행 = A-line, 행 시작 = 근거리). 뷰어가 부채꼴로 바꾼다. 여기서는 candidate 로 그대로 저장한다.
# SL-2C 블록의 prefix 앞 3바이트(00 00 00) 는 C10UR 에서 0/1/3 으로 변한다 → 동기는 8바이트 core 로 잡는다.

BLOCK_CORE = b"\x5a\xa5\xff\x00\x5a\xa5\xff\x00"
CORE_HEADER_SIZE = len(BLOCK_CORE) + 2
CORE_BLOCK_SIZE = CORE_HEADER_SIZE + PAYLOAD_SIZE      # 522: core 앞 3바이트는 블록 사이 잡음으로 건너뛴다


@dataclass(frozen=True)
class ProbeProfile:
    """프로브별로 관찰된 바이트열과 프레임 기하. 관찰된 것만 담는다 — 추정 명령은 만들지 않는다."""

    name: str
    frame_shape: tuple[int, int]        # (rows, cols) — candidate 해석 (검증 전)
    blocks_per_frame: int
    control_initial: bytes
    control_ready: bytes
    control_active: bytes
    scanner_active: bytes               # 프로브 → 클라이언트: 스캔 중
    scanner_idle: bytes
    setup_a: bytes                      # 20바이트
    setup_b: bytes                      # 68 / 52바이트
    setup_a_delay_s: float
    setup_b_delay_s: float
    ready_after_s: float                # 이 시각 이후 keepalive 를 initial → ready 로
    control_start: bytes | None = None  # 클라이언트가 스캔 시작을 명령하는 프로브만
    control_stop: bytes | None = None
    note: str = ""


SL2C = ProbeProfile(
    name="sl2c", frame_shape=CANDIDATE_FRAME_SHAPE, blocks_per_frame=BLOCKS_PER_FRAME,
    control_initial=CONTROL_INITIAL, control_ready=CONTROL_READY, control_active=CONTROL_ACTIVE,
    scanner_active=SCANNER_ACTIVE, scanner_idle=SCANNER_IDLE,
    setup_a=SETUP_20, setup_b=SETUP_68,
    setup_a_delay_s=SETUP_20_DELAY_SECONDS, setup_b_delay_s=SETUP_68_DELAY_SECONDS, ready_after_s=SETUP_68_DELAY_SECONDS,
    note="2026-08-06 Windows direct-AP PCAP. 스캔 시작은 프로브 버튼.",
)

C10UR = ProbeProfile(
    name="c10ur", frame_shape=(320, 256), blocks_per_frame=160,
    control_initial=bytes.fromhex("5aa512d0"), control_ready=bytes.fromhex("5aa52250"),
    control_active=bytes.fromhex("5aa5a250"),
    scanner_active=bytes.fromhex("5aa58250"), scanner_idle=bytes.fromhex("5aa50250"),
    setup_a=bytes.fromhex("5aa512d05ff5000000005000000000020000005a"),
    setup_b=bytes.fromhex(
        "5aa512d05ee500007f7f7f7f7f7f7f7f00000000588500002840030000500005326e00c359950000500019ff2802500000000030"),
    setup_a_delay_s=0.2, setup_b_delay_s=0.5, ready_after_s=1.1,
    control_start=bytes.fromhex("5aa58250"), control_stop=bytes.fromhex("5aa50250"),
    note="2026-09-09 pktmon (Wi-Fi 동글, WirelessUSG 2.1.4 세션). 320 라인 × 256 깊이 표본, 극좌표 candidate. 10 fps 관찰.",
)

PROFILES = {SL2C.name: SL2C, C10UR.name: C10UR}


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
    """완성된 candidate 프레임 (SL-2C 65,536 바이트 256×256, C10UR 81,920 바이트 320×256)."""

    frame_id: int
    raw_pixels: bytes
    shape: tuple[int, int] = CANDIDATE_FRAME_SHAPE

    def as_uint8_image(self) -> np.ndarray:
        """변환 없이 shape 의 uint8 로만 해석해 반환한다."""
        return np.frombuffer(self.raw_pixels, dtype=np.uint8).reshape(self.shape)


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
            # 8바이트 core 로 동기한다. SL-2C 의 앞 3바이트(00 00 00) 와 C10UR 의 가변 3바이트는 블록 사이에서 버려진다.
            prefix_index = self._buffer.find(BLOCK_CORE)
            if prefix_index < 0:
                # read 경계에 걸친 core 를 인식할 만큼만 뒤를 남긴다.
                keep = len(BLOCK_CORE) - 1
                if len(self._buffer) > keep:
                    del self._buffer[:-keep]
                break
            if prefix_index:
                del self._buffer[:prefix_index]
            if len(self._buffer) < CORE_BLOCK_SIZE:
                break

            counter = int.from_bytes(self._buffer[len(BLOCK_CORE):CORE_HEADER_SIZE], "big")
            payload = bytes(self._buffer[CORE_HEADER_SIZE:CORE_BLOCK_SIZE])
            blocks.append(Tcp5002Block(counter=counter, payload=payload))
            del self._buffer[:CORE_BLOCK_SIZE]

        return blocks


class CandidateFrameAssembler:
    """blocks_per_frame 개 블록을 candidate 프레임 하나로 묶는다 (SL-2C 128, C10UR 160)."""

    def __init__(self, blocks_per_frame: int = BLOCKS_PER_FRAME,
                 frame_shape: tuple[int, int] = CANDIDATE_FRAME_SHAPE) -> None:
        self.blocks_per_frame = blocks_per_frame
        self.frame_shape = tuple(frame_shape)
        self._frame_id: int | None = None
        self._blocks: dict[int, bytes] = {}

    def add(self, block: Tcp5002Block) -> CandidateFrame | None:
        if block.block_index >= self.blocks_per_frame:
            return None

        if self._frame_id != block.frame_id:
            self._frame_id = block.frame_id
            self._blocks = {}

        self._blocks[block.block_index] = block.payload
        if len(self._blocks) != self.blocks_per_frame:
            return None

        raw_pixels = b"".join(self._blocks[index] for index in range(self.blocks_per_frame))
        frame = CandidateFrame(frame_id=block.frame_id, raw_pixels=raw_pixels, shape=self.frame_shape)
        self._frame_id = None
        self._blocks = {}
        return frame


def now_allows_active(session: "UsScannerSession") -> bool:
    """정지 명령을 보내는 동안(≈1 s) 프로브가 아직 보내는 active 응답은 무시한다."""
    return time.monotonic() >= session._stop_until


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
        profile: ProbeProfile = SL2C,
    ) -> None:
        self.scanner_host = scanner_host
        self.video_port = video_port
        self.control_port = control_port
        self.connect_timeout = connect_timeout
        self.profile = profile
        self.video_socket: socket.socket | None = None
        self.control_socket: socket.socket | None = None
        self._started_at = 0.0
        self._next_keepalive = 0.0
        self._sent_setup_20 = False
        self._sent_setup_68 = False
        self._scanner_active = False
        self._start_requested = False
        self._stop_until = 0.0
        self._block_parser = Tcp5002BlockParser()
        self._frame_assembler = CandidateFrameAssembler(profile.blocks_per_frame, profile.frame_shape)

    @property
    def scanner_active(self) -> bool:
        return self._scanner_active

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.profile.frame_shape

    # -- 스캔 시작/정지 (클라이언트가 명령하는 프로브: C10UR) --
    @property
    def can_command_scan(self) -> bool:
        return self.profile.control_start is not None

    def start_scan(self) -> None:
        """관찰된 시작 요청을 keepalive 자리에 보낸다. 프로브가 scanner_active 로 답하면 active keepalive 로 넘어간다."""
        if not self.can_command_scan:
            raise RuntimeError(f"{self.profile.name}: 스캔 시작 명령이 관찰되지 않음 (프로브 버튼 사용)")
        self._start_requested = True
        self._stop_until = 0.0

    def stop_scan(self) -> None:
        if self.profile.control_stop is None:
            raise RuntimeError(f"{self.profile.name}: 스캔 정지 명령이 관찰되지 않음")
        self._start_requested = False
        self._scanner_active = False
        self._stop_until = time.monotonic() + 1.0        # 관찰: 정지 바이트를 ~1 s 보낸 뒤 ready 로

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
        self._start_requested = False
        self._stop_until = 0.0
        self._block_parser = Tcp5002BlockParser()
        self._frame_assembler = CandidateFrameAssembler(self.profile.blocks_per_frame, self.profile.frame_shape)

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
        pr = self.profile
        elapsed = now - self._started_at

        if not self._sent_setup_20 and elapsed >= pr.setup_a_delay_s:
            self.control_socket.sendall(pr.setup_a)
            self._sent_setup_20 = True
        if not self._sent_setup_68 and elapsed >= pr.setup_b_delay_s:
            self.control_socket.sendall(pr.setup_b)
            self._sent_setup_68 = True

        if now < self._next_keepalive:
            return
        self.video_socket.sendall(VIDEO_KEEPALIVE)
        if now < self._stop_until and pr.control_stop is not None:
            self.control_socket.sendall(pr.control_stop)
        elif self._scanner_active:
            self.control_socket.sendall(pr.control_active)
        elif self._start_requested and pr.control_start is not None:
            self.control_socket.sendall(pr.control_start)
        elif self._sent_setup_68 and elapsed >= pr.ready_after_s:
            self.control_socket.sendall(pr.control_ready)
        else:
            self.control_socket.sendall(pr.control_initial)
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
                if self.profile.scanner_active in data and not self._scanner_active and now_allows_active(self):
                    self._scanner_active = True
                continue
            for block in self._block_parser.feed(data):
                frame = self._frame_assembler.add(block)
                if frame is not None:
                    frames.append(frame)
        return frames
