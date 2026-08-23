"""장비 없이 수신 경로를 검증하기 위한 mock 초음파 스캐너.

2026-08-06 PCAP 에서 관찰된 TCP 5002/5003 교환만 흉내낸다. 실제 장비의
동작을 규정하는 것이 아니라, 수신기·ROS 노드·뷰어를 하드웨어 없이
end-to-end 로 돌려보기 위한 테스트 더블이다.

내보내는 영상은 **합성 패턴**이며 초음파 데이터가 아니다. 방향 확인용으로
좌상단에 밝은 사각형 마커를, 생존 확인용으로 움직이는 원을 넣는다.

    python3 -m fr5_vision.us_mock_scanner --bind 127.0.0.1
    python3 -m fr5_vision.us_mock_scanner --bind 127.0.0.1 --fps 8
"""

from __future__ import annotations

import argparse
import select
import socket
import time

from fr5_vision.us_protocol import (
    BLOCKS_PER_FRAME,
    build_block,
    CANDIDATE_FRAME_SHAPE,
    CONTROL_PORT,
    PAYLOAD_SIZE,
    SCANNER_ACTIVE,
    SCANNER_IDLE,
    SETUP_68,
    VIDEO_PORT,
)
import numpy as np


SCANNER_TICK_SECONDS = 0.05  # 관찰된 scanner -> client 4바이트 간격
DEFAULT_FPS = 8.0            # PCAP 의 초당 약 7.97 런
MARKER_SIZE = 14


def _sector_mask_and_depth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """부채꼴 마스크, 정규화 깊이, 정규화 각도를 만든다."""
    height, width = CANDIDATE_FRAME_SHAPE
    rows = np.arange(height, dtype=np.float32).reshape(-1, 1)
    cols = np.arange(width, dtype=np.float32).reshape(1, -1)

    apex_col = (width - 1) / 2.0
    dx = cols - apex_col
    dy = rows + 12.0  # apex 를 화면 위쪽 바깥에 두어 근거리 폭을 확보

    radius = np.hypot(dx, dy)
    angle = np.arctan2(dx, dy)

    half_angle = np.deg2rad(38.0)
    max_radius = float(height) + 8.0
    mask = (np.abs(angle) <= half_angle) & (radius <= max_radius)

    return mask, radius / max_radius, angle / half_angle


_SECTOR_MASK, _DEPTH, _ANGLE = _sector_mask_and_depth()


def make_synthetic_frame(index: int, rng: np.random.Generator) -> np.ndarray:
    """합성 256x256 uint8 프레임 하나를 만든다."""
    # 깊이에 따른 감쇠 + 근거리 밝기
    image = 210.0 * np.exp(-2.4 * _DEPTH)

    # 고정된 조직 경계면 3개
    tissue_layers = ((0.22, 0.020, 70.0), (0.46, 0.028, 55.0), (0.72, 0.035, 40.0))
    for depth_center, thickness, strength in tissue_layers:
        image += strength * np.exp(-(((_DEPTH - depth_center) / thickness) ** 2))

    # 움직이는 저에코 병변 (생존 확인용)
    phase = index / 24.0
    lesion_depth = 0.52 + 0.10 * np.sin(2.0 * np.pi * phase)
    lesion_angle = 0.45 * np.cos(2.0 * np.pi * phase)
    lesion = np.exp(
        -((_DEPTH - lesion_depth) / 0.075) ** 2 - ((_ANGLE - lesion_angle) / 0.22) ** 2
    )
    image *= 1.0 - 0.85 * lesion

    # 스페클
    image *= rng.gamma(shape=4.0, scale=0.25, size=CANDIDATE_FRAME_SHAPE).astype(np.float32)

    image = np.where(_SECTOR_MASK, image, 0.0)
    frame = np.clip(image, 0.0, 255.0).astype(np.uint8)

    # 방향 확인용 좌상단 마커 (뷰어에서 왼쪽 위에 보이면 방향이 맞다)
    frame[2:2 + MARKER_SIZE, 2:2 + MARKER_SIZE] = 255
    frame[2:2 + MARKER_SIZE, 2 + MARKER_SIZE:2 + 2 * MARKER_SIZE] = 0
    return frame


def frame_to_blocks(frame_id: int, frame: np.ndarray) -> list[bytes]:
    """프레임 하나를 관찰된 128개 블록으로 자른다."""
    raw = frame.tobytes()
    if len(raw) != BLOCKS_PER_FRAME * PAYLOAD_SIZE:
        raise ValueError(f"unexpected frame size: {len(raw)}")
    return [
        build_block(frame_id, index, raw[index * PAYLOAD_SIZE:(index + 1) * PAYLOAD_SIZE])
        for index in range(BLOCKS_PER_FRAME)
    ]


def _listen(bind_host: str, port: int) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((bind_host, port))
    listener.listen(1)
    return listener


def _accept_both(
    video_listener: socket.socket, control_listener: socket.socket
) -> tuple[socket.socket, socket.socket]:
    """수신기가 두 포트에 모두 붙을 때까지 기다린다 (접속 순서 무관)."""
    video_socket: socket.socket | None = None
    control_socket: socket.socket | None = None
    while video_socket is None or control_socket is None:
        paired = ((video_listener, video_socket), (control_listener, control_socket))
        pending = [listener for listener, peer in paired if peer is None]
        readable, _, _ = select.select(pending, [], [], 1.0)
        for listener in readable:
            connection, address = listener.accept()
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if listener is video_listener:
                video_socket = connection
                print(f"video  5002 <- {address[0]}:{address[1]}")
            else:
                control_socket = connection
                print(f"control 5003 <- {address[0]}:{address[1]}")
    return video_socket, control_socket


def serve_session(
    video_socket: socket.socket,
    control_socket: socket.socket,
    fps: float,
    active_delay: float,
    seed: int,
) -> int:
    """한 클라이언트 세션을 처리하고 보낸 프레임 수를 반환한다."""
    video_socket.setblocking(False)
    control_socket.setblocking(False)

    rng = np.random.default_rng(seed)
    now = time.monotonic()
    next_tick = now
    next_frame = 0.0
    setup_68_at: float | None = None
    active = False
    frame_id = 0x62
    frames_sent = 0
    frame_period = 1.0 / fps

    while True:
        readable, _, _ = select.select([video_socket, control_socket], [], [], 0.01)
        for peer in readable:
            data = peer.recv(65536)
            if not data:
                print("client closed the session")
                return frames_sent
            if peer is control_socket and SETUP_68 in data and setup_68_at is None:
                setup_68_at = time.monotonic()
                print(f"received observed 68-byte setup; going active in {active_delay:.1f}s")

        now = time.monotonic()
        if not active and setup_68_at is not None and now - setup_68_at >= active_delay:
            active = True
            next_frame = now
            print("scanner state -> active (streaming synthetic frames)")

        if now >= next_tick:
            control_socket.sendall(SCANNER_ACTIVE if active else SCANNER_IDLE)
            next_tick = now + SCANNER_TICK_SECONDS

        if active and now >= next_frame:
            frame = make_synthetic_frame(frames_sent, rng)
            for block in frame_to_blocks(frame_id, frame):
                video_socket.sendall(block)
            frames_sent += 1
            frame_id = (frame_id + 1) & 0xFF
            # 밀렸을 때 몰아치지 않도록 기준 시각을 재설정한다.
            next_frame = max(now, next_frame + frame_period)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1", help="바인딩 주소 (기본 127.0.0.1)")
    parser.add_argument("--video-port", type=int, default=VIDEO_PORT)
    parser.add_argument("--control-port", type=int, default=CONTROL_PORT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--active-delay", type=float, default=1.0,
                        help="68바이트 setup 수신 후 active 로 전환하기까지의 시간(초)")
    parser.add_argument("--seed", type=int, default=0, help="스페클 난수 시드")
    parser.add_argument("--once", action="store_true", help="세션 하나만 처리하고 종료")
    arguments = parser.parse_args()
    if arguments.fps <= 0:
        raise ValueError("--fps must be greater than zero")

    video_listener = _listen(arguments.bind, arguments.video_port)
    control_listener = _listen(arguments.bind, arguments.control_port)
    print(
        f"mock scanner listening on {arguments.bind}:{arguments.video_port}"
        f"/{arguments.control_port} (synthetic frames, not ultrasound data)"
    )

    try:
        while True:
            video_socket, control_socket = _accept_both(video_listener, control_listener)
            try:
                frames_sent = serve_session(
                    video_socket, control_socket,
                    arguments.fps, arguments.active_delay, arguments.seed,
                )
                print(f"session ended, frames_sent={frames_sent}")
            except (ConnectionError, OSError) as error:
                print(f"session ended with {type(error).__name__}: {error}")
            finally:
                video_socket.close()
                control_socket.close()
            if arguments.once:
                return 0
    except KeyboardInterrupt:
        print("\nmock scanner stopped")
        return 0
    finally:
        video_listener.close()
        control_listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
