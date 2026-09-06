"""PX6D 센서 값을 실시간으로 그리는 뷰어.

``px6d_probe`` 가 "붙는가"를 보는 도구라면, 이쪽은 "무엇이 찍히는가"를 보는
도구다. 1 kHz 스트림을 받아 스크롤 그래프와 축별 막대로 동시에 보여 준다.

    ros2 run fr5_control px6d_viz --port /dev/ttyACM0
    python3 -m fr5_control.px6d_viz --port /dev/ttyACM0 --window 20 --tare

``/dev/ttyACM*`` 는 ``dialout`` 그룹이라 세션에 그룹이 안 붙어 있으면 열리지
않는다. 그럴 때는 ``sg dialout -c "python3 -m fr5_control.px6d_viz ..."`` 로
감싼다.

**막대 화면은 축 배정 실측용이다.** :data:`~fr5_control.px6d_protocol.AXIS_ORDER`
는 매뉴얼 §5.3 과 §5.4 가 어긋나 잠정 상태다. 센서를 고정하고 한 축씩 눌러
어느 막대가 움직이는지 보면 배정과 부호가 바로 확인된다. 그 전까지 축 이름은
참고값이다.

키 조작:

===========  ==================================================================
``t``        센서 영점 보정 (CMD 0x10). 무부하 상태에서만 의미가 있다.
``z``        소프트 영점. 최근 값의 평균을 빼서 화면에서만 0 으로 만든다.
``x``        소프트 영점 해제.
``space``    일시 정지 / 재개. 수신은 계속되므로 재개하면 최신 구간이 보인다.
``q``        종료.
===========  ==================================================================

센서 영점(``t``)과 소프트 영점(``z``)은 다르다. 앞쪽은 센서 내부 기준을 바꾸고
뒤쪽은 화면에만 적용된다. 드리프트를 보려면 ``z`` 만 쓰고 ``t`` 는 건드리지
않는다 — 센서 기준이 바뀌면 드리프트를 관측할 대상이 사라진다.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

from fr5_control.px6d_protocol import (
    AXIS_ORDER,
    build_command,
    build_set_rate,
    CMD_GET_VERSION,
    CMD_STREAM_START,
    CMD_STREAM_STOP,
    CMD_TARE,
    FrameParser,
    SERIAL_BAUD,
    STREAM_START_DATA,
)
from matplotlib import transforms
from matplotlib.ticker import MaxNLocator
import numpy as np

#: 힘 3 축과 모멘트 3 축의 경계. AXIS_ORDER 가 (F,F,F,M,M,M) 순서라는 전제다.
_FORCE_AXES = 3

#: 매뉴얼 §1.2 의 측정 범위. 눈금은 값에 맞춰 움직이므로 여기 쓰이는 건
#: 세로 눈금의 최소 폭과 포화 경고 기준이다.
_FORCE_RANGE_N = 50.0
_TORQUE_RANGE_NM = 2.0

#: 축별 선 색. 힘과 모멘트에서 같은 축은 같은 색을 쓴다 (X 빨강 / Y 초록 / Z 파랑).
_AXIS_COLORS = ("#d1495b", "#2a9d8f", "#3d6fb4") * 2

#: 명령을 보내고 응답을 기다리는 시간.
_REPLY_TIMEOUT_S = 1.0

#: 파서 누적 버퍼. 1 kHz x 29 바이트 = 29 KB/s 이므로 2 초분이 넘는다.
_PARSER_BUFFER_BYTES = 1 << 16

#: 축당 눈금 개수 상한. 눈금 그리기가 전체 다시 그리기 비용의 큰 몫이라
#: (4 축 19.8 ms -> 12.4 ms) 읽는 데 지장 없는 선까지 줄였다.
_MAX_TICKS = 5

#: 측정 범위의 이 비율을 넘으면 숫자를 빨갛게 찍는다.
_SATURATION_FRACTION = 0.8

#: 소프트 영점(``z``)이 평균을 낼 구간.
_SOFT_ZERO_WINDOW_S = 0.5

#: 세로 축이 한 번 늘어난 뒤 다시 줄어들기까지 기다리는 시간. 값이 튈 때마다
#: 눈금이 춤추면 읽을 수가 없어서, 늘리는 건 즉시 / 줄이는 건 천천히 한다.
_YLIM_SHRINK_DELAY_S = 2.0


def _parse_args(argv):
    """명령행 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="PX6D 6축 F/T 센서 실시간 뷰어")
    parser.add_argument("--port", default="/dev/ttyACM0", help="시리얼 포트 (기본 %(default)s)")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD, help="보레이트 (기본 %(default)s)")
    parser.add_argument("--rate", type=int, default=1000, help="센서 회신 주기 Hz, 4 의 배수 (기본 %(default)s)")
    parser.add_argument("--window", type=float, default=10.0, help="그래프에 남길 시간 초 (기본 %(default)s)")
    parser.add_argument("--fps", type=float, default=30.0, help="화면 갱신 주기 (기본 %(default)s)")
    parser.add_argument(
        "--plot-rate",
        type=float,
        default=100.0,
        help="그래프에 남길 표본 주기 Hz. 원본은 블록 평균으로 줄인다 (기본 %(default)s)",
    )
    parser.add_argument("--tare", action="store_true", help="시작 전 센서 영점 보정. 반드시 무부하 상태에서")
    return parser.parse_args(argv)


class Ring:
    """고정 길이 원형 버퍼. 시각 하나와 축 6 개를 같이 담는다.

    ``deque`` 로도 되지만 매 프레임 numpy 배열로 바꾸는 비용이 커서 처음부터
    배열로 들고 있는다.
    """

    def __init__(self, capacity: int, channels: int = len(AXIS_ORDER)) -> None:
        """버퍼를 만든다.

        Args:
            capacity: 담을 표본 수.
            channels: 축 수.
        """
        self._time = np.zeros(capacity)
        self._data = np.zeros((capacity, channels))
        self._capacity = capacity
        self._head = 0
        self._count = 0
        self._lock = threading.Lock()

    def append(self, timestamp: float, values) -> None:
        """표본 하나를 넣는다. 가득 차 있으면 가장 오래된 것을 덮어쓴다.

        Args:
            timestamp: 수신 시각 (``time.monotonic``).
            values: 축 6 개 값.
        """
        with self._lock:
            self._time[self._head] = timestamp
            self._data[self._head] = values
            self._head = (self._head + 1) % self._capacity
            self._count = min(self._count + 1, self._capacity)

    def snapshot(self):
        """오래된 것부터 정렬된 사본을 돌려준다.

        Returns:
            ``(시각 배열, 값 배열)``. 비어 있으면 길이 0 인 배열 두 개.
        """
        with self._lock:
            if self._count < self._capacity:
                return self._time[: self._count].copy(), self._data[: self._count].copy()
            order = np.r_[self._head:self._capacity, 0:self._head]
            return self._time[order].copy(), self._data[order].copy()


class SensorReader(threading.Thread):
    """시리얼에서 프레임을 읽어 링 버퍼를 채우는 스레드.

    화면 갱신이 30 fps 인데 데이터는 1 kHz 로 들어오므로, 읽기를 그리기와 같은
    루프에 두면 버퍼가 밀린다. 그래서 읽기는 이 스레드가 전담하고 그리기는
    링 버퍼의 사본만 본다.

    시리얼 쓰기도 전부 이 스레드가 한다. 키 입력은 :meth:`send` 로 큐에 넣기만
    하고, 실제 write 는 read 루프 사이에서 일어난다 — 두 스레드가 같은 포트에
    동시에 쓰지 않게 하기 위해서다.
    """

    def __init__(self, port, ring: Ring, decimation: int) -> None:
        """스레드를 만든다.

        Args:
            port: 이미 열린 ``serial.Serial``.
            ring: 채울 버퍼.
            decimation: 원본 표본 몇 개를 평균해 한 점으로 남길지.
        """
        super().__init__(daemon=True)
        self._port = port
        self._ring = ring
        self._decimation = max(1, decimation)
        # 기본 4 KB 버퍼는 921600 bps 에서 45 ms 분량이다. 그리기 스레드가 GIL 을
        # 오래 쥐면 그 사이 들어온 바이트가 잘려 프레임이 조용히 사라진다.
        self._parser = FrameParser(max_buffer=_PARSER_BUFFER_BYTES)
        self._commands: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        self._accum = np.zeros(len(AXIS_ORDER))
        self._accum_count = 0

        # 상태 표시줄용. 그리기 스레드가 읽지만 정수·실수 한 개씩이라 락은 두지 않는다.
        self.latest = np.zeros(len(AXIS_ORDER))
        self.total_frames = 0
        self.measured_hz = 0.0
        self.error: str | None = None
        self._hz_mark = (time.monotonic(), 0)

    def send(self, cmd: int, data: int = 0x01) -> None:
        """명령을 큐에 넣는다. 실제 전송은 읽기 루프가 한다.

        Args:
            cmd: ``CMD_*`` 상수.
            data: 데이터 바이트.
        """
        self._commands.put(build_command(cmd, data))

    def stop(self) -> None:
        """루프를 멈춘다. 스트림 정지 명령은 :meth:`run` 이 나가면서 보낸다."""
        self._stop_event.set()

    def run(self) -> None:
        """포트가 닫히거나 :meth:`stop` 이 불릴 때까지 읽는다."""
        try:
            while not self._stop_event.is_set():
                self._flush_commands()
                chunk = self._port.read(self._port.in_waiting or 1)
                if chunk:
                    self._consume(chunk)
        except Exception as exc:  # 포트가 뽑히면 SerialException 이 난다.
            self.error = str(exc)
        finally:
            try:
                self._port.write(build_command(CMD_STREAM_STOP, 0x00))
            except Exception:
                pass

    def _flush_commands(self) -> None:
        """큐에 쌓인 명령을 모두 보낸다."""
        while True:
            try:
                self._port.write(self._commands.get_nowait())
            except queue.Empty:
                return

    def _consume(self, chunk: bytes) -> None:
        """바이트를 프레임으로 풀어 버퍼에 넣는다."""
        now = time.monotonic()
        for frame in self._parser.feed(chunk):
            if not frame.is_wrench:
                continue
            values = np.asarray(frame.wrench())
            self.latest = values
            self.total_frames += 1

            # 블록 평균으로 표본을 줄인다. 솎아내기(stride)와 달리 잡음 크기가
            # 그대로 보이므로, 화면에서 읽은 노이즈 폭을 스펙과 비교할 수 있다.
            self._accum += values
            self._accum_count += 1
            if self._accum_count >= self._decimation:
                self._ring.append(now, self._accum / self._accum_count)
                self._accum[:] = 0.0
                self._accum_count = 0

        elapsed = now - self._hz_mark[0]
        if elapsed >= 0.5:
            self.measured_hz = (self.total_frames - self._hz_mark[1]) / elapsed
            self._hz_mark = (now, self.total_frames)

    @property
    def crc_errors(self) -> int:
        """지금까지의 CRC 불일치 횟수."""
        return self._parser.crc_errors


class Visualizer:
    """스크롤 그래프 두 개와 축별 막대를 그린다."""

    def __init__(self, reader: SensorReader, ring: Ring, args) -> None:
        """화면을 구성한다.

        Args:
            reader: 상태 표시줄에 쓸 수치를 갖는 리더.
            ring: 그릴 데이터.
            args: 명령행 인자.
        """
        import matplotlib.pyplot as plt

        self._reader = reader
        self._ring = ring
        self._window = args.window
        self._paused = False
        self._offset = np.zeros(len(AXIS_ORDER))
        self._ylim = [None, None]  # 첫 프레임에서 반드시 한 번 설정되게 한다.
        self._ylim_peak = [(0.0, 0.0), (0.0, 0.0)]  # (값, 시각)

        self.fig = plt.figure(figsize=(13, 7))
        self.fig.canvas.manager.set_window_title(f"PX6D — {args.port}")
        grid = self.fig.add_gridspec(
            2, 2, width_ratios=(3, 1), height_ratios=(1, 1), hspace=0.25, wspace=0.2,
            left=0.07, right=0.97, top=0.90, bottom=0.08,
        )

        self._trace_axes = [self.fig.add_subplot(grid[0, 0]), None]
        self._trace_axes[1] = self.fig.add_subplot(grid[1, 0], sharex=self._trace_axes[0])
        self._bar_axes = [self.fig.add_subplot(grid[0, 1]), self.fig.add_subplot(grid[1, 1])]

        self._lines = []
        for block, (axis, label) in enumerate(
            zip(self._trace_axes, ("Force [N]", "Torque [N.m]"))
        ):
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25, linewidth=0.6)
            axis.yaxis.set_major_locator(MaxNLocator(_MAX_TICKS))
            axis.spines[["top", "right"]].set_visible(False)
            axis.axhline(0.0, color="0.5", linewidth=0.8, zorder=1)
            axis.set_xlim(-self._window, 0.0)
            for channel in range(block * _FORCE_AXES, (block + 1) * _FORCE_AXES):
                line, = axis.plot([], [], linewidth=1.1, color=_AXIS_COLORS[channel])
                self._lines.append(line)
        self._trace_axes[0].tick_params(labelbottom=False)
        self._trace_axes[1].xaxis.set_major_locator(MaxNLocator(_MAX_TICKS + 2))
        self._trace_axes[1].set_xlabel("time [s]   (0 = now)")

        self._bars = []
        self._bar_texts = []
        for block, axis in enumerate(self._bar_axes):
            channels = range(block * _FORCE_AXES, (block + 1) * _FORCE_AXES)
            names = [AXIS_ORDER[c] for c in channels]
            bars = axis.barh(
                names, [0.0] * _FORCE_AXES,
                color=[_AXIS_COLORS[c] for c in channels], height=0.55,
            )
            self._bars.extend(bars)
            axis.invert_yaxis()
            axis.xaxis.set_major_locator(MaxNLocator(_MAX_TICKS))
            axis.spines[["top", "right"]].set_visible(False)
            axis.axvline(0.0, color="0.4", linewidth=0.8)
            axis.grid(True, axis="x", alpha=0.25, linewidth=0.6)
            axis.set_xlabel("N" if block == 0 else "N.m", fontsize=9)
            # 막대 축 이름을 선 색으로 칠해 범례를 대신한다. 같은 이름이 어차피
            # 여기 있으니 범례는 자리와 그리기 시간만 먹는다.
            for channel, tick in zip(channels, axis.get_yticklabels()):
                tick.set_color(_AXIS_COLORS[channel])
                tick.set_fontweight("bold")
            # x 는 축 좌표(늘 오른쪽 끝), y 는 데이터 좌표(막대 높이)로 섞는다.
            anchored = transforms.blended_transform_factory(axis.transAxes, axis.transData)
            for offset, channel in enumerate(channels):
                self._bar_texts.append(
                    axis.text(
                        0.98, offset, "", transform=anchored, ha="right", va="center",
                        fontsize=10, family="monospace",
                    )
                )

        self._status = self.fig.text(0.07, 0.955, "", fontsize=10, family="monospace")

        # 블릿 대상. 축·눈금·격자는 배경으로 한 번만 그리고 이 artist 들만 매
        # 프레임 다시 얹는다. 전체 다시 그리기는 24 ms 인데 블릿은 3.5 ms 다.
        self._animated = [*self._lines, *self._bars, *self._bar_texts, self._status]
        for artist in self._animated:
            artist.set_animated(True)
        self._background = None
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)

        self.fig.text(
            0.97, 0.955,
            "t  sensor tare      z  soft zero      x  clear      space  pause      q  quit",
            fontsize=9, ha="right", color="0.35",
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_draw(self, _event) -> None:
        """전체를 다시 그릴 때마다 배경을 새로 뜬다.

        눈금이 바뀌거나 창 크기가 바뀌면 matplotlib 이 전체를 다시 그리는데,
        그때 배경 캐시가 낡으면 옛 눈금이 화면에 남는다.
        """
        self._background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _on_key(self, event) -> None:
        """키 입력을 처리한다."""
        if event.key == "t":
            self._reader.send(CMD_TARE)
        elif event.key == "z":
            times, data = self._ring.snapshot()
            if len(times):
                recent = data[times >= times[-1] - _SOFT_ZERO_WINDOW_S]
                self._offset = recent.mean(axis=0)
        elif event.key == "x":
            self._offset = np.zeros(len(AXIS_ORDER))
        elif event.key == " ":
            self._paused = not self._paused

    def update(self) -> None:
        """한 화면분을 갱신한다. 타이머가 부른다."""
        rescaled = False
        if not self._paused:
            times, data = self._ring.snapshot()
            if len(times):
                rescaled = self._redraw_traces(times, data - self._offset)
        self._redraw_bars(self._reader.latest - self._offset)
        self._redraw_status()

        canvas = self.fig.canvas
        if rescaled or self._background is None:
            # 눈금이 바뀌었으니 배경부터 다시 만든다. _on_draw 가 새 배경을 뜬다.
            # 전체 다시 그리기는 animated artist 를 건너뛰므로 여기서 끝내면
            # 그 프레임만 선이 사라진다. 그래서 이어서 블릿까지 한다.
            canvas.draw()
        canvas.restore_region(self._background)
        for artist in self._animated:
            self.fig.draw_artist(artist)
        canvas.blit()
        canvas.flush_events()

    def _redraw_traces(self, times, data) -> bool:
        """스크롤 그래프를 갱신한다. x 축은 현재를 0 으로 둔 상대 시간이다.

        Returns:
            세로 눈금을 바꿨는지. 바꿨다면 배경을 다시 떠야 한다.
        """
        relative = times - times[-1]
        visible = relative >= -self._window
        relative = relative[visible]
        data = data[visible]

        for channel, line in enumerate(self._lines):
            line.set_data(relative, data[:, channel])

        now = time.monotonic()
        rescaled = False
        for block, axis in enumerate(self._trace_axes):
            columns = data[:, block * _FORCE_AXES:(block + 1) * _FORCE_AXES]
            span = float(np.abs(columns).max()) if columns.size else 0.0
            before = self._ylim[block]
            limits = self._sticky_ylim(block, span, now)
            # set_ylim 은 값이 같아도 축을 통째로 무효화한다. 바뀐 때만 부른다.
            if self._ylim[block] != before:
                axis.set_ylim(*limits)
                # 막대도 같은 눈금을 쓴다. 두 그림의 크기 감각이 어긋나지 않게.
                self._bar_axes[block].set_xlim(*limits)
                rescaled = True
        return rescaled

    def _sticky_ylim(self, block: int, span: float, now: float):
        """세로 눈금을 정한다. 늘릴 때는 즉시, 줄일 때는 뜸을 들인다.

        Args:
            block: 0 이면 힘, 1 이면 모멘트.
            span: 이번 화면의 최대 절대값.
            now: 현재 시각.

        Returns:
            ``(아래, 위)`` 눈금 쌍.
        """
        floor = (_FORCE_RANGE_N if block == 0 else _TORQUE_RANGE_NM) * 0.02
        wanted = max(span * 1.2, floor)
        peak, marked = self._ylim_peak[block]
        if wanted >= peak:
            self._ylim_peak[block] = (wanted, now)
            self._ylim[block] = wanted
        elif now - marked > _YLIM_SHRINK_DELAY_S:
            self._ylim_peak[block] = (wanted, now)
            self._ylim[block] = wanted
        return -self._ylim[block], self._ylim[block]

    def _redraw_bars(self, latest) -> None:
        """막대와 숫자를 갱신한다."""
        for channel, (bar, text) in enumerate(zip(self._bars, self._bar_texts)):
            value = float(latest[channel])
            bar.set_width(value)
            if channel < _FORCE_AXES:
                unit, full_scale = "N", _FORCE_RANGE_N
            else:
                unit, full_scale = "N.m", _TORQUE_RANGE_NM
            text.set_text(f"{value:+8.3f} {unit}")
            # 숫자를 막대 반대편에 둔다. 같은 편에 두면 막대가 커질 때 글자를 덮는다.
            if value >= 0.0:
                text.set_position((0.02, text.get_position()[1]))
                text.set_horizontalalignment("left")
            else:
                text.set_position((0.98, text.get_position()[1]))
                text.set_horizontalalignment("right")
            # 막대 눈금이 값에 따라 움직이니 포화가 눈에 안 띈다. 색으로 알린다.
            text.set_color("#c1121f" if abs(value) > _SATURATION_FRACTION * full_scale else "0.15")

    def _redraw_status(self) -> None:
        """상태 표시줄을 갱신한다."""
        if self._reader.error:
            self._status.set_text(f"serial error: {self._reader.error}")
            self._status.set_color("#c1121f")
            return
        zeroed = "soft-zero ON " if self._offset.any() else "soft-zero off"
        paused = "   [PAUSED]" if self._paused else ""
        self._status.set_text(
            f"{self._reader.measured_hz:6.1f} Hz   frames {self._reader.total_frames:,}"
            f"   CRC errors {self._reader.crc_errors}   {zeroed}{paused}"
        )


def _read_version(port, timeout: float = _REPLY_TIMEOUT_S) -> str | None:
    """버전 문자열을 읽는다. 스트림을 켜기 전에 한 번만 부른다.

    Args:
        port: 열린 시리얼 포트.
        timeout: 응답 대기 시간.

    Returns:
        버전 문자열. 응답이 없으면 ``None``.
    """
    parser = FrameParser()
    port.write(build_command(CMD_GET_VERSION))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in parser.feed(port.read(port.in_waiting or 1)):
            if frame.cmd == CMD_GET_VERSION:
                return frame.payload.rstrip(b"\x00").lstrip(b"\x00").decode("ascii", "replace")
    return None


def main(argv=None) -> int:
    """센서에 붙어 스트림을 켜고 창을 띄운다.

    Args:
        argv: 명령행 인자. ``None`` 이면 :data:`sys.argv` 를 쓴다.

    Returns:
        종료 코드. 0 은 정상 종료.
    """
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    try:
        import serial
    except ImportError:
        print("pyserial 이 필요하다: pip install pyserial", file=sys.stderr)
        return 2

    try:
        port = serial.Serial(args.port, args.baud, timeout=0.05)
    except serial.SerialException as exc:
        print(f"{args.port} 를 열 수 없다: {exc}", file=sys.stderr)
        print('권한 문제라면 sg dialout -c "..." 로 감싼다.', file=sys.stderr)
        return 2

    import matplotlib.pyplot as plt

    with port:
        # 이전 세션이 자동 회신을 켜둔 채 죽었을 수 있다. 조용한 상태에서 시작한다.
        port.write(build_command(CMD_STREAM_STOP, 0x00))
        time.sleep(0.1)
        port.reset_input_buffer()

        version = _read_version(port)
        print(f"펌웨어 버전: {version!r}" if version else "버전 응답이 없다 — 배선을 의심한다")

        if args.tare:
            print("영점 보정 — 무부하 상태여야 한다")
            print("⚠️ 센서 쪽 영점을 다시 쓴다. 저장된 교정 프로파일의 전자영점\n"
                  "   (~/.ros/fr5_px6d_calibration.json 의 bias) 이 그 순간 무효가 되고,\n"
                  "   보상은 있지도 않은 오프셋을 계속 빼게 된다 — 자세와 무관한 수 N 의\n"
                  "   잔차로 나타난다. 이걸 보냈으면 --calib 로 교정을 다시 하라.",
                  file=sys.stderr)
            port.write(build_command(CMD_TARE))
            time.sleep(0.5)
            port.reset_input_buffer()

        port.write(build_set_rate(args.rate))
        time.sleep(0.05)
        port.write(build_command(CMD_STREAM_START, STREAM_START_DATA))

        # 원본 주기를 --plot-rate 까지 블록 평균으로 줄인다. 1 kHz 를 그대로
        # 그리면 창 하나에 수만 점이 쌓여 화면이 데이터를 못 따라간다.
        decimation = max(1, round(args.rate / args.plot_rate))
        ring = Ring(capacity=max(2, int(args.window * args.rate / decimation)))

        reader = SensorReader(port, ring, decimation)
        reader.start()

        viz = Visualizer(reader, ring, args)
        # FuncAnimation 대신 백엔드 타이머를 쓴다. 블릿 배경을 언제 다시 뜰지는
        # 눈금이 바뀌었는지에 달려 있어서 Visualizer 가 직접 쥐고 있어야 한다.
        timer = viz.fig.canvas.new_timer(interval=int(1000.0 / args.fps))
        timer.add_callback(viz.update)
        timer.start()
        try:
            plt.show()
        finally:
            timer.stop()
            reader.stop()
            reader.join(timeout=1.0)

    if reader.error:
        print(f"시리얼 오류: {reader.error}", file=sys.stderr)
        return 1
    print(f"프레임 {reader.total_frames:,} 개, CRC 불일치 {reader.crc_errors} 회")
    return 0


if __name__ == "__main__":
    sys.exit(main())
