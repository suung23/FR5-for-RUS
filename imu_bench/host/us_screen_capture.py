"""WirelessUSG 뷰어 창의 영상 영역을 화면 캡처해 candidate 프레임으로 만든다 (Windows 전용).

왜 화면 캡처인가. C10UR 프로브는 USB 로 붙으면 Cypress FX3 벤더 전용 장치(VID 04B4 / PID BC0C)
이고, 벤더 뷰어(WirelessUSG)가 CyUSB.dll 로 직접 말한다. 그 USB 프로토콜은 아직 모른다.
Wi-Fi 경로(`fr5_vision.us_protocol`, TCP 5002/5003)는 다른 프로브(SL-2C)에서 역공학한 것이라
이 프로브에 맞는지도 모른다. 오늘 데이터를 받으려면 **뷰어가 그리는 화면을 그대로 떠 오는 것**이
유일하게 확실한 길이다. 대가는 지연이다 — 프로브 획득 → 뷰어 렌더 → 화면 → 캡처 경로의 고정 지연이
붙고, 그것은 `policy_learning/scripts/inspect_session.py --latency` 로 실측해 `timing.us_latency_s`
에 넣는다 (POLICY_LEARNING_MATH §7.1). 이 모듈은 그 지연을 지우지 않는다.

산출 프레임은 기존 수집기(`us_imu_collect.py`, `us_imu_gui.py`)와 **같은 세션 포맷**이다:
    us_frames.bin   size×size uint8 프레임을 그대로 이어붙인 스트림 (기본 256×256)
    us_index.csv    pc_unix, us_seq, frame_id, byte_offset
그래서 `policy_learning` 의 `inspect_session.py` / `build_dataset.py` 가 수정 없이 읽는다.

프레임 만들기: 뷰어 클라이언트 영역 안의 ROI(분율) 를 잘라 → 그레이스케일 → **정방형 레터박스**
(검정 패딩) → size×size 로 축소. 패딩을 넣는 이유: 비등방 축소를 하면 convex 부채꼴이 타원이 되어
`Q_raw` 의 fan 기하(반경 = hypot) 가 틀어진다. 패딩·스케일은 세션 메타에 남긴다.

중복 제거: 뷰어는 화면 주사율로 다시 그리지만 프로브 fps 는 그보다 낮다. 직전 저장 프레임과
평균 절대차가 임계 아래면 같은 프레임으로 보고 저장하지 않는다 (fps 표시는 **새 프레임** 기준이다).
ROI 에 시계·카운터 같은 매 프레임 변하는 텍스트가 들어가면 중복 제거가 무력해지니 영상만 잡는다.

겹침: 화면 영역을 그대로 잡으면(mss/PIL) 뷰어 위에 올라온 창 — 이 GUI 자신 — 이 찍힌다. 그래서 기본은
`PrintWindow(PW_RENDERFULLCONTENT)` 로 **뷰어 창의 내용을 직접 렌더**해 오는 것이다. 다른 창에 가려져도 되고,
창을 옮겨도 되지만 최소화는 안 된다 (WPF 가 그리지 않는다). PrintWindow 가 검은 화면만 주는 환경이면
자동으로 화면 영역 캡처(mss) 로 물러나고, 그때는 뷰어를 가리지 말아야 한다.

시각: `pc_unix = time.time()` 을 **캡처 직전**에 찍는다. 화면에 있는 픽셀은 그보다 조금 전의 것이므로
지연은 양수 쪽으로만 치우친다. IMU 와 같은 시계다.
"""

from __future__ import annotations

import csv
import ctypes
import ctypes.wintypes as wt
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # noqa: BLE001
    cv2 = None
    _HAVE_CV2 = False
from PIL import Image, ImageGrab

try:
    import mss  # 영역만 BitBlt 한다 — PIL.ImageGrab 은 전체 화면을 잡은 뒤 자르므로 2560×1600 에서 ~95 ms
    _HAVE_MSS = True
except Exception:  # noqa: BLE001
    _HAVE_MSS = False

DEFAULT_WINDOW_TITLE = "WirelessUSG"
DEFAULT_FRAME_SIZE = 256


# --------------------------------------------------------------------------- Win32
def set_dpi_awareness() -> None:
    """물리 픽셀 좌표로 일하게 한다. 이걸 안 하면 150 % 배율 노트북에서 캡처 영역이 어긋난다."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)      # PER_MONITOR_AWARE
    except Exception:  # noqa: BLE001
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


def find_window(title_substr: str) -> Optional[int]:
    """제목에 title_substr 이 들어가는 **보이는** 최상위 창의 HWND. 없으면 None."""
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if title_substr.lower() in buf.value.lower():
            found.append(int(hwnd))
            return False
        return True

    user32.EnumWindows(_cb, 0)
    return found[0] if found else None


def client_rect_on_screen(hwnd: int) -> tuple[int, int, int, int]:
    """클라이언트 영역(제목표시줄·테두리 제외) 의 화면 좌표 (x0, y0, x1, y1)."""
    user32 = ctypes.windll.user32
    rc = wt.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
        raise OSError("GetClientRect 실패")
    pt = wt.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        raise OSError("ClientToScreen 실패")
    return pt.x, pt.y, pt.x + rc.right, pt.y + rc.bottom


def window_is_minimized(hwnd: int) -> bool:
    return bool(ctypes.windll.user32.IsIconic(hwnd))


def restore_window(hwnd: int) -> None:
    """최소화된 창을 복원한다 (SW_RESTORE). 화면에 보여야 캡처되므로 수신기가 필요할 때 부른다."""
    ctypes.windll.user32.ShowWindow(hwnd, 9)


def grab_gray(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """화면 bbox 를 그레이스케일 uint8 (H, W) 로 (PIL 경로, 스레드 무관). 느리지만 항상 된다."""
    img = ImageGrab.grab(bbox=bbox, all_screens=True).convert("L")
    return np.asarray(img, dtype=np.uint8)


def _bgra_to_gray(bgra: np.ndarray) -> np.ndarray:
    if _HAVE_CV2:
        return cv2.cvtColor(np.ascontiguousarray(bgra), cv2.COLOR_BGRA2GRAY)
    # ITU-R 601 luma, BGRA 순서
    return ((bgra[..., 2].astype(np.uint16) * 77 + bgra[..., 1].astype(np.uint16) * 151
             + bgra[..., 0].astype(np.uint16) * 28) >> 8).astype(np.uint8)


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG), ("biPlanes", wt.WORD),
                ("biBitCount", wt.WORD), ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                ("biClrImportant", wt.DWORD)]


PW_CLIENTONLY = 0x1
PW_RENDERFULLCONTENT = 0x2


class _PrintWindowGrabber:
    """PrintWindow 용 GDI 객체(메모리 DC·비트맵·DIB 버퍼) 를 창 크기별로 재사용한다. 스레드당 하나."""

    def __init__(self):
        self._key = None
        self._hdc_mem = None
        self._bmp = None
        self._buf = None
        self._bmi = None

    def _ensure(self, hwnd: int, w: int, h: int) -> None:
        if self._key == (w, h) and self._hdc_mem:
            return
        self.close()
        user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
        hdc_win = user32.GetDC(hwnd)
        try:
            self._hdc_mem = gdi32.CreateCompatibleDC(hdc_win)
            self._bmp = gdi32.CreateCompatibleBitmap(hdc_win, w, h)
            gdi32.SelectObject(self._hdc_mem, self._bmp)
        finally:
            user32.ReleaseDC(hwnd, hdc_win)
        self._buf = ctypes.create_string_buffer(w * h * 4)
        bmi = _BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.biWidth, bmi.biHeight = w, -h          # 음수 높이 = top-down
        bmi.biPlanes, bmi.biBitCount, bmi.biCompression = 1, 32, 0
        self._bmi = bmi
        self._key = (w, h)

    def grab(self, hwnd: int) -> Optional[np.ndarray]:
        """(H, W, 4) BGRA **뷰** — 다음 grab 때 덮어써지므로 필요한 부분은 바로 잘라 써야 한다."""
        user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
        rc = wt.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
            return None
        w, h = int(rc.right), int(rc.bottom)
        if w <= 0 or h <= 0:
            return None
        self._ensure(hwnd, w, h)
        if not user32.PrintWindow(hwnd, self._hdc_mem, PW_CLIENTONLY | PW_RENDERFULLCONTENT):
            return None
        if not gdi32.GetDIBits(self._hdc_mem, self._bmp, 0, h, self._buf, ctypes.byref(self._bmi), 0):
            return None
        return np.frombuffer(self._buf, dtype=np.uint8).reshape(h, w, 4)

    def close(self) -> None:
        gdi32 = ctypes.windll.gdi32
        if self._bmp:
            gdi32.DeleteObject(self._bmp)
        if self._hdc_mem:
            gdi32.DeleteDC(self._hdc_mem)
        self._bmp = self._hdc_mem = self._buf = self._bmi = None
        self._key = None


def print_window_bgra(hwnd: int) -> Optional[np.ndarray]:
    """뷰어 창의 **클라이언트 영역**을 PrintWindow 로 렌더해 (H, W, 4) BGRA 복사본으로. 실패면 None.

    다른 창에 가려져 있어도 창 자신의 내용이 나온다 (PW_RENDERFULLCONTENT: WPF/DirectComposition 포함).
    일회성 호출용 — 반복 캡처는 FrameGrabber 가 _PrintWindowGrabber 를 재사용한다.
    """
    g = _PrintWindowGrabber()
    try:
        out = g.grab(hwnd)
        return None if out is None else out.copy()
    finally:
        g.close()


class FrameGrabber:
    """뷰어 프레임 그래버. 백엔드: printwindow (기본, 가려져도 됨) | mss | pil (화면 영역).

    mss 는 만든 스레드에서만 써야 하므로 수신 스레드의 run() 안에서 만든다.
    `auto` 는 PrintWindow 를 먼저 쓰고, 연속으로 검은 화면(전부 0) 만 돌아오면 mss/pil 로 물러난다.
    """

    def __init__(self, backend: str = "auto"):
        if backend not in ("auto", "printwindow", "mss", "pil"):
            raise ValueError(f"backend 는 auto|printwindow|mss|pil: {backend!r}")
        self.requested = backend
        self._pw = _PrintWindowGrabber()
        self._sct = mss.mss() if (_HAVE_MSS and backend in ("auto", "mss")) else None
        self._screen_backend = "mss" if self._sct is not None else "pil"
        self.backend = self._screen_backend if backend in ("mss", "pil") else "printwindow"
        self._black_streak = 0

    def grab_screen(self, bbox: tuple[int, int, int, int]) -> np.ndarray:
        if self._sct is None:
            return grab_gray(bbox)
        x0, y0, x1, y1 = bbox
        shot = self._sct.grab({"left": int(x0), "top": int(y0), "width": int(x1 - x0), "height": int(y1 - y0)})
        bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return _bgra_to_gray(bgra)

    def grab(self, hwnd: int, client: tuple[int, int, int, int], roi: "CaptureRoi") -> np.ndarray:
        """ROI 의 그레이스케일 프레임. client 는 화면 좌표의 클라이언트 영역."""
        if self.backend == "printwindow":
            bgra = self._pw.grab(hwnd)
            # 실패 판정은 **창 전체** 비트맵이 0 인지로 한다. 영상 영역(ROI) 은 프로브 스트림 전이면
            # 정당하게 검다 — 그걸 실패로 보면 화면 캡처로 물러나 이 GUI 자신을 찍는다.
            if bgra is not None and int(bgra[..., :3].max()) > 0:
                self._black_streak = 0
                h, w = bgra.shape[:2]
                x0, y0, x1, y1 = roi.to_bbox((0, 0, w, h))
                return _bgra_to_gray(bgra[y0:y1, x0:x1])
            self._black_streak += 1
            if self.requested == "printwindow" or self._black_streak < 10:
                return np.zeros((8, 8), np.uint8)
            self.backend = self._screen_backend          # auto: PrintWindow 가 아무것도 못 그리는 환경
        return self.grab_screen(roi.to_bbox(client))

    def close(self) -> None:
        self._pw.close()
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- 순수 함수 (테스트 대상)
@dataclass
class CaptureRoi:
    """뷰어 클라이언트 영역 안의 영상 ROI — 분율 (0..1). 창 크기가 바뀌어도 그대로 쓸 수 있다."""
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 1.0
    y1: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.x0 < self.x1 <= 1.0 and 0.0 <= self.y0 < self.y1 <= 1.0):
            raise ValueError(f"ROI 분율은 0 ≤ x0 < x1 ≤ 1, 0 ≤ y0 < y1 ≤ 1: {asdict(self)}")

    def to_bbox(self, client: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        cx0, cy0, cx1, cy1 = client
        w, h = cx1 - cx0, cy1 - cy0
        x0 = cx0 + int(round(self.x0 * w)); x1 = cx0 + int(round(self.x1 * w))
        y0 = cy0 + int(round(self.y0 * h)); y1 = cy0 + int(round(self.y1 * h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError("ROI 가 너무 작습니다 (8 px 미만)")
        return x0, y0, x1, y1

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "CaptureRoi":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(**json.load(fh))


def letterbox_resize(gray: np.ndarray, size: int = DEFAULT_FRAME_SIZE, pad_value: int = 0
                     ) -> tuple[np.ndarray, dict]:
    """정방형으로 패딩한 뒤 등방 축소. (frame size×size uint8, info)."""
    if gray.ndim != 2:
        raise ValueError("gray 는 (H, W) 여야 합니다")
    h, w = gray.shape
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    scale = float(size) / float(side)
    # 축소를 먼저 하고 (실제 픽셀만), 작은 캔버스에 패딩한다 — 큰 정방형 캔버스를 만들어 통째로 줄이는 것보다 1.5–2 배 싸다
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    if (sw, sh) != (w, h):
        if _HAVE_CV2:
            interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
            small = cv2.resize(gray, (sw, sh), interpolation=interp)
        else:
            small = np.asarray(Image.fromarray(gray).resize((sw, sh), Image.BOX if side > size else Image.BILINEAR),
                               dtype=np.uint8)
    else:
        small = gray
    frame = np.full((size, size), pad_value, np.uint8)
    oy = (size - sh) // 2
    ox = (size - sw) // 2
    frame[oy:oy + sh, ox:ox + sw] = small[:size - oy, :size - ox]
    info = {"src_wh": [int(w), int(h)], "square_side": int(side),
            "pad_left_top": [int(left), int(top)], "scale": scale,
            "mm_note": "pixel->mm is calibrated separately from the viewer depth scale (TODO)"}
    return frame, info


def frame_changed(prev: Optional[np.ndarray], cur: np.ndarray, thresh: float,
                  pix_thresh: int = 16, min_changed_frac: float = 0.005) -> bool:
    """직전 저장 프레임과 다르면 True.

    두 기준 중 하나: (a) 평균 절대차 > thresh (0..255 스케일), (b) |차| > pix_thresh 인 픽셀 비율 >
    min_changed_frac. (b) 가 있는 이유: ROI 에 정적 UI 가 많이 섞이면 영상 부분만 바뀌어도 평균차가
    희석돼 (a) 만으로는 새 프레임을 놓친다.
    """
    if prev is None:
        return True
    d = np.abs(cur.astype(np.int16) - prev.astype(np.int16))
    if float(d.mean()) > thresh:
        return True
    return float((d > pix_thresh).mean()) > min_changed_frac


# --------------------------------------------------------------------------- 수신 스레드
class ScreenUsReceiver(threading.Thread):
    """`us_imu_gui.UsReceiver` 와 같은 인터페이스. 소스만 화면 캡처다.

    latest / frame_count / scanner_active / connected / last_frame_wall / snapshot()
    start_recording(session_dir) / stop_recording() / recording / rec_count / host
    """

    def __init__(self, title: str = DEFAULT_WINDOW_TITLE, roi: Optional[CaptureRoi] = None,
                 capture_hz: float = 30.0, frame_size: int = DEFAULT_FRAME_SIZE,
                 dedupe_thresh: float = 0.5, dedupe: bool = True, auto_restore: bool = True,
                 backend: str = "auto"):
        super().__init__(daemon=True)
        self.title = title
        self.requested_backend = backend
        self.auto_restore = bool(auto_restore)   # 최소화된 뷰어를 복원한다 (10 s 에 한 번만)
        self._last_restore = 0.0
        self.roi = roi or CaptureRoi()
        self.capture_hz = float(capture_hz)
        self.frame_size = int(frame_size)
        self.dedupe_thresh = float(dedupe_thresh)
        self.dedupe = bool(dedupe)
        self.host = f"screen:{title}"          # 세션 메타 호환 (UsReceiver.host 자리)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.hwnd: Optional[int] = None
        self.client_rect: Optional[tuple[int, int, int, int]] = None
        self.bbox: Optional[tuple[int, int, int, int]] = None
        self.letterbox_info: dict = {}
        self.error: Optional[str] = None
        self.backend = "pil"

        self.latest: Optional[np.ndarray] = None
        self.frame_count = 0                    # 새(중복 아닌) 프레임 수
        self.capture_count = 0                  # 캡처 시도 수
        self.dup_count = 0
        self.scanner_active = False             # 최근 2 s 안에 새 프레임이 있었나
        self.connected = False                  # 창을 찾았나
        self.last_frame_wall = 0.0
        self._fps_stamps: list[float] = []

        self._recording = False
        self._bin = None
        self._index = None
        self._index_writer = None
        self._byte_offset = 0
        self.rec_count = 0
        self.rec_dir: Optional[str] = None

    # -- 제어 (GUI 스레드) --
    def set_roi(self, roi: CaptureRoi) -> None:
        with self._lock:
            self.roi = roi
            self.latest = None

    def refind_window(self) -> None:
        with self._lock:
            self.hwnd = None

    def start_recording(self, session_dir: str) -> None:
        with self._lock:
            if self._recording:
                return
            self._bin = open(os.path.join(session_dir, "us_frames.bin"), "wb")
            self._index = open(os.path.join(session_dir, "us_index.csv"), "w", newline="")
            self._index_writer = csv.writer(self._index)
            self._index_writer.writerow(["pc_unix", "us_seq", "frame_id", "byte_offset"])
            self._byte_offset = 0
            self.rec_count = 0
            self.rec_dir = session_dir
            self._recording = True

    def stop_recording(self) -> int:
        with self._lock:
            if not self._recording:
                return 0
            n = self.rec_count
            self._recording = False
            for fh in (self._bin, self._index):
                if fh is not None:
                    fh.close()
            self._bin = self._index = self._index_writer = None
            return n

    @property
    def recording(self) -> bool:
        return self._recording

    def capture_meta(self) -> dict:
        """세션 메타의 us.capture 에 들어갈 것."""
        with self._lock:
            return {
                "source": "screen_capture", "window_title": self.title,
                "roi_fraction": asdict(self.roi), "client_rect_px": self.client_rect,
                "bbox_px": self.bbox, "letterbox": dict(self.letterbox_info),
                "capture_hz": self.capture_hz, "backend": self.backend,
                "dedupe": self.dedupe, "dedupe_thresh": self.dedupe_thresh,
                "timestamp_note": "pc_unix 는 캡처 직전 time.time(). 화면 픽셀은 그보다 앞선 시각 — 지연은 양수",
            }

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "img": self.latest, "fps": self._fps(), "active": self.scanner_active,
                "connected": self.connected, "frames": self.frame_count,
                "recording": self._recording, "rec_count": self.rec_count,
                "stale": (time.time() - self.last_frame_wall) > 2.0 if self.last_frame_wall else True,
                "captures": self.capture_count, "dupes": self.dup_count, "bbox": self.bbox,
                "error": self.error, "backend": self.backend, "hwnd": self.hwnd,
                "latest_mean": None if self.latest is None else float(self.latest.mean()),
            }

    def _fps(self) -> float:
        now = time.monotonic()
        self._fps_stamps = [t for t in self._fps_stamps if now - t < 2.0]
        return len(self._fps_stamps) / 2.0

    def stop(self) -> None:
        self._stop.set()

    # -- 루프 --
    def _locate(self) -> bool:
        hwnd = self.hwnd
        if hwnd is None or not ctypes.windll.user32.IsWindow(hwnd):
            hwnd = find_window(self.title)
        if hwnd is None:
            with self._lock:
                self.hwnd = None; self.connected = False; self.bbox = None
                self.error = f"창을 찾지 못함: 제목에 '{self.title}' 포함된 보이는 창 없음"
            return False
        if window_is_minimized(hwnd):
            if self.auto_restore and time.monotonic() - self._last_restore > 10.0:
                self._last_restore = time.monotonic()
                restore_window(hwnd)
                time.sleep(0.3)
            if window_is_minimized(hwnd):
                with self._lock:
                    self.hwnd = hwnd; self.connected = False
                    self.error = "뷰어 창이 최소화됨 — 화면에 보여야 캡처된다"
                return False
        try:
            client = client_rect_on_screen(hwnd)
            bbox = self.roi.to_bbox(client)
        except (OSError, ValueError) as exc:
            with self._lock:
                self.hwnd = hwnd; self.connected = False; self.error = str(exc)
            return False
        with self._lock:
            self.hwnd = hwnd; self.client_rect = client; self.bbox = bbox
            self.connected = True; self.error = None
        return True

    def run(self) -> None:
        set_dpi_awareness()
        grabber = FrameGrabber(self.requested_backend)
        self.backend = grabber.backend
        period = 1.0 / max(self.capture_hz, 1.0)
        prev_saved: Optional[np.ndarray] = None
        last_locate = 0.0
        while not self._stop.is_set():
            t_loop = time.monotonic()
            if t_loop - last_locate > 0.5 or self.bbox is None:
                last_locate = t_loop
                if not self._locate():
                    self._stop.wait(0.5)
                    continue
            hwnd, client = self.hwnd, self.client_rect
            try:
                pc_unix = time.time()
                gray = grabber.grab(hwnd, client, self.roi)
                self.backend = grabber.backend
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.error = f"캡처 실패: {type(exc).__name__}: {exc}"
                self._stop.wait(0.5)
                continue
            frame, info = letterbox_resize(gray, self.frame_size)
            with self._lock:
                self.capture_count += 1
                self.letterbox_info = info
                changed = (not self.dedupe) or frame_changed(prev_saved, frame, self.dedupe_thresh)
                if changed:
                    prev_saved = frame
                    self.latest = frame
                    self.frame_count += 1
                    self.last_frame_wall = pc_unix
                    self._fps_stamps.append(time.monotonic())
                    if self._recording and self._bin is not None:
                        raw = frame.tobytes()
                        self._bin.write(raw)
                        self._index_writer.writerow(
                            [f"{pc_unix:.6f}", self.rec_count, self.capture_count - 1, self._byte_offset])
                        self._byte_offset += len(raw)
                        self.rec_count += 1
                else:
                    self.dup_count += 1
                    if self.latest is None:
                        self.latest = frame
                self.scanner_active = (time.time() - self.last_frame_wall) < 2.0 if self.last_frame_wall else False
            # 주기 유지
            dt = time.monotonic() - t_loop
            if dt < period:
                self._stop.wait(period - dt)
        grabber.close()
        self.stop_recording()


# --------------------------------------------------------------------------- ROI 선택 (대화형)
def select_roi_interactive(gray: np.ndarray, current: Optional[CaptureRoi] = None,
                           title: str = "영상 영역을 드래그로 지정 — Enter 확정, Esc 취소") -> Optional[CaptureRoi]:
    """뷰어 클라이언트 영역의 스크린샷 위에서 사각형을 드래그해 ROI(분율) 를 고른다.

    matplotlib RectangleSelector 를 쓰고, 애니메이션 중인 다른 창이 있어도 동작하도록
    `start_event_loop` 로 블록한다. Enter 로 확정, Esc 또는 창 닫기로 취소(None).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import RectangleSelector

    h, w = gray.shape
    fig, ax = plt.subplots(figsize=(10, 10 * h / max(w, 1)))
    fig.canvas.manager.set_window_title("ROI 선택")
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    state = {"roi": None, "done": False}
    if current is not None:
        ax.add_patch(Rectangle((current.x0 * w, current.y0 * h), (current.x1 - current.x0) * w,
                               (current.y1 - current.y0) * h, fill=False, ec="#8ab4ff", ls="--", lw=1))

    def _on_select(eclick, erelease):
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        x0 = max(0.0, x0) / w; x1 = min(float(w), x1) / w
        y0 = max(0.0, y0) / h; y1 = min(float(h), y1) / h
        try:
            state["roi"] = CaptureRoi(round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))
            ax.set_title(f"ROI x {x0:.3f}–{x1:.3f}  y {y0:.3f}–{y1:.3f}   Enter 확정 / Esc 취소", fontsize=10)
        except ValueError as exc:
            ax.set_title(str(exc), fontsize=10)
        fig.canvas.draw_idle()

    sel = RectangleSelector(ax, _on_select, useblit=True, button=[1], minspanx=8, minspany=8,
                            spancoords="pixels", interactive=True)

    def _on_key(ev):
        if ev.key in ("enter", "return"):
            state["done"] = True
        elif ev.key == "escape":
            state["roi"] = None; state["done"] = True

    def _on_close(_ev):
        state["done"] = True

    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("close_event", _on_close)
    fig.show()
    while not state["done"]:
        fig.canvas.start_event_loop(0.1)
    sel.set_active(False)
    plt.close(fig)
    return state["roi"]


def grab_client_gray(title: str = DEFAULT_WINDOW_TITLE) -> np.ndarray:
    """뷰어 클라이언트 영역 전체를 한 장 잡는다 (ROI 선택용)."""
    set_dpi_awareness()
    hwnd = find_window(title)
    if hwnd is None:
        raise RuntimeError(f"제목에 '{title}' 이 들어가는 보이는 창이 없습니다. 뷰어를 먼저 실행하십시오.")
    if window_is_minimized(hwnd):
        restore_window(hwnd)
        time.sleep(0.3)
        if window_is_minimized(hwnd):
            raise RuntimeError("뷰어 창이 최소화되어 있습니다.")
    bgra = print_window_bgra(hwnd)
    if bgra is not None and bgra.max() > 0:
        return _bgra_to_gray(bgra)
    return grab_gray(client_rect_on_screen(hwnd))
