"""WirelessUSG 뷰어를 UI Automation 으로 조종한다 (Windows) — freeze/live, 프로브 연결, 게인·깊이, 상태 읽기.

뷰어는 프로브(USB, Cypress FX3) 와 말하는 유일한 프로세스라 없앨 수 없다. 대신 **뷰어를 우리 GUI 가 뒤에서
조종**한다. WPF 라서 버튼이 UI Automation 에 AutomationId 로 노출된다 (2026-09-09 실측, v2.1.4):

    btnFreeze        Freeze ↔ Live 토글 (InvokePattern — 포커스·마우스 불필요, 창이 가려져 있어도 됨)
    btnConnection    프로브 연결 대화상자/재연결
    btnNegativeGain / btnPositiveGain / btnDepth / btnPlay / btnBViewer …
    StateLabel       "FREEZE" | "LIVE"        ImageCountLabel  "71/71"       depthLabel "D:220mm"
    gainLabel        "GN:80dB"                프로브 이름 텍스트 "US-1C CGBA010"

이 모듈은 pywinauto 가 깔아 주는 comtypes 의 UIAutomationCore 를 직접 쓴다 (pywinauto 의 descendants 필터는
automation_id 조건을 못 만들고, 트리 전체 순회는 3 s 라 실시간엔 못 쓴다). 상태 읽기는 FindFirst 로 ~ms.

주의: 뷰어의 내부 상태를 바꾸는 것은 사용자가 버튼을 누르는 것과 같다. 녹화 중 freeze 를 누르면 프레임이 멈춘다.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import threading
import time
from typing import Optional

from us_screen_capture import find_window, window_is_minimized, restore_window, DEFAULT_WINDOW_TITLE

__all__ = ["ViewerControl", "ViewerState", "ViewerPoller", "DEFAULT_VIEWER_EXE", "OFFSCREEN_X"]

DEFAULT_VIEWER_EXE = r"C:\Program Files (x86)\WirelessUSG\WirelessUSG.exe"
OFFSCREEN_X = -4000          # 숨김 = 최소화가 아니라 화면 밖 이동. 최소화하면 WPF 가 그리지 않아 PrintWindow 가 빈다


class ViewerState(dict):
    """{'state': 'LIVE'|'FREEZE'|None, 'count': '71/71', 'depth': 'D:220mm', 'gain': 'GN:80dB', 'probe': 'US-1C …'}"""

    @property
    def live(self) -> Optional[bool]:
        s = self.get("state")
        return None if s is None else (s.strip().upper() == "LIVE")

    @property
    def depth_mm(self) -> Optional[float]:
        d = self.get("depth") or ""
        digits = "".join(ch for ch in d if ch.isdigit() or ch == ".")
        try:
            return float(digits) if digits else None
        except ValueError:
            return None


class ViewerControl:
    def __init__(self, title: str = DEFAULT_WINDOW_TITLE):
        if sys.platform != "win32":
            raise RuntimeError("Windows 전용")
        from pywinauto.uia_defines import IUIA      # comtypes 로 UIAutomationCore 를 연다
        self._uia = IUIA()
        self._dll = self._uia.UIA_dll
        self._auto = self._uia.iuia
        self.title = title
        self.hwnd: Optional[int] = None
        self._root = None
        self._cache: dict = {}          # automation_id → IUIAutomationElement (트리 탐색은 비싸다: 요소를 들고 있는다)
        self.attach()

    # ------------------------------------------------------------------ 연결
    def attach(self) -> bool:
        hwnd = find_window(self.title)
        if hwnd is None:
            self.hwnd, self._root = None, None
            return False
        if hwnd != self.hwnd or self._root is None:
            self.hwnd = hwnd
            self._root = self._auto.ElementFromHandle(hwnd)
            self._cache.clear()
        return True

    @property
    def attached(self) -> bool:
        return self._root is not None and bool(ctypes.windll.user32.IsWindow(self.hwnd or 0))

    # ------------------------------------------------------------------ 실행 · 창 위치
    @staticmethod
    def launch(exe: str = DEFAULT_VIEWER_EXE, wait_s: float = 20.0, title: str = DEFAULT_WINDOW_TITLE) -> Optional[int]:
        """뷰어가 없으면 실행하고 창이 뜰 때까지 기다린다. HWND 또는 None."""
        hwnd = find_window(title)
        if hwnd is not None:
            return hwnd
        if not os.path.isfile(exe):
            raise FileNotFoundError(f"뷰어 실행 파일이 없습니다: {exe}")
        subprocess.Popen([exe], cwd=os.path.dirname(exe), creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
        t_end = time.time() + wait_s
        while time.time() < t_end:
            time.sleep(0.5)
            hwnd = find_window(title)
            if hwnd is not None:
                time.sleep(1.5)          # WPF 초기화
                return hwnd
        return None

    def hide(self) -> bool:
        """창을 화면 밖으로 옮긴다 (세션·렌더는 유지). 사용자가 '창을 띄우지 말라' 고 한 방식."""
        if not self.attach():
            return False
        u = ctypes.windll.user32
        if window_is_minimized(self.hwnd):
            restore_window(self.hwnd)
            time.sleep(0.3)
        r = wt.RECT()
        u.GetWindowRect(self.hwnd, ctypes.byref(r))
        self._shown_pos = (r.left, r.top) if r.left > -2000 else getattr(self, "_shown_pos", (0, 0))
        return bool(u.SetWindowPos(self.hwnd, 0, OFFSCREEN_X, 100, 0, 0, 0x0001 | 0x0004 | 0x0010))

    def show(self) -> bool:
        if not self.attach():
            return False
        u = ctypes.windll.user32
        if window_is_minimized(self.hwnd):
            restore_window(self.hwnd)
        x, y = getattr(self, "_shown_pos", (0, 0))
        return bool(u.SetWindowPos(self.hwnd, 0, int(x), int(y), 0, 0, 0x0001 | 0x0004 | 0x0010))

    @property
    def hidden(self) -> bool:
        if not self.attach():
            return False
        r = wt.RECT()
        ctypes.windll.user32.GetWindowRect(self.hwnd, ctypes.byref(r))
        return r.left <= -2000

    def close_dialogs(self) -> int:
        """뷰어가 띄운 보조 대화상자(Wi-Fi 목록 등) 를 자기 CloseButton 으로 닫는다. 닫은 수."""
        if not self.attach():
            return 0
        u = ctypes.windll.user32
        pid = wt.DWORD()
        u.GetWindowThreadProcessId(self.hwnd, ctypes.byref(pid))
        found: list[int] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
        def _cb(hh, _l):
            p = wt.DWORD()
            u.GetWindowThreadProcessId(hh, ctypes.byref(p))
            if p.value == pid.value and u.IsWindowVisible(hh) and int(hh) != self.hwnd:
                found.append(int(hh))
            return True

        u.EnumWindows(_cb, 0)
        n = 0
        for hh in found:
            try:
                el = self._auto.ElementFromHandle(hh)
                cond = self._auto.CreatePropertyCondition(self._dll.UIA_AutomationIdPropertyId, "CloseButton")
                btn = el.FindFirst(self._dll.TreeScope_Descendants, cond)
                if btn is not None:
                    btn.GetCurrentPattern(self._dll.UIA_InvokePatternId).QueryInterface(
                        self._dll.IUIAutomationInvokePattern).Invoke()
                    n += 1
            except Exception:  # noqa: BLE001
                pass
        return n

    def _find(self, automation_id: str, use_cache: bool = True):
        """요소를 찾는다. 캐시된 요소는 CurrentName 을 한 번 읽어 살아 있는지 확인하고 쓴다 (FindFirst 회피)."""
        if not self.attach():
            return None
        if use_cache:
            el = self._cache.get(automation_id)
            if el is not None:
                try:
                    _ = el.CurrentName
                    return el
                except Exception:  # noqa: BLE001  요소가 사라짐 (모드 전환 등) → 다시 찾는다
                    self._cache.pop(automation_id, None)
        cond = self._auto.CreatePropertyCondition(self._dll.UIA_AutomationIdPropertyId, automation_id)
        try:
            el = self._root.FindFirst(self._dll.TreeScope_Descendants, cond)
        except Exception:  # noqa: BLE001  창이 사라지는 순간
            self._root = None
            self._cache.clear()
            return None
        if el is not None:
            self._cache[automation_id] = el
        return el

    def _find_text_containing(self, needle: str) -> Optional[str]:
        """텍스트 요소 전체 탐색 (FindAll ~1 s) — 자주 부르지 말 것. 찾은 요소는 캐시해 다음엔 이름만 읽는다."""
        if not self.attach():
            return None
        key = "__text__" + needle
        el = self._cache.get(key)
        if el is not None:
            try:
                name = el.CurrentName or ""
                if needle in name:
                    return name
            except Exception:  # noqa: BLE001
                pass
            self._cache.pop(key, None)
        cond = self._auto.CreatePropertyCondition(self._dll.UIA_ControlTypePropertyId, self._dll.UIA_TextControlTypeId)
        try:
            arr = self._root.FindAll(self._dll.TreeScope_Descendants, cond)
        except Exception:  # noqa: BLE001
            return None
        for i in range(arr.Length):
            el = arr.GetElement(i)
            name = el.CurrentName or ""
            if needle in name:
                self._cache[key] = el
                return name
        return None

    # ------------------------------------------------------------------ 읽기
    def text(self, automation_id: str) -> Optional[str]:
        el = self._find(automation_id)
        return None if el is None else (el.CurrentName or "")

    def state(self) -> ViewerState:
        return ViewerState(state=self.text("StateLabel"), count=self.text("ImageCountLabel"),
                           depth=self.text("depthLabel"), gain=self.text("gainLabel"),
                           probe=self._find_text_containing("US-"))

    # ------------------------------------------------------------------ 누르기
    def invoke(self, automation_id: str) -> bool:
        el = self._find(automation_id)
        if el is None:
            return False
        try:
            if not el.CurrentIsEnabled:
                return False
            pat = el.GetCurrentPattern(self._dll.UIA_InvokePatternId)
            pat = pat.QueryInterface(self._dll.IUIAutomationInvokePattern)
            pat.Invoke()
            return True
        except Exception:  # noqa: BLE001
            return False

    def toggle_freeze(self) -> bool:
        return self.invoke("btnFreeze")

    def set_live(self, live: bool, timeout: float = 2.0) -> Optional[bool]:
        """원하는 상태로 맞춘다. 결과 상태(True=LIVE) 를 돌려주고, 못 읽으면 None."""
        cur = self.state().live
        if cur is None:
            return None
        if cur == live:
            return cur
        if not self.toggle_freeze():
            return cur
        t_end = time.time() + timeout
        while time.time() < t_end:
            time.sleep(0.1)
            cur = self.state().live
            if cur == live:
                break
        return cur

    def connect_probe(self) -> bool:
        return self.invoke("btnConnection")

    def gain(self, up: bool) -> bool:
        return self.invoke("btnPositiveGain" if up else "btnNegativeGain")

    def depth(self) -> bool:
        return self.invoke("btnDepth")

    def play(self) -> bool:
        return self.invoke("btnPlay")

    def freeze_enabled(self) -> Optional[bool]:
        el = self._find("btnFreeze")
        try:
            return None if el is None else bool(el.CurrentIsEnabled)
        except Exception:  # noqa: BLE001
            return None


class ViewerPoller(threading.Thread):
    """뷰어 상태를 백그라운드에서 주기적으로 읽는다 (UIA FindFirst 가 호출당 ~0.2 s 라 GUI 루프에서 직접 못 읽는다)."""

    def __init__(self, control: ViewerControl, period_s: float = 1.0, slow_period_s: float = 10.0):
        super().__init__(daemon=True)
        self.vc = control
        self.period = period_s
        self.slow_period = slow_period_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.latest: dict = {"attached": False}
        self._slow: dict = {}
        self._slow_stamp = 0.0

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.latest)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            d: dict = {"attached": self.vc.attach()}
            if d["attached"]:
                d["state"] = self.vc.text("StateLabel")           # 캐시된 요소 → ~ms
                d["count"] = self.vc.text("ImageCountLabel")
                d["hidden"] = self.vc.hidden
                if time.monotonic() - self._slow_stamp > self.slow_period:   # 트리 탐색이 필요한 것들은 드물게
                    self._slow = {
                        "depth": self.vc.text("depthLabel"), "gain": self.vc.text("gainLabel"),
                        "freeze_enabled": self.vc.freeze_enabled(),
                        "probe": self.vc._find_text_containing("US-"),
                    }
                    self._slow_stamp = time.monotonic()
                d.update(self._slow)
            with self._lock:
                self.latest = d
            dt = time.monotonic() - t0
            self._stop.wait(max(0.1, self.period - dt))


if __name__ == "__main__":     # 간단 점검:  python us_viewer_control.py [toggle]
    vc = ViewerControl()
    print("attached", vc.attached, "hwnd", vc.hwnd)
    print("state", dict(vc.state()))
    if len(sys.argv) > 1 and sys.argv[1] == "toggle":
        print("toggle ->", vc.toggle_freeze())
        time.sleep(1.0)
        print("state", dict(vc.state()))
