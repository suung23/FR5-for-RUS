#!/usr/bin/env python3
"""Windows: Wi-Fi 동글로 프로브 AP 에 붙고, 기존 TCP 5002/5003 프로토콜이 이 프로브에 맞는지 판정한다.

    python imu_bench\\host\\probe_wifi_win.py                 # 스캔 → 접속 → 포트 → 프레임 파싱 (5 s)
    python imu_bench\\host\\probe_wifi_win.py --scan-only
    python imu_bench\\host\\probe_wifi_win.py --iface "Wi-Fi 2" --ssid "US-1C GRCGBA010" --password 12345678

판정 기준 (2026-09-09):
  * 뷰어(WirelessUSG) 의 Wi-Fi 킷 DLL 이 192.168.1.1 / 192.168.156.1 / 192.168.157.1 을 쓰고, WLAN 프로필
    템플릿 옆에 '12345678' 과 '#88888888' 이 있다 → 같은 제조사(SonopTek) 프로토콜일 가능성. 두 비밀번호를 차례로 시도.
  * 기존 수신기 `fr5_vision.us_protocol.UsScannerSession` (SL-2C 에서 역공학, 관찰된 바이트열만) 으로 프레임이
    파싱되면 **뷰어 없이 원시 프레임** — `us_imu_gui.py --host <ip> --port COMx` 가 그대로 쓰인다.
  * 포트는 열리는데 프레임이 안 오면 프로토콜이 다른 것 — USBPcap/Wi-Fi PCAP 으로 다시 역공학해야 한다.

netsh 로 프로필을 만들어 접속한다 (관리자 불필요). 동글 인터페이스만 쓰므로 내장 Wi-Fi 의 인터넷은 유지된다.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, "..", "..", "fr5_control", "fr5_vision"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CANDIDATE_HOSTS = ("192.168.1.1", "192.168.156.1", "192.168.157.1")
CANDIDATE_PASSWORDS = ("12345678", "88888888")
VIDEO_PORT, CONTROL_PORT = 5002, 5003


def _run(cmd: list[str]) -> str:
    """netsh/ipconfig 출력. 콘솔 코드페이지에 따라 UTF-8 또는 CP949 로 온다 — 둘 다 시도."""
    raw = subprocess.run(cmd, capture_output=True).stdout
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def wifi_interfaces() -> list[str]:
    out = _run(["netsh", "wlan", "show", "interfaces"])
    return [m.group(1).strip() for m in re.finditer(r"^\s*(?:이름|Name)\s*:\s*(.+)$", out, re.M)]


def scan(iface: str, needle: str = "US-", tries: int = 3) -> list[str]:
    for _ in range(tries):
        out = _run(["netsh", "wlan", "show", "networks", f"interface={iface}"])
        ssids = [m.group(1).strip() for m in re.finditer(r"^\s*SSID \d+ : (.*)$", out, re.M)]
        hits = [s for s in ssids if needle.lower() in s.lower()]
        if hits:
            return hits
        time.sleep(3.0)
    return []


def make_profile(ssid: str, password: str) -> str:
    return f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{ssid}</name>
  <SSIDConfig><SSID><name>{ssid}</name></SSID><nonBroadcast>false</nonBroadcast></SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>auto</connectionMode>
  <MSM><security>
    <authEncryption><authentication>WPA2PSK</authentication><encryption>AES</encryption><useOneX>false</useOneX></authEncryption>
    <sharedKey><keyType>passPhrase</keyType><protected>false</protected><keyMaterial>{password}</keyMaterial></sharedKey>
  </security></MSM>
</WLANProfile>
"""


def link_state(iface: str) -> str:
    out = _run(["netsh", "wlan", "show", "interfaces"])
    blk = out.split(iface, 1)[-1] if iface in out else out
    m = re.search(r"(?:상태|State)\s*:\s*(.+)", blk)
    return m.group(1).strip() if m else "?"


def connected(iface: str) -> bool:
    return bool(re.search(r"연결됨|connected", link_state(iface), re.I))


def connect(iface: str, ssid: str, password: str, timeout: float = 20.0) -> bool:
    path = os.path.join(tempfile.gettempdir(), "probe_wlan_profile.xml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(make_profile(ssid, password))
    _run(["netsh", "wlan", "add", "profile", f"filename={path}", f"interface={iface}", "user=current"])
    _run(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}", f"interface={iface}"])
    t_end = time.time() + timeout
    while time.time() < t_end:
        time.sleep(1.0)
        out = _run(["netsh", "wlan", "show", "interfaces"])
        block = out.split(iface, 1)[-1] if iface in out else out
        if re.search(r"(상태|State)\s*:\s*(연결됨|connected)", block, re.I) and ssid in block:
            return True
    return False


def iface_ipv4(iface: str) -> tuple[str | None, str | None]:
    """(주소, 게이트웨이) — ipconfig 파싱."""
    out = _run(["ipconfig"])
    m = re.search(re.escape(iface) + r".*?(?=\n\S|\Z)", out, re.S)
    if not m:
        return None, None
    blk = m.group(0)
    ip = re.search(r"IPv4[^:]*:\s*([\d.]+)", blk)
    gw = re.search(r"(?:기본 게이트웨이|Default Gateway)[^:]*:\s*([\d.]+)", blk)
    return (ip.group(1) if ip else None), (gw.group(1) if gw else None)


def tcp_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def try_frames(host: str, seconds: float = 5.0) -> dict:
    """기존 수신기로 프레임이 파싱되는지. {'frames': n, 'active': bool, 'error': str|None}"""
    try:
        from fr5_vision.us_protocol import UsScannerSession
    except Exception as exc:  # noqa: BLE001
        return {"frames": 0, "active": False, "error": f"us_protocol import 실패: {exc}"}
    sess = UsScannerSession(host)
    n = 0
    err = None
    try:
        sess.open()
        t_end = time.time() + seconds
        while time.time() < t_end:
            try:
                n += len(sess.poll(0.05))
            except (ConnectionError, OSError) as exc:
                err = f"{type(exc).__name__}: {exc}"
                break
        active = bool(getattr(sess, "scanner_active", False))
    except OSError as exc:
        return {"frames": 0, "active": False, "error": f"open 실패: {exc}"}
    finally:
        try:
            sess.close()
        except Exception:  # noqa: BLE001
            pass
    return {"frames": n, "active": active, "error": err}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default=None, help="Wi-Fi 인터페이스 이름 (기본: 'Wi-Fi 2' 등 두 번째 어댑터)")
    ap.add_argument("--ssid", default=None, help="프로브 SSID (기본: 'US-' 로 시작하는 것을 스캔)")
    ap.add_argument("--password", default=None, help="AP 비밀번호 (기본: 12345678 → 88888888 순서로 시도)")
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--connect-only", action="store_true", help="AP 접속과 포트 확인까지만 (프레임 시험 없음)")
    ap.add_argument("--seconds", type=float, default=5.0, help="프레임 수신 시험 시간")
    args = ap.parse_args()

    ifaces = wifi_interfaces()
    print("무선 인터페이스:", ifaces)
    iface = args.iface or (ifaces[1] if len(ifaces) > 1 else (ifaces[0] if ifaces else None))
    if iface is None:
        print("무선 인터페이스가 없습니다"); return 1
    print("사용 인터페이스:", iface)

    ssid = args.ssid
    if ssid is None:
        hits = scan(iface)
        if not hits:
            print("프로브 AP('US-…') 가 스캔에 없습니다 — 프로브 전원(배터리) 을 켜고 USB 를 뽑은 상태인지 확인하십시오.")
            return 2
        ssid = hits[0]
    print("프로브 SSID:", ssid)
    if args.scan_only:
        return 0

    pws = [args.password] if args.password else list(CANDIDATE_PASSWORDS)
    ok = False
    for pw in pws:
        print(f"접속 시도 {ssid!r} (비밀번호 {pw}) …", flush=True)
        if connect(iface, ssid, pw):
            ok = True
            print("  접속됨")
            break
        print("  실패")
    if not ok:
        print("접속 실패 — 비밀번호가 다르거나 프로브가 다른 클라이언트에 점유됨 (뷰어 Wi-Fi 연결 해제 필요)")
        return 3

    time.sleep(2.0)
    ip, gw = iface_ipv4(iface)
    print(f"IPv4 {ip}  게이트웨이 {gw}")
    if args.connect_only:
        h = gw or CANDIDATE_HOSTS[0]
        for i in range(5):
            v, c = tcp_open(h, VIDEO_PORT, 3), tcp_open(h, CONTROL_PORT, 3)
            print(f"  {h}: TCP {VIDEO_PORT} {'OK' if v else 'x'}   TCP {CONTROL_PORT} {'OK' if c else 'x'}")
            if v and c:
                return 0
            time.sleep(3)
        print("포트가 열리지 않음 — 뷰어가 프로브를 잡고 있거나(종료할 것) 프로브가 절전. 동글을 재연결하면 프로브 세션이 리셋된다.")
        return 4
    hosts = [h for h in ([gw] if gw else []) + list(CANDIDATE_HOSTS) if h]
    seen = []
    for h in hosts:
        if h in seen:
            continue
        seen.append(h)
        v, c = tcp_open(h, VIDEO_PORT), tcp_open(h, CONTROL_PORT)
        print(f"  {h}: TCP {VIDEO_PORT} {'OK' if v else 'x'}   TCP {CONTROL_PORT} {'OK' if c else 'x'}")
        if v and c:
            res = try_frames(h, args.seconds)
            print(f"  기존 프로토콜로 {args.seconds:.0f} s 수신: 프레임 {res['frames']} 개, active={res['active']}, "
                  f"오류={res['error']}")
            if res["frames"] > 0:
                print(f"\n판정: 프로토콜 일치 — 뷰어 없이 원시 프레임 수신 가능.\n"
                      f"      python imu_bench\\host\\us_imu_gui.py --host {h} --port COM3")
                return 0
            print("판정: 포트는 열리지만 프레임이 파싱되지 않음 — 프로토콜이 다르다 (PCAP 역공학 필요)")
            return 4
    print("판정: 후보 주소에서 5002/5003 이 열리지 않음 — 다른 포트/프로토콜 (PCAP 필요)")
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
