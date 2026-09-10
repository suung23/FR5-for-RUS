#!/usr/bin/env python3
"""리눅스: Wi-Fi 동글로 프로브 AP 에 붙고, 포트·프레임까지 판정한다.

`probe_wifi_win.py` 의 리눅스 짝이다. netsh 대신 **nmcli** 를 쓰고, 나머지 판정 순서는
같다 — 스캔 → 접속 → ping → (선택) 프레임 파싱.

    python3 imu_bench/host/probe_wifi_linux.py                      # 스캔 → 접속 → ping
    python3 imu_bench/host/probe_wifi_linux.py --scan-only
    python3 imu_bench/host/probe_wifi_linux.py --frames --probe c10ur   # 프레임 수신까지 시험
    python3 imu_bench/host/probe_wifi_linux.py --disconnect            # 동글을 프로브 AP 에서 뗀다

**내장 Wi-Fi 의 인터넷을 건드리지 않는다.** 프로브 AP 는 인터넷이 없으므로 프로필을
`ipv4.never-default yes` 로 만든다 — 안 그러면 기본 경로가 동글로 넘어가 LAN·인터넷이
같이 끊긴다. 자동 접속도 꺼 둔다 (`connection.autoconnect no`): 프로브 전원이 켜질
때마다 동글이 제멋대로 옮겨 붙으면 다른 실험 중에 링크가 흔들린다.

⚠️ **프로브는 클라이언트를 하나만 받는다.** TCP 를 열었다 닫으면 한동안 다음 접속을
거부하고, Wi-Fi 재연결로만 풀린다. 그래서 기본 판정은 **ping 까지만** 하고 5002/5003 은
GUI 가 연다. `--frames` 는 그 대가를 알고 쓰는 진단용이다 (쓰고 나면 GUI 를 열기 전에
`--disconnect` 후 다시 접속하는 편이 안전하다).
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, "..", "..", "fr5_control", "fr5_vision"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CANDIDATE_HOSTS = ("192.168.1.1", "192.168.156.1", "192.168.157.1")
#: 첫 값은 이 프로브(US-1C GRCGBA010) 의 키 — 뷰어가 만든 Windows 프로필에서 복구했다.
CANDIDATE_PASSWORDS = ("usccgba010", "12345678", "88888888")
VIDEO_PORT, CONTROL_PORT = 5002, 5003

#: nmcli 프로필 이름. SSID 와 분리해 둬야 프로브를 바꿔도 프로필 관리가 한 곳이다.
CONN_NAME = "us-probe"


def _nmcli(*args: str, check: bool = False, timeout: float = 60.0) -> tuple[int, str]:
    """nmcli 를 돌리고 (반환코드, stdout+stderr) 를 준다."""
    proc = subprocess.run(["nmcli", *args], capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    if check and proc.returncode != 0:
        raise RuntimeError(f"nmcli {' '.join(args)} 실패:\n{out.strip()}")
    return proc.returncode, out


def wifi_interfaces() -> list[str]:
    """무선 인터페이스 이름들. USB 동글(`wlx…`) 을 앞에 둔다."""
    _, out = _nmcli("-t", "-f", "DEVICE,TYPE", "device")
    names = [ln.split(":")[0] for ln in out.splitlines() if ln.endswith(":wifi")]
    # USB 어댑터는 MAC 기반 이름(wlx…) 을 받는다. 내장은 보통 wlp… 다.
    return sorted(names, key=lambda n: (not n.startswith("wlx"), n))


def default_route_iface() -> str | None:
    """기본 경로가 걸린 인터페이스. 여기에는 절대 프로브 AP 를 붙이지 않는다."""
    proc = subprocess.run(["ip", "-4", "route", "show", "default"], capture_output=True, text=True)
    m = re.search(r"\bdev\s+(\S+)", proc.stdout or "")
    return m.group(1) if m else None


def scan(iface: str, rescan: bool = True) -> list[str]:
    """프로브로 보이는 SSID 목록 ('US-' 로 시작)."""
    _nmcli("device", "wifi", "rescan", "ifname", iface, timeout=30.0) if rescan else None
    time.sleep(2.0 if rescan else 0.0)
    _, out = _nmcli("-t", "-f", "SSID,SIGNAL", "device", "wifi", "list", "ifname", iface)
    hits = []
    for line in out.splitlines():
        ssid = line.rsplit(":", 1)[0].strip()
        if ssid.upper().startswith("US-") and ssid not in hits:
            hits.append(ssid)
    return hits


def connect(iface: str, ssid: str, password: str, timeout: float = 30.0) -> bool:
    """프로브 AP 프로필을 만들고 올린다. 기존 동명 프로필은 지우고 다시 만든다.

    비밀번호가 틀리면 NetworkManager 가 프로필을 남긴 채 인증 대기로 멈춘다 — 다음 시도
    전에 지워야 후보 비밀번호를 순서대로 볼 수 있다.
    """
    _nmcli("connection", "delete", CONN_NAME)      # 없으면 실패해도 그만
    rc, out = _nmcli(
        "connection", "add", "type", "wifi", "ifname", iface, "con-name", CONN_NAME,
        "ssid", ssid,
        "wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", password,
        # 인터넷은 내장 Wi-Fi 가 계속 쥔다. 프로브 AP 로 기본 경로가 넘어가면 안 된다.
        "ipv4.never-default", "yes", "ipv6.never-default", "yes",
        "ipv6.method", "ignore",
        "connection.autoconnect", "no",
    )
    if rc != 0:
        print(f"  프로필 생성 실패: {out.strip()}")
        return False
    rc, out = _nmcli("connection", "up", CONN_NAME, "ifname", iface, timeout=timeout)
    if rc != 0:
        print(f"  접속 실패: {out.strip().splitlines()[-1] if out.strip() else rc}")
        _nmcli("connection", "delete", CONN_NAME)
        return False
    return True


def iface_ipv4(iface: str) -> tuple[str | None, str | None]:
    """(주소, 게이트웨이). 게이트웨이가 프로브의 주소다."""
    _, out = _nmcli("-t", "-f", "IP4.ADDRESS,IP4.GATEWAY", "device", "show", iface)
    ip = gw = None
    for line in out.splitlines():
        if line.startswith("IP4.ADDRESS") and ip is None:
            ip = line.split(":", 1)[1].split("/")[0].strip() or None
        elif line.startswith("IP4.GATEWAY"):
            gw = line.split(":", 1)[1].strip() or None
    return ip, gw


def ping(host: str, iface: str | None = None, timeout_s: int = 1) -> bool:
    """프로브가 살아 있는지. 인터페이스를 묶어 다른 경로로 새지 않게 한다."""
    cmd = ["ping", "-c", "1", "-W", str(timeout_s)]
    if iface:
        cmd += ["-I", iface]
    return subprocess.run(cmd + [host], capture_output=True).returncode == 0


def tcp_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def try_frames(host: str, seconds: float, probe: str) -> dict:
    """기존 수신기로 프레임이 파싱되는지. C10UR 은 스캔 시작을 클라이언트가 명령한다."""
    try:
        from fr5_vision.us_protocol import PROFILES, UsScannerSession
    except Exception as exc:  # noqa: BLE001
        return {"frames": 0, "active": False, "error": f"us_protocol import 실패: {exc}"}
    profile = PROFILES.get(probe)
    if profile is None:
        return {"frames": 0, "active": False, "error": f"알 수 없는 프로브 {probe!r} (있는 것: {sorted(PROFILES)})"}
    sess = UsScannerSession(host, profile=profile)
    n, err, started = 0, None, False
    try:
        sess.open()
        t_end = time.time() + seconds
        while time.time() < t_end:
            try:
                n += len(sess.poll(0.05))
            except (ConnectionError, OSError) as exc:
                err = f"{type(exc).__name__}: {exc}"
                break
            # setup 바이트가 다 나간 뒤에 스캔을 명령한다 (ready_after_s 이후).
            if not started and sess.can_command_scan and time.time() > t_end - seconds + profile.ready_after_s:
                sess.start_scan()
                started = True
        active = bool(sess.scanner_active)
    except OSError as exc:
        return {"frames": 0, "active": False, "error": f"open 실패: {exc}"}
    finally:
        try:
            sess.close()
        except Exception:  # noqa: BLE001
            pass
    return {"frames": n, "active": active, "error": err}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default=None, help="Wi-Fi 인터페이스 (기본: 기본 경로가 아닌 동글)")
    ap.add_argument("--ssid", default=None, help="프로브 SSID (기본: 'US-' 로 시작하는 것을 스캔)")
    ap.add_argument("--password", default=None, help=f"AP 비밀번호 (기본: {', '.join(CANDIDATE_PASSWORDS)} 순서)")
    ap.add_argument("--probe", default="c10ur", help="프로브 프로파일 (기본 %(default)s)")
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--frames", action="store_true",
                    help="프레임 수신까지 시험한다. ⚠ 프로브가 한동안 다음 접속을 거부한다")
    ap.add_argument("--seconds", type=float, default=6.0, help="프레임 수신 시험 시간")
    ap.add_argument("--disconnect", action="store_true", help="프로브 AP 프로필을 내리고 지운다")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    if args.disconnect:
        _nmcli("connection", "down", CONN_NAME)
        _nmcli("connection", "delete", CONN_NAME)
        print(f"{CONN_NAME} 내림. 동글은 다시 비어 있다.")
        return 0

    ifaces = wifi_interfaces()
    if not ifaces:
        print("무선 인터페이스가 없습니다 — 동글이 꽂혀 있는지 확인하십시오.")
        return 1
    default_iface = default_route_iface()
    print(f"무선 인터페이스: {ifaces}   (기본 경로: {default_iface})")

    iface = args.iface
    if iface is None:
        candidates = [n for n in ifaces if n != default_iface]
        if not candidates:
            print(f"동글을 못 찾았습니다. 무선이 {ifaces} 뿐이고 기본 경로도 여기 걸려 있습니다.\n"
                  f"인터넷을 끊고 쓰려면 --iface 로 직접 지정하십시오.")
            return 1
        iface = candidates[0]
    if iface == default_iface:
        print(f"⚠ {iface} 에 기본 경로가 걸려 있습니다 — 프로브 AP 에 붙이면 인터넷이 끊깁니다.")
    print(f"사용 인터페이스: {iface}")

    ssid = args.ssid
    if ssid is None:
        hits = scan(iface)
        if not hits:
            print("프로브 AP('US-…') 가 스캔에 없습니다.\n"
                  "  · 프로브 배터리 전원을 켜고 (USB 는 뽑은 상태)\n"
                  "  · 몇 초 기다린 뒤 다시 실행하십시오.")
            return 2
        if len(hits) > 1:
            print(f"프로브 후보가 여럿입니다: {hits} — 첫 번째를 씁니다. --ssid 로 고를 수 있습니다.")
        ssid = hits[0]
    print(f"프로브 SSID: {ssid}")
    if args.scan_only:
        return 0

    for pw in ([args.password] if args.password else list(CANDIDATE_PASSWORDS)):
        print(f"접속 시도 {ssid!r} (비밀번호 {pw}) …", flush=True)
        if connect(iface, ssid, pw):
            print("  접속됨")
            break
        print("  실패")
    else:
        print("접속 실패 — 비밀번호가 다르거나 프로브가 다른 클라이언트에 점유됐습니다\n"
              "  (Windows 뷰어가 붙어 있으면 그쪽 Wi-Fi 를 먼저 끊으십시오).")
        return 3

    time.sleep(2.0)
    ip, gw = iface_ipv4(iface)
    print(f"IPv4 {ip}   게이트웨이 {gw}")
    still_default = default_route_iface()
    if still_default != default_iface:
        print(f"⚠ 기본 경로가 {default_iface} → {still_default} 로 바뀌었습니다. 인터넷을 확인하십시오.")

    host = gw or CANDIDATE_HOSTS[0]
    for i in range(5):
        if ping(host, iface):
            print(f"  {host}: ping OK")
            break
        print(f"  {host}: ping x  ({i+1}/5)")
        time.sleep(2.0)
    else:
        print("프로브에 ping 이 안 됩니다 — 전원/절전, 동글 연결을 확인하십시오.")
        return 4

    if not args.frames:
        print(f"\n판정: AP 접속 OK. 포트는 GUI 가 엽니다 (프로브는 클라이언트를 하나만 받습니다).\n"
              f"      python3 phantom_stiffness/stiffness_gui.py --us-host {host} --probe {args.probe}")
        return 0

    v, c = tcp_open(host, VIDEO_PORT), tcp_open(host, CONTROL_PORT)
    print(f"  {host}: TCP {VIDEO_PORT} {'OK' if v else 'x'}   TCP {CONTROL_PORT} {'OK' if c else 'x'}")
    if not (v and c):
        print("판정: 5002/5003 이 열리지 않습니다 — 다른 포트/프로토콜 (PCAP 필요)")
        return 5
    res = try_frames(host, args.seconds, args.probe)
    print(f"  {args.seconds:.0f} s 수신: 프레임 {res['frames']} 개, active={res['active']}, 오류={res['error']}")
    if res["frames"] > 0:
        print(f"\n판정: 프로토콜 일치 — 뷰어 없이 원시 프레임 수신 가능.\n"
              f"      ⚠ 방금 소켓을 닫았으니 GUI 를 열기 전에 --disconnect 후 다시 접속하십시오.")
        return 0
    print("판정: 포트는 열리지만 프레임이 파싱되지 않습니다 — 프로토콜이 다릅니다 (PCAP 역공학 필요)")
    return 6


if __name__ == "__main__":
    raise SystemExit(main())
