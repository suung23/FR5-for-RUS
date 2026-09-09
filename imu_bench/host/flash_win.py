#!/usr/bin/env python3
"""Windows 수집 노트북에서 XIAO nRF52840 Sense 의 IMU 펌웨어를 확인하고 올린다.

    python imu_bench\\host\\flash_win.py --check              # 보드의 펌웨어 태그 ('V') 를 읽어 소스 FW_TAG 와 비교
    python imu_bench\\host\\flash_win.py --uf2 <file.uf2>     # UF2 복사로 플래시 (권장 — 아래 이유)
    python imu_bench\\host\\flash_win.py --build              # arduino-cli 로 직접 빌드+업로드 시도 (SAC 켜진 PC 에선 막힌다)

왜 UF2 인가 (2026-09-10): 이 노트북은 Windows Smart App Control 이 켜져 있어 서명 없는 실행파일 — arduino-cli, arm gcc,
adafruit-nrfutil — 이 전부 "애플리케이션 제어 정책" (WinError 4551) 에 막힌다. WSL 도 없다. 그래서 **빌드는 리눅스**
(`imu_bench/scripts/build_uf2.sh`, flash.sh 와 같은 툴체인) 에서 하고 결과 .uf2 를 git 으로 가져와, 여기서는 파일 복사만
한다. XIAO 의 UF2 부트로더는 1200 bps touch (또는 리셋 두 번) 에 "XIAO-SENSE" 드라이브를 열고, UF2 를 넣으면 스스로 쓰고
재부팅한다. 서명이 필요한 실행파일이 없다.

순서 (--uf2, 2026-09-10 실측으로 확정):
  1. 앱 모드 포트 (VID 2886 / PID 8045) 를 찾아 1200 bps touch  →  부트로더 **시리얼 DFU 전용** 모드 (PID 0045 COM 만,
     드라이브 없음). 드라이브 XIAO-SENSE 는 리셋 버튼 두 번에만 열린다.
  2. 시리얼 DFU: UF2 → HEX → DFU zip (sd-req 0xFFFE) → `python -m nordicsemi dfu serial` (pip install adafruit-nrfutil,
     순수 파이썬이라 SAC 에 안 걸린다). 드라이브가 이미 열려 있으면 그냥 복사한다.
  3. 앱 포트가 돌아오면 'V' 로 FW_TAG 를 읽어 소스의 태그와 같은지 확인 — 다르면 실패

보드가 GUI 에 잡혀 있으면 안 된다 (us_imu_gui 를 먼저 닫는다).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
SKETCH = os.path.join(ROOT, "firmware", "umi_device_hardware")
BUILD = os.path.join(ROOT, ".build", "win", "imu_only")
FQBN = "Seeeduino:nrf52:xiaonRF52840Sense"
BOARD_URL = "https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json"
LIBS = ["Adafruit BNO08x", "Adafruit_VL53L0X", "Adafruit BusIO", "Adafruit Unified Sensor"]
VID, PID_APP, PID_BOOT = 0x2886, 0x8045, 0x0045
UF2_LABELS = ("XIAO-SENSE", "XIAO-BOOT", "NRF52BOOT")

sys.path.insert(0, _HERE)


# ----------------------------------------------------------------------------- 보드 찾기 / 태그 읽기
def source_tag() -> str | None:
    src = open(os.path.join(SKETCH, "umi_device_hardware.ino"), encoding="utf-8").read()
    m = re.search(r'#define\s+FW_TAG\s+"([^"]+)"', src)
    return m.group(1) if m else None


def ports(pid: int):
    from serial.tools import list_ports
    return [p.device for p in list_ports.comports() if p.vid == VID and p.pid == pid]


def wait_for(fn, timeout: float, period: float = 0.25):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = fn()
        if r:
            return r
        time.sleep(period)
    return fn()


def read_tag(port: str, tries: int = 6):
    import serial
    from umi_protocol import read_board_name, read_fw_tag
    for _ in range(tries):
        try:
            with serial.Serial(port, 115200, timeout=0.05) as ser:
                time.sleep(0.3)
                return read_board_name(ser), read_fw_tag(ser)
        except (serial.SerialException, OSError):
            time.sleep(1.0)
    return None, None


def check() -> int:
    app = ports(PID_APP)
    if not app:
        print("앱 모드 보드 (VID 2886 / PID 8045) 가 없습니다. 부트로더 포트:", ports(PID_BOOT), " UF2 드라이브:", uf2_drives())
        return 2
    name, tag = read_tag(app[0])
    want = source_tag()
    print("보드 %s: 이름 %s, 펌웨어 태그 %s (소스 %s)" % (app[0], name, tag, want))
    if tag is None:
        print("→ 구 펌웨어 ('V' 응답 없음): 자이로 동적 보정이 꺼져 있어 cal_gyr 0 / cal_rv 1 에 머문다. 플래시 필요.")
        return 1
    if tag != want:
        print("→ 태그가 소스와 다르다. 플래시 필요.")
        return 1
    print("✔ 최신 펌웨어")
    return 0


# ----------------------------------------------------------------------------- UF2 복사 플래시
def uf2_drives():
    """UF2 부트로더가 연 이동식 드라이브 — 볼륨 이름 또는 INFO_UF2.TXT 로 판별."""
    found = []
    for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
        root = letter + ":\\"
        if not os.path.exists(root):
            continue
        if os.path.isfile(os.path.join(root, "INFO_UF2.TXT")):
            found.append(root)
            continue
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(261)
            ctypes.windll.kernel32.GetVolumeInformationW(root, buf, 261, None, None, None, None, 0)
            if buf.value.upper() in UF2_LABELS:
                found.append(root)
        except Exception:  # noqa: BLE001
            pass
    return found


def enter_bootloader():
    """부트로더로 넣고 (드라이브 목록, 부트로더 COM 목록) 을 돌려준다.

    실측 (2026-09-10): 1200 bps touch 는 Adafruit 부트로더를 **시리얼 DFU 전용** 모드 (PID 0045, CDC 만) 로 넣는다 —
    UF2 드라이브는 뜨지 않는다. 드라이브는 리셋 버튼을 빠르게 두 번 눌렀을 때만 (DFU_MAGIC_UF2_RESET) 열린다.
    그래서 touch 뒤에는 시리얼 DFU (adafruit-nrfutil, 순수 파이썬) 로 올리고, 드라이브가 이미 열려 있으면 복사한다."""
    drives = uf2_drives()
    if drives:
        return drives, []
    boot = ports(PID_BOOT)
    if boot:
        return [], boot
    app = ports(PID_APP)
    if not app:
        print("보드를 찾지 못했습니다 (VID 2886). USB 연결과 GUI 종료를 확인하십시오. 리셋 버튼을 빠르게 두 번 누르면 수동 진입.")
        return [], []
    import serial
    print("== 부트로더 진입: 1200 bps touch on", app[0])
    try:
        s = serial.Serial(app[0], 1200)
        s.dtr = False
        time.sleep(0.1)
        s.close()
    except (serial.SerialException, OSError) as e:
        print("   touch 실패:", e, "— 포트를 다른 프로그램(GUI) 이 잡고 있는지 확인")
        return [], []
    wait_for(lambda: uf2_drives() or ports(PID_BOOT), 15.0)
    return uf2_drives(), ports(PID_BOOT)


def uf2_to_hex(uf2_path: str, hex_path: str) -> None:
    """UF2 → Intel HEX (nrfutil genpkg 입력). 블록 주소를 그대로 쓴다."""
    import struct
    d = open(uf2_path, "rb").read()
    mem = {}
    for i in range(0, len(d), 512):
        m0, _m1, _fl, a, sz, _bi, _tot, fam = struct.unpack("<IIIIIIII", d[i:i + 32])
        if m0 != 0x0A324655:
            raise ValueError("UF2 매직 불일치 (블록 %d)" % (i // 512))
        for k, b in enumerate(d[i + 32:i + 32 + sz]):
            mem[a + k] = b

    def rec(t, addr, payload):
        raw = bytes([len(payload), addr >> 8, addr & 0xFF, t]) + payload
        return ":" + (raw + bytes([(-sum(raw)) & 0xFF])).hex().upper()

    lines, upper, addrs, i = [], None, sorted(mem), 0
    while i < len(addrs):
        a = addrs[i]
        if (a >> 16) != upper:
            upper = a >> 16
            lines.append(rec(4, 0, bytes([upper >> 8, upper & 0xFF])))
        chunk = bytearray()
        while i < len(addrs) and len(chunk) < 16 and addrs[i] == a + len(chunk) and (addrs[i] >> 16) == upper:
            chunk.append(mem[addrs[i]])
            i += 1
        lines.append(rec(0, a & 0xFFFF, bytes(chunk)))
    lines.append(rec(1, 0, b""))
    with open(hex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def dfu_serial(uf2_path: str, port: str) -> bool:
    """시리얼 DFU: UF2 → HEX → DFU zip (sd-req 0xFFFE = 어떤 SoftDevice 든) → adafruit-nrfutil dfu serial.
    adafruit-nrfutil 은 pip 패키지라 python.exe 아래에서 돌아 Smart App Control 에 안 걸린다 (`python -m nordicsemi`;
    Scripts\\adafruit-nrfutil.exe 런처는 서명이 없어 막힌다). 실측: sd-req 0x00B6 은 init 패킷에서 거부·재부팅, 0xFFFE 는 성공."""
    try:
        import nordicsemi  # noqa: F401
    except ImportError:
        print("adafruit-nrfutil 이 없습니다:  pip install adafruit-nrfutil")
        return False
    work = os.path.join(ROOT, ".build", "win", "dfu")
    os.makedirs(work, exist_ok=True)
    hexf = os.path.join(work, "app.hex")
    pkg = os.path.join(work, "app_dfu.zip")
    uf2_to_hex(uf2_path, hexf)
    cmd = [sys.executable, "-m", "nordicsemi", "dfu", "genpkg", "--dev-type", "0x0052", "--sd-req", "0xFFFE",
           "--application", hexf, pkg]
    print("  $", " ".join(cmd[1:]))
    if subprocess.run(cmd).returncode != 0:
        return False
    cmd = [sys.executable, "-m", "nordicsemi", "dfu", "serial", "-pkg", pkg, "-p", port, "-b", "115200", "--singlebank"]
    print("  $", " ".join(cmd[1:]))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("   " + (r.stdout.strip().splitlines() or ["(no output)"])[-1])
    if "Device programmed" not in r.stdout:
        print(r.stdout[-1500:], r.stderr[-800:])
        return False
    return True


def flash_uf2(path: str) -> int:
    if not os.path.isfile(path):
        print("UF2 파일이 없습니다:", path)
        return 2
    want = source_tag()
    drives, boot = enter_bootloader()
    if drives:
        dst = os.path.join(drives[0], os.path.basename(path))
        print("== 복사 %s → %s (%d 바이트)" % (path, dst, os.path.getsize(path)))
        try:
            shutil.copyfile(path, dst)
        except OSError as e:
            # 부트로더가 쓰기를 마치면 드라이브를 즉시 떼므로 마지막 쓰기/닫기에서 오류가 날 수 있다 — 아래 검증으로 판정
            print("   복사 중 예외 (부트로더가 먼저 재부팅했을 수 있음):", e)
        print("== 재부팅 대기 (드라이브 사라짐 → 앱 포트)")
        wait_for(lambda: not uf2_drives(), 20.0)
    elif boot:
        print("== 시리얼 DFU on", boot[0], "(1200 bps touch 는 드라이브 없이 시리얼 DFU 모드로 들어간다)")
        if not dfu_serial(path, boot[0]):
            print("✘ 시리얼 DFU 실패. 보드는 부트로더에 남아 있다 (COM, PID 0045) — 다시 시도하거나 리셋 두 번 → 드라이브 복사.")
            return 3
    else:
        print("부트로더 (UF2 드라이브 또는 PID 0045 포트) 가 나타나지 않았습니다.")
        return 3
    app = wait_for(lambda: ports(PID_APP), 20.0)
    if not app:
        print("앱 포트가 돌아오지 않았습니다. 드라이브가 아직 열려 있으면 파일이 거부된 것 (INFO_UF2.TXT 의 보드/패밀리 확인).")
        return 4
    time.sleep(1.5)
    name, tag = read_tag(app[0])
    print("보드 %s: 이름 %s, 펌웨어 태그 %s (소스 %s)" % (app[0], name, tag, want))
    if tag != want:
        print("✘ 태그 불일치 — 업로드가 반영되지 않았습니다.")
        return 5
    print("✔ 플래시 완료. 다음: GUI 에서 K → 8 자 워밍업 (rv 3) → S (DCD 저장) → Z → R")
    return 0


# ----------------------------------------------------------------------------- arduino-cli 직접 경로 (SAC 꺼진 PC 용)
def build_and_upload() -> int:
    cli = shutil.which("arduino-cli") or os.path.join(ROOT, ".toolchain", "win", "arduino-cli.exe")
    if not os.path.isfile(cli):
        print("arduino-cli 가 없습니다. Smart App Control 이 꺼진 PC 라면 https://arduino.github.io/arduino-cli 에서 받아 PATH 에.")
        return 2
    try:
        def run(cmd):
            print("  $", " ".join(cmd))
            subprocess.run(cmd, check=True)
        cfg = subprocess.run([cli, "config", "dump"], capture_output=True, text=True).stdout
        if BOARD_URL not in cfg:
            subprocess.run([cli, "config", "init", "--overwrite"], check=True, capture_output=True)
            run([cli, "config", "add", "board_manager.additional_urls", BOARD_URL])
        if "Seeeduino:nrf52" not in subprocess.run([cli, "core", "list"], capture_output=True, text=True).stdout:
            run([cli, "core", "update-index"])
            run([cli, "core", "install", "Seeeduino:nrf52"])
        have = subprocess.run([cli, "lib", "list"], capture_output=True, text=True).stdout
        missing = [l for l in LIBS if l not in have]
        if missing:
            run([cli, "lib", "install", *missing])
        os.makedirs(BUILD, exist_ok=True)
        run([cli, "compile", "--fqbn", FQBN, "--build-property", "compiler.cpp.extra_flags=-DUMI_IMU_ONLY=1",
             "--output-dir", BUILD, SKETCH])
    except OSError as e:
        if getattr(e, "winerror", None) == 4551:
            print("✘ Windows Smart App Control 이 서명 없는 실행파일을 막았습니다 (WinError 4551).\n"
                  "   이 PC 에서는 빌드할 수 없습니다 — 리눅스에서 `imu_bench/scripts/build_uf2.sh` 로 .uf2 를 만들고\n"
                  "   여기서 `flash_win.py --uf2 <file>` 로 복사하십시오.")
            return 6
        raise
    hexf = os.path.join(BUILD, "umi_device_hardware.ino.hex")
    uf2 = os.path.join(BUILD, "umi_device_hardware.uf2")
    subprocess.run([sys.executable, os.path.join(_HERE, "hex2uf2.py"), hexf, "-o", uf2], check=True)
    return flash_uf2(uf2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--uf2", metavar="FILE")
    g.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.check:
        return check()
    if args.uf2:
        return flash_uf2(args.uf2)
    return build_and_upload()


if __name__ == "__main__":
    raise SystemExit(main())
