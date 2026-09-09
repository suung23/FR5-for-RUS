#!/usr/bin/env python3
"""Intel HEX → UF2 (nRF52840, Adafruit/Seeed UF2 부트로더용). 외부 의존성 없음.

    python3 host/hex2uf2.py .build/imu_only/umi_device_hardware.ino.hex -o .build/imu_only/umi_device_hardware.uf2

왜 있나 (2026-09-10): Windows 수집 노트북은 Smart App Control 이 켜져 있어 서명 없는 arduino-cli·gcc·nrfutil 이 실행되지
않는다. 그래서 빌드는 리눅스(scripts/flash.sh 와 같은 툴체인)에서 하고, 결과를 UF2 로 바꿔 Windows 에서는 **파일 복사만**
으로 올린다 — XIAO nRF52840 의 부트로더가 1200 bps touch / 리셋 두 번에 "XIAO-SENSE" 드라이브를 열고, 거기 UF2 를 넣으면
스스로 쓰고 재부팅한다 (host/flash_win.py --uf2 가 이 순서를 자동으로 한다).

UF2 규격 (github.com/microsoft/uf2): 512 바이트 블록 = 32 바이트 헤더 + 476 바이트 데이터 + 4 바이트 끝 매직.
nRF52840 family ID 0xADA52840, 블록당 페이로드 256 바이트, flags 0x2000 (family ID 있음).
"""

from __future__ import annotations

import argparse
import struct
import sys

UF2_MAGIC0, UF2_MAGIC1, UF2_MAGIC_END = 0x0A324655, 0x9E5D5157, 0x0AB16F30
UF2_FLAG_FAMILY = 0x00002000
FAMILY_NRF52840 = 0xADA52840
PAYLOAD = 256


def parse_ihex(text: str) -> dict[int, int]:
    """Intel HEX → {절대 주소: 바이트}. 레코드 00 (data), 01 (EOF), 02 (ext. segment), 04 (ext. linear) 를 다룬다."""
    mem: dict[int, int] = {}
    base = 0
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line[0] != ":":
            raise ValueError("line %d: ':' 로 시작하지 않음" % ln)
        raw = bytes.fromhex(line[1:])
        if (sum(raw) & 0xFF) != 0:
            raise ValueError("line %d: 체크섬 불일치" % ln)
        n, addr, rtype = raw[0], (raw[1] << 8) | raw[2], raw[3]
        data = raw[4:4 + n]
        if rtype == 0x00:
            for i, b in enumerate(data):
                mem[base + addr + i] = b
        elif rtype == 0x01:
            break
        elif rtype == 0x02:
            base = ((data[0] << 8) | data[1]) << 4
        elif rtype == 0x04:
            base = ((data[0] << 8) | data[1]) << 16
        elif rtype in (0x03, 0x05):
            continue
        else:
            raise ValueError("line %d: 모르는 레코드 타입 %02X" % (ln, rtype))
    return mem


def to_uf2(mem: dict[int, int], family: int = FAMILY_NRF52840, payload: int = PAYLOAD) -> bytes:
    if not mem:
        raise ValueError("데이터 없음")
    lo = min(mem) - (min(mem) % payload)
    hi = max(mem)
    addrs = list(range(lo, hi + 1, payload))
    # 비어 있는 페이지(0xFF 만) 는 건너뛴다 — 부트로더가 그 영역을 지우지 않도록
    chunks = []
    for a in addrs:
        block = bytes(mem.get(a + i, 0xFF) for i in range(payload))
        if any(b != 0xFF for b in block):
            chunks.append((a, block))
    out = bytearray()
    total = len(chunks)
    for i, (a, block) in enumerate(chunks):
        hdr = struct.pack("<IIIIIIII", UF2_MAGIC0, UF2_MAGIC1, UF2_FLAG_FAMILY, a, payload, i, total, family)
        out += hdr + block + bytes(476 - payload) + struct.pack("<I", UF2_MAGIC_END)
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hex")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--family", type=lambda s: int(s, 0), default=FAMILY_NRF52840)
    args = ap.parse_args()
    mem = parse_ihex(open(args.hex, encoding="ascii").read())
    data = to_uf2(mem, args.family)
    with open(args.out, "wb") as fh:
        fh.write(data)
    print("%s → %s: %d 블록, 0x%05X..0x%05X (%d 바이트)" % (
        args.hex, args.out, len(data) // 512, min(mem), max(mem), len(mem)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
