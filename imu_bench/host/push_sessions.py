#!/usr/bin/env python3
"""수집 세션을 git 없이 LAN 으로 리눅스 학습 PC 에 보낸다 (scp, Windows 내장 OpenSSH).

    python host\\push_sessions.py                     # 새 세션만 → rosotauser@192.168.77.1:~/FR5-for-RUS/imu_bench/logs
    python host\\push_sessions.py --dry-run           # 무엇을 보낼지만
    python host\\push_sessions.py --host 192.168.0.197  # 이더넷 대신 공유기 Wi-Fi 경로
    python host\\push_sessions.py --pull-masks        # 리눅스에서 그린 마스크(Unet_seg/data/phantom_c10ur/masks) 가져오기

경로: 리눅스 PC 는 이 노트북과 두 길로 이어져 있다 (같은 호스트 키 확인, 2026-09-10):
  * 192.168.77.1  — USB 이더넷 직결 (Realtek FE, 100 Mbps ≈ 11 MB/s, 세션 90 MB ≈ 8 s)   ← 기본
  * 192.168.0.197 — 공유기 Wi-Fi (노트북 AX201 링크 576 Mbps; 실제 속도는 공유기에 달림)
기본은 이더넷을 먼저 시도하고 안 되면 Wi-Fi 주소로 넘어간다. 프로브 AP(192.168.1.x, 동글) 는 건드리지 않는다.

한 번만 할 것 (비밀번호 없이 보내려면):
    ssh-keygen -t ed25519 -N "" -f %USERPROFILE%\\.ssh\\id_ed25519
    type %USERPROFILE%\\.ssh\\id_ed25519.pub | ssh rosotauser@192.168.77.1 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
키가 없으면 이 스크립트는 바로 멈춘다 (BatchMode). 위 두 줄은 비밀번호를 한 번만 묻는다.

보내는 것: us_imu_* 중 프레임 ≥10 인 세션 폴더 전체 (에피소드 포함, images/ 제외). 이미 리눅스에 같은 크기의 us_frames.bin 이 있으면 건너뛴다.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "logs"
REPO = HERE.parent.parent
HOSTS = ("192.168.77.1", "192.168.0.197")
# BatchMode: 키 인증만 쓴다 — 없으면 세션마다 비밀번호를 묻는 대신 바로 실패시키고 키 설치를 안내한다.
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=accept-new"]


def ssh(host: str, user: str, cmd: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, f"{user}@{host}", cmd], capture_output=True, text=True, timeout=timeout)


def reachable(host: str, user: str) -> bool:
    try:
        return ssh(host, user, "true", timeout=12).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def local_sessions(min_frames: int = 10) -> list[Path]:
    """유효한 session.meta.json 이 있고 프레임이 min_frames 이상인 세션 (에피소드 모드의 60–300 프레임 세션 포함)."""
    import json
    out = []
    for d in sorted(LOGS.glob("us_imu_*")):
        fb = d / "us_frames.bin"
        mp = d / "session.meta.json"
        if not mp.is_file() or not fb.is_file():
            continue
        try:
            json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue                      # 깨진 시험 세션 (NaN 메타 등)
        if fb.stat().st_size < min_frames * 65536:
            continue
        out.append(d)
    return out


def remote_sizes(host: str, user: str, remote_dir: str) -> dict[str, int]:
    r = ssh(host, user, f"mkdir -p {remote_dir} && cd {remote_dir} && stat -c '%n %s' us_imu_*/us_frames.bin 2>/dev/null")
    sizes = {}
    for line in r.stdout.splitlines():
        try:
            name, size = line.split()
            sizes[name.split("/")[0]] = int(size)
        except ValueError:
            pass
    return sizes


def session_mb(d: Path) -> float:
    return sum(p.stat().st_size for p in d.iterdir() if p.is_file()) / 1e6


def push_session(d: Path, host: str, user: str, remote_dir: str, attempts: int = 3) -> float:
    """tar 스트림 한 번으로 폴더 전송 (images/ 제외): ssh 접속이 세션당 하나라 mkdir+scp 두 번보다 빠르고
    sshd 의 접속 제한에도 덜 걸린다. 실패하면 attempts 회까지 재시도. 전송한 MB 를 돌려준다."""
    mb = session_mb(d)
    last = None
    for k in range(attempts):
        tar = subprocess.Popen(["tar", "-cf", "-", "--exclude", "images", "-C", str(LOGS), d.name],
                               stdout=subprocess.PIPE)
        try:
            r = subprocess.run(["ssh", *SSH_OPTS, f"{user}@{host}",
                                f"mkdir -p {remote_dir} && tar -xf - -C {remote_dir}"],
                               stdin=tar.stdout, capture_output=True, text=True, timeout=60 + mb * 2)
            tar.stdout.close(); tar.wait(timeout=10)
            if r.returncode == 0 and tar.returncode == 0:
                return mb
            last = (r.stderr or "").strip()[-200:] or f"exit {r.returncode}/{tar.returncode}"
        except subprocess.TimeoutExpired:
            last = "timeout"
            tar.kill()
        time.sleep(2 * (k + 1))
        print("    재시도 %d/%d (%s)" % (k + 2, attempts, last)) if k + 1 < attempts else None
    raise RuntimeError(f"전송 실패 ({d.name}): {last}")


def pull_masks(host: str, user: str, remote_repo: str) -> None:
    local = REPO / "Unet_seg" / "data" / "phantom_c10ur" / "masks"
    local.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["scp", "-q", "-r", *SSH_OPTS, f"{user}@{host}:{remote_repo}/Unet_seg/data/phantom_c10ur/masks/.", str(local)])
    n = len(list(local.glob("*.png")))
    print("마스크 %s: 로컬 %d 장 (%s)" % ("가져옴" if r.returncode == 0 else "실패", n, local))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=None, help="기본: 192.168.77.1(이더넷) → 192.168.0.197(Wi-Fi) 순으로 시도")
    ap.add_argument("--user", default="rosotauser")
    ap.add_argument("--remote-repo", default="~/FR5-for-RUS", help="리눅스 쪽 저장소 경로")
    ap.add_argument("--sessions", nargs="*", default=None, help="세션 폴더 이름 (기본: 전부)")
    ap.add_argument("--force", action="store_true", help="리눅스에 있어도 다시 보낸다")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pull-masks", action="store_true", help="세션 전송 대신 리눅스의 masks/ 를 가져온다")
    args = ap.parse_args()

    hosts = [args.host] if args.host else list(HOSTS)
    host = None
    for h in hosts:
        t = time.time()
        if reachable(h, args.user):
            host = h
            print("리눅스 PC: %s@%s  (ssh %.1f s)" % (args.user, h, time.time() - t))
            break
        print("  %s 응답 없음 또는 인증 실패" % h)
    if host is None:
        print("연결 실패. 이더넷 케이블/전원을 확인하고, 키를 아직 안 넣었으면 파일 머리의 한 번만 할 것 을 실행하십시오.")
        return 1

    remote_dir = f"{args.remote_repo}/imu_bench/logs"
    if args.pull_masks:
        pull_masks(host, args.user, args.remote_repo)
        return 0

    have = remote_sizes(host, args.user, remote_dir)
    todo = []
    for d in local_sessions():
        if args.sessions and d.name not in args.sessions:
            continue
        size = (d / "us_frames.bin").stat().st_size
        if not args.force and have.get(d.name) == size:
            continue
        todo.append(d)
    print("리눅스에 있는 세션 %d, 보낼 세션 %d" % (len(have), len(todo)))
    if args.dry_run:
        for d in todo:
            print("  ", d.name, "%.0f MB" % (sum(p.stat().st_size for p in d.iterdir() if p.is_file()) / 1e6))
        return 0
    total_mb, t0 = 0.0, time.time()
    failed: list[str] = []
    others = [h for h in HOSTS if h != host]
    for k, d in enumerate(todo, 1):
        t = time.time()
        try:
            mb = push_session(d, host, args.user, remote_dir)
        except RuntimeError as e:
            # 이 경로가 죽었으면 다른 주소(공유기 Wi-Fi)로 넘어가 본다
            print("  ! %s" % e)
            switched = False
            for h in others:
                if reachable(h, args.user):
                    print("  → %s 로 전환" % h)
                    host, others, switched = h, [x for x in HOSTS if x != h], True
                    break
            if not switched or not reachable(host, args.user):
                failed.append(d.name)
                continue
            try:
                mb = push_session(d, host, args.user, remote_dir)
            except RuntimeError as e2:
                print("  ! %s" % e2)
                failed.append(d.name)
                continue
        total_mb += mb
        print("  [%d/%d] %s  %.0f MB  %.1f MB/s" % (k, len(todo), d.name, mb, mb / max(time.time() - t, 1e-3)))
    if todo:
        print("전송: %.0f MB, %.0f s, 평균 %.1f MB/s → %s:%s" % (
            total_mb, time.time() - t0, total_mb / max(time.time() - t0, 1e-3), host, remote_dir))
    else:
        print("보낼 새 세션이 없습니다.")
    # 검증: 리눅스 쪽 us_frames.bin 크기가 로컬과 같은지
    have = remote_sizes(host, args.user, remote_dir)
    checked = [d for d in local_sessions() if not args.sessions or d.name in args.sessions]
    bad = [d.name for d in checked if have.get(d.name) != (d / "us_frames.bin").stat().st_size]
    if bad or failed:
        print("⚠ 리눅스에 없거나 크기가 다른 세션 %d: %s" % (len(bad), " ".join(bad[:10]) + (" ..." if len(bad) > 10 else "")))
        print("  다시 실행하면 이것들만 보냅니다.")
        return 1
    print("검증 OK: 세션 %d 개가 리눅스에 같은 크기로 있습니다 (리눅스 전체 %d)." % (len(checked), len(have)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
