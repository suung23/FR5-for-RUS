#!/usr/bin/env python3
"""GT 마스크 수동 편집기 — 로컬 서버.

라벨이 방광 내강 밖까지 삼킨 환자를 사람이 직접 고치기 위한 도구다.
기존 GT 를 불러와 **지우개로 덜어내는 것**이 주 작업이라 브러시 방식으로 만들었다.
419 프레임을 다뤄야 하므로 이동 시 자동 저장하고, 이전 프레임 복사를 넣었다
(초음파 연속 프레임은 거의 같아서 이게 가장 큰 시간 절약이다).

원본은 건드리지 않는다. 편집본은 masks_edited/ 에 따로 쓴다.

    python3 server.py --patients P097 P033 P034
    → http://127.0.0.1:8777

의존성 없음 (stdlib + PIL + numpy).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import posixpath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

DATA = "/home/rosotauser/datasets/pfus/"
EDIT = DATA + "masks_edited/"
PROG = EDIT + "progress.json"
HERE = os.path.dirname(os.path.abspath(__file__))

FRAMES: dict[str, list[dict]] = {}


def load_frames(patients):
    rows = list(csv.DictReader(open(DATA + "manifest.csv")))
    for pid in patients:
        rs = [r for r in rows if r["patient_id"] == pid]
        rs.sort(key=lambda r: int(r["frame_index"]))
        FRAMES[pid] = rs
        print("  %s  %d 프레임" % (pid, len(rs)))


def read_progress() -> dict:
    try:
        return json.load(open(PROG))
    except Exception:
        return {}


def write_progress(pid: str, idx: int) -> None:
    """마지막으로 저장한 프레임을 남긴다 — 재시작 시 그 지점부터 잇는다."""
    d = read_progress()
    d[pid] = idx
    os.makedirs(os.path.dirname(PROG), exist_ok=True)
    json.dump(d, open(PROG, "w"), indent=1)


def edited_path(rec) -> str:
    return EDIT + rec["mask_path"].split("masks/", 1)[1]


def mask_png(rec, force_original: bool = False) -> bytes:
    """편집본이 있으면 그것을, 없으면(또는 원본 요청 시) 원본 GT 를 준다."""
    p = edited_path(rec)
    src = DATA + rec["mask_path"] if force_original or not os.path.exists(p) else p
    a = np.asarray(Image.open(src).convert("L"))
    buf = io.BytesIO()
    Image.fromarray(((a > 127) * 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


class H(BaseHTTPRequestHandler):
    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):     # 콘솔을 조용히
        pass

    def do_GET(self):
        raw = self.path.split("?", 1)
        path = posixpath.normpath(raw[0])
        query = raw[1] if len(raw) > 1 else ""
        if path in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8",
                              open(HERE + "/index.html", "rb").read())
        if path == "/api/patients":
            body = json.dumps({p: len(v) for p, v in FRAMES.items()}).encode()
            return self._send(200, "application/json", body)
        if path.startswith("/api/resume/"):
            pid = path.rsplit("/", 1)[1]
            idx = int(read_progress().get(pid, 0))
            idx = max(0, min(idx, len(FRAMES.get(pid, [1])) - 1))
            return self._send(200, "application/json", json.dumps({"index": idx}).encode())
        if path.startswith("/api/state/"):
            pid = path.rsplit("/", 1)[1]
            done = sum(os.path.exists(edited_path(r)) for r in FRAMES.get(pid, []))
            return self._send(200, "application/json",
                              json.dumps({"n": len(FRAMES.get(pid, [])), "edited": done}).encode())
        orig = "orig=1" in query
        for kind, fn in (("image", lambda r: open(DATA + r["image_path"], "rb").read()),
                         ("mask", lambda r: mask_png(r, orig))):
            pre = "/api/%s/" % kind
            if path.startswith(pre):
                pid, idx = path[len(pre):].split("/")
                rec = FRAMES[pid][int(idx)]
                return self._send(200, "image/png", fn(rec))
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if not self.path.startswith("/api/save/"):
            return self._send(404, "text/plain", b"not found")
        pid, idx = self.path[len("/api/save/"):].split("/")
        rec = FRAMES[pid][int(idx)]
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        a = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
        p = edited_path(rec)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        Image.fromarray(((a > 127) * 255).astype(np.uint8)).save(p)
        write_progress(pid, int(idx))
        self._send(200, "application/json", json.dumps({"ok": True}).encode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="+", default=["P097", "P033", "P034"])
    ap.add_argument("--port", type=int, default=8777)
    a = ap.parse_args()
    load_frames(a.patients)
    os.makedirs(EDIT, exist_ok=True)
    print("\n  편집본 → %s" % EDIT)
    print("  원본은 건드리지 않습니다.\n")
    res = read_progress()
    if res:
        print("  이어서 시작할 지점: " + ", ".join("%s=%d" % kv for kv in sorted(res.items())))
    print("  http://127.0.0.1:%d   (Ctrl+C 로 종료 — 저장분은 남습니다)" % a.port)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  종료. 편집본은 %s 에 있습니다." % EDIT)
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
