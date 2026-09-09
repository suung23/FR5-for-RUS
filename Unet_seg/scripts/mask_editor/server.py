#!/usr/bin/env python3
"""수동 마스크 편집기 (브라우저) — manifest 데이터셋 어디에나 붙는다. 빈 마스크에서 그리기 시작 가능.

experiments/label_review/mask_editor 는 PFUS GT 를 *덜어내는* 용도로 경로가 고정돼 있었다. 이것은
`scripts/prepare_phantom_sessions.py` 가 만든 데이터셋(또는 같은 형식의 어떤 것)에 붙어 **처음부터 그리는** 편집기다.

    python scripts/mask_editor/server.py --data data/phantom_c10ur --queue     # label_queue.csv 의 프레임만
    python scripts/mask_editor/server.py --data data/phantom_c10ur --queue --queue-file label_queue_round2.csv
    python scripts/mask_editor/server.py --data data/phantom_c10ur             # manifest 의 모든 프레임
    → http://127.0.0.1:8778

저장: <data>/masks/<image name>.png (0/255).  획마다 자동 저장.  진행 상황은 <data>/masks/progress.json.
그리는 법: P 다각형(클릭으로 꼭짓점, Enter 로 채움) 이 가장 빠르다. D/E 브러시로 손보기. N = "방광 없음"(빈 마스크 저장).
끝나면  python scripts/prepare_phantom_sessions.py --output-dir <data> --refresh  로 manifest 의 mask_path 를 채운다.

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

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = ""
MASKS = ""
PROG = ""
FRAMES: dict[str, list[dict]] = {}


def load_frames(data: str, queue_only: bool, queue_name: str = "label_queue.csv") -> None:
    man = os.path.join(data, "manifest.csv")
    rows = list(csv.DictReader(open(man, encoding="utf-8")))
    keep = None
    if queue_only:
        qp = queue_name if os.path.isabs(queue_name) else os.path.join(data, queue_name)
        keep = {(r["patient_id"], int(r["frame_index"])) for r in csv.DictReader(open(qp, encoding="utf-8"))}
    for r in rows:
        key = (r["patient_id"], int(r["frame_index"]))
        if keep is not None and key not in keep:
            continue
        r["mask_path"] = r.get("mask_path") or ("masks/" + os.path.basename(r["image_path"]))
        FRAMES.setdefault(r["patient_id"], []).append(r)
    for pid, rs in FRAMES.items():
        rs.sort(key=lambda r: int(r["frame_index"]))
        print("  %s  %d 프레임  (라벨 %d)" % (pid, len(rs), sum(os.path.exists(mask_file(r)) for r in rs)))


def mask_file(rec) -> str:
    return os.path.join(DATA, rec["mask_path"])


def read_progress() -> dict:
    try:
        return json.load(open(PROG, encoding="utf-8"))
    except Exception:
        return {}


def write_progress(pid: str, idx: int) -> None:
    d = read_progress()
    d[pid] = idx
    os.makedirs(os.path.dirname(PROG), exist_ok=True)
    json.dump(d, open(PROG, "w", encoding="utf-8"), indent=1)


def image_png(rec) -> bytes:
    return open(os.path.join(DATA, rec["image_path"]), "rb").read()


def mask_png(rec) -> bytes:
    p = mask_file(rec)
    if os.path.exists(p):
        a = np.asarray(Image.open(p).convert("L"))
    else:
        w, h = Image.open(os.path.join(DATA, rec["image_path"])).size
        a = np.zeros((h, w), np.uint8)
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

    def log_message(self, *a):
        pass

    def do_GET(self):
        raw = self.path.split("?", 1)
        path = posixpath.normpath(raw[0])
        if path in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8", open(os.path.join(HERE, "index.html"), "rb").read())
        if path == "/api/patients":
            body = json.dumps({p: {"n": len(v), "done": sum(os.path.exists(mask_file(r)) for r in v)}
                               for p, v in FRAMES.items()}).encode()
            return self._send(200, "application/json", body)
        if path.startswith("/api/resume/"):
            pid = path.rsplit("/", 1)[1]
            idx = int(read_progress().get(pid, 0))
            idx = max(0, min(idx, len(FRAMES.get(pid, [1])) - 1))
            return self._send(200, "application/json", json.dumps({"index": idx}).encode())
        if path.startswith("/api/state/"):
            pid = path.rsplit("/", 1)[1]
            rs = FRAMES.get(pid, [])
            flags = [os.path.exists(mask_file(r)) for r in rs]
            body = json.dumps({"n": len(rs), "edited": sum(flags), "labeled": flags,
                               "frames": [int(r["frame_index"]) for r in rs],
                               "t": [r.get("timestamp", "") for r in rs]}).encode()
            return self._send(200, "application/json", body)
        for kind, fn in (("image", image_png), ("mask", mask_png)):
            pre = "/api/%s/" % kind
            if path.startswith(pre):
                pid, idx = path[len(pre):].split("/")
                return self._send(200, "image/png", fn(FRAMES[pid][int(idx)]))
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path.startswith("/api/delete/"):
            pid, idx = self.path[len("/api/delete/"):].split("/")
            p = mask_file(FRAMES[pid][int(idx)])
            if os.path.exists(p):
                os.remove(p)
            return self._send(200, "application/json", json.dumps({"ok": True}).encode())
        if not self.path.startswith("/api/save/"):
            return self._send(404, "text/plain", b"not found")
        pid, idx = self.path[len("/api/save/"):].split("/")
        rec = FRAMES[pid][int(idx)]
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        a = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
        p = mask_file(rec)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        Image.fromarray(((a > 127) * 255).astype(np.uint8)).save(p)
        write_progress(pid, int(idx))
        self._send(200, "application/json", json.dumps({"ok": True}).encode())


def main() -> int:
    global DATA, MASKS, PROG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="manifest.csv 가 있는 데이터셋 폴더")
    ap.add_argument("--queue", action="store_true", help="라벨 큐 CSV 의 프레임만")
    ap.add_argument("--queue-file", default="label_queue.csv",
                    help="큐 CSV (데이터 폴더 기준 상대경로 또는 절대경로). 라운드마다 큐를 갈아끼울 때 쓴다")
    ap.add_argument("--port", type=int, default=8778)
    a = ap.parse_args()
    DATA = os.path.abspath(a.data)
    MASKS = os.path.join(DATA, "masks")
    PROG = os.path.join(MASKS, "progress.json")
    os.makedirs(MASKS, exist_ok=True)
    load_frames(DATA, a.queue, a.queue_file)
    if not FRAMES:
        print("프레임이 없습니다. prepare_phantom_sessions.py 를 먼저 실행하십시오.")
        return 1
    print("\n  마스크 → %s" % MASKS)
    print("  http://127.0.0.1:%d   (Ctrl+C 로 종료 — 저장분은 남습니다)" % a.port)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  종료. 마스크는 %s 에 있습니다." % MASKS)
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
