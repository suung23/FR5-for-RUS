#!/usr/bin/env python3
"""GT 라벨 육안 판정기 — 로컬 서버.

자동 라벨품질 스크리닝은 이 데이터셋에서 판별력이 없었다(AUC 0.45~0.49). 시각적으로
멀쩡한 환자를 매번 P021 아래로 랭크했으므로 폐기했고, 판정은 사람이 눈으로 한다.
이 도구는 그 판정을 빠르게(환자당 10~20초) 기록하기 위한 것이다.

batch_A/batch_B 의 정적 HTML 을 대체한다. 그쪽은 라디오 버튼이 저장되지 않아
verdicts.csv 를 손으로 채워야 했다. 여기서는 키 한 번이 곧 디스크 기록이다.

시트(PNG)는 없으면 그 자리에서 만든다. batch_A/B 에 이미 있는 48명 분은 재사용하고,
시트조차 없던 34명은 처음 열릴 때 렌더링한다(백그라운드로 미리 당겨둔다).

원본 마스크도 매니페스트도 건드리지 않는다. 쓰는 것은 verdicts.csv 하나뿐이다.

    python3 server.py                 # 미판정 82명
    python3 server.py --all           # 110명 전부 (기존 판정 재확인)
    python3 server.py --patients P022 P025
    → http://127.0.0.1:8778

의존성: stdlib + PIL + numpy + matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import posixpath
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = "/home/rosotauser/datasets/pfus/"
HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.dirname(HERE)                      # experiments/label_review
SHEETS = os.path.join(HERE, "sheets")
VERDICTS = os.path.join(HERE, "verdicts.csv")
LEGACY = os.path.join(REVIEW, "VERDICTS.csv")
BATCH_DIRS = [os.path.join(REVIEW, "batch_A"), os.path.join(REVIEW, "batch_B")]

VALID = ("label_ok", "partial", "label_wrong", "unjudgeable")
FIELDS = ["patient_id", "split", "n_frames", "verdict", "note", "dice"]

RENDER_LOCK = threading.Lock()      # matplotlib 은 스레드 안전하지 않다
WRITE_LOCK = threading.Lock()

PATIENTS: dict[str, dict] = {}      # pid -> {split, n_frames, rows, dice}
QUEUE: list[str] = []
VERD: dict[str, dict] = {}


# ---------------------------------------------------------------- 데이터 적재

def load_patients() -> None:
    rows = list(csv.DictReader(open(DATA + "manifest.csv")))
    for r in rows:
        p = PATIENTS.setdefault(r["patient_id"], {"split": r["split"], "rows": []})
        p["rows"].append(r)
    for p in PATIENTS.values():
        p["rows"].sort(key=lambda r: int(r["frame_index"]))
        p["n_frames"] = len(p["rows"])


def load_dice() -> None:
    """runs/ 아래 모든 frame_metrics.csv 에서 프레임별 Dice 를 모은다.

    train 환자는 어떤 평가에도 없으므로 Dice 가 없다. 그게 옳다 — 학습 데이터에
    대한 모델 점수는 암기를 재는 것이라 라벨 판정의 근거가 될 수 없다.
    최신 평가가 이기도록 mtime 순으로 덮어쓴다.
    """
    root = os.path.join(os.path.dirname(REVIEW), "..", "runs")
    root = os.path.normpath(root)
    found = []
    for dirpath, _, files in os.walk(root):
        if "frame_metrics.csv" in files:
            p = os.path.join(dirpath, "frame_metrics.csv")
            found.append((os.path.getmtime(p), p))
    # 'last' 평가를 먼저 깔고 'best' 로 덮는다 — 모델 선택 기준이 best.pt 이므로
    # 화면에 뜨는 Dice 도 best.pt 것이어야 한다. mtime 만 쓰면 마지막에 돌린
    # last.pt 평가가 이겨서 환자가 실제보다 나쁘게 보인다.
    found.sort(key=lambda t: (0 if "last" in t[1] else 1, t[0]))
    for _, path in found:
        try:
            for r in csv.DictReader(open(path)):
                pid = r.get("patient_id")
                if pid not in PATIENTS or not r.get("dice"):
                    continue
                PATIENTS[pid].setdefault("dice", {})[int(r["frame_index"])] = float(r["dice"])
        except Exception:
            continue
    for pid, p in PATIENTS.items():
        d = p.get("dice") or {}
        p["dice_mean"] = round(float(np.mean(list(d.values()))), 3) if d else None


def seed_verdicts() -> None:
    """기존 판정을 하나의 CSV 로 모은다. verdicts.csv 가 이미 있으면 그것이 진실."""
    if os.path.exists(VERDICTS):
        for r in csv.DictReader(open(VERDICTS)):
            VERD[r["patient_id"]] = r
        return
    if os.path.exists(LEGACY):
        for r in csv.reader(open(LEGACY)):
            if len(r) >= 4 and r[0] in PATIENTS:
                VERD[r[0]] = {"patient_id": r[0], "split": PATIENTS[r[0]]["split"],
                              "n_frames": PATIENTS[r[0]]["n_frames"], "verdict": r[3],
                              "note": (r[4] if len(r) > 4 else ""),
                              "dice": PATIENTS[r[0]].get("dice_mean") or ""}
    flush()


def flush() -> None:
    with WRITE_LOCK:
        tmp = VERDICTS + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for pid in sorted(VERD):
                w.writerow({k: VERD[pid].get(k, "") for k in FIELDS})
        os.replace(tmp, VERDICTS)


# ---------------------------------------------------------------- 시트 렌더링

def contour(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    inner = m.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        inner &= np.roll(m, (dy, dx), axis=(0, 1))
    return m & ~inner


def existing_sheet(pid: str, kind: str) -> str | None:
    name = "%s_%s.png" % (pid, kind)
    for d in [SHEETS] + BATCH_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def render(pid: str, kind: str) -> str:
    """<PID>_zoom.png / <PID>_context.png 를 만든다. 이미 있으면 그대로 쓴다."""
    have = existing_sheet(pid, kind)
    if have:
        return have
    with RENDER_LOCK:
        have = existing_sheet(pid, kind)
        if have:
            return have
        os.makedirs(SHEETS, exist_ok=True)
        out = os.path.join(SHEETS, "%s_%s.png" % (pid, kind))
        rows = PATIENTS[pid]["rows"]
        dices = PATIENTS[pid].get("dice") or {}
        zoom = kind == "zoom"
        pick = np.linspace(0, len(rows) - 1, min(20, len(rows))).astype(int)
        cols = 5
        nrow = int(np.ceil(len(pick) / cols))
        fig, axes = plt.subplots(nrow, cols, figsize=(cols * 3.0, nrow * 3.0))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, i in zip(axes, pick):
            r = rows[i]
            img = np.asarray(Image.open(DATA + r["image_path"]).convert("L"), np.float32)
            gt = np.asarray(Image.open(DATA + r["mask_path"])) > 0
            rgb = np.stack([img] * 3, -1) / 255.0
            if gt.any():
                rgb[contour(gt)] = [0.1, 1.0, 0.3]
            if zoom and gt.any():
                ys, xs = np.nonzero(gt)
                cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
                h = int(max(np.ptp(ys), np.ptp(xs)) * 0.72) + 14
                rgb = rgb[max(cy - h, 0):min(cy + h, rgb.shape[0]),
                          max(cx - h, 0):min(cx + h, rgb.shape[1])]
            ax.imshow(np.clip(rgb, 0, 1))
            fid = os.path.splitext(os.path.basename(r["image_path"]))[0]
            dc = dices.get(int(r["frame_index"]))
            ax.set_title("%s   Dice %s" % (fid, "n/a" if dc is None else "%.2f" % dc),
                         fontsize=8, color=("#b0452b" if (dc is not None and dc < 0.5) else "#333"))
        fig.suptitle("%s  —  GT 라벨 검토, %s   (초록 = GT 윤곽, 모델 예측 없음)"
                     % (pid, "확대" if zoom else "전체 프레임"), fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return out


def prerender_worker() -> None:
    """다음 환자들의 시트를 미리 만들어 둔다 — 판정 중 기다리지 않도록."""
    for pid in QUEUE:
        for kind in ("zoom", "context"):
            try:
                render(pid, kind)
            except Exception as exc:      # 한 환자가 깨져도 전체는 계속
                print("  ! %s %s 렌더 실패: %s" % (pid, kind, exc))


# ---------------------------------------------------------------------- 서버

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
        path = posixpath.normpath(self.path.split("?", 1)[0])
        if path in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8",
                              open(os.path.join(HERE, "index.html"), "rb").read())
        if path == "/api/queue":
            body = [{"patient_id": pid, "split": PATIENTS[pid]["split"],
                     "n_frames": PATIENTS[pid]["n_frames"],
                     "dice": PATIENTS[pid].get("dice_mean"),
                     "verdict": VERD.get(pid, {}).get("verdict", ""),
                     "note": VERD.get(pid, {}).get("note", "")} for pid in QUEUE]
            return self._send(200, "application/json", json.dumps(body).encode())
        if path.startswith("/api/sheet/"):
            _, _, _, pid, kind = path.split("/")
            if pid not in PATIENTS or kind not in ("zoom", "context"):
                return self._send(404, "text/plain", b"not found")
            try:
                return self._send(200, "image/png", open(render(pid, kind), "rb").read())
            except Exception as exc:
                return self._send(500, "text/plain", str(exc).encode())
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if not self.path.startswith("/api/verdict"):
            return self._send(404, "text/plain", b"not found")
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        pid = req.get("patient_id")
        verdict = req.get("verdict", "")
        if pid not in PATIENTS or (verdict and verdict not in VALID):
            return self._send(400, "application/json", b'{"ok":false}')
        if verdict == "":                       # 판정 취소
            VERD.pop(pid, None)
        else:
            VERD[pid] = {"patient_id": pid, "split": PATIENTS[pid]["split"],
                         "n_frames": PATIENTS[pid]["n_frames"], "verdict": verdict,
                         "note": req.get("note", ""),
                         "dice": PATIENTS[pid].get("dice_mean") or ""}
        flush()
        done = sum(1 for p in QUEUE if p in VERD)
        self._send(200, "application/json",
                   json.dumps({"ok": True, "done": done, "total": len(QUEUE)}).encode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*", default=None, help="이 환자들만 본다")
    ap.add_argument("--all", action="store_true", help="이미 판정된 환자도 큐에 넣는다")
    ap.add_argument("--port", type=int, default=8778)
    ap.add_argument("--no-prerender", action="store_true")
    a = ap.parse_args()

    load_patients()
    load_dice()
    seed_verdicts()

    if a.patients:
        QUEUE.extend([p for p in a.patients if p in PATIENTS])
    else:
        QUEUE.extend(sorted(p for p in PATIENTS if a.all or p not in VERD))

    done = sum(1 for p in QUEUE if p in VERD)
    print("  환자 %d명 (전체 %d, 기존 판정 %d)" % (len(QUEUE), len(PATIENTS), len(VERD)))
    print("  판정 기록 → %s" % VERDICTS)
    print("  원본 마스크와 매니페스트는 건드리지 않습니다.")
    if not a.no_prerender:
        threading.Thread(target=prerender_worker, daemon=True).start()
        print("  시트를 백그라운드로 미리 렌더링합니다 (없는 34명 분).")
    print("\n  http://127.0.0.1:%d      진행 %d/%d   (Ctrl+C 종료 — 판정은 남습니다)\n"
          % (a.port, done, len(QUEUE)))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  종료. 판정 %d건이 %s 에 있습니다." % (len(VERD), VERDICTS))
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
