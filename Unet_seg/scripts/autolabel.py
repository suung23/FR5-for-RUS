#!/usr/bin/env python3
"""Semi-automatic bladder-lumen labelling, driven entirely from one terminal.

The lumen is an anechoic (dark), compact, near-field region inside the imaging
sector. That is enough structure for a classical detector to propose a contour
on every frame; a human then keeps or drops each proposal. Proposing is the
expensive part, judging is cheap -- so the human never draws, only arbitrates.

No display server is required. Previews are rendered as ANSI-coloured ASCII in
the terminal, and a PNG overlay is written for every frame so the same decision
can be made by opening the file in an editor instead.

    # 1. propose a lumen contour for every image (batch, no interaction)
    python3 scripts/autolabel.py run --images <dir> --out <dir>

    # 2. arbitrate in the terminal: keep / drop / retune, one keypress each
    python3 scripts/autolabel.py review --out <dir>

    # 3. collect what survived
    python3 scripts/autolabel.py export --out <dir> --masks <dir>

`run` enters `review` automatically unless `--no-review` is given, so the usual
invocation is a single command.

Masks written here are ALGORITHMIC proposals confirmed by a human, not
independent expert annotation. They are recorded as `reference=autolabel` and
must not be relabelled `manual` in any downstream metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

__all__ = ["Candidate", "fan_mask", "propose", "confidence_of"]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
RECORD_NAME = "autolabel.csv"
CONFIG_NAME = "autolabel_config.json"
RAMP = " .:-=+*#%@"

FIELDS = (
    "sample_id",
    "image_path",
    "mask_path",
    "status",            # keep | drop | empty | pending
    "quality",           # good | moderate | poor
    "uncertain_boundary",
    "reviewed",
    "area_px",
    "area_frac",
    "solidity",
    "contrast",
    "depth",
    "confidence",
    "dark_percentile",
    "candidate_rank",
    "num_candidates",
    "notes",
)


# --------------------------------------------------------------------------
# proposal
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    """One lumen proposal within a frame."""

    mask: np.ndarray
    area_px: int
    area_frac: float
    solidity: float
    contrast: float
    depth: float
    score: float

    def metrics(self) -> dict[str, float]:
        """Return the scalar descriptors, rounded for CSV output."""
        return {
            "area_px": int(self.area_px),
            "area_frac": round(self.area_frac, 4),
            "solidity": round(self.solidity, 3),
            "contrast": round(self.contrast, 1),
            "depth": round(self.depth, 3),
        }


def fan_mask(image: np.ndarray) -> np.ndarray:
    """Boolean mask of the imaging sector, excluding the black surround.

    Everything downstream is restricted to this region: the letterbox around a
    B-mode sector is the darkest thing in the frame and would otherwise win any
    "find the dark region" search outright.
    """
    binary = (image > 8).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count < 2:
        return np.ones(image.shape, dtype=bool)
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def propose(
    image: np.ndarray,
    dark_percentile: float = 22.0,
    min_area_frac: float = 0.008,
    max_area_frac: float = 0.55,
) -> list[Candidate]:
    """Rank anechoic candidates for the bladder lumen, best first.

    Args:
        image: Grayscale frame.
        dark_percentile: Intensity percentile (within the sector) treated as the
            fluid/tissue threshold. Lower is stricter.
        min_area_frac: Reject candidates below this fraction of the sector.
        max_area_frac: Reject candidates above it -- a region covering half the
            sector is the far field, not a bladder.

    Returns:
        Candidates sorted by descending score; empty if nothing qualifies.
    """
    sector = fan_mask(image)
    inner = cv2.erode(sector.astype(np.uint8), _ellipse(15)).astype(bool)
    if not inner.any():
        return []

    smooth = cv2.medianBlur(cv2.bilateralFilter(image, 9, 45, 45), 7)
    threshold = max(6.0, float(np.percentile(smooth[inner], dark_percentile)))
    dark = ((smooth <= threshold) & inner).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, _ellipse(11))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, _ellipse(11))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    sector_area = float(inner.sum())
    rows, cols = np.where(inner)
    top, bottom = rows.min(), rows.max()

    candidates: list[Candidate] = []
    for index in range(1, count):
        raw_area = stats[index, cv2.CC_STAT_AREA]
        if not (min_area_frac * sector_area <= raw_area <= max_area_frac * sector_area):
            continue
        component = cv2.morphologyEx(
            (labels == index).astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(25)
        )
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(component)
        cv2.drawContours(filled, [contour], -1, 1, -1)
        mask = (filled > 0) & inner
        area = float(mask.sum())
        if area < min_area_frac * sector_area:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = float(cv2.contourArea(contour) / hull_area) if hull_area > 0 else 0.0
        ring = (
            cv2.dilate(mask.astype(np.uint8), _ellipse(31)).astype(bool) & inner & ~mask
        )
        contrast = float(smooth[ring].mean() - smooth[mask].mean()) if ring.any() else 0.0
        depth = float((np.where(mask)[0].mean() - top) / max(1.0, float(bottom - top)))

        score = (
            1.8 * float(np.sqrt(area / sector_area))
            + 1.2 * solidity
            + 0.020 * contrast
            - 1.6 * max(0.0, depth - 0.55)   # the bladder is near-field
        )
        candidates.append(
            Candidate(mask, int(area), area / sector_area, solidity, contrast, depth, score)
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def confidence_of(candidate: Candidate) -> float:
    """Heuristic confidence in ``[0, 1]`` combining shape, contrast and size."""
    shape = min(1.0, max(0.0, (candidate.solidity - 0.75) / 0.22))
    contrast = min(1.0, max(0.0, candidate.contrast / 35.0))
    size = min(1.0, max(0.0, (candidate.area_frac - 0.008) / 0.10))
    depth = 1.0 - min(1.0, max(0.0, (candidate.depth - 0.5) / 0.4))
    return float(0.35 * shape + 0.35 * contrast + 0.20 * size + 0.10 * depth)


def grade(confidence: float) -> tuple[str, str]:
    """Map a confidence to ``(quality, suggested status)``."""
    if confidence >= 0.62:
        return "good", "keep"
    if confidence >= 0.38:
        return "moderate", "keep"
    return "poor", "drop"


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def ascii_preview(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    reference: Optional[np.ndarray] = None,
    columns: Optional[int] = None,
) -> str:
    """Render the frame as ANSI-coloured ASCII, mask in red, reference in blue."""
    if columns is None:
        columns = max(48, min(shutil.get_terminal_size((100, 40)).columns - 2, 120))
    height, width = image.shape
    rows = max(8, int(columns * height / width * 0.5))
    small = cv2.resize(image, (columns, rows), interpolation=cv2.INTER_AREA)

    def shrink(binary: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if binary is None:
            return None
        resized = cv2.resize(
            (binary > 0).astype(np.uint8) * 255, (columns, rows), interpolation=cv2.INTER_AREA
        )
        return resized > 100

    small_mask = shrink(mask)
    small_reference = shrink(reference)
    if small_reference is not None:
        eroded = cv2.erode(small_reference.astype(np.uint8), np.ones((3, 3), np.uint8))
        small_reference = small_reference & ~(eroded.astype(bool))

    lines = []
    for y in range(rows):
        parts = []
        for x in range(columns):
            char = RAMP[min(len(RAMP) - 1, int(small[y, x]) * len(RAMP) // 256)]
            if small_reference is not None and small_reference[y, x]:
                parts.append(f"\033[94m{char}\033[0m")
            elif small_mask is not None and small_mask[y, x]:
                parts.append(f"\033[91m{char}\033[0m")
            else:
                parts.append(char)
        lines.append("".join(parts))
    return "\n".join(lines)


def overlay_png(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Colour overlay: filled mask tinted red, contour drawn solid."""
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if mask.any():
        tint = canvas.copy()
        tint[mask > 0] = (0, 0, 255)
        canvas = cv2.addWeighted(canvas, 0.72, tint, 0.28, 0.0)
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, (0, 0, 255), 2)
    return canvas


def side_by_side(image: np.ndarray, mask: np.ndarray, caption: str) -> np.ndarray:
    """Raw frame beside the proposal, captioned -- the view the editor shows."""
    left = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    right = overlay_png(image, mask)
    divider = np.full((left.shape[0], 6, 3), 40, dtype=np.uint8)
    pair = np.hstack([left, divider, right])
    scale = min(1.0, 1200.0 / pair.shape[1])
    if scale < 1.0:
        pair = cv2.resize(pair, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    banner = np.zeros((32, pair.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, caption, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([banner, pair])


def find_code_cli() -> Optional[str]:
    """Path to the VS Code CLI, including the remote-server copy, or ``None``."""
    found = shutil.which("code") or shutil.which("code-insiders")
    if found:
        return found
    candidates = sorted(
        Path.home().glob(".vscode-server/cli/servers/*/server/bin/remote-cli/code")
    )
    return str(candidates[-1]) if candidates else None


def open_in_editor(path: Path) -> bool:
    """Show ``path`` in the running VS Code window. Silent when unavailable.

    The reviewer keeps overwriting this one file, and the editor's image preview
    reloads on change -- so a single tab tracks the whole session.
    """
    cli = find_code_cli()
    if cli is None:
        return False
    try:
        subprocess.run(
            [cli, "--reuse-window", str(path)],
            check=False,
            timeout=20,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# record I/O
# --------------------------------------------------------------------------


def read_record(path: Path) -> list[dict[str, str]]:
    """Load ``autolabel.csv``; empty list when absent."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_record(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Atomically rewrite ``autolabel.csv``."""
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})
    temporary.replace(path)


def find_images(directory: Path) -> list[Path]:
    """Sorted image files directly inside ``directory``."""
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def load_gray(path: Path) -> np.ndarray:
    """Read an image as 8-bit grayscale."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"cannot read image: {path}")
    return image


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def command_run(args: argparse.Namespace) -> int:
    """Propose a contour for every image and write masks, overlays and record."""
    images_dir = Path(args.images).resolve()
    out_dir = Path(args.out).resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"--images is not a directory: {images_dir}")
    paths = find_images(images_dir)
    if not paths:
        raise SystemExit(f"no images found in {images_dir}")

    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    existing = {row["sample_id"]: row for row in read_record(out_dir / RECORD_NAME)}
    rows: list[dict[str, str]] = []
    kept_reviews = 0

    for position, path in enumerate(paths, start=1):
        sample_id = path.stem
        previous = existing.get(sample_id)
        if previous and previous.get("reviewed") == "1" and not args.force:
            rows.append(previous)
            kept_reviews += 1
            continue

        image = load_gray(path)
        candidates = propose(image, dark_percentile=args.dark_percentile)
        if candidates:
            best = candidates[0]
            confidence = confidence_of(best)
            quality, status = grade(confidence)
            mask = best.mask
            metrics = best.metrics()
        else:
            best = None
            confidence = 0.0
            quality, status = "poor", "empty"
            mask = np.zeros(image.shape, dtype=bool)
            metrics = {"area_px": 0, "area_frac": 0.0, "solidity": 0.0, "contrast": 0.0, "depth": 0.0}

        mask_path = masks_dir / f"{sample_id}.png"
        cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255)
        cv2.imwrite(str(overlays_dir / f"{sample_id}.png"), overlay_png(image, mask))

        rows.append(
            {
                "sample_id": sample_id,
                "image_path": os.path.relpath(path, out_dir),
                "mask_path": os.path.relpath(mask_path, out_dir),
                "status": status,
                "quality": quality,
                "uncertain_boundary": "1" if confidence < 0.62 else "0",
                "reviewed": "0",
                "confidence": round(confidence, 3),
                "dark_percentile": args.dark_percentile,
                "candidate_rank": 0,
                "num_candidates": len(candidates),
                "notes": "",
                **metrics,
            }
        )
        if position % 20 == 0 or position == len(paths):
            print(f"  proposed {position}/{len(paths)}", flush=True)

    write_record(out_dir / RECORD_NAME, rows)
    (out_dir / CONFIG_NAME).write_text(
        json.dumps(
            {
                "images": str(images_dir),
                "dark_percentile": args.dark_percentile,
                "reference": "autolabel",
                "note": "algorithmic proposals; human-arbitrated in review, not expert annotation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n{len(rows)} frames -> {out_dir}")
    if kept_reviews:
        print(f"  {kept_reviews} already-reviewed rows preserved (--force to redo)")
    summarise(rows)
    if not args.no_review:
        return command_review(
            argparse.Namespace(
                out=str(out_dir),
                all=False,
                show_reference=args.show_reference,
                reference_dir=args.reference_dir,
                viewer=args.viewer,
                ascii_preview=args.ascii_preview,
            )
        )
    print(f"\nnext: python3 scripts/autolabel.py review --out {out_dir}")
    return 0


HELP = """\
 keys   enter/k keep      d drop        e mark empty (no defensible lumen)
        g good  m moderate  p poor      u toggle uncertain-boundary
        [ / ]   darker / brighter threshold (re-segments live)
        a       next candidate region    r reset to best candidate
        j / n   next frame               b / h previous frame
        s       skip to next unreviewed   ? this help    q save and quit\
"""


def _read_key() -> str:
    """Read one keypress without waiting for Enter; falls back to line input."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line.strip()[:1] or "q"
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
    if char == "\x03":
        raise KeyboardInterrupt
    return char


def command_review(args: argparse.Namespace) -> int:
    """Interactive terminal arbitration of the proposals."""
    out_dir = Path(args.out).resolve()
    record_path = out_dir / RECORD_NAME
    rows = read_record(record_path)
    if not rows:
        raise SystemExit(f"{record_path} not found -- run `autolabel.py run` first.")

    reference_dir = Path(args.reference_dir).resolve() if args.reference_dir else None
    order = list(range(len(rows)))
    if not args.all:
        pending = [i for i in order if rows[i].get("reviewed") != "1"]
        order = pending or order

    index = 0
    cache: dict[str, tuple[np.ndarray, list[Candidate]]] = {}
    show_help = False
    viewer_mode = getattr(args, "viewer", "auto")
    use_viewer = viewer_mode != "none" and (viewer_mode == "code" or find_code_cli() is not None)
    viewer_opened = False
    use_viewer_note = False
    current_view = out_dir / "_current.png"
    ascii_mode = getattr(args, "ascii_preview", "auto")
    show_ascii = ascii_mode == "on" or (ascii_mode == "auto" and not use_viewer)

    def load(row: dict[str, str]) -> tuple[np.ndarray, list[Candidate]]:
        key = f"{row['sample_id']}@{row['dark_percentile']}"
        if key not in cache:
            image = load_gray((out_dir / row["image_path"]).resolve())
            cache[key] = (image, propose(image, dark_percentile=float(row["dark_percentile"])))
        return cache[key]

    def apply_candidate(row: dict[str, str], candidates: list[Candidate], image: np.ndarray) -> np.ndarray:
        if not candidates:
            row["status"] = "empty" if row["status"] != "drop" else "drop"
            return np.zeros(image.shape, dtype=bool)
        rank = min(int(row["candidate_rank"]), len(candidates) - 1)
        row["candidate_rank"] = rank
        candidate = candidates[rank]
        confidence = confidence_of(candidate)
        row.update({k: v for k, v in candidate.metrics().items()})
        row["confidence"] = round(confidence, 3)
        row["num_candidates"] = len(candidates)
        return candidate.mask

    def persist(row: dict[str, str], image: np.ndarray, mask: np.ndarray) -> None:
        mask_path = (out_dir / row["mask_path"]).resolve()
        cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255)
        cv2.imwrite(str(out_dir / "overlays" / f"{row['sample_id']}.png"), overlay_png(image, mask))

    dirty = False
    while True:
        row = rows[order[index]]
        image, candidates = load(row)
        mask = apply_candidate(row, candidates, image)
        reference = None
        if args.show_reference and reference_dir is not None:
            candidate_path = reference_dir / f"{row['sample_id']}.png"
            if candidate_path.exists():
                reference = load_gray(candidate_path) > 0

        print("\033[2J\033[H", end="")
        reviewed = sum(1 for r in rows if r.get("reviewed") == "1")
        headline = (
            f"[{index + 1}/{len(order)}]  {row['sample_id']}   "
            f"reviewed {reviewed}/{len(rows)}   "
            f"status={row['status']} quality={row['quality']} "
            f"uncertain={row['uncertain_boundary']}"
        )
        detail = (
            f"  conf {float(row['confidence']):.2f}  area {row['area_px']}px "
            f"({float(row['area_frac']):.3f})  solidity {row['solidity']}  "
            f"contrast {row['contrast']}  depth {row['depth']}  "
            f"cand {int(row['candidate_rank']) + 1}/{row['num_candidates']}  "
            f"dark_pct {row['dark_percentile']}"
        )
        print(headline)
        print(detail)

        if use_viewer:
            caption = (
                f"{row['sample_id']}   [{index + 1}/{len(order)}]   "
                f"conf {float(row['confidence']):.2f}  sol {row['solidity']}  "
                f"ctr {row['contrast']}  depth {row['depth']}      raw | proposal"
            )
            cv2.imwrite(str(current_view), side_by_side(image, mask, caption))
            if not viewer_opened:
                viewer_opened = open_in_editor(current_view)
                if not viewer_opened:
                    print(f"  (could not auto-open; open {current_view} once, it refreshes)")
                    use_viewer_note = True
        if show_ascii:
            print(ascii_preview(image, mask, reference))
        if show_help:
            print(HELP)
        else:
            print(" enter=keep  d=drop  e=empty  [ ]=threshold  a=candidate  j/b=move  ?=help  q=quit")
            if use_viewer and viewer_opened:
                print(f" image: {current_view.name} (VS Code tab refreshes each frame)")

        try:
            key = _read_key()
        except KeyboardInterrupt:
            key = "q"

        if key in ("\r", "\n", "k"):
            row.update(status="keep", reviewed="1")
            persist(row, image, mask)
            dirty = True
            index = min(index + 1, len(order) - 1)
        elif key == "d":
            row.update(status="drop", reviewed="1")
            dirty = True
            index = min(index + 1, len(order) - 1)
        elif key == "e":
            row.update(status="empty", quality="poor", uncertain_boundary="1", reviewed="1")
            persist(row, image, np.zeros(image.shape, dtype=bool))
            dirty = True
            index = min(index + 1, len(order) - 1)
        elif key in ("g", "m", "p"):
            row["quality"] = {"g": "good", "m": "moderate", "p": "poor"}[key]
            dirty = True
        elif key == "u":
            row["uncertain_boundary"] = "0" if row["uncertain_boundary"] == "1" else "1"
            dirty = True
        elif key in ("[", "]"):
            step = -3.0 if key == "[" else 3.0
            row["dark_percentile"] = round(
                min(45.0, max(5.0, float(row["dark_percentile"]) + step)), 1
            )
            row["candidate_rank"] = 0
            dirty = True
        elif key == "a":
            row["candidate_rank"] = int(row["candidate_rank"]) + 1
            if int(row["candidate_rank"]) >= max(1, int(row["num_candidates"])):
                row["candidate_rank"] = 0
            dirty = True
        elif key == "r":
            row["candidate_rank"] = 0
            dirty = True
        elif key in ("j", "n"):
            index = min(index + 1, len(order) - 1)
        elif key in ("b", "h"):
            index = max(index - 1, 0)
        elif key == "s":
            nxt = [i for i in range(index + 1, len(order)) if rows[order[i]].get("reviewed") != "1"]
            index = nxt[0] if nxt else index
        elif key == "?":
            show_help = not show_help
        elif key == "q":
            break

        if dirty:
            write_record(record_path, rows)
            dirty = False

    write_record(record_path, rows)
    print("\033[2J\033[H", end="")
    summarise(rows)
    print(f"\nrecord: {record_path}")
    return 0


def summarise(rows: Sequence[dict[str, str]]) -> None:
    """Print status/quality counts."""
    def tally(field: str) -> str:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.get(field, "")] = counts.get(row.get(field, ""), 0) + 1
        return "  ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k)

    reviewed = sum(1 for row in rows if row.get("reviewed") == "1")
    print(f"status   {tally('status')}")
    print(f"quality  {tally('quality')}")
    print(f"reviewed {reviewed}/{len(rows)}")


def command_export(args: argparse.Namespace) -> int:
    """Copy masks of kept frames into a destination directory."""
    out_dir = Path(args.out).resolve()
    destination = Path(args.masks).resolve()
    rows = read_record(out_dir / RECORD_NAME)
    if not rows:
        raise SystemExit(f"{out_dir / RECORD_NAME} not found.")
    destination.mkdir(parents=True, exist_ok=True)

    wanted = {"keep"} if not args.include_empty else {"keep", "empty"}
    if args.include_unreviewed:
        selected = [r for r in rows if r["status"] in wanted]
    else:
        selected = [r for r in rows if r["status"] in wanted and r.get("reviewed") == "1"]

    for row in selected:
        source = (out_dir / row["mask_path"]).resolve()
        if source.exists():
            shutil.copy2(source, destination / f"{row['sample_id']}.png")
    print(f"exported {len(selected)}/{len(rows)} masks -> {destination}")
    dropped = [r["sample_id"] for r in rows if r["status"] == "drop"]
    if dropped:
        print(f"dropped {len(dropped)}: {', '.join(dropped[:8])}{' ...' if len(dropped) > 8 else ''}")
    return 0


def command_stats(args: argparse.Namespace) -> int:
    """Print the current record summary."""
    rows = read_record(Path(args.out).resolve() / RECORD_NAME)
    if not rows:
        raise SystemExit("no record found.")
    summarise(rows)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="batch-propose contours, then review")
    run.add_argument("--images", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--dark-percentile", type=float, default=22.0)
    run.add_argument("--no-review", action="store_true")
    run.add_argument("--force", action="store_true", help="re-propose already-reviewed frames")
    run.add_argument("--show-reference", action="store_true")
    run.add_argument("--reference-dir", default=None)
    run.add_argument("--viewer", choices=("auto", "code", "none"), default="auto")
    run.add_argument("--ascii", dest="ascii_preview", choices=("auto", "on", "off"), default="auto")
    run.set_defaults(func=command_run)

    review = sub.add_parser("review", help="terminal keep/drop arbitration")
    review.add_argument("--out", required=True)
    review.add_argument("--all", action="store_true", help="revisit already-reviewed frames too")
    review.add_argument("--show-reference", action="store_true", help="overlay an existing mask in blue")
    review.add_argument("--reference-dir", default=None)
    review.add_argument("--viewer", choices=("auto", "code", "none"), default="auto",
                        help="auto-open a live preview in VS Code (default: auto-detect)")
    review.add_argument("--ascii", dest="ascii_preview", choices=("auto", "on", "off"), default="auto",
                        help="terminal ASCII preview; auto = only when no viewer")
    review.set_defaults(func=command_review)

    export = sub.add_parser("export", help="copy kept masks out")
    export.add_argument("--out", required=True)
    export.add_argument("--masks", required=True)
    export.add_argument("--include-empty", action="store_true")
    export.add_argument("--include-unreviewed", action="store_true")
    export.set_defaults(func=command_export)

    stats = sub.add_parser("stats", help="print record summary")
    stats.add_argument("--out", required=True)
    stats.set_defaults(func=command_stats)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
