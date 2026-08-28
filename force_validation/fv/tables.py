"""CSV 읽기·쓰기와 입력 검증.

평면 CSV 몇 개를 다루는 데 pandas 를 들이지 않는다. 표준 ``csv`` 로 충분하고,
그 대신 **무엇이 왜 잘못됐는지** 를 말하는 데 자리를 쓴다 — 빠진 값, 어긋난
sample id, 중복 id, 단위 오류는 조용히 넘어가면 결과가 조용히 틀린다.
"""

from __future__ import annotations

import csv
import math
import os


class InputError(Exception):
    """입력 파일이 쓸 수 없는 상태다. 메시지는 사람이 읽고 고칠 수 있어야 한다."""


def read_rows(path: str) -> list:
    """CSV 한 장을 dict 목록으로 읽는다.

    Raises:
        InputError: 파일이 없거나, 머리글이 없거나, 비어 있을 때.
    """
    if not os.path.exists(path):
        raise InputError(f"파일이 없다: {path}")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InputError(f"머리글이 없다: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise InputError(f"내용이 없다 (머리글만 있다): {path}")
    return rows


def write_rows(path: str, columns, rows) -> None:
    """CSV 를 쓴다. 열 순서는 ``columns`` 를 따른다."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def require_columns(rows, needed, path: str) -> None:
    """필요한 열이 다 있는지 본다.

    Raises:
        InputError: 빠진 열이 있을 때. 어느 열인지 전부 말한다 — 하나씩 알려 주면
            고치고 다시 돌리기를 반복하게 된다.
    """
    present = set(rows[0].keys())
    missing = [name for name in needed if name not in present]
    if missing:
        raise InputError(
            f"{path} 에 없는 열: {', '.join(missing)}\n"
            f"  있는 열: {', '.join(sorted(present))}"
        )


def as_float(value, field: str, sample: str, allow_blank=False):
    """Str 을 실수로. 빈 값은 ``None`` 또는 오류다.

    Raises:
        InputError: 숫자가 아니거나, 빈 값이 허용되지 않을 때.
    """
    text = (value or "").strip()
    if not text:
        if allow_blank:
            return None
        raise InputError(f"sample {sample}: '{field}' 가 비어 있다")
    try:
        number = float(text)
    except ValueError:
        raise InputError(f"sample {sample}: '{field}' 가 숫자가 아니다 ({text!r})") from None
    if not math.isfinite(number):
        raise InputError(f"sample {sample}: '{field}' 가 유한하지 않다 ({text!r})")
    return number


def as_bool(value, default=None):
    """Bool 문자열을 읽는다.

    ``TRUE``/``FALSE``/``1``/``0``/``yes``/``no`` 를 받는다. 비면 기본값.
    """
    text = (value or "").strip().lower()
    if not text:
        return default
    if text in ("true", "1", "yes", "y", "t"):
        return True
    if text in ("false", "0", "no", "n", "f"):
        return False
    return default


def as_sample_id(value, path: str) -> int:
    """Sample id 는 정수여야 한다.

    Raises:
        InputError: 비어 있거나 정수가 아닐 때.
    """
    text = (value or "").strip()
    if not text:
        raise InputError(f"{path}: sample_id 가 빈 행이 있다")
    try:
        return int(float(text))
    except ValueError:
        raise InputError(f"{path}: sample_id 가 정수가 아니다 ({text!r})") from None


def index_by_sample(rows, path: str) -> dict:
    """Sample id 로 색인한다.

    Raises:
        InputError: id 가 중복될 때. 중복은 병합 결과를 조용히 바꾸므로 막는다.
    """
    out: dict = {}
    for row in rows:
        key = as_sample_id(row.get("sample_id"), path)
        if key in out:
            raise InputError(f"{path}: sample_id {key} 가 두 번 나온다")
        out[key] = row
    return out
