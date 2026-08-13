"""Structured logging plus JSONL and CSV writers.

The writers are used by the evaluation and real-time monitoring paths. Both
flush per record so a log survives an abrupt shutdown of the monitor, and both
are context managers so file handles are always released.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, TextIO

__all__ = ["setup_logging", "JsonlWriter", "CsvWriter"]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    stream: Optional[TextIO] = None,
) -> logging.Logger:
    """Configure the root logger.

    Args:
        level: Logging level name.
        log_file: Optional file to mirror the log into.
        stream: Stream for console output; defaults to stderr.

    Returns:
        The configured root logger.

    Raises:
        ValueError: If ``level`` is not a valid logging level.
    """
    numeric = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown logging level {level!r}.")

    root = logging.getLogger()
    root.setLevel(numeric)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(stream or sys.stderr)
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(file_handler)
    return root


class JsonlWriter:
    """Append-only JSON Lines writer.

    Every record is written as one JSON object per line and flushed immediately.
    """

    def __init__(self, path: str | Path, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: Optional[TextIO] = self.path.open(
            "a" if append else "w", encoding="utf-8"
        )
        self.count = 0

    def write(self, record: Mapping[str, Any]) -> None:
        """Write one record.

        Raises:
            RuntimeError: If the writer has been closed.
            TypeError: If the record is not JSON-serialisable.
        """
        if self._handle is None:
            raise RuntimeError(f"JsonlWriter for {self.path} is already closed.")
        self._handle.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
        self._handle.flush()
        self.count += 1

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class CsvWriter:
    """CSV writer that fixes its column set from the first record.

    Fields absent from a later record are written empty; fields that appear
    later and were not in the header are dropped, with a one-time warning, so
    the file stays rectangular and machine-readable.
    """

    def __init__(self, path: str | Path, fieldnames: Optional[Iterable[str]] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: Optional[TextIO] = self.path.open("w", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list[str]] = list(fieldnames) if fieldnames else None
        self._warned = False
        self.count = 0
        if self._fieldnames:
            self._start(self._fieldnames)

    def _start(self, fieldnames: list[str]) -> None:
        assert self._handle is not None
        self._fieldnames = fieldnames
        self._writer = csv.DictWriter(self._handle, fieldnames=fieldnames, extrasaction="ignore")
        self._writer.writeheader()

    def write(self, record: Mapping[str, Any]) -> None:
        """Write one record.

        Raises:
            RuntimeError: If the writer has been closed.
        """
        if self._handle is None:
            raise RuntimeError(f"CsvWriter for {self.path} is already closed.")
        if self._writer is None:
            self._start(list(record.keys()))
        assert self._writer is not None and self._fieldnames is not None
        extra = set(record) - set(self._fieldnames)
        if extra and not self._warned:
            logging.getLogger(__name__).warning(
                "CsvWriter %s: dropping field(s) not present in the header: %s",
                self.path,
                sorted(extra),
            )
            self._warned = True
        self._writer.writerow({key: record.get(key, "") for key in self._fieldnames})
        self._handle.flush()
        self.count += 1

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "CsvWriter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
