"""Frame sources for real-time inference: video files, image directories, cameras.

Every source yields :class:`Frame` records carrying an index and a timestamp, and
every source releases its underlying handle in :meth:`FrameSource.release`, which
is also invoked by the context-manager protocol.

Device paths are never hard-coded: a camera is addressed by integer index or by
an explicit URI supplied through configuration or the command line.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "Frame",
    "FrameSource",
    "VideoFileSource",
    "ImageDirectorySource",
    "CameraSource",
    "open_source",
    "IMAGE_EXTENSIONS",
]

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


@dataclass
class Frame:
    """One captured frame."""

    index: int
    timestamp: float
    image: np.ndarray
    source_id: str = ""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying image array."""
        return tuple(self.image.shape)


class FrameSource(abc.ABC):
    """Abstract source of ultrasound frames."""

    #: Short identifier used in logs.
    name: str = "abstract"

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        """Yield frames until the stream ends."""

    def release(self) -> None:
        """Release any underlying handle. Safe to call more than once."""

    @property
    def is_live(self) -> bool:
        """Whether frames arrive in real time and cannot be replayed.

        Stale-frame dropping is only correct for live sources: for a file or a
        directory every frame is still available a moment later, so discarding
        one loses data for no latency benefit.
        """
        return False

    @property
    def frame_rate(self) -> Optional[float]:
        """Nominal frame rate, when the source reports one."""
        return None

    @property
    def total_frames(self) -> Optional[int]:
        """Total frame count, when known in advance."""
        return None

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Reduce a frame to ``H x W`` float32 in ``[0, 1]``."""
    array = np.asarray(image)
    if array.ndim == 3:
        array = array.mean(axis=2)
    array = array.astype(np.float32)
    if array.max() > 1.0 + 1e-6:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


class VideoFileSource(FrameSource):
    """Frames read sequentially from a video file.

    Args:
        path: Video file.
        grayscale: Convert frames to single-channel ``[0, 1]`` floats.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If OpenCV cannot open it.
    """

    name = "video"

    def __init__(self, path: str | Path, grayscale: bool = True) -> None:
        import cv2

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self.grayscale = grayscale
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(
                f"OpenCV could not open the video {self.path}. Check the codec and file integrity."
            )
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS)) or 0.0
        self._count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame_rate(self) -> Optional[float]:
        return self._fps if self._fps > 0 else None

    @property
    def total_frames(self) -> Optional[int]:
        return self._count if self._count > 0 else None

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            timestamp = index / self._fps if self._fps > 0 else time.perf_counter()
            yield Frame(
                index=index,
                timestamp=float(timestamp),
                image=_to_grayscale(frame) if self.grayscale else frame,
                source_id=self.path.name,
            )
            index += 1

    def release(self) -> None:
        if getattr(self, "_capture", None) is not None:
            self._capture.release()
            self._capture = None  # type: ignore[assignment]


class ImageDirectorySource(FrameSource):
    """Frames read from a directory of images, sorted by file name.

    Sorting by name preserves acquisition order for the usual
    ``frame_000123.png`` convention. Frame timestamps are synthesised from
    ``frame_rate``.

    Raises:
        FileNotFoundError: If the directory does not exist or holds no images.
    """

    name = "images"

    def __init__(
        self,
        directory: str | Path,
        grayscale: bool = True,
        frame_rate: float = 30.0,
        extensions: Sequence[str] = IMAGE_EXTENSIONS,
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.directory}")
        self.grayscale = grayscale
        self._fps = float(frame_rate)
        allowed = {ext.lower() for ext in extensions}
        self.files = sorted(
            path for path in self.directory.iterdir()
            if path.is_file() and path.suffix.lower() in allowed
        )
        if not self.files:
            raise FileNotFoundError(
                f"No images with extensions {sorted(allowed)} found in {self.directory}."
            )

    @property
    def frame_rate(self) -> Optional[float]:
        return self._fps if self._fps > 0 else None

    @property
    def total_frames(self) -> Optional[int]:
        return len(self.files)

    def __iter__(self) -> Iterator[Frame]:
        from ..data.io import load_grayscale

        for index, path in enumerate(self.files):
            image = load_grayscale(path)
            yield Frame(
                index=index,
                timestamp=index / self._fps if self._fps > 0 else float(index),
                image=image,
                source_id=path.name,
            )

    def release(self) -> None:
        return None


class CameraSource(FrameSource):
    """Live frames from a camera index or a stream URI.

    Args:
        source: Integer camera index, or a stream URI/device path string.
        grayscale: Convert frames to single-channel ``[0, 1]`` floats.
        warmup_frames: Frames discarded after opening, while auto-exposure and
            gain settle.
        max_consecutive_failures: Consecutive read failures tolerated before the
            stream is declared ended, so a transient dropout does not stop
            acquisition.

    Raises:
        RuntimeError: If the device or stream cannot be opened.
    """

    name = "camera"

    @property
    def is_live(self) -> bool:
        return True

    def __init__(
        self,
        source: int | str = 0,
        grayscale: bool = True,
        warmup_frames: int = 2,
        max_consecutive_failures: int = 30,
    ) -> None:
        import cv2

        self.source = source
        self.grayscale = grayscale
        self.max_consecutive_failures = int(max_consecutive_failures)
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open camera/stream {source!r}. Check the index, URI or permissions."
            )
        for _ in range(max(0, warmup_frames)):
            self._capture.read()
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS)) or 0.0

    @property
    def frame_rate(self) -> Optional[float]:
        return self._fps if self._fps > 0 else None

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        failures = 0
        start = time.perf_counter()
        while True:
            ok, frame = self._capture.read()
            if not ok:
                failures += 1
                if failures >= self.max_consecutive_failures:
                    logger.warning(
                        "Camera %r produced %d consecutive read failures; ending the stream.",
                        self.source,
                        failures,
                    )
                    break
                continue
            failures = 0
            yield Frame(
                index=index,
                timestamp=time.perf_counter() - start,
                image=_to_grayscale(frame) if self.grayscale else frame,
                source_id=str(self.source),
            )
            index += 1

    def release(self) -> None:
        if getattr(self, "_capture", None) is not None:
            self._capture.release()
            self._capture = None  # type: ignore[assignment]


def open_source(
    spec: str | int,
    grayscale: bool = True,
    frame_rate: float = 30.0,
    **kwargs: Any,
) -> FrameSource:
    """Open the right :class:`FrameSource` for ``spec``.

    Resolution order:

    #. an integer (or an all-digit string) -> :class:`CameraSource` by index;
    #. a string containing ``"://"`` -> :class:`CameraSource` by stream URI;
    #. an existing directory -> :class:`ImageDirectorySource`;
    #. an existing file -> :class:`VideoFileSource`.

    Raises:
        FileNotFoundError: If a path-like spec does not exist.
    """
    if isinstance(spec, int):
        return CameraSource(spec, grayscale=grayscale, **kwargs)
    text = str(spec)
    if text.isdigit():
        return CameraSource(int(text), grayscale=grayscale, **kwargs)
    if "://" in text:
        return CameraSource(text, grayscale=grayscale, **kwargs)

    path = Path(text)
    if path.is_dir():
        return ImageDirectorySource(path, grayscale=grayscale, frame_rate=frame_rate, **kwargs)
    if path.is_file():
        return VideoFileSource(path, grayscale=grayscale, **kwargs)
    raise FileNotFoundError(
        f"Source {text!r} is neither an existing file or directory, a camera index, nor a URI."
    )
