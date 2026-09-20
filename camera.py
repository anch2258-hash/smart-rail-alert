"""OpenCV video capture wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameSource:
    capture: cv2.VideoCapture | None = None
    description: str = "Не подключено"
    frame_interval_seconds: float | None = None

    def open_camera(self, index: int, width: int, height: int) -> None:
        self.release()
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Не удалось открыть камеру с индексом {index}")
        self.capture, self.description = capture, f"Веб-камера {index}"
        self.frame_interval_seconds = None

    def open_video(self, path: str | Path) -> None:
        video_path = Path(path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
        self.release()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        self.capture, self.description = capture, str(video_path)
        self.frame_interval_seconds = 1 / source_fps if source_fps > 0 else None

    def read(self) -> np.ndarray | None:
        if self.capture is None:
            return None
        is_read, frame = self.capture.read()
        return frame if is_read else None

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.frame_interval_seconds = None
