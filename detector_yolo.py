"""Ultralytics inference with portable device selection.

Startup picks cuda:0 when torch reports CUDA, otherwise CPU fallback.
Intel/AMD GPUs, DirectML and ROCm are intentionally not probed.
Same model, confidence and image size on every device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from roi import RoiPoints, is_bbox_inside_roi


@dataclass(frozen=True)
class GpuStatus:
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    selected_device: str
    error: str | None = None


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    bottom_center: tuple[int, int]
    inside_track_roi: bool


def get_gpu_status(device: str = "cuda:0") -> GpuStatus:
    try:
        import torch
    except ImportError as error:
        return GpuStatus("not installed", False, None, None, device, str(error))
    is_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if is_available else None
    error = (
        None
        if is_available
        else "CUDA недоступна; используется CPU (сниженная производительность)."
    )
    return GpuStatus(
        torch.__version__, is_available, torch.version.cuda, gpu_name, device, error
    )


def resolve_inference_device(preferred: str = "cuda:0") -> tuple[str, GpuStatus]:
    """Single startup selection point: CUDA when present, else CPU fallback."""
    status = get_gpu_status(preferred)
    if status.cuda_available:
        return preferred, status
    fallback = GpuStatus(
        status.torch_version,
        False,
        status.cuda_version,
        None,
        "cpu",
        status.error,
    )
    return "cpu", fallback


def format_device_status(status: GpuStatus) -> str:
    if status.cuda_available:
        return (
            f"GPU: {status.gpu_name or 'Не обнаружен'}\nCUDA: ВКЛ\n"
            f"PyTorch: {status.torch_version}\nИнференс: {status.selected_device}"
        )
    return (
        "Ускорение: CPU\nCUDA: НЕДОСТУПНА\nСниженная производительность\n"
        f"PyTorch: {status.torch_version}\nИнференс: {status.selected_device}"
    )


class YoloDetector:
    def __init__(
        self,
        model_path: Path,
        device: str,
        confidence: float,
        image_size: int,
        relevant_classes: tuple[str, ...],
    ) -> None:
        self.device, self.confidence, self.image_size = device, confidence, image_size
        self.relevant_classes = set(relevant_classes)
        self.model: Any | None = None
        self.names: dict[int, str] = {}
        self.model_path = model_path

    def load(self) -> None:
        from ultralytics import YOLO

        # Ultralytics downloads the named pretrained model once when it is absent.
        model_reference = (
            str(self.model_path) if self.model_path.exists() else self.model_path.name
        )
        self.model = YOLO(model_reference)
        if not self.model_path.exists() and Path(model_reference).exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            Path(model_reference).replace(self.model_path)
        self.names = {int(key): str(value) for key, value in self.model.names.items()}

    def detect(
        self, frame: np.ndarray, roi_points: RoiPoints | None
    ) -> tuple[list[Detection], float]:
        if self.model is None:
            raise RuntimeError("YOLO-модель не загружена")
        started_at = perf_counter()
        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )
        latency_ms = (perf_counter() - started_at) * 1000
        detections: list[Detection] = []
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            class_name = self.names.get(class_id, str(class_id))
            if class_name not in self.relevant_classes:
                continue
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            bottom_center = ((x1 + x2) // 2, y2)
            detections.append(
                Detection(
                    class_id,
                    class_name,
                    float(box.conf.item()),
                    x1,
                    y1,
                    x2,
                    y2,
                    bottom_center,
                    is_bbox_inside_roi(x1, y1, x2, y2, roi_points),
                )
            )
        return detections, latency_ms
