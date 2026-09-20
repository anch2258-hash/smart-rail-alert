from pathlib import Path

import cv2
import numpy as np

from detector_rail import RailDetector
from detector_yolo import (
    Detection,
    format_device_status,
    resolve_inference_device,
)
from report import save_report
from risk_engine import RiskEngine
from roi import is_bbox_inside_roi, load_roi, point_in_roi, save_roi
from statistics import RuntimeStatistics


WEIGHTS = {
    "person": 70,
    "vehicle": 80,
    "object": 50,
    "rail_anomaly": 40,
    "rail_discontinuity": 50,
    "multiple_issues": 20,
}


def detection(name: str = "person", inside: bool = True) -> Detection:
    return Detection(0, name, 0.9, 10, 10, 30, 40, (20, 40), inside)


def test_roi_membership_and_persistence(tmp_path: Path) -> None:
    points = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_roi((50, 50), points)
    assert not point_in_roi((101, 50), points)
    path = tmp_path / "roi.json"
    save_roi(path, points)
    assert load_roi(path) == points


def test_temporal_confirmation_and_risk_boundary() -> None:
    engine = RiskEngine(3, 2.0, WEIGHTS)
    rail = RailDetector().analyze(np.zeros((100, 100, 3), dtype=np.uint8), None)
    assert engine.evaluate([detection()], rail).status == "NORMAL"
    assert engine.evaluate([detection()], rail).status == "NORMAL"
    result = engine.evaluate([detection()], rail)
    assert result.status == "HIGH RISK" and result.score == 70
    assert 0 <= result.score <= 100


def test_irrelevant_or_outside_detection_has_no_risk() -> None:
    engine = RiskEngine(1, 2.0, WEIGHTS)
    rail = RailDetector().analyze(np.zeros((100, 100, 3), dtype=np.uint8), None)
    assert engine.evaluate([detection(inside=False)], rail).score == 0


def test_bbox_fully_outside_roi_is_not_inside() -> None:
    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert not is_bbox_inside_roi(120, 120, 180, 180, roi)


def test_bbox_with_bottom_center_inside_roi_is_inside() -> None:
    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert is_bbox_inside_roi(40, 40, 60, 90, roi)


def test_bbox_partially_inside_with_center_outside_is_inside() -> None:
    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert is_bbox_inside_roi(80, 40, 140, 90, roi)


def test_bbox_touching_roi_border_only_is_not_inside() -> None:
    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert not is_bbox_inside_roi(100, 100, 150, 150, roi)


def test_rail_detector_returns_structured_result() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.line(image, (30, 230), (130, 30), (255, 255, 255), 3)
    cv2.line(image, (290, 230), (190, 30), (255, 255, 255), 3)
    result = RailDetector().analyze(image, [(0, 239), (320, 239), (230, 0), (90, 0)])
    assert isinstance(result.rail_detected, bool)
    assert hasattr(result, "possible_anomaly") and hasattr(result, "reason")


def test_report_contains_measured_fields(tmp_path: Path) -> None:
    statistics = RuntimeStatistics()
    statistics.record(
        analyzed=True,
        inference_ms=12.5,
        processing_ms=20,
        detections=["person"],
        confirmed=1,
        risk=70,
        status="HIGH RISK",
        anomaly=None,
    )
    path = save_report(tmp_path, "Веб-камера 0", statistics, "HIGH RISK")
    contents = path.read_text(encoding="utf-8")
    assert (
        "Средний FPS:" in contents
        and "Средний инференс YOLO:" in contents
        and "Кадров: 1" in contents
        and "ВЫСОКИЙ РИСК" in contents
    )


def test_resolve_prefers_cuda_when_available(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device, status = resolve_inference_device("cuda:0")
    assert device == "cuda:0"
    assert status.cuda_available
    assert "CUDA: ВКЛ" in format_device_status(status)


def test_resolve_falls_back_to_cpu(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, status = resolve_inference_device("cuda:0")
    assert device == "cpu"
    assert not status.cuda_available
    assert status.selected_device == "cpu"


def test_cpu_mode_status_text(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _, status = resolve_inference_device("cuda:0")
    text = format_device_status(status)
    assert "Ускорение: CPU" in text
    assert "НЕДОСТУПНА" in text
    assert "Сниженная производительность" in text


def test_report_records_inference_device(tmp_path: Path) -> None:
    statistics = RuntimeStatistics()
    statistics.record(
        analyzed=True,
        inference_ms=12.5,
        processing_ms=20,
        detections=["person"],
        confirmed=1,
        risk=70,
        status="HIGH RISK",
        anomaly=None,
    )
    path = save_report(tmp_path, "Веб-камера 0", statistics, "HIGH RISK", "cpu")
    assert "Режим инференса: cpu" in path.read_text(encoding="utf-8")
