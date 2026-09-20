"""Controlled synthetic rail-geometry scenes.

- stable rail pair -> detected, no anomaly (NORMAL);
- controlled horizontal shift -> possible geometry deviation;
- room-like decoy lines -> no pair, no confirmed anomaly.

These images also double as demo material (saved under results/).
"""

import cv2
import numpy as np

import detector_rail
from detector_rail import CONFIDENCE_THRESHOLD, RailDetector

WIDTH, HEIGHT = 640, 480
ROI = [(40, 470), (620, 470), (460, 120), (210, 120)]
SHIFT = 80


def make_rail_frame(shift: int = 0) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.line(frame, (140 + shift, 470), (280 + shift, 120), (255, 255, 255), 4)
    cv2.line(frame, (500 + shift, 470), (360 + shift, 120), (255, 255, 255), 4)
    return frame


def make_decoy_frame() -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.line(frame, (50, 150), (590, 150), (200, 200, 200), 3)
    cv2.line(frame, (50, 300), (590, 300), (200, 200, 200), 3)
    cv2.line(frame, (100, 50), (100, 430), (200, 200, 200), 3)
    cv2.line(frame, (480, 420), (560, 280), (255, 255, 255), 3)
    cv2.line(frame, (380, 200), (460, 380), (255, 255, 255), 3)
    cv2.line(frame, (300, 400), (330, 370), (255, 255, 255), 3)
    return frame


def draw_pair(path: str, frame: np.ndarray, detector: RailDetector) -> None:
    result = detector.analyze(frame, ROI)
    overlay = frame.copy()
    for line in (result.left_rail, result.right_rail):
        if line:
            cv2.line(overlay, line[:2], line[2:], (0, 255, 0), 2)
    cv2.imwrite(path, overlay)


def test_stable_rails_have_no_anomaly() -> None:
    detector = RailDetector()
    for _ in range(5):
        result = detector.analyze(make_rail_frame(), ROI)
        assert result.rail_detected
        assert not result.possible_anomaly
        assert result.anomaly_type is None
        assert result.score >= CONFIDENCE_THRESHOLD


def test_controlled_shift_triggers_possible_anomaly() -> None:
    detector = RailDetector()
    for _ in range(5):
        baseline = detector.analyze(make_rail_frame(), ROI)
        assert not baseline.possible_anomaly
    shifted = [detector.analyze(make_rail_frame(SHIFT), ROI) for _ in range(5)]
    assert any(
        result.possible_anomaly and result.anomaly_type == "geometry_deviation"
        for result in shifted
    )
    draw_pair("results/rail_demo_normal.png", make_rail_frame(), RailDetector())
    draw_pair("results/rail_demo_shifted.png", make_rail_frame(SHIFT), detector)


def test_room_decoys_raise_no_confirmed_anomaly() -> None:
    detector = RailDetector()
    for _ in range(5):
        result = detector.analyze(make_decoy_frame(), ROI)
        assert not result.possible_anomaly


def test_discontinuity_uses_video_timestamps_not_frame_count() -> None:
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    detector = RailDetector()
    for tick in (0.0, 0.05, 0.10, 0.15):
        assert detector.analyze(make_rail_frame(), ROI, timestamp=tick).rail_detected
    # 20 consecutive misses in 0.2 s of video time: still inside grace.
    for index in range(20):
        lost = detector.analyze(blank, ROI, timestamp=0.20 + index * 0.01)
        assert not lost.possible_anomaly
        assert lost.anomaly_type is None
    # Same loss crossing 0.45 s of video time: discontinuity fires.
    lost = detector.analyze(blank, ROI, timestamp=0.70)
    assert lost.possible_anomaly
    assert lost.anomaly_type == "line_discontinuity"


def test_discontinuity_arms_after_time_based_grace(monkeypatch) -> None:
    clock = iter((0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3, 1.4, 1.9))
    monkeypatch.setattr(detector_rail, "monotonic", lambda: next(clock))
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    flicker = RailDetector()
    assert flicker.analyze(make_rail_frame(), ROI).rail_detected
    for _ in range(4):
        lost = flicker.analyze(blank, ROI)
        assert not lost.possible_anomaly
        assert lost.anomaly_type is None
    stable = RailDetector()
    for _ in range(4):
        assert stable.analyze(make_rail_frame(), ROI).rail_detected
    stable.analyze(blank, ROI)
    lost = stable.analyze(blank, ROI)
    assert lost.possible_anomaly
    assert lost.anomaly_type == "line_discontinuity"
