"""Measured session statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


@dataclass
class RuntimeStatistics:
    started_at: float = field(default_factory=monotonic)
    total_frames: int = 0
    analyzed_frames: int = 0
    detections: int = 0
    confirmed_detections: int = 0
    alerts: int = 0
    total_inference_ms: float = 0.0
    total_processing_ms: float = 0.0
    max_risk: int = 0
    current_risk: int = 0
    detection_counts: dict[str, int] = field(default_factory=dict)
    anomaly_types: dict[str, int] = field(default_factory=dict)
    _previous_status: str = "NORMAL"

    def reset(self) -> None:
        self.__dict__.update(RuntimeStatistics().__dict__)

    def record(
        self,
        *,
        analyzed: bool,
        inference_ms: float,
        processing_ms: float,
        detections: list[str],
        confirmed: int,
        risk: int,
        status: str,
        anomaly: str | None,
    ) -> None:
        self.total_frames += 1
        self.analyzed_frames += int(analyzed)
        self.total_inference_ms += inference_ms
        self.total_processing_ms += processing_ms
        self.detections += len(detections)
        self.confirmed_detections = confirmed
        self.current_risk = risk
        self.max_risk = max(self.max_risk, risk)
        if status != "NORMAL" and self._previous_status == "NORMAL":
            self.alerts += 1
        self._previous_status = status
        for name in detections:
            self.detection_counts[name] = self.detection_counts.get(name, 0) + 1
        if anomaly:
            self.anomaly_types[anomaly] = self.anomaly_types.get(anomaly, 0) + 1

    @property
    def duration_seconds(self) -> float:
        return max(monotonic() - self.started_at, 0.001)

    @property
    def average_fps(self) -> float:
        return self.total_frames / self.duration_seconds

    @property
    def average_inference_ms(self) -> float:
        return self.total_inference_ms / max(self.analyzed_frames, 1)
