"""Temporal confirmation and prototype risk assessment."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from detector_rail import RailAnalysis
from detector_yolo import Detection


@dataclass(frozen=True)
class RiskResult:
    score: int
    status: str
    reasons: tuple[str, ...]
    confirmed_keys: tuple[str, ...]


class RiskEngine:
    def __init__(
        self, confirmation_frames: int, cooldown_seconds: float, weights: dict[str, int]
    ) -> None:
        self.confirmation_frames, self.cooldown_seconds, self.weights = (
            confirmation_frames,
            cooldown_seconds,
            weights,
        )
        self.counts: dict[str, int] = {}
        self.last_seen: dict[str, float] = {}
        self.active: set[str] = set()

    def reset(self) -> None:
        self.counts.clear()
        self.last_seen.clear()
        self.active.clear()

    def evaluate(self, detections: list[Detection], rail: RailAnalysis) -> RiskResult:
        now = monotonic()
        observed = {
            f"object:{detection.class_name}"
            for detection in detections
            if detection.inside_track_roi
        }
        if rail.possible_anomaly:
            observed.add(f"rail:{rail.anomaly_type or 'geometry'}")
        for key in observed:
            self.counts[key] = self.counts.get(key, 0) + 1
            self.last_seen[key] = now
            if self.counts[key] >= self.confirmation_frames:
                self.active.add(key)
        for key in list(self.active):
            if now - self.last_seen.get(key, 0) > self.cooldown_seconds:
                self.active.remove(key)
                self.counts.pop(key, None)
        score, reasons = self._score()
        status = "HIGH RISK" if score >= 61 else "WARNING" if score >= 31 else "NORMAL"
        return RiskResult(
            min(score, 100),
            status,
            tuple(reasons) or ("No confirmed anomalies",),
            tuple(sorted(self.active)),
        )

    def _score(self) -> tuple[int, list[str]]:
        score, reasons = 0, []
        for key in sorted(self.active):
            if key == "object:person":
                score += self.weights["person"]
                reasons.append("Person confirmed inside track region")
            elif key.startswith("object:") and key.split(":", 1)[1] in {
                "car",
                "bus",
                "truck",
                "motorcycle",
            }:
                score += self.weights["vehicle"]
                reasons.append("Vehicle confirmed inside track region")
            elif key.startswith("object:"):
                score += self.weights["object"]
                reasons.append("Relevant object confirmed inside track region")
            elif key == "rail:line_discontinuity":
                score += self.weights["rail_discontinuity"]
                reasons.append("Possible rail line discontinuity")
            else:
                score += self.weights["rail_anomaly"]
                reasons.append("Possible rail geometry anomaly")
        if len(self.active) > 1:
            score += self.weights["multiple_issues"]
            reasons.append("Multiple confirmed issues")
        return score, reasons
