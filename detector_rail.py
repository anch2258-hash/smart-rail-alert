"""Controlled-view rail geometry analysis using Canny and Hough lines.

Robustness model (prototype, never a damage diagnostic):
- Hough candidates must converge upward inside the calibrated ROI;
- gauge, symmetry and slope agreement select a single rail pair;
- a 0..1 confidence blends orientation, symmetry, gauge, support, stability;
- geometry below CONFIDENCE_THRESHOLD never raises a rail anomaly;
- every finding is reported as a *possible* anomaly only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, degrees, hypot
from time import monotonic

import cv2
import numpy as np

from roi import RoiPoints, is_valid_roi, point_in_roi

CONFIDENCE_THRESHOLD = 0.55
# Lower bound admits near-vertical rails (~6-7 deg on 1920x1080 footage);
# pair-level gates (symmetry/gauge/spread/confidence) still reject strays.
_MIN_ANGLE_FROM_VERTICAL = 5.0
_MAX_ANGLE_FROM_VERTICAL = 45.0
_IDEAL_ANGLE_FROM_VERTICAL = 27.5
# Shorter floor keeps Hough fragments (maxGap 25) of real rails alive;
# absolute 45 px floor and all downstream gates are unchanged.
_MIN_LENGTH_FRACTION = 0.10
# Lower floor admits real narrow-gauge-at-ROI perspective pairs
# (measured bottom_sep 264 px on 1920 px footage); all other gates stay.
_MIN_GAUGE_FRACTION = 0.12
_MAX_GAUGE_FRACTION = 0.95
_IDEAL_GAUGE_LO = 0.35
_IDEAL_GAUGE_HI = 0.80
_SYMMETRY_TOLERANCE_FRACTION = 0.18
# Converging rails nearly meet at the vanishing point (measured top_sep
# 34 px); spread gate still rejects duplicate-line pairs.
_MIN_TOP_SEPARATION_FRACTION = 0.01
_MIN_SPREAD_DEG = 5.0
_MAX_SPREAD_DEG = 60.0
_MIN_PAIR_SCORE = 0.35
_POOL_SHORTLIST = 8
_BASELINE_ALPHA = 0.3
_LINE_DISCONTINUITY_GRACE_SECONDS = 0.45
_VISUAL_PAIR_HOLD_SECONDS = 0.18


@dataclass(frozen=True)
class RailAnalysis:
    rail_detected: bool
    left_rail: tuple[int, int, int, int] | None
    right_rail: tuple[int, int, int, int] | None
    possible_anomaly: bool
    anomaly_type: str | None
    score: float
    reason: str


@dataclass
class RailDebugInfo:
    """Per-frame gate statistics.

    Filled only when passed as ``debug=`` to :meth:`RailDetector.analyze`.
    Observes the pipeline without changing any decision.
    """

    hough_lines: int = 0
    after_orientation: int = 0
    after_length: int = 0
    after_roi: int = 0
    left_pool: int = 0
    right_pool: int = 0
    stage: str = "no_lines"
    symmetry_pass: int = 0
    gauge_pass: int = 0
    final_pairs: int = 0
    all_lines: list[tuple[int, int, int, int]] = field(default_factory=list)
    oriented_lines: list[tuple[int, int, int, int]] = field(default_factory=list)


def _orient(
    line: tuple[int, int, int, int],
) -> tuple[float, float, float, tuple[int, int], tuple[int, int]] | None:
    """Return (slope, angle_from_vertical, length, top, bottom) or None if flat."""
    x1, y1, x2, y2 = line
    if y1 == y2:
        return None
    top, bottom = (x1, y1), (x2, y2)
    if y1 > y2:
        top, bottom = bottom, top
    dy = bottom[1] - top[1]
    slope = (bottom[0] - top[0]) / dy
    angle = abs(degrees(atan2(abs(bottom[0] - top[0]), dy)))
    return slope, angle, hypot(x2 - x1, y2 - y1), top, bottom


def _seg_length(line: tuple[int, int, int, int]) -> float:
    return hypot(line[2] - line[0], line[3] - line[1])


class RailDetector:
    def __init__(self) -> None:
        self.previous_pair: (
            tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None
        ) = None
        self._baseline: tuple[float, float] | None = None
        self._stable_count = 0
        self._rails_stable = False
        self._last_valid_at: float | None = None
        self._miss_started_at: float | None = None

    def analyze(
        self,
        frame: np.ndarray,
        roi_points: RoiPoints | None,
        debug: RailDebugInfo | None = None,
        timestamp: float | None = None,
    ) -> RailAnalysis:
        # Timestamp is video time when the caller knows it (file: frame/fps),
        # otherwise wall clock. All grace/hold intervals use this one clock,
        # so loss duration never depends on processing speed or source FPS.
        now = timestamp if timestamp is not None else monotonic()
        if frame is None or frame.size == 0:
            if debug is not None:
                debug.stage = "invalid_frame"
            return RailAnalysis(False, None, None, False, None, 0.0, "Invalid frame")
        if not is_valid_roi(roi_points):
            if debug is not None:
                debug.stage = "invalid_roi"
            return RailAnalysis(
                False, None, None, False, None, 0.0, "Track ROI is not calibrated"
            )
        height, width = frame.shape[:2]
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(roi_points, dtype=np.int32)], 255)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
        lines = cv2.HoughLinesP(
            cv2.bitwise_and(edges, mask),
            1,
            np.pi / 180,
            35,
            minLineLength=45,
            maxLineGap=25,
        )
        if debug is not None:
            raw = (
                []
                if lines is None
                else [tuple(map(int, segment)) for segment in lines[:, 0]]
            )
            debug.hough_lines = len(raw)
            debug.all_lines = raw
            debug.stage = "no_lines" if not raw else "filtered"
        candidates = self._filter_lines(lines, height, roi_points, debug)
        # Symmetry is measured against the monitored zone (ROI centroid),
        # not the frame center: the track, not the image, is the reference.
        roi_center_x = sum(p[0] for p in roi_points) / len(roi_points)
        left, right, pair_score = self._select_pair(
            candidates, width, height, roi_center_x, debug
        )
        if left is None or right is None:
            if self._miss_started_at is None:
                self._miss_started_at = now
            self._stable_count = 0
            missed_for = now - self._miss_started_at
            held_pair = (
                self.previous_pair
                if self._last_valid_at is not None
                and now - self._last_valid_at <= _VISUAL_PAIR_HOLD_SECONDS
                else None
            )
            if (
                self._rails_stable
                and self.previous_pair is not None
                and missed_for >= _LINE_DISCONTINUITY_GRACE_SECONDS
            ):
                return RailAnalysis(
                    False,
                    left,
                    right,
                    True,
                    "line_discontinuity",
                    0.7,
                    "Expected rail line is missing",
                )
            return RailAnalysis(
                False,
                held_pair[0] if held_pair else None,
                held_pair[1] if held_pair else None,
                False,
                None,
                0.0,
                "Rail pair not found",
            )
        confidence = self._confidence(left, right, width, height, pair_score)
        if confidence < CONFIDENCE_THRESHOLD:
            self._stable_count = 0
            if debug is not None:
                debug.stage = "low_confidence"
            return RailAnalysis(
                True,
                left,
                right,
                False,
                None,
                confidence,
                "Low-confidence rail geometry; possible anomaly suppressed",
            )
        self._last_valid_at = now
        self._miss_started_at = None
        self._stable_count += 1
        if self._stable_count >= 3:
            self._rails_stable = True
        midpoint = (left[0] + left[2] + right[0] + right[2]) / 4
        gauge = abs((left[0] + left[2]) / 2 - (right[0] + right[2]) / 2)
        if self._baseline is None:
            self._baseline = (midpoint, gauge)
            self.previous_pair = (left, right)
            return RailAnalysis(
                True,
                left,
                right,
                False,
                None,
                confidence,
                "Rail geometry consistent",
            )
        base_mid, base_gauge = self._baseline
        shifted = abs(midpoint - base_mid) > max(40.0, 0.10 * width) or abs(
            gauge - base_gauge
        ) > max(30.0, 0.12 * width)
        self.previous_pair = (left, right)
        if not shifted:
            alpha = _BASELINE_ALPHA
            self._baseline = (
                (1 - alpha) * base_mid + alpha * midpoint,
                (1 - alpha) * base_gauge + alpha * gauge,
            )
            return RailAnalysis(
                True, left, right, False, None, confidence, "Rail geometry consistent"
            )
        return RailAnalysis(
            True,
            left,
            right,
            True,
            "geometry_deviation",
            confidence,
            "Large rail position change",
        )

    @staticmethod
    def _filter_lines(
        lines: np.ndarray | None,
        height: int,
        roi_points: RoiPoints,
        debug: RailDebugInfo | None = None,
    ) -> list[tuple[int, int, int, int]]:
        if lines is None:
            return []
        min_length = max(45.0, _MIN_LENGTH_FRACTION * height)
        candidates = []
        for x1, y1, x2, y2 in lines[:, 0]:
            segment = (int(x1), int(y1), int(x2), int(y2))
            oriented = _orient(segment)
            if oriented is None:
                continue
            _, angle, length, _, _ = oriented
            if not _MIN_ANGLE_FROM_VERTICAL <= angle <= _MAX_ANGLE_FROM_VERTICAL:
                continue
            if debug is not None:
                debug.after_orientation += 1
                debug.oriented_lines.append(segment)
            if length < min_length:
                continue
            if debug is not None:
                debug.after_length += 1
            midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            if not point_in_roi((int(midpoint[0]), int(midpoint[1])), roi_points):
                continue
            if debug is not None:
                debug.after_roi += 1
            candidates.append(segment)
        return candidates

    @staticmethod
    def _select_pair(
        candidates: list[tuple[int, int, int, int]],
        width: int,
        height: int,
        roi_center_x: float,
        debug: RailDebugInfo | None = None,
    ) -> tuple[
        tuple[int, int, int, int] | None, tuple[int, int, int, int] | None, float
    ]:
        def fail(stage: str) -> tuple[None, None, float]:
            if debug is not None:
                debug.stage = stage
            return None, None, 0.0

        lefts = [
            line
            for line in candidates
            if (_orient(line) or (0, 0, 0, (0, 0), (0, 0)))[0] < 0
        ]
        rights = [
            line
            for line in candidates
            if (_orient(line) or (0, 0, 0, (0, 0), (0, 0)))[0] > 0
        ]
        if debug is not None:
            debug.left_pool = len(lefts)
            debug.right_pool = len(rights)
        if not lefts or not rights:
            return fail("no_pools")
        # Best-pair selection: longest-first loses to clutter (e.g. sun streaks
        # on ballast), so shortlist by length and score every combo by geometry.
        short_lefts = sorted(lefts, key=_seg_length, reverse=True)[:_POOL_SHORTLIST]
        short_rights = sorted(rights, key=_seg_length, reverse=True)[:_POOL_SHORTLIST]
        best: (
            tuple[tuple[int, int, int, int], tuple[int, int, int, int], float] | None
        ) = None
        failures: dict[str, int] = {}
        for left_cand in short_lefts:
            for right_cand in short_rights:
                passed, stage, combo_score = RailDetector._score_combo(
                    left_cand, right_cand, width, height, roi_center_x
                )
                if passed and (best is None or combo_score > best[2]):
                    best = (left_cand, right_cand, combo_score)
                elif not passed:
                    failures[stage] = failures.get(stage, 0) + 1
        if best is None:
            top_failure = (
                max(failures, key=lambda s: failures[s]) if failures else "no_pools"
            )
            return fail(top_failure)
        left, right, pair_score = best
        if debug is not None:
            debug.symmetry_pass = 1
            debug.gauge_pass = 1
            debug.stage = "accepted"
            debug.final_pairs = 1
        return left, right, pair_score

    @staticmethod
    def _score_combo(
        left: tuple[int, int, int, int],
        right: tuple[int, int, int, int],
        width: int,
        height: int,
        roi_center_x: float,
    ) -> tuple[bool, str, float]:
        """Validate one candidate pair and score its geometry (0..1)."""
        center_x = roi_center_x
        left_info = _orient(left)
        right_info = _orient(right)
        if left_info is None or right_info is None:
            return False, "pools", 0.0
        _, angle_left, len_left, top_left, bottom_left = left_info
        _, angle_right, len_right, top_right, bottom_right = right_info
        mid_left = (left[0] + left[2]) / 2
        mid_right = (right[0] + right[2]) / 2
        # Converging rails always diverge downward: order by ends, never by
        # midpoints (midpoints invert when fragments sit at different heights)
        # and never against a magic center line.
        if not bottom_left[0] < bottom_right[0]:
            return False, "position", 0.0
        if not top_left[0] < top_right[0]:
            return False, "position", 0.0
        bottom_sep = abs(bottom_left[0] - bottom_right[0])
        top_sep = abs(top_left[0] - top_right[0])
        if not _MIN_GAUGE_FRACTION * width <= bottom_sep <= _MAX_GAUGE_FRACTION * width:
            return False, "gauge", 0.0
        if not top_sep < bottom_sep:
            return False, "convergence", 0.0
        if top_sep < _MIN_TOP_SEPARATION_FRACTION * width:
            return False, "convergence", 0.0
        spread = angle_left + angle_right
        if not _MIN_SPREAD_DEG <= spread <= _MAX_SPREAD_DEG:
            return False, "spread", 0.0
        pair_center = (mid_left + mid_right) / 2
        symmetry = max(
            0.0,
            1 - abs(pair_center - center_x) / (_SYMMETRY_TOLERANCE_FRACTION * width),
        )
        gauge_fraction = bottom_sep / width
        if _IDEAL_GAUGE_LO <= gauge_fraction <= _IDEAL_GAUGE_HI:
            gauge_fit = 1.0
        elif gauge_fraction < _IDEAL_GAUGE_LO:
            gauge_fit = max(
                0.0,
                (gauge_fraction - _MIN_GAUGE_FRACTION)
                / (_IDEAL_GAUGE_LO - _MIN_GAUGE_FRACTION),
            )
        else:
            gauge_fit = max(
                0.0,
                (_MAX_GAUGE_FRACTION - gauge_fraction)
                / (_MAX_GAUGE_FRACTION - _IDEAL_GAUGE_HI),
            )
        spread_fit = max(0.0, 1 - abs(spread - 30.0) / 30.0)
        length_fit = min(1.0, (len_left + len_right) / (1.5 * height))
        pair_score = (
            0.35 * symmetry + 0.25 * gauge_fit + 0.15 * spread_fit + 0.25 * length_fit
        )
        if pair_score < _MIN_PAIR_SCORE:
            return False, "pair_score", pair_score
        return True, "accepted", pair_score

    def _confidence(
        self,
        left: tuple[int, int, int, int],
        right: tuple[int, int, int, int],
        width: int,
        height: int,
        pair_score: float,
    ) -> float:
        left_info = _orient(left)
        right_info = _orient(right)
        if left_info is None or right_info is None:
            return 0.0
        _, angle_left, len_left, _, _ = left_info
        _, angle_right, len_right, _, _ = right_info
        orientation = (
            sum(
                max(
                    0.0,
                    1
                    - abs(angle - _IDEAL_ANGLE_FROM_VERTICAL)
                    / (_IDEAL_ANGLE_FROM_VERTICAL - _MIN_ANGLE_FROM_VERTICAL),
                )
                for angle in (angle_left, angle_right)
            )
            / 2
        )
        strength = min(1.0, (len_left + len_right) / height)
        if self.previous_pair is None:
            temporal = 0.6
        else:
            old_left, old_right = self.previous_pair
            old_mid = (old_left[0] + old_left[2] + old_right[0] + old_right[2]) / 4
            midpoint = (left[0] + left[2] + right[0] + right[2]) / 4
            temporal = max(0.0, 1 - abs(midpoint - old_mid) / (0.10 * width))
        return float(
            0.35 * orientation + 0.30 * pair_score + 0.20 * strength + 0.15 * temporal
        )
