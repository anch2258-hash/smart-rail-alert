"""Track-region persistence and polygon helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

Point: TypeAlias = tuple[int, int]
RoiPoints: TypeAlias = list[Point]


def is_valid_roi(points: RoiPoints | None) -> bool:
    return (
        points is not None
        and len(points) == 4
        and len({tuple(point) for point in points}) == 4
    )


def point_in_roi(point: Point, points: RoiPoints | None) -> bool:
    if not is_valid_roi(points):
        return False
    polygon = np.asarray(points, dtype=np.int32)
    return cv2.pointPolygonTest(polygon, point, False) >= 0


REQUIRED_INSIDE_POINTS = 2


def bbox_control_points(x1: int, y1: int, x2: int, y2: int) -> list[Point]:
    """Lower-body control points: bottom row plus the 75% height row."""
    y75 = y1 + int(0.75 * (y2 - y1))
    center_x = (x1 + x2) // 2
    return [
        (x1, y2),
        (center_x, y2),
        (x2, y2),
        (x1, y75),
        (center_x, y75),
        (x2, y75),
    ]


def is_bbox_inside_roi(
    x1: int, y1: int, x2: int, y2: int, points: RoiPoints | None
) -> bool:
    """A box counts as inside when at least two control points are inside."""
    if not is_valid_roi(points):
        return False
    inside = sum(
        1
        for point in bbox_control_points(x1, y1, x2, y2)
        if point_in_roi(point, points)
    )
    return inside >= REQUIRED_INSIDE_POINTS


def draw_roi(frame: np.ndarray, points: RoiPoints | None) -> np.ndarray:
    if not is_valid_roi(points):
        return frame
    cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], True, (0, 215, 255), 2)
    return frame


def save_roi(path: Path, points: RoiPoints) -> None:
    if not is_valid_roi(points):
        raise ValueError("ROI должна содержать четыре различные точки")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"points": points}, indent=2), encoding="utf-8")


def load_roi(path: Path) -> RoiPoints | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        points = [tuple(map(int, point)) for point in payload["points"]]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return points if is_valid_roi(points) else None
