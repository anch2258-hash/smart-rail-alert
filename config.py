"""Central configuration for the local SmartRail Alert prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    model_path: Path = ROOT_DIR / "models" / "yolo11n.pt"
    roi_path: Path = ROOT_DIR / "results" / "track_roi.json"
    reports_dir: Path = ROOT_DIR / "results"
    logs_dir: Path = ROOT_DIR / "logs"
    # Preferred device; main.py resolves it at startup to cuda:0 or CPU.
    device: str = "cuda:0"
    confidence_threshold: float = 0.45
    image_size: int = 416
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    yolo_interval: int = 2
    # Rail geometry is quasi-static: re-detect every Nth frame and hold the
    # pair between runs. Kills per-frame Hough flicker and saves CPU.
    rail_interval: int = 5
    debug_mode: bool = False
    confirmation_frames: int = 3
    alert_cooldown_seconds: float = 2.0
    relevant_classes: tuple[str, ...] = (
        "person",
        "bicycle",
        "motorcycle",
        "car",
        "bus",
        "truck",
        "suitcase",
        "backpack",
        "foreign_object",
    )
    risk_weights: dict[str, int] = field(
        default_factory=lambda: {
            "person": 70,
            "vehicle": 80,
            "object": 50,
            "rail_anomaly": 40,
            "rail_discontinuity": 50,
            "multiple_issues": 20,
        }
    )

    def prepare_directories(self) -> None:
        for directory in (
            self.model_path.parent,
            self.roi_path.parent,
            self.reports_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
