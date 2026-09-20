"""SmartRail Alert application entry point."""

from __future__ import annotations

import logging
from dataclasses import replace
from logging.handlers import RotatingFileHandler

from config import SETTINGS
from detector_yolo import resolve_inference_device


def configure_logging() -> None:
    SETTINGS.prepare_directories()
    handler = RotatingFileHandler(
        SETTINGS.logs_dir / "smartrail.log",
        maxBytes=1_000_000,
        backupCount=2,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[handler, logging.StreamHandler()],
    )


def main() -> None:
    configure_logging()
    device, status = resolve_inference_device(SETTINGS.device)
    logging.info(
        "GPU status: torch=%s cuda=%s gpu=%s device=%s",
        status.torch_version,
        status.cuda_available,
        status.gpu_name,
        device,
    )
    from ui import SmartRailApp

    SmartRailApp(replace(SETTINGS, device=device), status).run()


if __name__ == "__main__":
    main()
