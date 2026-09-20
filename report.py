"""Human-readable local report generation (Russian)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from labels import anomaly_label, class_label, status_label
from statistics import RuntimeStatistics


def save_report(
    output_dir: Path,
    source: str,
    statistics: RuntimeStatistics,
    final_status: str,
    inference_device: str = "cuda:0",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone()
    report_path = output_dir / f"smartrail_report_{timestamp:%Y%m%d_%H%M%S}.txt"
    detection_counts = {
        class_label(name): count for name, count in statistics.detection_counts.items()
    } or "Нет"
    anomaly_types = {
        anomaly_label(name): count for name, count in statistics.anomaly_types.items()
    } or "Нет"
    contents = "\n".join(
        (
            "ОТЧЁТ SMART RAIL ALERT",
            f"Метка времени: {timestamp.isoformat()}",
            f"Источник: {source}",
            f"Длительность: {statistics.duration_seconds:.1f} с",
            f"Кадров: {statistics.total_frames}",
            f"Проанализировано кадров: {statistics.analyzed_frames}",
            f"Средний FPS: {statistics.average_fps:.2f}",
            f"Средний инференс YOLO: {statistics.average_inference_ms:.2f} мс",
            f"Режим инференса: {inference_device}",
            f"Детекции: {statistics.detections}",
            f"Подтверждённые детекции: {statistics.confirmed_detections}",
            f"Тревоги: {statistics.alerts}",
            f"Максимальный риск: {statistics.max_risk} / 100",
            f"Итоговый статус: {status_label(final_status)}",
            f"Счётчики детекций: {detection_counts}",
            f"Возможные аномалии рельсов: {anomaly_types}",
            "",
            "Отчёт содержит измеренные данные прототипа мониторинга и не является сертификацией безопасности.",
        )
    )
    report_path.write_text(contents, encoding="utf-8")
    return report_path
