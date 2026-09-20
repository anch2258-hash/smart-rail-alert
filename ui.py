"""Thread-safe Tkinter monitoring dashboard."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from camera import FrameSource
from config import Settings
from detector_rail import RailAnalysis, RailDebugInfo, RailDetector
from detector_yolo import (
    Detection,
    GpuStatus,
    YoloDetector,
    format_device_status,
)
from labels import (
    REASON_LABELS,
    REASON_WEIGHT_KEYS,
    STATUS_LABELS,
    reason_label,
)
from report import save_report
from risk_engine import RiskEngine, RiskResult
from roi import RoiPoints, draw_roi, load_roi, save_roi
from statistics import RuntimeStatistics

LOGGER = logging.getLogger(__name__)


@dataclass
class FrameUpdate:
    frame: object
    detections: list[Detection]
    rail: RailAnalysis
    risk: RiskResult
    inference_ms: float
    processing_ms: float
    rail_debug: RailDebugInfo | None = None


class SmartRailApp:
    def __init__(self, settings: Settings, gpu_status: GpuStatus) -> None:
        self.settings, self.gpu_status = settings, gpu_status
        self.root = tk.Tk()
        self.root.title("SMART RAIL ALERT")
        self.root.configure(bg="#111827")
        self.source = FrameSource()
        self.statistics = RuntimeStatistics()
        self.rail_detector = RailDetector()
        self.risk_engine = RiskEngine(
            settings.confirmation_frames,
            settings.alert_cooldown_seconds,
            settings.risk_weights,
        )
        self.roi_points: RoiPoints | None = load_roi(settings.roi_path)
        self.yolo: YoloDetector | None = None
        self.is_running = False
        self.is_analyzing = False
        self.is_closing = False
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.updates: queue.Queue[FrameUpdate | Exception] = queue.Queue(maxsize=2)
        self.latest_frame = None
        self.canvas_image_id: int | None = None
        self.canvas_status_id: int | None = None
        self.canvas_reasons_id: int | None = None
        self.calibration_points: RoiPoints = []
        self.is_calibrating = False
        self.display_scale = 1.0
        self.source_label = "Не подключено"
        self.current_status = "NORMAL"
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(30, self._drain_updates)

    def _build(self) -> None:
        title = tk.Label(
            self.root,
            text="SMART RAIL ALERT",
            fg="#f8fafc",
            bg="#111827",
            font=("Segoe UI", 20, "bold"),
        )
        title.grid(row=0, column=0, columnspan=2, padx=14, pady=(12, 2), sticky="w")
        subtitle = tk.Label(
            self.root,
            text="Прототип визуального мониторинга в реальном времени | поддержка принятия решений",
            fg="#94a3b8",
            bg="#111827",
        )
        subtitle.grid(row=1, column=0, columnspan=2, padx=14, sticky="w")
        self.canvas = tk.Canvas(
            self.root, width=900, height=500, bg="#020617", highlightthickness=0
        )
        self.canvas.grid(row=2, column=0, padx=(14, 8), pady=12)
        self.canvas.bind("<Button-1>", self._roi_click)
        panel = tk.Frame(self.root, bg="#1e293b", padx=16, pady=14)
        panel.grid(row=2, column=1, padx=(8, 14), pady=12, sticky="nsew")
        self.status_var = tk.StringVar(value=STATUS_LABELS["NORMAL"])
        self.risk_var = tk.StringVar(value="0 / 100")
        self.details_var = tk.StringVar(value=REASON_LABELS["No confirmed anomalies"])
        self.metrics_var = tk.StringVar(value="FPS: 0.00\nДетекции: 0\nТревоги: 0")
        self.rail_status_var = tk.StringVar(value="Рельсы: не обнаружены")
        self.gpu_var = tk.StringVar(value=self._gpu_text())
        self.hint_var = tk.StringVar(
            value="Откройте источник и нажмите «Калибровать ROI»."
        )
        panel_rows = [
            ("СОСТОЯНИЕ СИСТЕМЫ", None, ("Segoe UI", 10, "bold")),
            ("", self.status_var, ("Segoe UI", 18, "bold")),
            ("УРОВЕНЬ РИСКА", None, ("Segoe UI", 10, "bold")),
            ("", self.risk_var, ("Segoe UI", 16, "bold")),
            ("ПРИЧИНА", None, ("Segoe UI", 10, "bold")),
            ("", self.details_var, ("Segoe UI", 9)),
            ("", self.rail_status_var, ("Segoe UI", 10, "bold")),
            ("МЕТРИКИ", None, ("Segoe UI", 10, "bold")),
            ("", self.metrics_var, ("Consolas", 9)),
            ("GPU", None, ("Segoe UI", 10, "bold")),
            ("", self.gpu_var, ("Segoe UI", 9)),
        ]
        if self.settings.debug_mode:
            self.rail_diag_var = tk.StringVar(
                value="Hough-линии: 0\nПосле orientation: 0\nПосле length: 0\nПосле ROI: 0\nКандидаты L/R: 0 / 0\nStage: -\nRail detected: НЕТ\nConfidence: 0.00"
            )
            panel_rows.extend(
                (
                    ("ДИАГНОСТИКА РЕЛЬСОВ", None, ("Segoe UI", 10, "bold")),
                    ("", self.rail_diag_var, ("Consolas", 9)),
                )
            )
        for text, variable, font in panel_rows:
            tk.Label(
                panel,
                text=text if variable is None else "",
                textvariable=variable,
                justify="left",
                wraplength=250,
                anchor="w",
                fg="#f8fafc" if variable else "#94a3b8",
                bg="#1e293b",
                font=font,
            ).pack(fill="x", pady=(0, 5))
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=(8, 8))
        tk.Label(
            panel,
            text="ПОДСКАЗКА",
            justify="left",
            anchor="w",
            fg="#94a3b8",
            bg="#1e293b",
            font=("Segoe UI", 10, "bold"),
        ).pack(fill="x", pady=(0, 5))
        tk.Label(
            panel,
            textvariable=self.hint_var,
            justify="left",
            wraplength=250,
            anchor="w",
            fg="#f8fafc",
            bg="#1e293b",
            font=("Segoe UI", 9),
        ).pack(fill="x", pady=(0, 5))
        controls = tk.Frame(self.root, bg="#111827")
        controls.grid(row=3, column=0, columnspan=2, pady=(0, 14))
        buttons = (
            ("Запустить камеру", self.start_camera),
            ("Открыть видео", self.open_video),
            ("Остановить видео", self.pause_video),
            ("Продолжить", self.resume_video),
            ("Начать анализ", self.start_analysis),
            ("Остановить анализ", self.stop_analysis),
            ("Калибровать ROI", self.calibrate_roi),
            ("Сбросить ROI", self.reset_roi),
            ("Сохранить ROI", self.save_roi),
            ("Сохранить отчёт", self.save_report),
        )
        for text, command in buttons:
            ttk.Button(controls, text=text, command=command).pack(side="left", padx=3)

    def _gpu_text(self) -> str:
        return format_device_status(self.gpu_status)

    def run(self) -> None:
        self.root.mainloop()

    def start_camera(self) -> None:
        try:
            self.stop_source()
            self.source.open_camera(
                self.settings.camera_index,
                self.settings.frame_width,
                self.settings.frame_height,
            )
            self.source_label = self.source.description
            self._start_worker()
        except Exception as error:
            self._show_error(error)

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Открыть видео",
            filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            self.stop_source()
            self.source.open_video(path)
            self.source_label = self.source.description
            self._start_worker()
        except Exception as error:
            self._show_error(error)

    def _start_worker(self) -> None:
        self.statistics.reset()
        self.risk_engine.reset()
        self.stop_event.clear()
        self.resume_event.set()
        self.is_running = True
        self.worker = threading.Thread(
            target=self._worker_loop, name="smartrail-frame-worker", daemon=True
        )
        self.worker.start()
        LOGGER.info("Frame source opened: %s", self.source_label)

    def start_analysis(self) -> None:
        if not self.is_running:
            return self._show_error("Сначала откройте камеру или видеофайл.")
        self.is_analyzing = True
        LOGGER.info("Analysis requested")

    def stop_analysis(self) -> None:
        self.is_analyzing = False
        self.risk_engine.reset()
        self.current_status = "NORMAL"
        LOGGER.info("Analysis stopped")

    def pause_video(self) -> None:
        if not self.is_running:
            return
        self.resume_event.clear()
        LOGGER.info("Frame source paused")

    def resume_video(self) -> None:
        if not self.is_running:
            return
        self.resume_event.set()
        LOGGER.info("Frame source resumed")

    def stop_source(self) -> None:
        self.is_analyzing = False
        self.is_running = False
        self.stop_event.set()
        self.resume_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.5)
        self.source.release()
        LOGGER.info("Frame source closed")

    def _worker_loop(self) -> None:
        frame_index = 0
        cached_detections: list[Detection] = []
        cached_rail = None
        cached_rail_debug = None
        cached_rail_roi = None
        next_frame_due = perf_counter()
        while not self.stop_event.is_set():
            if not self.resume_event.is_set():
                self.resume_event.wait()
                next_frame_due = perf_counter()
                continue
            if self.stop_event.is_set():
                break
            frame = self.source.read()
            if frame is None:
                break
            started_at = perf_counter()
            frame_index += 1
            inference_ms = 0.0
            interval = self.source.frame_interval_seconds
            frame_ts = (
                frame_index * interval if interval is not None else perf_counter()
            )
            if (
                cached_rail is None
                or cached_rail_roi != self.roi_points
                or frame_index % self.settings.rail_interval == 0
            ):
                rail_debug = RailDebugInfo() if self.settings.debug_mode else None
                rail = self.rail_detector.analyze(
                    frame, self.roi_points, debug=rail_debug, timestamp=frame_ts
                )
                cached_rail = rail
                cached_rail_debug = rail_debug
                cached_rail_roi = (
                    list(self.roi_points) if self.roi_points is not None else None
                )
            else:
                rail = cached_rail
                rail_debug = cached_rail_debug
            if self.is_analyzing:
                try:
                    if self.yolo is None:
                        self.yolo = YoloDetector(
                            self.settings.model_path,
                            self.settings.device,
                            self.settings.confidence_threshold,
                            self.settings.image_size,
                            self.settings.relevant_classes,
                        )
                        self.yolo.load()
                        LOGGER.info("YOLO model loaded on %s", self.settings.device)
                    if frame_index % self.settings.yolo_interval == 0:
                        cached_detections, inference_ms = self.yolo.detect(
                            frame, self.roi_points
                        )
                    risk = self.risk_engine.evaluate(cached_detections, rail)
                    analyzed = frame_index % self.settings.yolo_interval == 0
                except Exception as error:
                    try:
                        self.updates.put_nowait(error)
                    except queue.Full:
                        pass
                    self.is_analyzing = False
                    continue
            else:
                cached_detections = []
                risk = self.risk_engine.evaluate([], rail)
                analyzed = False
            processing_ms = (perf_counter() - started_at) * 1000
            self.statistics.record(
                analyzed=analyzed,
                inference_ms=inference_ms,
                processing_ms=processing_ms,
                detections=[d.class_name for d in cached_detections]
                if analyzed
                else [],
                confirmed=len(risk.confirmed_keys),
                risk=risk.score,
                status=risk.status,
                anomaly=rail.anomaly_type if rail.possible_anomaly else None,
            )
            update = FrameUpdate(
                frame,
                cached_detections,
                rail,
                risk,
                inference_ms,
                processing_ms,
                rail_debug,
            )
            try:
                self.updates.put_nowait(update)
            except queue.Full:
                try:
                    self.updates.get_nowait()
                except queue.Empty:
                    pass
                self.updates.put_nowait(update)
            interval = self.source.frame_interval_seconds
            if interval is not None:
                next_frame_due += interval
                delay = next_frame_due - perf_counter()
                if delay > 0:
                    self.stop_event.wait(delay)
                else:
                    next_frame_due = perf_counter()

    def _drain_updates(self) -> None:
        if self.is_closing:
            return
        try:
            while True:
                update = self.updates.get_nowait()
                if isinstance(update, Exception):
                    self._show_error(update)
                    continue
                self._render(update)
        except queue.Empty:
            pass
        if not self.is_closing:
            self.root.after(30, self._drain_updates)

    def _render(self, update: FrameUpdate) -> None:
        frame = update.frame.copy()
        self.latest_frame = update.frame
        draw_roi(frame, self.roi_points)
        for detection in update.detections:
            color = (0, 0, 255) if detection.inside_track_roi else (120, 120, 120)
            cv2.rectangle(
                frame,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                color,
                2,
            )
            cv2.putText(
                frame,
                f"{detection.class_name} {detection.confidence:.2f}",
                (detection.x1, max(20, detection.y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        for line in (update.rail.left_rail, update.rail.right_rail):
            if line:
                cv2.line(frame, line[:2], line[2:], (0, 255, 0), 3)
        height, width = frame.shape[:2]
        self.display_scale = min(900 / width, 500 / height)
        display = cv2.resize(
            frame, (int(width * self.display_scale), int(height * self.display_scale))
        )
        image = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        )
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(
                0, 0, image=image, anchor="nw"
            )
        else:
            self.canvas.itemconfigure(self.canvas_image_id, image=image)
        self.canvas.image = image
        status_color = {
            "HIGH RISK": "#ff4040",
            "WARNING": "#fbbf24",
            "NORMAL": "#4ade80",
        }.get(update.risk.status, "#f8fafc")
        status_text = (
            f"{self._status_label(update.risk.status)}  РИСК: {update.risk.score}/100"
        )
        if self.canvas_status_id is None:
            self.canvas_status_id = self.canvas.create_text(
                15,
                15,
                anchor="nw",
                fill=status_color,
                font=("Segoe UI", 14, "bold"),
                text=status_text,
            )
        else:
            self.canvas.itemconfigure(
                self.canvas_status_id, text=status_text, fill=status_color
            )
        reason_lines = []
        for reason in update.risk.reasons:
            weight_key = REASON_WEIGHT_KEYS.get(reason)
            points = self.settings.risk_weights.get(weight_key) if weight_key else None
            text = reason_label(reason)
            reason_lines.append(f"• {text}  +{points}" if points is not None else text)
        reasons_text = "Причины:\n" + "\n".join(reason_lines)
        if self.canvas_reasons_id is None:
            self.canvas_reasons_id = self.canvas.create_text(
                15,
                42,
                anchor="nw",
                fill="#e2e8f0",
                font=("Segoe UI", 10),
                justify="left",
                text=reasons_text,
            )
        else:
            self.canvas.itemconfigure(self.canvas_reasons_id, text=reasons_text)
        self.current_status = update.risk.status
        self.status_var.set(self._status_label(update.risk.status))
        self.risk_var.set(f"{update.risk.score} / 100")
        self.details_var.set(
            "; ".join(self._reason_label(reason) for reason in update.risk.reasons)
        )
        self.metrics_var.set(
            f"FPS: {self.statistics.average_fps:.2f}\nДетекции: {self.statistics.detections}\nТревоги: {self.statistics.alerts}"
        )
        self.rail_status_var.set(
            "Рельсы: обнаружены"
            if update.rail.rail_detected
            else "Рельсы: не обнаружены"
        )
        if self.settings.debug_mode and update.rail_debug is not None:
            debug = update.rail_debug
            self.rail_diag_var.set(
                f"Hough-линии: {debug.hough_lines}\n"
                f"После orientation: {debug.after_orientation}\n"
                f"После length: {debug.after_length}\n"
                f"После ROI: {debug.after_roi}\n"
                f"Кандидаты L/R: {debug.left_pool} / {debug.right_pool}\n"
                f"Stage: {debug.stage}\n"
                f"Rail detected: {'ДА' if update.rail.rail_detected else 'НЕТ'}\n"
                f"Confidence: {update.rail.score:.2f}"
            )
        self._draw_calibration_markers()

    def _draw_calibration_markers(self) -> None:
        self.canvas.delete("calibration")
        if not self.is_calibrating or not self.calibration_points:
            return
        for index, (x, y) in enumerate(self.calibration_points, start=1):
            dx = x * self.display_scale
            dy = y * self.display_scale
            radius = 8
            self.canvas.create_oval(
                dx - radius,
                dy - radius,
                dx + radius,
                dy + radius,
                outline="#38bdf8",
                width=2,
                tags="calibration",
            )
            self.canvas.create_text(
                dx,
                dy - radius - 8,
                text=str(index),
                fill="#38bdf8",
                font=("Segoe UI", 10, "bold"),
                tags="calibration",
            )

    def calibrate_roi(self) -> None:
        if self.latest_frame is None:
            return self._show_error("Запустите камеру или видео перед калибровкой.")
        self.calibration_points = []
        self.is_calibrating = True
        self.hint_var.set("Выберите четыре угла наблюдаемой зоны пути на видео.")

    def _roi_click(self, event: tk.Event) -> None:
        if self.latest_frame is None or not self.is_calibrating:
            return
        if len(self.calibration_points) >= 4:
            return
        self.calibration_points.append(
            (int(event.x / self.display_scale), int(event.y / self.display_scale))
        )
        if len(self.calibration_points) == 4:
            self.roi_points = self.calibration_points.copy()
            self.calibration_points = []
            self.is_calibrating = False
            self.hint_var.set("ROI откалибрована. Нажмите «Сохранить ROI».")
        else:
            self.hint_var.set(f"Выбрано: {len(self.calibration_points)}/4 точек")

    def reset_roi(self) -> None:
        self.roi_points = None
        self.calibration_points = []
        self.is_calibrating = False
        self.hint_var.set("ROI сброшена.")

    def save_roi(self) -> None:
        try:
            if self.roi_points is None:
                raise ValueError("Нет откалиброванной ROI для сохранения")
            save_roi(self.settings.roi_path, self.roi_points)
            self.details_var.set("ROI сохранена")
        except Exception as error:
            self._show_error(error)

    def save_report(self) -> Path | None:
        try:
            path = save_report(
                self.settings.reports_dir,
                self.source_label,
                self.statistics,
                self.current_status,
                self.settings.device,
            )
            self.details_var.set(f"Отчёт сохранён: {path.name}")
            LOGGER.info("Report saved: %s", path)
            return path
        except Exception as error:
            self._show_error(error)
            return None

    def _show_error(self, error: Exception | str) -> None:
        LOGGER.error("Application error: %s", error)
        messagebox.showerror("Ошибка SmartRail Alert", str(error))

    @staticmethod
    def _status_label(status: str) -> str:
        return STATUS_LABELS.get(status, status)

    @staticmethod
    def _reason_label(reason: str) -> str:
        return REASON_LABELS.get(reason, reason)

    def close(self) -> None:
        self.is_closing = True
        self.stop_source()
        self.root.destroy()
