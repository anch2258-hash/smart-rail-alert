"""Russian display labels for user-visible interface strings."""

from __future__ import annotations

STATUS_LABELS = {
    "NORMAL": "НОРМА",
    "WARNING": "ПРЕДУПРЕЖДЕНИЕ",
    "HIGH RISK": "ВЫСОКИЙ РИСК",
}

REASON_LABELS = {
    "No confirmed anomalies": "Подтверждённые аномалии отсутствуют",
    "Person confirmed inside track region": "Человек подтверждён в зоне пути",
    "Vehicle confirmed inside track region": "Транспорт подтверждён в зоне пути",
    "Relevant object confirmed inside track region": "Релевантный объект подтверждён в зоне пути",
    "Possible rail line discontinuity": "Возможная потеря непрерывности рельса",
    "Possible rail geometry anomaly": "Возможная аномалия геометрии рельса",
    "Multiple confirmed issues": "Несколько подтверждённых событий",
}

CLASS_LABELS = {
    "person": "человек",
    "bicycle": "велосипед",
    "motorcycle": "мотоцикл",
    "car": "автомобиль",
    "bus": "автобус",
    "truck": "грузовик",
    "suitcase": "чемодан",
    "backpack": "рюкзак",
    "foreign_object": "посторонний объект",
}

REASON_WEIGHT_KEYS = {
    "Person confirmed inside track region": "person",
    "Vehicle confirmed inside track region": "vehicle",
    "Relevant object confirmed inside track region": "object",
    "Possible rail line discontinuity": "rail_discontinuity",
    "Possible rail geometry anomaly": "rail_anomaly",
    "Multiple confirmed issues": "multiple_issues",
}

ANOMALY_LABELS = {
    "line_discontinuity": "разрыв линии",
    "geometry_deviation": "отклонение геометрии",
    "geometry": "геометрия",
}


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status)


def reason_label(reason: str) -> str:
    return REASON_LABELS.get(reason, reason)


def class_label(class_name: str) -> str:
    return CLASS_LABELS.get(class_name, class_name)


def anomaly_label(anomaly: str) -> str:
    return ANOMALY_LABELS.get(anomaly, anomaly)
