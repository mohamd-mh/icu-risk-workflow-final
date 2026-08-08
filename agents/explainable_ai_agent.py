"""Explainability helper for the trained AI risk model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORTANCE_PATH = PROJECT_ROOT / "models" / "ai_feature_importance.json"

DISPLAY_NAMES = {
    "spo2_avg": "Oxygen saturation",
    "systolic_bp_avg": "Systolic blood pressure",
    "respiratory_rate_avg": "Respiratory rate",
    "creatinine_max": "Creatinine",
    "heart_rate_avg": "Heart rate",
    "heart_rate_max": "Maximum heart rate",
    "wbc_max": "White blood cell count",
    "hemoglobin_min": "Hemoglobin",
    "age": "Age",
    "temperature_avg": "Temperature",
    "glucose_max": "Glucose",
    "gender": "Gender",
    "admission_type": "Admission type",
}
EXPLANATION_NOTE = (
    "Feature contribution is an explainability aid for the trained AI model, "
    "provided for quality-review support."
)
_FEATURE_IMPORTANCE_CACHE: list[dict[str, Any]] | None = None


def clean_feature_name(feature_name: str) -> str:
    if "__" in feature_name:
        feature_name = feature_name.split("__", 1)[1]
    for prefix in ["gender_", "admission_type_"]:
        if feature_name.startswith(prefix):
            return prefix[:-1]
    return feature_name


def is_missing(value: Any) -> bool:
    return value is None or str(value).lower() == "nan" or value != value


def numeric_value(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def interpret_feature(feature: str, display_name: str, value: Any) -> str:
    number = numeric_value(value)
    if is_missing(value):
        return f"{display_name} was missing, which the AI model handled using imputation."
    if feature == "spo2_avg" and number is not None and number < 92:
        return "Low oxygen saturation contributed to higher predicted risk."
    if feature == "systolic_bp_avg" and number is not None and number < 90:
        return "Lower systolic blood pressure was an important model signal."
    if feature == "respiratory_rate_avg" and number is not None and number > 24:
        return "Elevated respiratory rate was an important model signal."
    if feature == "creatinine_max" and number is not None and number > 1.5:
        return "Higher creatinine was an important model signal."
    if feature in ["heart_rate_avg", "heart_rate_max"] and number is not None and number > 100:
        return "Elevated heart rate was an important model signal."
    if feature == "wbc_max" and number is not None and number > 12:
        return "Higher white blood cell count was an important model signal."
    if feature == "hemoglobin_min" and number is not None and number < 10:
        return "Lower hemoglobin was an important model signal."
    if feature == "age" and number is not None and number > 70:
        return "Older age was an important model signal."
    return f"{display_name} was one of the higher-weighted features for this AI model."


def is_abnormal_model_signal(feature: str, value: Any) -> bool:
    number = numeric_value(value)
    if number is None:
        return False
    return (
        (feature == "spo2_avg" and number < 92)
        or (feature == "systolic_bp_avg" and number < 90)
        or (feature == "respiratory_rate_avg" and number > 24)
        or (feature == "creatinine_max" and number > 1.5)
        or (feature in ["heart_rate_avg", "heart_rate_max"] and number > 100)
        or (feature == "wbc_max" and number > 12)
        or (feature == "hemoglobin_min" and number < 10)
        or (feature == "age" and number > 70)
    )


def load_feature_importance() -> list[dict[str, Any]]:
    global _FEATURE_IMPORTANCE_CACHE
    if _FEATURE_IMPORTANCE_CACHE is not None:
        return _FEATURE_IMPORTANCE_CACHE
    if not IMPORTANCE_PATH.exists():
        return []
    try:
        payload = json.loads(IMPORTANCE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    _FEATURE_IMPORTANCE_CACHE = payload.get("feature_importance", [])
    return _FEATURE_IMPORTANCE_CACHE


def explain_ai_prediction(case_row: dict[str, Any], limit: int = 5) -> dict[str, Any]:
    importance_rows = load_feature_importance()
    if not importance_rows:
        return {
            "top_features": [],
            "explanation_note": EXPLANATION_NOTE,
            "message": "AI feature importance file is not available yet.",
        }

    seen_features: set[str] = set()
    top_features: list[dict[str, Any]] = []
    for row in importance_rows:
        feature = clean_feature_name(str(row.get("feature", "")))
        if not feature or feature in seen_features:
            continue
        seen_features.add(feature)
        display_name = DISPLAY_NAMES.get(feature, feature.replace("_", " ").title())
        value = case_row.get(feature)
        missing = is_missing(value)
        abnormal = is_abnormal_model_signal(feature, value)
        top_features.append(
            {
                "feature": feature,
                "display_name": display_name,
                "value": None if missing else value,
                "importance": row.get("importance", 0),
                "is_missing": missing,
                "is_abnormal": abnormal,
                "signal_type": "missing" if missing else "present_abnormal" if abnormal else "present",
                "interpretation": interpret_feature(feature, display_name, value),
            }
        )
        if len(top_features) >= limit:
            break

    return {
        "top_features": top_features,
        "explanation_note": EXPLANATION_NOTE,
    }
