"""AI prediction agent for the trained supervised ICU risk model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ai_risk_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "ai_model_metadata.json"

EXCLUDED_COLUMNS = {
    "subject_id",
    "hadm_id",
    "icustay_id",
    "hospital_expire_flag",
}
INPUT_FEATURES = [
    "gender",
    "age",
    "admission_type",
    "heart_rate_avg",
    "heart_rate_max",
    "systolic_bp_avg",
    "respiratory_rate_avg",
    "temperature_avg",
    "spo2_avg",
    "creatinine_max",
    "glucose_max",
    "wbc_max",
    "hemoglobin_min",
]
THRESHOLD_BASIS = "Low < 0.33, Medium 0.33-0.67, High >= 0.67"
LABEL_NOTE = (
    "hospital_expire_flag was used only for training/evaluation, "
    "not as an input feature"
)
_MODEL_CACHE = None
_METADATA_CACHE: dict[str, Any] | None = None


def get_risk_level(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    if probability < 0.67:
        return "Medium"
    return "High"


def load_model_metadata() -> dict[str, Any]:
    global _METADATA_CACHE
    if _METADATA_CACHE is not None:
        return _METADATA_CACHE
    if not METADATA_PATH.exists():
        return {}
    try:
        _METADATA_CACHE = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return _METADATA_CACHE
    except json.JSONDecodeError:
        return {}


def load_ai_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE


def prepare_case_row(case_row: dict[str, Any]) -> pd.DataFrame:
    model_input = {
        feature: case_row.get(feature)
        for feature in INPUT_FEATURES
        if feature not in EXCLUDED_COLUMNS
    }
    return pd.DataFrame([model_input], columns=INPUT_FEATURES)


def predict_ai_risk(case_row: dict[str, Any]) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {
            "available": False,
            "model_name": "AI model not trained yet",
            "risk_probability": None,
            "ai_risk_level": "Unavailable",
            "threshold_basis": THRESHOLD_BASIS,
            "label_note": LABEL_NOTE,
            "message": "AI model not trained yet",
        }

    try:
        model = load_ai_model()
        probability = float(model.predict_proba(prepare_case_row(case_row))[0][1])
        metadata = load_model_metadata()
        model_name = metadata.get("model_name") or model.named_steps[
            "classifier"
        ].__class__.__name__

        return {
            "available": True,
            "model_name": model_name,
            "risk_probability": round(probability, 4),
            "ai_risk_level": get_risk_level(probability),
            "threshold_basis": THRESHOLD_BASIS,
            "label_note": LABEL_NOTE,
        }
    except Exception as error:  # noqa: BLE001 - keep UI available on model issues.
        return {
            "available": False,
            "model_name": "AI model not trained yet",
            "risk_probability": None,
            "ai_risk_level": "Unavailable",
            "threshold_basis": THRESHOLD_BASIS,
            "label_note": LABEL_NOTE,
            "message": f"AI prediction could not be generated: {error}",
        }
