"""Uncertainty estimation for supervised AI risk predictions."""

from __future__ import annotations

import math
from typing import Any

LOW_THRESHOLD = 0.33
HIGH_THRESHOLD = 0.67


def _entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-12), 1 - 1e-12)
    return round(float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p))), 4)


def estimate_prediction_uncertainty(
    ai_prediction: dict[str, Any],
    missing_important_values: list[dict[str, Any]] | None = None,
    data_quality: str | None = None,
) -> dict[str, Any]:
    if not ai_prediction.get("available") or ai_prediction.get("risk_probability") is None:
        return {"available": False, "message": "AI prediction probability is not available for uncertainty estimation."}

    probability = float(ai_prediction["risk_probability"])
    distances = [abs(probability - LOW_THRESHOLD), abs(probability - HIGH_THRESHOLD)]
    nearest_distance = round(float(min(distances)), 4)
    entropy = _entropy(probability)
    missing_count = len(missing_important_values or [])

    if nearest_distance < 0.05 or entropy > 0.9 or missing_count >= 3:
        confidence = "Borderline"
    elif nearest_distance < 0.15 or entropy > 0.65 or missing_count > 0 or data_quality in {"Partial", "Poor"}:
        confidence = "Moderate"
    else:
        confidence = "Strong"

    warning = "No major completeness warning from top AI features."
    if missing_count:
        warning = f"{missing_count} important AI feature value(s) were missing and imputed."
    elif data_quality in {"Partial", "Poor"}:
        warning = f"Overall workflow data quality is {data_quality}; interpret probability with caution."

    return {
        "available": True,
        "confidence_label": confidence,
        "probability_entropy": entropy,
        "distance_from_nearest_threshold": nearest_distance,
        "missing_important_feature_count": missing_count,
        "data_quality": data_quality or "Not available",
        "data_completeness_warning": warning,
        "thresholds": {"Low": "probability < 0.33", "Medium": "0.33 <= probability < 0.67", "High": "probability >= 0.67"},
        "explanation": "Uncertainty is computed from probability distance to risk thresholds, probability entropy, missing important AI features, and data quality.",
    }
