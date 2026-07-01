"""Nearest-neighbor similar case retrieval from the processed ICU feature dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "patient_features_ai.csv"
EXCLUDED_COLUMNS = {"subject_id", "hadm_id", "icustay_id", "hospital_expire_flag"}


def _load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


def _numeric_features(data: pd.DataFrame) -> list[str]:
    candidates = [col for col in data.columns if col not in EXCLUDED_COLUMNS]
    numeric = data[candidates].select_dtypes(include="number").columns.tolist()
    return [col for col in numeric if data[col].notna().any()]


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def find_similar_cases(case_data: dict[str, Any], top_n: int = 5) -> dict[str, Any]:
    data = _load_dataset()
    if data.empty:
        return {"available": False, "message": "Processed feature dataset was not found.", "cases": [], "summary": {}}

    features = _numeric_features(data)
    if not features:
        return {"available": False, "message": "No usable numeric features are available for similarity search.", "cases": [], "summary": {}}

    current_id = str(case_data.get("icustay_id", ""))
    feature_matrix = data[features]
    current_row = pd.DataFrame([{feature: case_data.get(feature) for feature in features}], columns=features)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformed = pipeline.fit_transform(feature_matrix)
    current_transformed = pipeline.transform(current_row)

    neighbor_count = min(len(data), top_n + 1)
    model = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    model.fit(transformed)
    distances, indices = model.kneighbors(current_transformed)

    similar = []
    for distance, index in zip(distances[0], indices[0]):
        row = data.iloc[int(index)]
        if str(row.get("icustay_id", "")) == current_id:
            continue
        similar.append({
            "icustay_id": str(_clean_value(row.get("icustay_id"))),
            "subject_id": str(_clean_value(row.get("subject_id"))) if "subject_id" in row else "Not available",
            "age": _clean_value(row.get("age")),
            "admission_type": _clean_value(row.get("admission_type")) or "Not available",
            "distance": round(float(distance), 4),
            "similarity_score": round(float(1 / (1 + distance)), 4),
            "historical_outcome_label": _clean_value(row.get("hospital_expire_flag")) if "hospital_expire_flag" in data.columns else None,
        })
        if len(similar) >= top_n:
            break

    similar_frame = pd.DataFrame(similar)
    mortality_pct = None
    if not similar_frame.empty and "historical_outcome_label" in similar_frame:
        labels = pd.to_numeric(similar_frame["historical_outcome_label"], errors="coerce").dropna()
        if not labels.empty:
            mortality_pct = round(float(labels.mean() * 100), 1)

    avg_age = None
    if not similar_frame.empty and "age" in similar_frame:
        ages = pd.to_numeric(similar_frame["age"], errors="coerce").dropna()
        if not ages.empty:
            avg_age = round(float(ages.mean()), 1)

    common_admission_type = None
    if not similar_frame.empty and "admission_type" in similar_frame:
        modes = similar_frame["admission_type"].dropna().mode()
        if not modes.empty:
            common_admission_type = str(modes.iloc[0])

    return {
        "available": True,
        "method": "sklearn.neighbors.NearestNeighbors",
        "features_used": features,
        "cases": similar,
        "summary": {
            "similar_cases_found": len(similar),
            "mortality_label_percentage": mortality_pct,
            "average_age": avg_age,
            "common_admission_type": common_admission_type or "Not available",
        },
        "note": "Similar cases are retrieved from the processed academic dataset using nearest-neighbor similarity. They are review context only, not treatment guidance.",
    }
