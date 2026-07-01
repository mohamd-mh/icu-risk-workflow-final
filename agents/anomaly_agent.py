"""IsolationForest anomaly context for ICU case review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "patient_features_ai.csv"
EXCLUDED_COLUMNS = {"subject_id", "hadm_id", "icustay_id", "hospital_expire_flag"}


def _load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


def _feature_columns(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = [col for col in data.columns if col not in EXCLUDED_COLUMNS]
    numeric = data[candidates].select_dtypes(include="number").columns.tolist()
    categorical = [col for col in candidates if col not in numeric]
    numeric = [col for col in numeric if data[col].notna().any()]
    categorical = [col for col in categorical if data[col].notna().any()]
    return numeric, categorical


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers)


def score_case_anomaly(case_data: dict[str, Any]) -> dict[str, Any]:
    data = _load_dataset()
    if data.empty:
        return {"available": False, "message": "Processed feature dataset was not found."}

    numeric, categorical = _feature_columns(data)
    if not numeric and not categorical:
        return {"available": False, "message": "No usable features are available for anomaly detection."}

    features = numeric + categorical
    preprocessor = _preprocessor(numeric, categorical)
    transformed = preprocessor.fit_transform(data[features])
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    model.fit(transformed)
    dataset_scores = model.score_samples(transformed)

    current_row = pd.DataFrame([{feature: case_data.get(feature) for feature in features}], columns=features)
    current_score = float(model.score_samples(preprocessor.transform(current_row))[0])
    percentile = round(float((dataset_scores <= current_score).mean() * 100), 1)

    if percentile < 5:
        label = "Highly unusual"
    elif percentile < 20:
        label = "Somewhat unusual"
    else:
        label = "Typical"

    return {
        "available": True,
        "method": "sklearn.ensemble.IsolationForest",
        "anomaly_score": round(current_score, 4),
        "score_percentile": percentile,
        "label": label,
        "features_used": features,
        "explanation": f"The case score is at approximately the {percentile}th percentile compared with the processed dataset; lower percentiles are more unusual.",
        "note": "This shows how unusual the case is compared with the processed dataset, not whether the patient is clinically abnormal.",
    }
