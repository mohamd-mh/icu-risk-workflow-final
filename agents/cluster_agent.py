"""KMeans cluster context for ICU case review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "patient_features_ai.csv"
MODEL_PATH = BASE_DIR / "models" / "ai_risk_model.pkl"
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


def _cluster_ai_probability(cluster_rows: pd.DataFrame) -> float | None:
    if not MODEL_PATH.exists() or cluster_rows.empty:
        return None
    try:
        model = joblib.load(MODEL_PATH)
        inputs = cluster_rows.drop(columns=[col for col in EXCLUDED_COLUMNS if col in cluster_rows.columns], errors="ignore")
        probabilities = model.predict_proba(inputs)[:, 1]
        return round(float(probabilities.mean()), 4)
    except Exception:
        return None


def assign_case_cluster(case_data: dict[str, Any], cluster_count: int = 5) -> dict[str, Any]:
    data = _load_dataset()
    if data.empty:
        return {"available": False, "message": "Processed feature dataset was not found."}

    numeric, categorical = _feature_columns(data)
    if not numeric and not categorical:
        return {"available": False, "message": "No usable features are available for clustering."}

    features = numeric + categorical
    n_clusters = max(1, min(cluster_count, len(data)))
    pipeline = Pipeline([
        ("preprocess", _preprocessor(numeric, categorical)),
        ("cluster", KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
    ])
    labels = pipeline.fit_predict(data[features])
    current_row = pd.DataFrame([{feature: case_data.get(feature) for feature in features}], columns=features)
    current_cluster = int(pipeline.predict(current_row)[0])

    data_with_cluster = data.copy()
    data_with_cluster["_cluster"] = labels
    cluster_rows = data_with_cluster[data_with_cluster["_cluster"] == current_cluster]

    mortality_pct = None
    if "hospital_expire_flag" in cluster_rows.columns:
        labels_series = pd.to_numeric(cluster_rows["hospital_expire_flag"], errors="coerce").dropna()
        if not labels_series.empty:
            mortality_pct = round(float(labels_series.mean() * 100), 1)

    avg_age = None
    if "age" in cluster_rows.columns:
        ages = pd.to_numeric(cluster_rows["age"], errors="coerce").dropna()
        if not ages.empty:
            avg_age = round(float(ages.mean()), 1)

    common_admission_type = "Not available"
    if "admission_type" in cluster_rows.columns:
        modes = cluster_rows["admission_type"].dropna().mode()
        if not modes.empty:
            common_admission_type = str(modes.iloc[0])

    return {
        "available": True,
        "method": "sklearn.cluster.KMeans",
        "cluster_id": current_cluster,
        "cluster_count": n_clusters,
        "cases_in_cluster": int(len(cluster_rows)),
        "average_ai_probability": _cluster_ai_probability(cluster_rows),
        "historical_mortality_label_percentage": mortality_pct,
        "common_admission_type": common_admission_type,
        "average_age": avg_age,
        "features_used": features,
        "note": "This is an unsupervised grouping from the processed academic dataset for review context only.",
    }
