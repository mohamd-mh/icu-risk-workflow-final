"""Train a supervised ICU risk model from prepared patient features.

This script uses hospital_expire_flag only as the supervised training label.
It never includes that label or patient identifiers as prediction inputs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPANDED_DATA_PATH = PROJECT_ROOT / "data" / "patient_features_ai.csv"
FALLBACK_DATA_PATH = PROJECT_ROOT / "data" / "patient_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "ai_risk_model.pkl"
METRICS_PATH = MODELS_DIR / "ai_model_metrics.json"
IMPORTANCE_PATH = MODELS_DIR / "ai_feature_importance.json"
METADATA_PATH = MODELS_DIR / "ai_model_metadata.json"

TARGET_LABEL = "hospital_expire_flag"
EXCLUDED_COLUMNS = [
    "hospital_expire_flag",
    "subject_id",
    "hadm_id",
    "icustay_id",
]
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
CATEGORICAL_FEATURES = ["gender", "admission_type"]
NUMERIC_FEATURES = [feature for feature in INPUT_FEATURES if feature not in CATEGORICAL_FEATURES]
RISK_THRESHOLDS = {
    "Low": "probability < 0.33",
    "Medium": "0.33 <= probability < 0.67",
    "High": "probability >= 0.67",
}


def get_training_data_path() -> Path:
    return EXPANDED_DATA_PATH if EXPANDED_DATA_PATH.exists() else FALLBACK_DATA_PATH


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest Classifier": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        class_weight="balanced",
                        n_estimators=200,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def load_training_data() -> tuple[pd.DataFrame, pd.Series, list[str], Path]:
    warnings: list[str] = []
    data_path = get_training_data_path()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    missing_columns = [
        column for column in [TARGET_LABEL, *INPUT_FEATURES] if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    data = data[[TARGET_LABEL, *INPUT_FEATURES]].copy()
    data = data.dropna(subset=[TARGET_LABEL])
    if data.empty:
        raise ValueError("No rows remain after dropping missing target labels.")

    y = data[TARGET_LABEL].astype(int)
    X = data[INPUT_FEATURES]

    class_counts = y.value_counts().to_dict()
    if len(class_counts) < 2:
        raise ValueError("The target label must contain at least two classes.")
    if min(class_counts.values()) < 2:
        warnings.append(
            "At least one class has fewer than two rows; stratified split may not be possible."
        )

    return X, y, warnings, data_path


def split_training_data(
    X: pd.DataFrame, y: pd.Series, warnings: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    if len(X) < 4:
        warnings.append(
            "Dataset is very small; evaluating on the training data instead of a train/test split."
        )
        return X, X, y, y, warnings

    class_counts = y.value_counts()
    stratify = y if len(class_counts) > 1 and class_counts.min() >= 2 else None
    if stratify is None:
        warnings.append("Stratified split was skipped because class counts are too small.")

    try:
        return (*train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify), warnings)
    except ValueError as error:
        warnings.append(
            f"Train/test split failed ({error}); evaluating on the training data instead."
        )
        return X, X, y, y, warnings


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "roc_auc": None,
    }

    if len(set(y_test)) < 2:
        metrics["roc_auc_warning"] = "ROC-AUC was skipped because y_test has one class."
        return metrics

    try:
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities)
    except (AttributeError, ValueError) as error:
        metrics["roc_auc_warning"] = f"ROC-AUC could not be computed: {error}"

    return metrics


def select_operating_threshold(y_train: pd.Series, train_probabilities) -> float:
    """Pick the probability cutoff that maximizes F1 on training predictions.

    class_weight="balanced" shifts the default 0.5 cutoff toward high recall and
    low precision. precision_recall_curve is used here (on training data, not the
    test set, to avoid tuning on the data the reported metrics are evaluated on)
    to find a cutoff that trades some recall back for precision.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_train, train_probabilities)
    precisions, recalls = precisions[:-1], recalls[:-1]
    f1_scores = [
        (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        for precision, recall in zip(precisions, recalls)
    ]
    best_index = max(range(len(f1_scores)), key=lambda index: f1_scores[index])
    return float(thresholds[best_index])


def build_tuned_threshold_metrics(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> dict | None:
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "predict_proba") or len(set(y_train)) < 2:
        return None

    train_probabilities = model.predict_proba(X_train)[:, 1]
    threshold = select_operating_threshold(y_train, train_probabilities)
    test_probabilities = model.predict_proba(X_test)[:, 1]
    tuned_predictions = (test_probabilities >= threshold).astype(int)

    return {
        "decision_threshold": round(threshold, 4),
        "selection_method": "F1-maximizing cutoff from precision_recall_curve on training predictions",
        "accuracy": accuracy_score(y_test, tuned_predictions),
        "precision": precision_score(y_test, tuned_predictions, zero_division=0),
        "recall": recall_score(y_test, tuned_predictions, zero_division=0),
        "f1_score": f1_score(y_test, tuned_predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, tuned_predictions).tolist(),
        "tradeoff_note": (
            "The default 0.5 cutoff (see 'metrics') favors recall because the classifier "
            "uses class_weight=\"balanced\" on an imbalanced label. This tuned cutoff was "
            "chosen to maximize F1 on training predictions and generally raises precision "
            "at the cost of some recall; compare the two blocks to judge the tradeoff for "
            "your review workflow."
        ),
    }


def select_best_model(results: dict[str, dict]) -> str:
    def score(model_name: str) -> tuple[float, float]:
        model_metrics = results[model_name]["metrics"]
        roc_auc = model_metrics["roc_auc"]
        return (
            model_metrics["f1_score"],
            roc_auc if roc_auc is not None else -1.0,
        )

    return max(results, key=score)


def extract_feature_importance(model_name: str, model: Pipeline) -> list[dict]:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        importance_type = "random_forest_feature_importance"
    elif hasattr(classifier, "coef_"):
        importances = abs(classifier.coef_[0])
        importance_type = "absolute_logistic_regression_coefficient"
    else:
        return []

    ranked = sorted(
        [
            {
                "feature": feature_name,
                "importance": float(importance),
                "importance_type": importance_type,
                "model_name": model_name,
            }
            for feature_name, importance in zip(feature_names, importances, strict=False)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranked


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    try:
        X, y, load_warnings, data_path = load_training_data()
        warnings.extend(load_warnings)
        X_train, X_test, y_train, y_test, warnings = split_training_data(X, y, warnings)

        results: dict[str, dict] = {}
        trained_models: dict[str, Pipeline] = {}
        for model_name, model in build_models().items():
            try:
                model.fit(X_train, y_train)
                results[model_name] = {
                    "metrics": evaluate_model(model, X_test, y_test),
                    "tuned_threshold_metrics": build_tuned_threshold_metrics(
                        model, X_train, y_train, X_test, y_test
                    ),
                    "train_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                }
                trained_models[model_name] = model
            except Exception as error:  # noqa: BLE001 - save error in metrics JSON.
                results[model_name] = {"error": str(error)}
                warnings.append(f"{model_name} failed during training or evaluation: {error}")

        successful_results = {
            name: result for name, result in results.items() if "metrics" in result
        }
        if not successful_results:
            raise RuntimeError("No model trained successfully.")

        best_model_name = select_best_model(successful_results)
        best_model = trained_models[best_model_name]
        joblib.dump(best_model, MODEL_PATH)

        metrics_payload = {
            "best_model": best_model_name,
            "models": results,
            "warnings": warnings,
            "class_distribution": {
                str(label): int(count) for label, count in y.value_counts().sort_index().items()
            },
            "label_distribution": {
                str(label): int(count) for label, count in y.value_counts().sort_index().items()
            },
            "selected_dataset_path": str(data_path.relative_to(PROJECT_ROOT)),
            "selected_dataset_filename": data_path.name,
            "training_row_count": int(len(y)),
            "target_label": TARGET_LABEL,
        }
        write_json(METRICS_PATH, metrics_payload)

        write_json(
            IMPORTANCE_PATH,
            {
                "model_name": best_model_name,
                "feature_importance": extract_feature_importance(best_model_name, best_model),
            },
        )

        write_json(
            METADATA_PATH,
            {
                "model_name": best_model_name,
                "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "selected_dataset_path": str(data_path.relative_to(PROJECT_ROOT)),
                "selected_dataset_filename": data_path.name,
                "training_row_count": int(len(y)),
                "label_distribution": {
                    str(label): int(count) for label, count in y.value_counts().sort_index().items()
                },
                "input_features": INPUT_FEATURES,
                "excluded_columns": EXCLUDED_COLUMNS,
                "target_label": TARGET_LABEL,
                "risk_thresholds": RISK_THRESHOLDS,
                "tuned_decision_threshold": results[best_model_name].get("tuned_threshold_metrics"),
                "note": (
                    "hospital_expire_flag is used only as training/evaluation label "
                    "and never as prediction input."
                ),
            },
        )

        print(f"Training complete. Best model: {best_model_name}")
        print(f"Selected dataset: {data_path}")
        print(f"Training rows: {len(y)}")
        print(
            "Class distribution: "
            + json.dumps(
                {str(label): int(count) for label, count in y.value_counts().sort_index().items()},
                sort_keys=True,
            )
        )
        print(f"Saved model: {MODEL_PATH}")
        print(f"Saved metrics: {METRICS_PATH}")
        print(f"Saved feature importance: {IMPORTANCE_PATH}")
        print(f"Saved metadata: {METADATA_PATH}")
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"- {warning}")

    except Exception as error:  # noqa: BLE001 - preserve failure details for the user.
        failure_payload = {
            "error": str(error),
            "warnings": warnings,
            "training_failed": True,
        }
        write_json(METRICS_PATH, failure_payload)
        raise


if __name__ == "__main__":
    main()
