"""Flask app for the ICU risk workflow website."""

import json
import math
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

from agents.ai_prediction_agent import EXCLUDED_COLUMNS as AI_EXCLUDED_COLUMNS
from agents.ai_prediction_agent import INPUT_FEATURES as AI_INPUT_FEATURES
from agents.ai_prediction_agent import get_risk_level as get_ai_risk_level
from agents.ai_prediction_agent import load_ai_model, predict_ai_risk
from agents.anomaly_agent import score_case_anomaly
from agents.cluster_agent import assign_case_cluster
from agents.explainable_ai_agent import explain_ai_prediction
from agents.similar_cases_agent import find_similar_cases
from agents.uncertainty_agent import estimate_prediction_uncertainty
from agents.generative_ai_agent import answer_case_question, generate_case_summary
from agents.multi_agent_workflow import run_data_validator, run_workflow
from baseline.baseline_pipeline import run_baseline
from evaluation.evaluate import run_evaluation
from review_store import (
    PRIORITIES as WORKFLOW_PRIORITIES,
    STATUSES as WORKFLOW_STATUSES,
    add_review_note,
    archive_review_item,
    create_dataset_review_item,
    create_manual_case,
    create_manual_review_item,
    delete_review_item,
    get_active_review_item_for_icustay,
    get_audit_events,
    get_manual_case,
    get_review_item,
    get_review_notes,
    get_workflow_stats,
    list_review_items,
    update_review_item,
)

app = Flask(__name__)

EXPANDED_DATA_PATH = Path(__file__).parent / "data" / "patient_features_ai.csv"
FALLBACK_DATA_PATH = Path(__file__).parent / "data" / "patient_features.csv"
DATA_PATH = EXPANDED_DATA_PATH if EXPANDED_DATA_PATH.exists() else FALLBACK_DATA_PATH
REVIEWS_PATH = Path(__file__).parent / "data" / "case_reviews.json"
FEATURE_AVAILABILITY_PATH = Path(__file__).parent / "data" / "feature_availability_report.csv"
EVALUATION_RESULTS_PATH = Path(__file__).parent / "data" / "evaluation_results.json"
AI_MODEL_PATH = Path(__file__).parent / "models" / "ai_risk_model.pkl"
AI_METADATA_PATH = Path(__file__).parent / "models" / "ai_model_metadata.json"
AI_METRICS_PATH = Path(__file__).parent / "models" / "ai_model_metrics.json"

CLINICAL_MEASUREMENTS = [
    ("heart_rate_avg", "Average heart rate", "bpm"),
    ("heart_rate_max", "Maximum heart rate", "bpm"),
    ("systolic_bp_avg", "Average systolic blood pressure", "mmHg"),
    ("respiratory_rate_avg", "Average respiratory rate", "breaths/min"),
    ("temperature_avg", "Average temperature", "C"),
    ("spo2_avg", "Average SpO2", "%"),
    ("creatinine_max", "Maximum creatinine", "mg/dL"),
    ("glucose_max", "Maximum glucose", "mg/dL"),
    ("wbc_max", "Maximum white blood cell count", "K/uL"),
    ("hemoglobin_min", "Minimum hemoglobin", "g/dL"),
]

REVIEW_STATUSES = ["Not Reviewed", "In Review", "Needs Follow-up", "Ready for Quality Team Review", "Reviewed"]
REVIEW_PRIORITIES = ["Normal", "Watch", "Urgent Review"]

SAFETY_SENTENCE = (
    "This system supports quality-review prioritization and documentation only. "
    "It does not replace hospital review, bedside assessment, or care protocols."
)
CASE_PAGE_SIZE_OPTIONS = [25, 50]
DEFAULT_CASE_PAGE_SIZE = 25
REVIEW_OUTCOMES = [
    "Pending team review",
    "No further action",
    "Needs follow-up documentation",
    "Data quality issue",
    "Escalated to quality team",
    "Reviewed and closed",
]

NAV_SECTIONS = [
    {
        "label": "Main Workflow",
        "pages": [
            {"endpoint": "software_home", "label": "Dashboard"},
            {"endpoint": "review_queue", "label": "Review Queue"},
            {"endpoint": "icu_cases", "label": "Case Library"},
            {"endpoint": "new_case", "label": "Create Review Ticket"},
        ],
    },
    {
        "label": "System Transparency",
        "pages": [
            {"endpoint": "ai_evaluation", "label": "AI Risk Engine"},
            {"endpoint": "results", "label": "Model Validation"},
            {"endpoint": "user_guide", "label": "User Guide"},
        ],
    }
]
PAGES = [page for section in NAV_SECTIONS for page in section["pages"]]

WORKFLOW_MESSAGES = {
    "ticket_in_review": {
        "level": "success",
        "text": "Ticket moved to In Review. Next step: review signals and add a Coordinator Note.",
    },
    "note_saved": {
        "level": "success",
        "text": "Coordinator Note saved. Next step: generate the Prepared Review Summary.",
    },
    "summary_prepared": {
        "level": "success",
        "text": "Review summary prepared. Next step: mark the ticket Ready for Quality Team Review.",
    },
    "ready_quality_team": {
        "level": "success",
        "text": "Ticket is ready for the Quality & Patient Safety Review Team.",
    },
    "ticket_reviewed": {
        "level": "success",
        "text": "Ticket marked as Reviewed. Review is complete. Archive the ticket when documentation is no longer active.",
    },
    "ticket_archived": {
        "level": "info",
        "text": "This ticket is archived and removed from the active Review Queue. It is kept for audit/history.",
    },
    "archive_requires_reviewed": {
        "level": "warning",
        "text": "Archive is a final closure action and is available only after the review workflow status is Reviewed.",
    },
    "ticket_updated": {
        "level": "success",
        "text": "Review Ticket updated.",
    },
}


def workflow_message_from_code(code):
    return WORKFLOW_MESSAGES.get(code or "")


def local_redirect_target(target, fallback):
    if not target or not target.startswith("/") or target.startswith("//"):
        return fallback
    return target


def add_query_params(target, **params):
    split = urlsplit(target)
    query = dict(parse_qsl(split.query, keep_blank_values=True))
    for key, value in params.items():
        if value is not None:
            query[key] = value
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(query), split.fragment))


def status_message_code(status):
    return {
        "In Review": "ticket_in_review",
        "Ready for Quality Team Review": "ready_quality_team",
        "Reviewed": "ticket_reviewed",
        "Archived": "ticket_archived",
    }.get(status, "ticket_updated")


def can_archive_review_item(item):
    return bool(item and item.get("status") == "Reviewed")


def editable_workflow_statuses(item):
    if item.get("status") == "Archived":
        return WORKFLOW_STATUSES
    return [status for status in WORKFLOW_STATUSES if status != "Archived"]

@app.context_processor
def inject_navigation():
    """Make the sidebar links available to every template."""
    return {
        "pages": PAGES,
        "nav_sections": NAV_SECTIONS,
        "safety_sentence": SAFETY_SENTENCE,
        "workflow_message": workflow_message_from_code(request.args.get("msg")),
    }


def load_patient_data():
    """Load patient feature rows used by the Demo page."""
    return pd.read_csv(DATA_PATH)


def get_active_feature_dataset_info():
    data = load_patient_data()
    return {
        "path": str(DATA_PATH.relative_to(Path(__file__).parent)),
        "filename": DATA_PATH.name,
        "row_count": int(len(data)),
    }


def load_case_reviews():
    """Load saved local clinical review notes."""
    if not REVIEWS_PATH.exists():
        return {}

    with REVIEWS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_case_reviews(reviews):
    """Save local clinical review notes."""
    REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEWS_PATH.open("w", encoding="utf-8") as file:
        json.dump(reviews, file, indent=2, sort_keys=True)


def normalize_review_history(review):
    history = review.get("history", [])
    if not isinstance(history, list):
        history = []
    return [
        {
            "status": entry.get("status", entry.get("review_status", "Not Reviewed")),
            "priority": entry.get("priority", "Normal"),
            "reviewer": entry.get("reviewer", ""),
            "note": entry.get("note", entry.get("reviewer_note", "")),
            "updated_at": entry.get("updated_at", ""),
        }
        for entry in history
        if isinstance(entry, dict)
    ]


def load_feature_availability():
    if not FEATURE_AVAILABILITY_PATH.exists():
        return []
    return pd.read_csv(FEATURE_AVAILABILITY_PATH).to_dict(orient="records")


def load_evaluation_results_file():
    if not EVALUATION_RESULTS_PATH.exists():
        return None
    with EVALUATION_RESULTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_json_file(path):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return {}


def safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_probability_percent(probability):
    if probability is None:
        return "Not available"
    try:
        return f"{float(probability) * 100:.1f}%"
    except (TypeError, ValueError):
        return "Not available"


def get_sirs_screening_label(baseline_result):
    return "SIRS Positive" if baseline_result.get("sirs_positive") else "SIRS Negative"


def get_sirs_screening_key(baseline_result):
    return "positive" if baseline_result.get("sirs_positive") else "negative"


def is_low_confidence_or_missing(probability, data_quality):
    if data_quality in {"Partial", "Poor"}:
        return True
    if probability is None:
        return True
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        return True
    return min(abs(probability - 0.33), abs(probability - 0.67)) < 0.05


def suggest_review_priority(ai_risk_level, sirs_positive=False, confidence_label="", data_quality=""):
    """Combine model output and review context into a workflow priority."""
    if ai_risk_level == "High" or data_quality == "Poor":
        return "Urgent"
    if (
        ai_risk_level == "Medium"
        or sirs_positive
        or confidence_label == "Borderline"
        or data_quality == "Partial"
    ):
        return "Watch"
    return "Low"


def get_priority_sort_value(priority):
    return {"Urgent": 0, "Watch": 1, "Low": 2, "Urgent Review": 0, "Normal": 2}.get(priority or "", 3)


def get_risk_sort_value(risk_level):
    return {"High": 0, "Medium": 1, "Low": 2}.get(risk_level or "", 3)


def get_status_sort_value(status):
    return {
        "Needs Follow-up": 0,
        "New": 1,
        "Not Reviewed": 1,
        "In Queue": 1,
        "In Review": 2,
        "Ready for Quality Team Review": 3,
        "Reviewed": 4,
        "Closed": 5,
        "Archived": 6,
    }.get(status or "", 7)


def get_confidence_label(probability, data_quality):
    if data_quality in {"Partial", "Poor"}:
        return "Low confidence / missing data"
    if probability is None:
        return "Not available"
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        return "Not available"
    if min(abs(probability - 0.33), abs(probability - 0.67)) < 0.05:
        return "Borderline"
    return "Standard"


def build_recommended_action(
    ai_risk_level,
    sirs_positive=False,
    confidence_label="",
    data_quality="",
    status="",
    priority="",
    source="",
):
    if status == "Archived":
        return "This ticket is archived and removed from the active Review Queue. It is kept for audit/history."
    if status == "Ready for Quality Team Review":
        return "Ready for handoff. Keep the prepared summary, notes, and audit trail available for the Quality & Patient Safety Review Team."
    if status == "Reviewed":
        return "Review is complete. Archive the ticket when documentation is no longer active."
    if status == "Needs Follow-up":
        return "Needs follow-up documentation. Add a coordinator note and keep ownership assigned."
    if data_quality in {"Partial", "Poor"} or "missing" in (confidence_label or "").lower():
        return "Check data completeness before relying on the risk estimate. Add a note documenting missing values."
    if ai_risk_level == "High" and sirs_positive:
        return "Prioritize this case for quality review. Review model signals and SIRS criteria, then mark the item In Review or Needs Follow-up."
    if ai_risk_level == "High":
        return "Prioritize for review and mark In Review."
    if sirs_positive or priority == "Watch":
        return "Review SIRS screening context and document whether follow-up is needed."
    return "Open the profile, review AI-assisted priority, and document the coordinator decision."


def build_why_flagged_summary(ai_prediction, model_signal_text, baseline_result, uncertainty_result, anomaly_result, data_quality):
    probability = format_probability_percent(ai_prediction.get("risk_probability"))
    risk_level = ai_prediction.get("ai_risk_level", "Not available")
    sirs_text = f"{get_sirs_screening_label(baseline_result)} ({baseline_result.get('sirs_criteria_met', 0)} of 4 criteria)"
    confidence_text = uncertainty_result.get("confidence_label", get_confidence_label(ai_prediction.get("risk_probability"), data_quality))
    anomaly_text = anomaly_result.get("label", "Anomaly context not available")
    return (
        f"The trained model estimated {probability} probability, mapped to {risk_level} AI-assisted review priority. "
        f"Main model signals: {model_signal_text}. SIRS screening: {sirs_text}. "
        f"Confidence/data quality: {confidence_text} / {data_quality}. Anomaly status: {anomaly_text}."
    )


def get_review_state_from_item(item):
    if not item:
        return "Not Reviewed"
    if item.get("status") == "Archived":
        return "Archived"
    if item.get("status") == "Ready for Quality Team Review":
        return "Ready for Quality Team Review"
    if item.get("status") == "Reviewed":
        return "Reviewed"
    return "In Queue"


def triage_score(item):
    status = item.get("status") or item.get("review_status")
    updated = item.get("updated_at") or ""
    return (
        get_priority_sort_value(item.get("priority") or item.get("suggested_review_priority")),
        get_risk_sort_value(item.get("ai_risk_level")),
        0 if item.get("sirs_screening_key") == "positive" else 1,
        0 if status == "Needs Follow-up" else 1,
        get_status_sort_value(status),
        0 if item.get("low_confidence_or_missing") or item.get("data_quality") in {"Partial", "Poor"} else 1,
        -int(str(updated).replace("-", "").replace(":", "").replace(" ", "") or 0) if str(updated).replace("-", "").replace(":", "").replace(" ", "").isdigit() else 0,
    )


def start_next_review_score(item):
    status = item.get("status") or item.get("review_status")
    updated = item.get("updated_at") or ""
    numeric_updated = str(updated).replace("-", "").replace(":", "").replace(" ", "")
    return (
        get_priority_sort_value(item.get("priority") or item.get("suggested_review_priority")),
        get_risk_sort_value(item.get("ai_risk_level")),
        0 if item.get("sirs_screening_key") == "positive" else 1,
        get_status_sort_value(status),
        0 if item.get("low_confidence_or_missing") or item.get("data_quality") in {"Partial", "Poor"} else 1,
        -int(numeric_updated) if numeric_updated.isdigit() else 0,
    )


def source_label(source_type):
    return "Dataset" if source_type == "dataset_case" else "Manual"


def build_ai_prediction_summaries(patients):
    unavailable = [
        {
            "available": False,
            "risk_probability": None,
            "ai_risk_level": "Unavailable",
            "probability_percent": "Not available",
        }
        for _ in range(len(patients))
    ]
    if not AI_MODEL_PATH.exists():
        return unavailable

    try:
        model = load_ai_model()
        model_input = patients.reindex(columns=AI_INPUT_FEATURES)
        probabilities = model.predict_proba(model_input)[:, 1]
    except Exception:
        return unavailable

    summaries = []
    for probability in probabilities:
        probability = round(float(probability), 4)
        summaries.append(
            {
                "available": True,
                "risk_probability": probability,
                "ai_risk_level": get_ai_risk_level(probability),
                "probability_percent": format_probability_percent(probability),
            }
        )
    return summaries


@lru_cache(maxsize=2)
def build_cached_dataset_case_summaries(data_path, modified_ns, file_size):
    patients = pd.read_csv(data_path)
    ai_predictions = build_ai_prediction_summaries(patients)
    summaries = []
    for index, (_, row) in enumerate(patients.iterrows()):
        patient = row.to_dict()
        icustay_id = str(int(patient["icustay_id"]))
        baseline_result = run_baseline(patient)
        ai_prediction = ai_predictions[index]
        data_quality = run_data_validator(patient)["data_quality"]
        confidence_label = get_confidence_label(ai_prediction.get("risk_probability"), data_quality)
        suggested_priority = suggest_review_priority(
            ai_prediction.get("ai_risk_level"),
            baseline_result.get("sirs_positive", False),
            confidence_label,
            data_quality,
        )
        summaries.append(
            {
                "patient": patient,
                "case_label": get_case_label(icustay_id),
                "subject_id": str(int(patient["subject_id"])),
                "hadm_id": str(int(patient["hadm_id"])),
                "icustay_id": icustay_id,
                "age": patient.get("age"),
                "gender": patient.get("gender"),
                "admission_type": patient.get("admission_type"),
                "ai_prediction_result": ai_prediction,
                "ai_risk_level": ai_prediction.get("ai_risk_level", "Unavailable"),
                "ai_probability": ai_prediction.get("risk_probability"),
                "ai_probability_percent": ai_prediction.get("probability_percent", "Not available"),
                "baseline_result": baseline_result,
                "sirs_screening": get_sirs_screening_label(baseline_result),
                "sirs_screening_key": get_sirs_screening_key(baseline_result),
                "sirs_criteria_met": baseline_result.get("sirs_criteria_met", 0),
                "data_quality": data_quality,
                "confidence_label": confidence_label,
                "suggested_review_priority": suggested_priority,
                "main_model_signal_text": "Open profile for model signals",
                "low_confidence_or_missing": is_low_confidence_or_missing(
                    ai_prediction.get("risk_probability"),
                    data_quality,
                ),
            }
        )
    return summaries


def build_review_item_map_by_icustay():
    items_by_icustay = {}
    for item in list_review_items({"status": "All"}):
        icustay_id = item.get("icustay_id")
        if not icustay_id:
            continue
        existing = items_by_icustay.get(str(icustay_id))
        if not existing or int(item.get("id", 0)) > int(existing.get("id", 0)):
            items_by_icustay[str(icustay_id)] = item
    return items_by_icustay


def build_case_summaries():
    stat = DATA_PATH.stat()
    base_summaries = build_cached_dataset_case_summaries(str(DATA_PATH), stat.st_mtime_ns, stat.st_size)
    queue_items = build_review_item_map_by_icustay()
    reviews = load_case_reviews()
    summaries = []
    for base_case in base_summaries:
        case = dict(base_case)
        queue_item = queue_items.get(case["icustay_id"])
        legacy_review = get_case_review(case["icustay_id"], reviews)
        case["review_ticket_id"] = queue_item.get("id") if queue_item else None
        case["review_status"] = queue_item.get("status") if queue_item else legacy_review["status"]
        case["priority"] = queue_item.get("priority") if queue_item else legacy_review["priority"]
        case["reviewer"] = queue_item.get("assigned_reviewer") if queue_item else legacy_review["reviewer"]
        case["updated_at"] = queue_item.get("updated_at") if queue_item else legacy_review["updated_at"]
        case["review_state"] = get_review_state_from_item(queue_item)
        case["recommended_action"] = build_recommended_action(
            case["ai_risk_level"],
            case["baseline_result"].get("sirs_positive", False),
            case["confidence_label"],
            case["data_quality"],
            case["review_status"],
            case["suggested_review_priority"],
            "Dataset",
        )
        summaries.append(case)
    return summaries


def count_by(items, key, expected=None):
    counts = {value: 0 for value in (expected or [])}
    for item in items:
        value = item.get(key) or "Unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def paginate_items(items, page, page_size):
    total = len(items)
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "has_previous": page > 1,
        "has_next": page < total_pages,
        "previous_page": page - 1,
        "next_page": page + 1,
        "start_index": start + 1 if total else 0,
        "end_index": min(end, total),
    }


def get_ai_system_status():
    metadata = load_json_file(AI_METADATA_PATH)
    metrics = load_json_file(AI_METRICS_PATH)
    model_name = metadata.get("model_name") or metrics.get("best_model") or "Not available"
    model_metrics = metrics.get("models", {}).get(model_name, {})
    train_rows = model_metrics.get("train_rows")
    test_rows = model_metrics.get("test_rows")
    training_rows = metadata.get("training_row_count")
    if training_rows is None and train_rows is not None and test_rows is not None:
        training_rows = train_rows + test_rows

    return {
        "model_status": "Trained" if AI_MODEL_PATH.exists() else "Missing",
        "model_name": model_name,
        "training_dataset_file": metadata.get("selected_dataset_path", "data/patient_features.csv"),
        "selected_dataset_filename": metadata.get("selected_dataset_filename", "patient_features.csv"),
        "training_rows": training_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "label_distribution": metadata.get("label_distribution", metrics.get("label_distribution", metrics.get("class_distribution", {}))),
        "model_metrics": model_metrics.get("metrics", {}),
        "target_label": metadata.get("target_label", "hospital_expire_flag"),
        "llm_assistant_mode": "Live LLM mode" if os.getenv("LLM_API_KEY") else "Offline fallback mode",
        "generative_ai_status": "configured" if os.getenv("LLM_API_KEY") else "not configured",
    }


def get_case_review(icustay_id, reviews):
    review = reviews.get(str(icustay_id), {})
    return {
        "status": review.get("status", review.get("review_status", "Not Reviewed")),
        "priority": review.get("priority", "Normal"),
        "reviewer": review.get("reviewer", ""),
        "note": review.get("note", review.get("reviewer_note", "")),
        "updated_at": review.get("updated_at", ""),
        "history": normalize_review_history(review),
    }


def clean_value(value):
    if value is None or str(value).lower() == "nan" or value != value:
        return None
    return value


def format_clinical_value(value):
    value = clean_value(value)
    if value is None:
        return "Missing"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def is_abnormal_measurement(name, value):
    value = clean_value(value)
    if value is None:
        return False

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return (
        (name in ["heart_rate_avg"] and number >= 100)
        or (name in ["heart_rate_max"] and number >= 130)
        or (name == "systolic_bp_avg" and number <= 90)
        or (name == "respiratory_rate_avg" and number >= 24)
        or (name == "temperature_avg" and number >= 38)
        or (name == "spo2_avg" and number <= 92)
        or (name == "creatinine_max" and number >= 2)
        or (name == "glucose_max" and number >= 200)
        or (name == "wbc_max" and number >= 15)
        or (name == "hemoglobin_min" and number <= 9)
    )


def get_patient_by_icustay_id(icustay_id):
    patients = load_patient_data()
    matches = patients[patients["icustay_id"].astype(str) == str(icustay_id)]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def get_case_label(icustay_id):
    return f"ICU Case {icustay_id}"


def get_top_risk_drivers(baseline_result, multi_agent_result, limit=3):
    factors = []
    for factor in multi_agent_result.get("contributing_factors", []):
        if factor not in factors:
            factors.append(factor)
    for factor in baseline_result.get("contributing_factors", []):
        if factor not in factors:
            factors.append(factor)

    return factors[:limit]


def format_risk_drivers(drivers):
    if not drivers:
        return "No major abnormal factors"
    return "; ".join(drivers)


def format_model_signals(top_features, limit=3):
    """Present explainability features as model signals, not medical rules."""
    labels = []
    for feature in top_features[:limit]:
        name = str(feature.get("display_name", "feature")).strip().lower()
        labels.append("Model signal: admission context" if "admission" in name else f"Model signal: {name} feature")
    return "; ".join(labels) or "No model signals available"


def get_review_focus_items(case):
    factor_text = " ".join(case["main_risk_drivers"]).lower()
    items = []

    if "spo2" in factor_text or "oxygen" in factor_text or "respiratory rate" in factor_text:
        items.append("Review oxygen/respiratory indicators.")
    if "systolic blood pressure" in factor_text or "blood pressure" in factor_text:
        items.append("Review blood-pressure indicators.")
    if "creatinine" in factor_text:
        items.append("Review kidney/lab-related indicators.")
    if "white blood cell" in factor_text or "wbc" in factor_text or "temperature" in factor_text:
        items.append("Review infection/inflammation-related indicators.")
    if "hemoglobin" in factor_text:
        items.append("Review blood/lab-related indicators.")
    if case["data_quality"] in ["Partial", "Poor"]:
        items.append("Check missing or incomplete clinical data.")

    items.append("Compare SIRS-style screening, modular workflow context, and AI risk before documenting review.")
    return items


def get_priority_rank(priority):
    return {"Urgent Review": 0, "Watch": 1, "Normal": 2}.get(priority, 3)


def get_dashboard_queue_rank(case):
    return (
        get_priority_rank(case["priority"]),
        0 if case["risk_level"] == "High" else 1,
        0 if case["review_status"] == "Needs Follow-up" else 1,
        0 if case["data_quality"] in ["Poor", "Partial"] else 1,
        int(case["icustay_id"]),
    )


def get_missing_ai_features(ai_explainability_result):
    return [
        feature
        for feature in ai_explainability_result.get("top_features", [])
        if feature.get("value") is None
    ]


def build_generated_review_summary(case):
    ai_prediction = case["ai_prediction_result"]
    ai_explainability = case["ai_explainability_result"]
    top_features = ai_explainability.get("top_features", [])
    missing_features = get_missing_ai_features(ai_explainability)

    if not ai_prediction.get("available"):
        return (
            "Template-based offline summary from local model outputs: the supervised AI model "
            "output is not available for this ICU stay. The reviewer can still "
            "use the SIRS-style baseline, modular workflow, measurements, "
            "and local review notes."
        )

    feature_names = ", ".join(
        feature["display_name"] for feature in top_features[:3]
    ) or "no feature-importance signals available"
    missing_text = (
        " Missing important values include "
        + ", ".join(feature["display_name"] for feature in missing_features)
        + "."
        if missing_features
        else " No missing values appeared among the top AI model signals."
    )

    return (
        "Template-based offline summary from local model outputs: the trained supervised model "
        f"estimated a {ai_prediction['risk_probability'] * 100:.1f}% risk "
        f"probability, mapped to {ai_prediction['ai_risk_level']} by the "
        "configured thresholds. The highest-weighted model signals shown for "
        f"this case are {feature_names}. The modular workflow data quality "
        f"assessment is {case['data_quality']}."
        f"{missing_text} The SIRS-style screening baseline is {case['sirs_screening']}, "
        f"while the modular workflow risk level is {case['risk_level']}."
    )


def build_case_assessment(patient, reviews, include_deep_analytics=False):
    baseline_result = run_baseline(patient)
    multi_agent_result = run_workflow(patient)
    ai_prediction_result = predict_ai_risk(patient)
    ai_explainability_result = explain_ai_prediction(patient)
    top_features = ai_explainability_result.get("top_features", [])
    icustay_id = str(int(patient["icustay_id"]))
    review = get_case_review(icustay_id, reviews)
    main_risk_drivers = get_top_risk_drivers(baseline_result, multi_agent_result)

    case = {
        "patient": patient,
        "case_label": get_case_label(icustay_id),
        "subject_id": str(int(patient["subject_id"])),
        "hadm_id": str(int(patient["hadm_id"])),
        "icustay_id": icustay_id,
        "age": patient.get("age"),
        "gender": patient.get("gender"),
        "admission_type": patient.get("admission_type"),
        "baseline_result": baseline_result,
        "multi_agent_result": multi_agent_result,
        "ai_prediction_result": ai_prediction_result,
        "ai_explainability_result": ai_explainability_result,
        "baseline_risk": baseline_result["risk_level"],
        "sirs_screening": get_sirs_screening_label(baseline_result),
        "sirs_screening_key": get_sirs_screening_key(baseline_result),
        "sirs_criteria_met": baseline_result.get("sirs_criteria_met", 0),
        "risk_level": multi_agent_result["final_risk_level"],
        "data_quality": multi_agent_result["data_quality"],
        "main_risk_drivers": main_risk_drivers,
        "main_risk_driver_text": format_risk_drivers(main_risk_drivers),
        "main_model_signal_text": format_model_signals(top_features),
        "review_status": review["status"],
        "priority": review["priority"],
        "reviewer": review["reviewer"],
        "reviewer_note": review["note"],
        "updated_at": review["updated_at"],
        "review_history": review["history"],
        "requires_attention": multi_agent_result["final_risk_level"] == "High"
        or multi_agent_result["data_quality"] in ["Partial", "Poor"]
        or multi_agent_result["verification_status"] == "Warning",
    }
    case["ai_missing_important_values"] = get_missing_ai_features(ai_explainability_result)
    if include_deep_analytics:
        case["uncertainty_result"] = estimate_prediction_uncertainty(
            ai_prediction_result,
            case["ai_missing_important_values"],
            case["data_quality"],
        )
        case["similar_cases_result"] = find_similar_cases(patient)
        case["cluster_result"] = assign_case_cluster(patient)
        case["anomaly_result"] = score_case_anomaly(patient)
        confidence_label = case["uncertainty_result"].get("confidence_label", "")
    else:
        case["uncertainty_result"] = {"available": False, "message": "Open the assessment page to compute uncertainty."}
        case["similar_cases_result"] = {"available": False, "message": "Open the assessment page to compute similar cases."}
        case["cluster_result"] = {"available": False, "message": "Open the assessment page to compute cluster context."}
        case["anomaly_result"] = {"available": False, "message": "Open the assessment page to compute anomaly context."}
        confidence_label = ""
    case["suggested_review_priority"] = suggest_review_priority(
        ai_prediction_result.get("ai_risk_level"),
        baseline_result.get("sirs_positive", False),
        confidence_label,
        case["data_quality"],
    )
    case["ai_generated_summary"] = build_generated_review_summary(case)
    return case


def build_case_assessments():
    patients = load_patient_data()
    reviews = load_case_reviews()
    return [build_case_assessment(row.to_dict(), reviews) for _, row in patients.iterrows()]


def get_measurement_rows(patient):
    rows = []
    for name, label, unit in CLINICAL_MEASUREMENTS:
        value = patient.get(name)
        rows.append(
            {
                "name": name,
                "label": label,
                "value": format_clinical_value(value),
                "unit": unit,
                "is_abnormal": is_abnormal_measurement(name, value),
            }
        )
    return rows


def summarize_note_timeline(notes):
    if not notes:
        return "No coordinator note has been added yet."
    latest = notes[0]
    note_text = (latest.get("note_text") or "").strip()
    reviewer = latest.get("reviewer_name") or "Coordinator"
    timestamp = latest.get("created_at") or "time not recorded"
    if len(note_text) > 320:
        note_text = note_text[:317].rstrip() + "..."
    return f"Latest note by {reviewer} at {timestamp}: {note_text}"


def format_missing_or_uncertain_text(item):
    missing_features = item.get("ai_missing_important_values") or []
    missing_names = [
        feature.get("display_name") or feature.get("feature")
        for feature in missing_features
        if isinstance(feature, dict) and (feature.get("display_name") or feature.get("feature"))
    ]
    confidence = item.get("confidence_label") or "Not available"
    data_quality = item.get("data_quality") or "Not available"
    if missing_names:
        return f"Data quality: {data_quality}; confidence: {confidence}; missing or imputed top AI signals: {', '.join(missing_names)}."
    return f"Data quality: {data_quality}; confidence: {confidence}; no missing top AI model signals are shown."


def build_prepared_review_summary(item, notes):
    case_reference = item.get("local_review_reference") or item.get("case_display") or f"Ticket #{item.get('id')}"
    case_identifier = item.get("icustay_id") or f"Manual Case #{item.get('manual_case_id', 'not available')}"
    review_outcome = item.get("review_outcome") or "Not selected"
    reason_flagged = (
        f"The case is queued because AI risk is {item.get('ai_risk_level') or 'not available'} "
        f"({item.get('ai_probability_percent', 'Not available')}), SIRS screening is "
        f"{item.get('sirs_screening', 'not available')}, and the current workflow priority is "
        f"{item.get('priority', 'not available')}."
    )
    return "\n".join(
        [
            f"Case reference: {case_reference}",
            f"Ticket: #{item.get('id')} | Source: {item.get('source_label', 'Not available')} | Case ID: {case_identifier}",
            f"Reason flagged: {reason_flagged}",
            f"AI risk and probability: {item.get('ai_risk_level') or 'Not available'} / {item.get('ai_probability_percent', 'Not available')}",
            f"SIRS screening: {item.get('sirs_screening', 'Not available')}",
            f"Main contributing signals: {item.get('main_model_signal_text') or 'Not available'}",
            f"Data quality / uncertainty notes: {format_missing_or_uncertain_text(item)}",
            f"Coordinator note summary: {summarize_note_timeline(notes)}",
            f"Current priority/status: {item.get('priority', 'Not available')} / {item.get('status', 'Not available')}",
            f"Review outcome: {review_outcome}",
            f"Recommended next review step: {item.get('recommended_action') or 'Open the ticket, document review context, and update status.'}",
        ]
    )


def build_ticket_assistant_outputs(item, prepared_review_summary):
    missing_text = format_missing_or_uncertain_text(item)
    return {
        "summary": prepared_review_summary,
        "flagged": (
            f"This ticket is prioritized because AI risk is {item.get('ai_risk_level') or 'not available'} "
            f"with probability {item.get('ai_probability_percent', 'Not available')}, SIRS screening is "
            f"{item.get('sirs_screening', 'not available')}, and current data quality/confidence is "
            f"{item.get('data_quality') or 'not available'} / {item.get('confidence_label') or 'not available'}."
        ),
        "missing": missing_text,
        "note": (
            f"Coordinator note draft: reviewed {item.get('case_display', 'this case')} for AI-assisted quality-review priority. "
            f"AI risk: {item.get('ai_risk_level') or 'not available'} ({item.get('ai_probability_percent', 'Not available')}); "
            f"SIRS screening: {item.get('sirs_screening', 'not available')}; data quality/confidence: "
            f"{item.get('data_quality') or 'not available'} / {item.get('confidence_label') or 'not available'}. "
            "Document local findings and update status based on review workflow."
        ),
        "compare": (
            f"AI risk is {item.get('ai_risk_level') or 'not available'} at {item.get('ai_probability_percent', 'Not available')}; "
            f"SIRS-style screening is {item.get('sirs_screening', 'not available')}. Use the comparison to prioritize review documentation and follow-up."
        ),
    }


def normalize_review_outcome(review_outcome):
    return review_outcome if review_outcome in REVIEW_OUTCOMES else ""


def is_summary_prepared(item, explicit_summary_prepared=False):
    return explicit_summary_prepared or item.get("status") in {
        "Ready for Quality Team Review",
        "Reviewed",
        "Archived",
    }


def build_next_step_guidance(item, summary_prepared=False):
    status = item.get("status")
    if status == "Ready for Quality Team Review":
        return "This ticket is prepared for the Quality & Patient Safety Review Team. After team review, mark it Reviewed or move it back to Needs Follow-up."
    if status == "Reviewed":
        return "Review is complete. Archive the ticket when documentation is no longer active."
    if status == "Archived":
        return "This ticket is archived and removed from the active Review Queue. It is kept for audit/history."
    if summary_prepared:
        return "Review summary prepared. Next step: mark the ticket Ready for Quality Team Review."
    return {
        "New": "Start by checking the AI/SIRS signals, then mark the ticket In Review.",
        "In Review": "Add a Coordinator Note and prepare the review summary.",
        "Needs Follow-up": "Document the missing follow-up information or assign ownership before preparing the handoff.",
    }.get(status, "Open the Review Ticket, check signals, add a Coordinator Note, and prepare the handoff summary.")


def build_ticket_stepper(item, notes, summary_prepared=False):
    status = item.get("status")
    step_1_completed = status in {"In Review", "Needs Follow-up", "Ready for Quality Team Review", "Reviewed", "Archived"}
    step_2_completed = bool(notes)
    step_3_completed = is_summary_prepared(item, summary_prepared)
    step_4_completed = status in {"Ready for Quality Team Review", "Reviewed", "Archived"}
    steps = [
        {"number": 1, "label": "Check Signals", "completed": step_1_completed},
        {"number": 2, "label": "Add Note", "completed": step_2_completed},
        {"number": 3, "label": "Prepare Summary", "completed": step_3_completed},
        {"number": 4, "label": "Mark Ready for Team", "completed": step_4_completed},
    ]
    current_assigned = False
    for step in steps:
        if step["completed"]:
            step["state"] = "completed"
        elif not current_assigned:
            step["state"] = "active"
            current_assigned = True
        else:
            step["state"] = "pending"
    if all(step["completed"] for step in steps):
        steps[-1]["state"] = "completed"
    return steps


def build_ticket_lifecycle(item):
    status = item.get("status")
    stages = [
        {"label": "New", "matches": {"New"}},
        {"label": "In Review", "matches": {"In Review"}},
        {
            "label": "Needs Follow-up / Ready for Quality Team Review",
            "matches": {"Needs Follow-up", "Ready for Quality Team Review"},
        },
        {"label": "Reviewed", "matches": {"Reviewed", "Closed"}},
        {"label": "Archived", "matches": {"Archived"}},
    ]
    for stage in stages:
        stage["active"] = status in stage["matches"]
    return stages


def primary_ticket_action(item, notes, summary_prepared=False):
    status = item.get("status")
    if status == "Ready for Quality Team Review":
        return "mark_reviewed"
    if status == "Reviewed":
        return "archive"
    if status == "Archived":
        return ""
    if summary_prepared:
        return "mark_ready"
    if status == "New":
        return "mark_in_review"
    if status == "In Review":
        if not notes:
            return "add_note"
        if not is_summary_prepared(item, summary_prepared):
            return "prepare_summary"
        return "mark_ready"
    if status == "Needs Follow-up":
        return "add_note"
    return ""


def build_case_ai_workflow_trace(case, summary_prepared=False):
    return [
        {"label": "Data validation checked", "detail": f"Data quality: {case.get('data_quality', 'Not available')}."},
        {
            "label": "ML risk prediction generated",
            "detail": f"AI risk: {case.get('ai_prediction_result', {}).get('ai_risk_level', 'Not available')} / {format_probability_percent(case.get('ai_prediction_result', {}).get('risk_probability'))}.",
        },
        {"label": "SIRS-style screening calculated", "detail": f"{case.get('sirs_screening', 'Not available')} ({case.get('sirs_criteria_met', 0)} of 4)."},
        {"label": "Main model signals extracted", "detail": case.get("main_model_signal_text", "Not available")},
        {
            "label": "Similar cases retrieved if available",
            "detail": "Available." if case.get("similar_cases_result", {}).get("available") else "Not available for this view.",
        },
        {
            "label": "Cluster/anomaly context checked if available",
            "detail": "Available." if case.get("cluster_result", {}).get("available") or case.get("anomaly_result", {}).get("available") else "Not available for this view.",
        },
        {
            "label": "Uncertainty estimated if available",
            "detail": case.get("uncertainty_result", {}).get("confidence_label", "Not available"),
        },
        {"label": "Coordinator action recommended", "detail": case.get("recommended_action", "Not available")},
        {
            "label": "Review summary prepared if available",
            "detail": "Prepared." if summary_prepared else "Available after opening or creating a Review Ticket.",
        },
    ]


def build_ticket_ai_workflow_trace(item, summary_prepared=False):
    return [
        {"label": "Data validation checked", "detail": f"Data quality: {item.get('data_quality') or 'Not available'}."},
        {
            "label": "ML risk prediction generated",
            "detail": f"AI risk: {item.get('ai_risk_level') or 'Not available'} / {item.get('ai_probability_percent', 'Not available')}.",
        },
        {"label": "SIRS-style screening calculated", "detail": item.get("sirs_screening", "Not available")},
        {"label": "Main model signals extracted", "detail": item.get("main_model_signal_text", "Not available")},
        {"label": "Similar cases retrieved if available", "detail": "Open the Case Safety Profile for this optional context."},
        {"label": "Cluster/anomaly context checked if available", "detail": "Open the Case Safety Profile for this optional context."},
        {"label": "Uncertainty estimated if available", "detail": item.get("confidence_label", "Not available")},
        {"label": "Coordinator action recommended", "detail": item.get("recommended_action", "Not available")},
        {
            "label": "Review summary prepared if available",
            "detail": "Prepared." if is_summary_prepared(item, summary_prepared) else "Use Generate Review Summary when ready.",
        },
    ]


def enrich_review_item(item):
    item = dict(item)
    item["review_outcome"] = item.get("review_outcome", "")
    item["source_label"] = source_label(item.get("source_type"))
    if item.get("icustay_id"):
        item["case_display"] = f"ICU Case {item['icustay_id']}"
        patient = get_patient_by_icustay_id(item["icustay_id"])
    else:
        item["case_display"] = f"Manual Review Case #{item.get('manual_case_id')}"
        patient = get_manual_case(item["manual_case_id"]) if item.get("manual_case_id") else None
    item["local_review_reference"] = item.get("local_review_reference") or (patient or {}).get("local_review_reference", "")

    if patient:
        baseline_result = run_baseline(patient)
        item["sirs_screening"] = get_sirs_screening_label(baseline_result)
        item["sirs_screening_key"] = get_sirs_screening_key(baseline_result)
        item["sirs_criteria_met"] = baseline_result.get("sirs_criteria_met", 0)
        explainability = explain_ai_prediction(patient)
        item["main_model_signal_text"] = format_model_signals(explainability.get("top_features", []), limit=2)
        item["ai_missing_important_values"] = get_missing_ai_features(explainability)
    else:
        baseline_result = {}
        item["sirs_screening"] = "Not available"
        item["sirs_screening_key"] = "unknown"
        item["sirs_criteria_met"] = None
        item["main_model_signal_text"] = "Not available"
        item["ai_missing_important_values"] = []

    item["ai_probability_percent"] = format_probability_percent(item.get("ai_probability"))
    item["confidence_label"] = get_confidence_label(item.get("ai_probability"), item.get("data_quality"))
    item["low_confidence_or_missing"] = is_low_confidence_or_missing(item.get("ai_probability"), item.get("data_quality"))
    item["recommended_action"] = build_recommended_action(
        item.get("ai_risk_level"),
        baseline_result.get("sirs_positive", False),
        item["confidence_label"],
        item.get("data_quality", ""),
        item.get("status", ""),
        item.get("priority", ""),
        item["source_label"],
    )
    item["case_profile_url"] = url_for("icu_case_detail", icustay_id=item["icustay_id"]) if item.get("icustay_id") else url_for("review_item_detail", item_id=item["id"])
    item["review_ticket_url"] = url_for("review_item_detail", item_id=item["id"])
    return item


def enrich_review_items(items):
    return sorted((enrich_review_item(item) for item in items), key=triage_score)


def build_dashboard_context():
    case_summaries = build_case_summaries()
    workflow_stats = get_workflow_stats()
    active_items = enrich_review_items(list_review_items({"status": "__active__"}))
    if active_items:
        top_attention_items = active_items[:8]
    else:
        top_attention_items = []
        for case in sorted(case_summaries, key=triage_score)[:8]:
            candidate = dict(case)
            candidate["source_label"] = "Dataset"
            candidate["case_display"] = case["case_label"]
            candidate["status"] = case["review_state"]
            candidate["priority"] = case["suggested_review_priority"]
            candidate["case_profile_url"] = url_for("icu_case_detail", icustay_id=case["icustay_id"])
            candidate["review_ticket_url"] = None
            top_attention_items.append(candidate)
    risk_distribution = count_by(
        case_summaries,
        "ai_risk_level",
        expected=["High", "Medium", "Low", "Unavailable"],
    )

    return {
        "total_cases": len(case_summaries),
        "active_review_items": len(active_items),
        "high_ai_risk_cases": sum(1 for case in case_summaries if case["ai_risk_level"] == "High"),
        "sirs_positive_cases": sum(1 for case in case_summaries if case["sirs_screening_key"] == "positive"),
        "needs_follow_up": workflow_stats["status_counts"].get("Needs Follow-up", 0),
        "low_confidence_or_missing_cases": sum(1 for case in case_summaries if case["low_confidence_or_missing"]),
        "risk_distribution": risk_distribution,
        "review_status_distribution": workflow_stats["status_counts"],
        "recent_activity": get_audit_events(limit=8),
        "attention_items": top_attention_items,
        "active_ticket_items": active_items[:8],
        "workflow_stats": workflow_stats,
        "ai_system_status": get_ai_system_status(),
    }


@app.route("/")
def home():
    return render_template("software_home.html", title="Dashboard", **build_dashboard_context())


@app.route("/software")
def software_home():
    return render_template("software_home.html", title="Dashboard", **build_dashboard_context())


@app.route("/icu-dashboard")
def icu_dashboard():
    return render_template("software_home.html", title="Dashboard", **build_dashboard_context())


@app.route("/start-next-review")
def start_next_review():
    active_items = sorted(
        enrich_review_items(list_review_items({"status": "__active__"})),
        key=start_next_review_score,
    )
    if active_items:
        return redirect(url_for("review_item_detail", item_id=active_items[0]["id"], started="1"))

    try:
        case_summaries = build_case_summaries()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        case_summaries = []
    if case_summaries:
        top_case = sorted(case_summaries, key=start_next_review_score)[0]
        return redirect(url_for("icu_case_detail", icustay_id=top_case["icustay_id"], start_next="create_ticket"))
    return redirect(url_for("review_queue", start_next="empty"))


@app.route("/icu-cases")
def icu_cases():
    all_cases = build_case_summaries()
    cases = all_cases
    ai_risk_filter = request.args.get("ai_risk_level", "")
    sirs_filter = request.args.get("sirs_screening", "")
    quality_filter = request.args.get("data_quality", "")
    status_filter = request.args.get("review_status", "")
    priority_filter = request.args.get("priority", "")
    admission_filter = request.args.get("admission_type", "")
    search = request.args.get("search", "").strip().lower()
    page_size = safe_int(request.args.get("page_size"), DEFAULT_CASE_PAGE_SIZE)
    if page_size not in CASE_PAGE_SIZE_OPTIONS:
        page_size = DEFAULT_CASE_PAGE_SIZE
    page = safe_int(request.args.get("page"), 1)

    if ai_risk_filter:
        cases = [case for case in cases if case["ai_risk_level"] == ai_risk_filter]
    if sirs_filter:
        cases = [case for case in cases if case["sirs_screening_key"] == sirs_filter]
    if quality_filter:
        cases = [case for case in cases if case["data_quality"] == quality_filter]
    if status_filter:
        cases = [case for case in cases if case["review_status"] == status_filter]
    if priority_filter:
        cases = [case for case in cases if case["priority"] == priority_filter]
    if admission_filter:
        cases = [case for case in cases if case["admission_type"] == admission_filter]
    if search:
        cases = [
            case
            for case in cases
            if search in case["icustay_id"].lower()
            or search in case["subject_id"].lower()
            or search in case["hadm_id"].lower()
        ]

    cases = sorted(
        cases,
        key=lambda case: (
            0 if case["ai_risk_level"] == "High" else 1 if case["ai_risk_level"] == "Medium" else 2,
            0 if case["sirs_screening_key"] == "positive" else 1,
            int(case["icustay_id"]),
        ),
    )
    pagination = paginate_items(cases, page, page_size)
    pagination_args = request.args.to_dict()
    page_link_args = {key: value for key, value in pagination_args.items() if key != "page"}
    pagination["previous_url"] = url_for("icu_cases", **{**page_link_args, "page": pagination["previous_page"]})
    pagination["next_url"] = url_for("icu_cases", **{**page_link_args, "page": pagination["next_page"]})

    return render_template(
        "icu_cases.html",
        title="Case Library",
        cases=pagination["items"],
        pagination=pagination,
        page_size_options=CASE_PAGE_SIZE_OPTIONS,
        pagination_args=pagination_args,
        review_statuses=REVIEW_STATUSES,
        priorities=REVIEW_PRIORITIES,
        admission_types=sorted(
            {
                str(case["admission_type"])
                for case in all_cases
                if clean_value(case["admission_type"]) is not None
            }
        ),
        filters={
            "ai_risk_level": ai_risk_filter,
            "sirs_screening": sirs_filter,
            "data_quality": quality_filter,
            "review_status": status_filter,
            "priority": priority_filter,
            "admission_type": admission_filter,
            "search": search,
            "page_size": page_size,
        },
    )


@app.route("/user-guide")
def user_guide():
    return render_template("user_guide.html", title="User Guide")



@app.route("/icu-cases/<icustay_id>/add-to-review-queue", methods=["POST"])
def add_dataset_case_to_review_queue(icustay_id):
    patient = get_patient_by_icustay_id(icustay_id)
    if patient is None:
        return render_template("icu_case_not_found.html", title="ICU Stay Not Found", icustay_id=icustay_id), 404
    case = build_case_assessment(patient, load_case_reviews())
    item, created = create_dataset_review_item(
        case,
        request.form.get("assigned_reviewer", ""),
        case.get("suggested_review_priority"),
    )
    if request.form.get("next") == "ticket":
        return redirect(url_for("review_item_detail", item_id=item.get("id")))
    return redirect(url_for("review_queue", created="1" if created else "0", existing_id=item.get("id")))


@app.route("/review-queue")
def review_queue():
    status_filter = request.args.get("status")
    filters = {
        "status": "__active__" if status_filter is None else status_filter,
        "priority": request.args.get("priority", ""),
        "ai_risk_level": request.args.get("ai_risk_level", ""),
        "source_type": request.args.get("source_type", ""),
        "assigned_reviewer": request.args.get("assigned_reviewer", ""),
        "sirs_screening": request.args.get("sirs_screening", ""),
        "search": request.args.get("search", ""),
    }
    db_filters = {key: value for key, value in filters.items() if key != "sirs_screening"}
    items = enrich_review_items(list_review_items(db_filters))
    if filters["sirs_screening"]:
        items = [item for item in items if item["sirs_screening_key"] == filters["sirs_screening"]]
    coordinator_items = list_review_items({"status": filters["status"]})
    coordinator_options = sorted(
        {
            item.get("assigned_reviewer", "").strip()
            for item in coordinator_items
            if item.get("assigned_reviewer", "").strip()
        }
    )
    return render_template(
        "review_queue.html",
        title="Review Queue",
        items=items,
        filters=filters,
        statuses=WORKFLOW_STATUSES,
        priorities=WORKFLOW_PRIORITIES,
        workflow_stats=get_workflow_stats(),
        coordinator_options=coordinator_options,
        audit_events=get_audit_events(limit=15),
        created=request.args.get("created"),
        existing_id=request.args.get("existing_id"),
        start_next=request.args.get("start_next"),
    )


@app.route("/review-item/<int:item_id>")
def review_item_detail(item_id):
    item = get_review_item(item_id)
    if not item:
        return redirect(url_for("review_queue"))
    item = enrich_review_item(item)
    manual_case = get_manual_case(item["manual_case_id"]) if item.get("manual_case_id") else None
    measurement_rows = get_measurement_rows(manual_case) if manual_case else []
    notes = get_review_notes(item_id)
    prepared_review_summary = build_prepared_review_summary(item, notes)
    summary_prepared = request.args.get("summary") == "prepared"
    return render_template(
        "review_item_detail.html",
        title="Review Ticket",
        item=item,
        manual_case=manual_case,
        measurement_rows=measurement_rows,
        notes=notes,
        audit_events=get_audit_events("review_item", item_id),
        statuses=editable_workflow_statuses(item),
        priorities=WORKFLOW_PRIORITIES,
        prepared_review_summary=prepared_review_summary,
        ticket_ai_outputs=build_ticket_assistant_outputs(item, prepared_review_summary),
        next_step_guidance=build_next_step_guidance(item, summary_prepared),
        ticket_steps=build_ticket_stepper(item, notes, summary_prepared),
        ticket_lifecycle=build_ticket_lifecycle(item),
        primary_action=primary_ticket_action(item, notes, summary_prepared),
        summary_prepared=summary_prepared,
        review_outcomes=REVIEW_OUTCOMES,
        ai_workflow_trace=build_ticket_ai_workflow_trace(item, summary_prepared),
        ai_system_status=get_ai_system_status(),
        started=request.args.get("started") == "1",
    )


@app.route("/review-item/<int:item_id>/edit", methods=["POST"])
def review_item_edit(item_id):
    item = get_review_item(item_id)
    if not item:
        return redirect(url_for("review_queue"))
    status = request.form.get("status", "New")
    review_outcome = normalize_review_outcome(request.form.get("review_outcome", ""))
    destination = local_redirect_target(request.form.get("return_to"), url_for("review_item_detail", item_id=item_id))
    if status == "Archived" and item.get("status") != "Archived":
        if not can_archive_review_item(item):
            return redirect(add_query_params(destination, msg="archive_requires_reviewed"))
        archive_review_item(item_id)
        return redirect(add_query_params(destination, msg="ticket_archived"))
    update_review_item(
        item_id,
        status,
        request.form.get("priority", "Low"),
        request.form.get("assigned_reviewer", ""),
        review_outcome,
    )
    return redirect(add_query_params(destination, msg=status_message_code(status)))


@app.route("/review-item/<int:item_id>/quick-action", methods=["POST"])
def review_item_quick_action(item_id):
    item = get_review_item(item_id)
    if not item:
        return redirect(url_for("review_queue"))
    action = request.form.get("action", "")
    message_code = "ticket_updated"
    if action == "archive":
        if can_archive_review_item(item):
            archive_review_item(item_id)
            message_code = "ticket_archived"
        else:
            message_code = "archive_requires_reviewed"
    else:
        status_map = {
            "mark_in_review": "In Review",
            "mark_follow_up": "Needs Follow-up",
            "mark_ready_quality_team": "Ready for Quality Team Review",
            "mark_reviewed": "Reviewed",
        }
        next_status = status_map.get(action, item.get("status", "New"))
        message_code = status_message_code(next_status)
        update_review_item(
            item_id,
            next_status,
            item.get("priority", "Low"),
            item.get("assigned_reviewer", ""),
        )
    fallback = url_for("review_queue") if action == "archive" and message_code == "ticket_archived" else url_for("review_item_detail", item_id=item_id)
    destination = local_redirect_target(request.form.get("return_to"), fallback)
    return redirect(add_query_params(destination, msg=message_code))


@app.route("/review-item/<int:item_id>/note", methods=["POST"])
def review_item_note(item_id):
    note_text = request.form.get("note_text", "").strip()
    if note_text:
        add_review_note(item_id, request.form.get("reviewer_name", ""), note_text)
        message_code = "note_saved"
    else:
        message_code = "ticket_updated"
    destination = local_redirect_target(request.form.get("return_to"), url_for("review_item_detail", item_id=item_id))
    return redirect(add_query_params(destination, msg=message_code))


@app.route("/review-item/<int:item_id>/archive", methods=["POST"])
def review_item_archive(item_id):
    item = get_review_item(item_id)
    if not item:
        return redirect(url_for("review_queue"))
    if not can_archive_review_item(item):
        return redirect(url_for("review_item_detail", item_id=item_id, msg="archive_requires_reviewed"))
    archive_review_item(item_id)
    return redirect(url_for("review_queue", msg="ticket_archived"))


@app.route("/review-item/<int:item_id>/delete", methods=["POST"])
def review_item_delete(item_id):
    item = get_review_item(item_id)
    if not item:
        return redirect(url_for("review_queue"))
    if not can_archive_review_item(item):
        return redirect(url_for("review_item_detail", item_id=item_id, msg="archive_requires_reviewed"))
    delete_review_item(item_id)
    return redirect(url_for("review_queue", msg="ticket_archived"))


MANUAL_RANGE_ERROR = "The value is outside the supported manual-entry range. Please check units or correct the entry."
NUMERIC_CASE_FIELDS = [
    ("age", "Age", 0, 120),
    ("heart_rate_avg", "Heart rate avg", 20, 250),
    ("heart_rate_max", "Heart rate max", 20, 300),
    ("systolic_bp_avg", "Systolic BP avg", 40, 250),
    ("respiratory_rate_avg", "Respiratory rate avg", 1, 80),
    ("temperature_avg", "Temperature avg", 25, 45),
    ("spo2_avg", "SpO2 avg", 0, 100),
    ("creatinine_max", "Creatinine max", 0, 20),
    ("glucose_max", "Glucose max", 10, 1000),
    ("wbc_max", "WBC max", 0, 300),
    ("hemoglobin_min", "Hemoglobin min", 1, 25),
]


def parse_manual_case_form(form):
    """Parse the Create Review Ticket form.

    Returns (values, raw_values, errors). Empty numeric fields are allowed
    (None, handled by the model pipeline's own imputation). Non-empty values
    that are not valid numbers are collected in `errors` instead of raising,
    so the route can safely re-render the form with the user's input intact.
    """
    raw_values = {
        "local_review_reference": form.get("local_review_reference", "").strip(),
        "gender": form.get("gender", ""),
        "admission_type": form.get("admission_type", ""),
    }
    values = {
        "local_review_reference": raw_values["local_review_reference"],
        "gender": raw_values["gender"],
        "admission_type": raw_values["admission_type"],
    }
    errors = {}

    for field, label, minimum, maximum in NUMERIC_CASE_FIELDS:
        raw = form.get(field, "").strip()
        raw_values[field] = raw
        if not raw:
            values[field] = None
            continue
        try:
            values[field] = float(raw)
        except ValueError:
            errors[field] = f"{label} must be a valid number."
            continue
        if values[field] < minimum or values[field] > maximum:
            errors[field] = f"{label}: {MANUAL_RANGE_ERROR}"

    return values, raw_values, errors


@app.route("/new-case", methods=["GET", "POST"])
def new_case():
    result = None
    form_values = {}
    errors = {}
    if request.method == "POST":
        case_values, form_values, errors = parse_manual_case_form(request.form)
        if not errors:
            ai_prediction = predict_ai_risk(case_values)
            explainability = explain_ai_prediction(case_values)
            missing = get_missing_ai_features(explainability)
            uncertainty = estimate_prediction_uncertainty(ai_prediction, missing, "Manual")
            baseline_result = run_baseline(case_values)
            suggested_priority = suggest_review_priority(
                ai_prediction.get("ai_risk_level"),
                baseline_result.get("sirs_positive", False),
                uncertainty.get("confidence_label", ""),
                "Manual",
            )
            similar = find_similar_cases(case_values)
            cluster = assign_case_cluster(case_values)
            anomaly = score_case_anomaly(case_values)
            manual_case = create_manual_case(case_values, ai_prediction)
            review_item = create_manual_review_item(manual_case, suggested_priority=suggested_priority)
            result = {
                "case_values": case_values,
                "ai_prediction": ai_prediction,
                "uncertainty": uncertainty,
                "baseline_result": baseline_result,
                "sirs_screening": get_sirs_screening_label(baseline_result),
                "sirs_screening_key": get_sirs_screening_key(baseline_result),
                "suggested_review_priority": suggested_priority,
                "recommended_action": build_recommended_action(
                    ai_prediction.get("ai_risk_level"),
                    baseline_result.get("sirs_positive", False),
                    uncertainty.get("confidence_label", ""),
                    "Manual",
                    "New",
                    suggested_priority,
                    "Manual",
                ),
                "main_model_signal_text": format_model_signals(explainability.get("top_features", []), limit=2),
                "similar_cases": similar,
                "cluster": cluster,
                "anomaly": anomaly,
                "manual_case": manual_case,
                "review_item": review_item,
            }
    return render_template("new_case.html", title="Create Review Ticket", result=result, form_values=form_values, errors=errors)

@app.route("/ai-evaluation")
@app.route("/ai-risk-engine")
def ai_evaluation():
    metadata = load_json_file(AI_METADATA_PATH)
    metrics = load_json_file(AI_METRICS_PATH)
    active_dataset = get_active_feature_dataset_info()
    model_name = metadata.get("model_name") or metrics.get("best_model") or "Not available"
    model_record = metrics.get("models", {}).get(model_name, {})
    return render_template(
        "ai_evaluation.html",
        title="AI Risk Engine",
        ai_system_status=get_ai_system_status(),
        metadata=metadata,
        metrics=metrics,
        model_name=model_name,
        model_metrics=model_record.get("metrics", {}),
        tuned_metrics=model_record.get("tuned_threshold_metrics"),
        active_dataset=active_dataset,
        input_features=metadata.get("input_features", AI_INPUT_FEATURES),
        excluded_columns=metadata.get("excluded_columns", sorted(AI_EXCLUDED_COLUMNS)),
    )

@app.route("/testing-validation")
def testing_validation():
    return render_template(
        "testing_validation.html",
        title="Testing & Validation",
        feature_availability=load_feature_availability(),
        evaluation_results=load_evaluation_results_file(),
        review_storage_path="data/case_reviews.json",
        active_feature_dataset=get_active_feature_dataset_info(),
    )


@app.route("/software-system")
def software_system():
    return render_template("software_system.html", title="Software System")


@app.route("/icu-cases/<icustay_id>", methods=["GET", "POST"])
def icu_case_detail(icustay_id):
    patient = get_patient_by_icustay_id(icustay_id)
    if patient is None:
        return render_template("icu_case_not_found.html", title="ICU Stay Not Found", icustay_id=icustay_id), 404

    reviews = load_case_reviews()
    return_to = request.values.get("return_to", "")
    if not return_to.startswith("/icu-cases"):
        return_to = ""

    if request.method == "POST":
        review_status = request.form.get("status", "Not Reviewed")
        if review_status not in REVIEW_STATUSES:
            review_status = "Not Reviewed"
        priority = request.form.get("priority", "Normal")
        if priority not in REVIEW_PRIORITIES:
            priority = "Normal"

        updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        review_entry = {
            "status": review_status,
            "priority": priority,
            "reviewer": request.form.get("reviewer", "").strip(),
            "note": request.form.get("note", "").strip(),
            "updated_at": updated_at,
        }
        existing_review = reviews.get(str(icustay_id), {})
        history = normalize_review_history(existing_review)
        history.append(review_entry.copy())
        reviews[str(icustay_id)] = {**review_entry, "history": history}
        save_case_reviews(reviews)
        redirect_args = {"icustay_id": icustay_id, "saved": "1"}
        if return_to:
            redirect_args["return_to"] = return_to
        return redirect(url_for("icu_case_detail", **redirect_args))

    case = build_case_assessment(patient, reviews, include_deep_analytics=True)
    linked_review_item = get_active_review_item_for_icustay(icustay_id)
    if linked_review_item:
        linked_review_item = enrich_review_item(linked_review_item)
        case["review_ticket_id"] = linked_review_item["id"]
        case["review_status"] = linked_review_item["status"]
        case["priority"] = linked_review_item["priority"]
        case["reviewer"] = linked_review_item.get("assigned_reviewer", "")
        case["recommended_action"] = build_recommended_action(
            case["ai_prediction_result"].get("ai_risk_level"),
            case["baseline_result"].get("sirs_positive", False),
            case["uncertainty_result"].get("confidence_label", ""),
            case["data_quality"],
            linked_review_item["status"],
            linked_review_item["priority"],
            "Dataset",
        )
    case["why_flagged_summary"] = build_why_flagged_summary(
        case["ai_prediction_result"],
        case["main_model_signal_text"],
        case["baseline_result"],
        case["uncertainty_result"],
        case["anomaly_result"],
        case["data_quality"],
    )
    back_to_worklist_url = return_to or url_for("icu_cases")
    return render_template(
        "icu_case_detail.html",
        title="Case Safety Profile",
        case=case,
        patient=patient,
        measurement_rows=get_measurement_rows(patient),
        review_statuses=REVIEW_STATUSES,
        priorities=REVIEW_PRIORITIES,
        focus_items=get_review_focus_items(case),
        saved=request.args.get("saved") == "1",
        return_to=return_to,
        back_to_worklist_url=back_to_worklist_url,
        ai_system_status=get_ai_system_status(),
        linked_review_item=linked_review_item,
        ai_workflow_trace=build_case_ai_workflow_trace(case),
    )


@app.route("/icu-case/<icustay_id>/ai-assistant", methods=["POST"])
def icu_case_ai_assistant(icustay_id):
    patient = get_patient_by_icustay_id(icustay_id)
    if patient is None:
        return jsonify({"answer": "ICU stay was not found."}), 404

    case = build_case_assessment(patient, load_case_reviews(), include_deep_analytics=True)
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        question = payload.get("question", "")
    else:
        question = request.form.get("question", "")

    if question.strip().lower() in {"generate case summary", "generate review summary"}:
        answer = generate_case_summary(
            case,
            case["ai_prediction_result"],
            case["ai_explainability_result"],
            case["data_quality"],
            case["baseline_result"],
            similar_cases=case.get("similar_cases_result"),
            cluster_result=case.get("cluster_result"),
            anomaly_result=case.get("anomaly_result"),
            uncertainty_result=case.get("uncertainty_result"),
            model_metadata=get_ai_system_status(),
        )
    else:
        answer = answer_case_question(
            question,
            case,
            case["ai_prediction_result"],
            case["ai_explainability_result"],
            case["data_quality"],
            case["baseline_result"],
            similar_cases=case.get("similar_cases_result"),
            cluster_result=case.get("cluster_result"),
            anomaly_result=case.get("anomaly_result"),
            uncertainty_result=case.get("uncertainty_result"),
            model_metadata=get_ai_system_status(),
        )

    if request.is_json:
        return jsonify({"answer": answer})
    return answer


@app.route("/research-question")
@app.route("/articles")
@app.route("/dataset")
@app.route("/system-architecture")
@app.route("/baseline-pipeline")
@app.route("/multi-agent-workflow")
@app.route("/ai-methodology")
@app.route("/stakeholders")
def legacy_documentation_redirect():
    return redirect(url_for("software_home"))

@app.route("/demo")
def demo():
    patients = load_patient_data()
    icustay_ids = patients["icustay_id"].astype(str).tolist()

    selected_id = request.args.get("icustay_id", icustay_ids[0] if icustay_ids else "")
    from_case = request.args.get("from_case", "")
    if from_case not in icustay_ids:
        from_case = ""
    selected_patient = None
    baseline_result = None
    multi_agent_result = None

    if selected_id:
        matches = patients[patients["icustay_id"].astype(str) == selected_id]
        if not matches.empty:
            selected_patient = matches.iloc[0].to_dict()
            baseline_result = run_baseline(selected_patient)
            multi_agent_result = run_workflow(selected_patient)

    return render_template(
        "demo.html",
        title="Demo",
        icustay_ids=icustay_ids,
        selected_id=selected_id,
        selected_patient=selected_patient,
        baseline_result=baseline_result,
        multi_agent_result=multi_agent_result,
        from_case=from_case,
    )


@app.route("/results")
def results():
    evaluation_result = load_evaluation_results_file() or run_evaluation()
    return render_template(
        "results.html",
        title="Model Validation",
        evaluation_result=evaluation_result,
        ai_system_status=get_ai_system_status(),
        model_metrics=get_ai_system_status().get("model_metrics", {}),
    )


@app.route("/technologies")
def technologies():
    return render_template("technologies.html", title="Technologies")


@app.route("/team")
def team():
    return render_template("team.html", title="Team")


if __name__ == "__main__":
    app.run(debug=True)















