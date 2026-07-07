"""Flask app for the ICU risk workflow website."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

from agents.ai_prediction_agent import EXCLUDED_COLUMNS as AI_EXCLUDED_COLUMNS
from agents.ai_prediction_agent import INPUT_FEATURES as AI_INPUT_FEATURES
from agents.ai_prediction_agent import predict_ai_risk
from agents.anomaly_agent import score_case_anomaly
from agents.cluster_agent import assign_case_cluster
from agents.explainable_ai_agent import explain_ai_prediction
from agents.similar_cases_agent import find_similar_cases
from agents.uncertainty_agent import estimate_prediction_uncertainty
from agents.generative_ai_agent import answer_case_question, generate_case_summary
from agents.multi_agent_workflow import run_workflow
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

REVIEW_STATUSES = ["Not Reviewed", "Reviewed", "Needs Follow-up"]
REVIEW_PRIORITIES = ["Normal", "Watch", "Urgent Review"]

NAV_SECTIONS = [
    {
        "label": "AI-Assisted ICU Quality Review System",
        "pages": [
            {"endpoint": "software_home", "label": "Home"},
            {"endpoint": "review_queue", "label": "Quality Review Queue"},
            {"endpoint": "new_case", "label": "Add New Case"},
            {"endpoint": "icu_cases", "label": "Case Library"},
            {"endpoint": "ai_evaluation", "label": "AI Model Report"},
            {"endpoint": "results", "label": "Results & Comparison"},
            {"endpoint": "user_guide", "label": "User Guide"},
        ],
    }
]
PAGES = [page for section in NAV_SECTIONS for page in section["pages"]]

@app.context_processor
def inject_navigation():
    """Make the sidebar links available to every template."""
    return {"pages": PAGES, "nav_sections": NAV_SECTIONS}


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

    items.append("Compare baseline and multi-agent assessment before documenting review.")
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
            "Structured review-support summary from computed model outputs: the supervised AI model "
            "output is not available for this ICU stay. The reviewer can still "
            "use the baseline workflow, multi-agent workflow, measurements, "
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
        "Structured review-support summary from computed model outputs: the trained supervised model "
        f"estimated a {ai_prediction['risk_probability'] * 100:.1f}% risk "
        f"probability, mapped to {ai_prediction['ai_risk_level']} by the "
        "configured thresholds. The highest-weighted model signals shown for "
        f"this case are {feature_names}. The multi-agent data quality "
        f"assessment is {case['data_quality']}."
        f"{missing_text} The baseline rule workflow is {case['baseline_risk']}, "
        f"while the multi-agent workflow is {case['risk_level']}."
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
    else:
        case["uncertainty_result"] = {"available": False, "message": "Open the assessment page to compute uncertainty."}
        case["similar_cases_result"] = {"available": False, "message": "Open the assessment page to compute similar cases."}
        case["cluster_result"] = {"available": False, "message": "Open the assessment page to compute cluster context."}
        case["anomaly_result"] = {"available": False, "message": "Open the assessment page to compute anomaly context."}
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


@app.route("/")
def home():
    return render_template("software_home.html", title="Home", workflow_stats=get_workflow_stats(), ai_system_status=get_ai_system_status())


@app.route("/software")
def software_home():
    return render_template("software_home.html", title="Home", workflow_stats=get_workflow_stats(), ai_system_status=get_ai_system_status())


@app.route("/icu-dashboard")
def icu_dashboard():
    cases = build_case_assessments()
    high_risk_cases = [case for case in cases if case["risk_level"] == "High"]
    medium_risk_cases = [case for case in cases if case["risk_level"] == "Medium"]
    low_risk_cases = [case for case in cases if case["risk_level"] == "Low"]
    ai_high_risk_cases = [case for case in cases if case["ai_prediction_result"].get("ai_risk_level") == "High"]
    ai_medium_risk_cases = [case for case in cases if case["ai_prediction_result"].get("ai_risk_level") == "Medium"]
    ai_low_risk_cases = [case for case in cases if case["ai_prediction_result"].get("ai_risk_level") == "Low"]
    good_data_cases = [case for case in cases if case["data_quality"] == "Good"]
    poor_or_partial_cases = [case for case in cases if case["data_quality"] in ["Poor", "Partial"]]
    follow_up_cases = [case for case in cases if case["review_status"] == "Needs Follow-up"]
    reviewed_cases = [case for case in cases if case["review_status"] == "Reviewed"]
    not_reviewed_cases = [case for case in cases if case["review_status"] == "Not Reviewed"]
    urgent_cases = [case for case in cases if case["priority"] == "Urgent Review"]
    priority_queue = sorted(cases, key=get_dashboard_queue_rank)[:12]

    return render_template(
        "icu_dashboard.html",
        title="Case Review Overview",
        total_cases=len(cases),
        high_risk_count=len(high_risk_cases),
        medium_risk_count=len(medium_risk_cases),
        low_risk_count=len(low_risk_cases),
        ai_high_risk_count=len(ai_high_risk_cases),
        ai_medium_risk_count=len(ai_medium_risk_cases),
        ai_low_risk_count=len(ai_low_risk_cases),
        good_data_count=len(good_data_cases),
        poor_or_partial_count=len(poor_or_partial_cases),
        follow_up_count=len(follow_up_cases),
        reviewed_count=len(reviewed_cases),
        not_reviewed_count=len(not_reviewed_cases),
        urgent_count=len(urgent_cases),
        priority_queue=priority_queue,
        ai_system_status=get_ai_system_status(),
        workflow_stats=get_workflow_stats(),
        recent_activity=get_audit_events(limit=8),
    )


@app.route("/icu-cases")
def icu_cases():
    all_cases = build_case_assessments()
    cases = all_cases
    risk_filter = request.args.get("risk_level", "")
    quality_filter = request.args.get("data_quality", "")
    status_filter = request.args.get("review_status", "")
    priority_filter = request.args.get("priority", "")
    admission_filter = request.args.get("admission_type", "")
    search = request.args.get("search", "").strip().lower()

    if risk_filter:
        cases = [case for case in cases if case["risk_level"] == risk_filter]
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
            if search in case["subject_id"].lower()
            or search in case["hadm_id"].lower()
            or search in case["icustay_id"].lower()
        ]

    return render_template(
        "icu_cases.html",
        title="Case Library",
        cases=sorted(cases, key=get_dashboard_queue_rank),
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
            "risk_level": risk_filter,
            "data_quality": quality_filter,
            "review_status": status_filter,
            "priority": priority_filter,
            "admission_type": admission_filter,
            "search": search,
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
    item, created = create_dataset_review_item(case, request.form.get("assigned_reviewer", ""))
    return redirect(url_for("review_queue", created="1" if created else "0", existing_id=item.get("id")))


@app.route("/review-queue")
def review_queue():
    status_filter = request.args.get("status")
    filters = {"status": "__active__" if status_filter is None else status_filter, "priority": request.args.get("priority", ""), "ai_risk_level": request.args.get("ai_risk_level", ""), "search": request.args.get("search", "")}
    return render_template("review_queue.html", title="Review Queue", items=list_review_items(filters), filters=filters, statuses=WORKFLOW_STATUSES, priorities=WORKFLOW_PRIORITIES, audit_events=get_audit_events(limit=15), created=request.args.get("created"), existing_id=request.args.get("existing_id"))


@app.route("/review-item/<int:item_id>")
def review_item_detail(item_id):
    item = get_review_item(item_id)
    if not item:
        return redirect(url_for("review_queue"))
    manual_case = get_manual_case(item["manual_case_id"]) if item.get("manual_case_id") else None
    return render_template("review_item_detail.html", title="Review Item", item=item, manual_case=manual_case, notes=get_review_notes(item_id), audit_events=get_audit_events("review_item", item_id), statuses=WORKFLOW_STATUSES, priorities=WORKFLOW_PRIORITIES)


@app.route("/review-item/<int:item_id>/edit", methods=["POST"])
def review_item_edit(item_id):
    update_review_item(item_id, request.form.get("status", "New"), request.form.get("priority", "Low"), request.form.get("assigned_reviewer", ""))
    return redirect(request.form.get("return_to") or url_for("review_item_detail", item_id=item_id))


@app.route("/review-item/<int:item_id>/note", methods=["POST"])
def review_item_note(item_id):
    note_text = request.form.get("note_text", "").strip()
    if note_text:
        add_review_note(item_id, request.form.get("reviewer_name", ""), note_text)
    return redirect(request.form.get("return_to") or url_for("review_item_detail", item_id=item_id))


@app.route("/review-item/<int:item_id>/archive", methods=["POST"])
def review_item_archive(item_id):
    archive_review_item(item_id)
    return redirect(url_for("review_queue"))


@app.route("/review-item/<int:item_id>/delete", methods=["POST"])
def review_item_delete(item_id):
    delete_review_item(item_id)
    return redirect(url_for("review_queue"))


def parse_manual_case_form(form):
    values = {"gender": form.get("gender", ""), "admission_type": form.get("admission_type", "")}
    for field in ["age", "heart_rate_avg", "heart_rate_max", "systolic_bp_avg", "respiratory_rate_avg", "temperature_avg", "spo2_avg", "creatinine_max", "glucose_max", "wbc_max", "hemoglobin_min"]:
        raw = form.get(field, "").strip()
        values[field] = float(raw) if raw else None
    return values


@app.route("/new-case", methods=["GET", "POST"])
def new_case():
    result = None
    if request.method == "POST":
        case_values = parse_manual_case_form(request.form)
        ai_prediction = predict_ai_risk(case_values)
        explainability = explain_ai_prediction(case_values)
        missing = get_missing_ai_features(explainability)
        uncertainty = estimate_prediction_uncertainty(ai_prediction, missing, "Manual")
        similar = find_similar_cases(case_values)
        cluster = assign_case_cluster(case_values)
        anomaly = score_case_anomaly(case_values)
        manual_case = create_manual_case(case_values, ai_prediction)
        review_item = create_manual_review_item(manual_case)
        result = {"case_values": case_values, "ai_prediction": ai_prediction, "uncertainty": uncertainty, "similar_cases": similar, "cluster": cluster, "anomaly": anomaly, "manual_case": manual_case, "review_item": review_item}
    return render_template("new_case.html", title="Add New Case", result=result)

@app.route("/ai-evaluation")
def ai_evaluation():
    metadata = load_json_file(AI_METADATA_PATH)
    metrics = load_json_file(AI_METRICS_PATH)
    active_dataset = get_active_feature_dataset_info()
    model_name = metadata.get("model_name") or metrics.get("best_model") or "Not available"
    model_record = metrics.get("models", {}).get(model_name, {})
    return render_template(
        "ai_evaluation.html",
        title="AI Model Report",
        ai_system_status=get_ai_system_status(),
        metadata=metadata,
        metrics=metrics,
        model_name=model_name,
        model_metrics=model_record.get("metrics", {}),
        tuned_metrics=model_record.get("tuned_threshold_metrics"),
        active_dataset=active_dataset,
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
    back_to_worklist_url = return_to or url_for("icu_cases")
    return render_template(
        "icu_case_detail.html",
        title="ICU Stay Assessment",
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
        linked_review_item=get_active_review_item_for_icustay(icustay_id),
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

    if question.strip().lower() == "generate case summary":
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
    evaluation_result = run_evaluation()
    return render_template(
        "results.html",
        title="Results & Comparison",
        evaluation_result=evaluation_result,
    )


@app.route("/technologies")
def technologies():
    return render_template("technologies.html", title="Technologies")


@app.route("/team")
def team():
    return render_template("team.html", title="Team")


if __name__ == "__main__":
    app.run(debug=True)















