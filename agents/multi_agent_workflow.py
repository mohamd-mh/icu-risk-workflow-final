"""Modular agent-inspired ICU quality review workflow.

Runs a fixed sequence of specialized rule-based components (data validation,
weighted clinical scoring, verification, explanation). This is not a full
autonomous AutoGen-style multi-agent negotiation system: there is no
independent planning, delegation, or LLM reasoning between steps - each
component is a plain function call whose output feeds the next step in a
fixed order.
"""

from baseline.baseline_pipeline import run_baseline


CLINICAL_FEATURES = [
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

MAX_SCORE = 15


def get_number(patient, feature_name):
    try:
        value = float(patient.get(feature_name))
    except (TypeError, ValueError):
        return None

    if value != value:
        return None
    return value


def is_missing(patient, feature_name):
    value = patient.get(feature_name)
    return value is None or str(value).strip() == "" or str(value).lower() == "nan" or get_number(patient, feature_name) is None


def run_planner():
    return [
        "Identify required clinical features.",
        "Validate data quality.",
        "Calculate weighted clinical score.",
        "Verify clinical consistency.",
        "Generate explanation.",
        "Build workflow trace.",
    ]


def run_data_validator(patient):
    missing_features = [feature for feature in CLINICAL_FEATURES if is_missing(patient, feature)]
    missing_count = len(missing_features)

    if missing_count <= 2:
        data_quality = "Good"
    elif missing_count <= 5:
        data_quality = "Partial"
    else:
        data_quality = "Poor"

    warnings = []
    if missing_features:
        warnings.append("Some clinical features are missing.")

    validation_summary = (
        f"Data quality is {data_quality}. "
        f"{missing_count} of {len(CLINICAL_FEATURES)} required clinical features are missing."
    )

    return {
        "data_quality": data_quality,
        "missing_features": missing_features,
        "validation_summary": validation_summary,
        "warnings": warnings,
    }


def add_factor(factors, score, condition, message, weight):
    if condition:
        factors.append(f"{message}: +{weight}")
        return score + weight
    return score


def score_to_level(weighted_score):
    if weighted_score <= 3:
        return "Low"
    if weighted_score <= 7:
        return "Medium"
    return "High"


def raise_one_level(risk_level):
    if risk_level == "Low":
        return "Medium"
    if risk_level == "Medium":
        return "High"
    return "High"


def run_weighted_risk_assessor(patient):
    heart_rate_avg = get_number(patient, "heart_rate_avg")
    heart_rate_max = get_number(patient, "heart_rate_max")
    systolic_bp_avg = get_number(patient, "systolic_bp_avg")
    respiratory_rate_avg = get_number(patient, "respiratory_rate_avg")
    temperature_avg = get_number(patient, "temperature_avg")
    spo2_avg = get_number(patient, "spo2_avg")
    creatinine_max = get_number(patient, "creatinine_max")
    glucose_max = get_number(patient, "glucose_max")
    wbc_max = get_number(patient, "wbc_max")
    hemoglobin_min = get_number(patient, "hemoglobin_min")

    factors = []
    score = 0
    score = add_factor(
        factors,
        score,
        (heart_rate_avg is not None and heart_rate_avg >= 100)
        or (heart_rate_max is not None and heart_rate_max >= 130),
        "High heart rate",
        1,
    )
    score = add_factor(factors, score, systolic_bp_avg is not None and systolic_bp_avg <= 90, "Low systolic blood pressure", 3)
    score = add_factor(factors, score, respiratory_rate_avg is not None and respiratory_rate_avg >= 24, "High respiratory rate", 2)
    score = add_factor(factors, score, temperature_avg is not None and temperature_avg >= 38, "High temperature", 1)
    score = add_factor(factors, score, spo2_avg is not None and spo2_avg <= 92, "Low oxygen saturation (SpO2 <= 92)", 3)
    score = add_factor(factors, score, creatinine_max is not None and creatinine_max >= 2, "High creatinine", 2)
    score = add_factor(factors, score, glucose_max is not None and glucose_max >= 200, "High glucose", 1)
    score = add_factor(factors, score, wbc_max is not None and wbc_max >= 15, "High white blood cell count", 1)
    score = add_factor(factors, score, hemoglobin_min is not None and hemoglobin_min <= 9, "Low hemoglobin", 1)

    return {
        "weighted_score": score,
        "weighted_risk_level": score_to_level(score),
        "contributing_factors": factors,
    }


def has_critical_pattern(patient):
    spo2_avg = get_number(patient, "spo2_avg")
    systolic_bp_avg = get_number(patient, "systolic_bp_avg")
    respiratory_rate_avg = get_number(patient, "respiratory_rate_avg")
    creatinine_max = get_number(patient, "creatinine_max")
    temperature_avg = get_number(patient, "temperature_avg")
    wbc_max = get_number(patient, "wbc_max")

    patterns = [
        spo2_avg is not None and spo2_avg <= 92 and respiratory_rate_avg is not None and respiratory_rate_avg >= 24,
        spo2_avg is not None and spo2_avg <= 92 and systolic_bp_avg is not None and systolic_bp_avg <= 90,
        creatinine_max is not None and creatinine_max >= 2 and systolic_bp_avg is not None and systolic_bp_avg <= 90,
        temperature_avg is not None and temperature_avg >= 38 and wbc_max is not None and wbc_max >= 15,
    ]
    return any(patterns)


def run_verifier(patient, validation_result, baseline_result, risk_result):
    initial_level = baseline_result["risk_level"]
    final_level = initial_level
    weighted_score = risk_result["weighted_score"]
    verification_status = "Passed"
    verification_notes = []

    if initial_level == "Medium" and weighted_score >= 8:
        final_level = "High"
        verification_notes.append("Risk was raised from Medium to High because weighted verification score is high.")

    if initial_level == "Low" and weighted_score >= 6:
        final_level = "Medium"
        verification_notes.append("Risk was raised from Low to Medium because weighted verification score is high.")

    if final_level == "Low" and validation_result["data_quality"] == "Poor":
        verification_status = "Warning"
        verification_notes.append("Data quality is poor, so the low-risk result should be interpreted carefully.")

    if has_critical_pattern(patient):
        raised_level = raise_one_level(final_level)
        if raised_level != final_level:
            final_level = raised_level
            verification_notes.append("Risk was raised one level because a critical clinical pattern was found.")

    if validation_result["data_quality"] in ["Partial", "Poor"] or final_level != initial_level:
        verification_status = "Warning"

    if not verification_notes:
        verification_notes.append("Verifier confirmed the weighted risk result.")

    return {
        "final_risk_level": final_level,
        "verification_status": verification_status,
        "verification_notes": verification_notes,
    }


def run_explainer(validation_result, baseline_result, risk_result, verification_result):
    baseline_factors = baseline_result.get("contributing_factors", [])
    weighted_factors = risk_result["contributing_factors"]
    all_factors = baseline_factors + weighted_factors
    factor_text = "; ".join(all_factors[:6]) if all_factors else "no clinical factors were triggered"
    changed_text = "confirmed"
    if verification_result["final_risk_level"] != baseline_result["risk_level"]:
        changed_text = "changed"

    explanation = (
        f"The final risk level is {verification_result['final_risk_level']}. "
        f"The baseline initial risk was {baseline_result['risk_level']} with a score of "
        f"{baseline_result['risk_score']}. The weighted verification score is "
        f"{risk_result['weighted_score']} out of {MAX_SCORE}. Data quality is "
        f"{validation_result['data_quality']}. The verifier {changed_text} the risk level. "
        f"Main contributing factors: {factor_text}."
    )
    return explanation


def build_trace():
    return [
        "Planner Agent defined the required workflow steps.",
        "Data Validator Agent checked data completeness.",
        "Baseline Risk Assessor produced the initial risk.",
        "Weighted Clinical Verifier checked severity patterns.",
        "Verifier Agent adjusted or confirmed the risk.",
        "Explainer Agent generated the final explanation.",
    ]


def run_workflow(patient):
    planner_steps = run_planner()
    validation_result = run_data_validator(patient)
    baseline_result = run_baseline(patient)
    risk_result = run_weighted_risk_assessor(patient)
    verification_result = run_verifier(patient, validation_result, baseline_result, risk_result)
    explanation = run_explainer(validation_result, baseline_result, risk_result, verification_result)
    contributing_factors = baseline_result.get("contributing_factors", []) + risk_result["contributing_factors"]

    return {
        "initial_baseline_risk_level": baseline_result["risk_level"],
        "final_risk_level": verification_result["final_risk_level"],
        "baseline_risk_score": baseline_result["risk_score"],
        "weighted_score": risk_result["weighted_score"],
        "max_score": MAX_SCORE,
        "data_quality": validation_result["data_quality"],
        "validation_summary": validation_result["validation_summary"],
        "verification_status": verification_result["verification_status"],
        "verification_notes": verification_result["verification_notes"],
        "contributing_factors": contributing_factors,
        "explanation": explanation,
        "workflow_trace": build_trace(),
        "planner_steps": planner_steps,
    }
