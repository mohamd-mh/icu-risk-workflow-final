"""Optional live generative AI assistant for ICU case review support.

The assistant uses a real LLM only when LLM_API_KEY is configured. Without a
key, it returns a template-based offline summary from local model outputs.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

SAFE_REFUSAL = (
    "This system supports quality-review prioritization and documentation only. "
    "It does not provide diagnosis, treatment, or medication recommendations. "
    "Please consult qualified clinical staff for medical decisions."
)
OFFLINE_MODE_NOTE = "Template-based offline summary from local model outputs; not live LLM output."
DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
UNSAFE_TERMS = [
    "diagnosis",
    "diagnose",
    "treatment",
    "treat",
    "medication",
    "medicine",
    "drug",
    "dose",
    "prescribe",
    "therapy",
    "what should be given",
    "doctor",
    "clinician",
    "clinically",
    "emergency",
    "urgent command",
    "clinical command",
    "should do clinically",
    "what should the doctor do",
]
ALLOWED_TERMS = [
    "risk",
    "missing",
    "similar",
    "cluster",
    "unusual",
    "anomaly",
    "uncertain",
    "confidence",
    "baseline",
    "sirs",
    "flagged",
    "draft",
    "note",
    "trained",
    "training",
    "evaluation",
    "workflow",
    "summary",
    "feature",
]


def is_unsafe_question(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in UNSAFE_TERMS)


def is_supported_question(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in ALLOWED_TERMS)


def _safe_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _safe_jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [_safe_jsonable(item) for item in value]
        return str(value)


def _format_probability(ai_prediction: dict[str, Any]) -> str:
    probability = ai_prediction.get("risk_probability")
    if probability is None:
        return "not available"
    try:
        return f"{float(probability) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(probability)


def _top_feature_text(explainability: dict[str, Any], limit: int = 3) -> str:
    features = explainability.get("top_features") or []
    lines = []
    for feature in features[:limit]:
        name = feature.get("display_name") or feature.get("feature") or "AI feature"
        if feature.get("is_missing") or feature.get("value") is None:
            lines.append(f"Missing important AI feature: {name}.")
        else:
            value = feature.get("value")
            interpretation = feature.get("interpretation") or "This feature influenced the model output."
            lines.append(f"{name}: value {value}. {interpretation}")
    return " ".join(lines) if lines else "No top-feature explanation is available for this case."


def _missing_feature_text(explainability: dict[str, Any]) -> str:
    missing = [
        feature.get("display_name") or feature.get("feature")
        for feature in (explainability.get("top_features") or [])
        if feature.get("is_missing") or feature.get("value") is None
    ]
    missing = [item for item in missing if item]
    if not missing:
        return "No missing values appear among the top AI explanation features."
    return "Missing important AI features: " + ", ".join(missing) + ". Prediction should be interpreted with caution."


def _baseline_text(ai_prediction: dict[str, Any], baseline_result: dict[str, Any] | None) -> str:
    ai_level = ai_prediction.get("ai_risk_level", "not available")
    if not baseline_result:
        return f"The AI model classified the case as {ai_level}. Baseline comparison is not available."
    return (
        f"The AI model classified the case as {ai_level}. The SIRS-style screening result is "
        f"{baseline_result.get('risk_level', 'not available')} with score "
        f"{baseline_result.get('risk_score', 'not available')}. Use this comparison as review context."
    )


def build_offline_case_summary(
    case_data: dict[str, Any],
    ai_prediction: dict[str, Any],
    explainability: dict[str, Any],
    data_quality: str,
    baseline_result: dict[str, Any] | None = None,
    **analytics: Any,
) -> str:
    case_label = case_data.get("case_label") or f"ICU stay {case_data.get('icustay_id', 'unknown')}"
    risk_level = ai_prediction.get("ai_risk_level", "not available")
    probability = _format_probability(ai_prediction)
    uncertainty = analytics.get("uncertainty_result") or {}
    confidence = uncertainty.get("confidence_label", "not available")
    return (
        f"{OFFLINE_MODE_NOTE} {case_label} is classified by the AI risk prediction model as "
        f"{risk_level} with probability {probability}. Confidence is {confidence}; data quality is {data_quality}. "
        f"Top explanation signals: {_top_feature_text(explainability)} {_missing_feature_text(explainability)} "
        f"{_baseline_text(ai_prediction, baseline_result)}"
    )


def build_structured_context(
    case_data: dict[str, Any],
    ai_prediction: dict[str, Any],
    explainability: dict[str, Any],
    data_quality: str,
    baseline_result: dict[str, Any] | None = None,
    similar_cases: dict[str, Any] | None = None,
    cluster_result: dict[str, Any] | None = None,
    anomaly_result: dict[str, Any] | None = None,
    uncertainty_result: dict[str, Any] | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _safe_jsonable({
        "case_identifiers": {
            "case_label": case_data.get("case_label"),
            "subject_id": case_data.get("subject_id"),
            "hadm_id": case_data.get("hadm_id"),
            "icustay_id": case_data.get("icustay_id"),
        },
        "case_features": case_data.get("patient", {}),
        "ai_prediction": ai_prediction,
        "explainable_ai": explainability,
        "similar_cases": similar_cases,
        "cluster_context": cluster_result,
        "anomaly_detection": anomaly_result,
        "uncertainty": uncertainty_result,
        "data_quality": data_quality,
        "baseline_comparison": baseline_result,
        "model_metadata": model_metadata,
        "safety_constraints": [
            "academic prototype",
            "review support and explanation only",
            "no diagnosis",
            "no medication advice",
            "no treatment recommendations",
            "no emergency or clinical action instructions",
        ],
    })


def call_llm_api(question: str, context: dict[str, Any]) -> str:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return ""

    api_url = os.getenv("LLM_API_URL", DEFAULT_API_URL)
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": ("You are an academic ICU case-review assistant. Answer only about AI risk explanation, missing data, similar cases, cluster context, anomaly score, uncertainty, baseline comparison, reviewer note drafts, model training/evaluation, and workflow guidance. Refuse diagnosis, medication, treatment, emergency, or clinical action advice.")},
            {"role": "user", "content": json.dumps({"question": question, "context": context}, indent=2)},
        ],
        "temperature": 0.2,
    }
    request = urllib.request.Request(api_url, data=json.dumps(payload).encode("utf-8"), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        return f"Live Generative AI request failed: {error}"


def generate_case_summary(
    case_data: dict[str, Any],
    ai_prediction: dict[str, Any],
    explainability: dict[str, Any],
    data_quality: str,
    baseline_result: dict[str, Any] | None = None,
    **analytics: Any,
) -> str:
    if not os.getenv("LLM_API_KEY"):
        return build_offline_case_summary(case_data, ai_prediction, explainability, data_quality, baseline_result, **analytics)
    context = build_structured_context(case_data, ai_prediction, explainability, data_quality, baseline_result, **analytics)
    return call_llm_api("Generate a concise reviewer-support case summary from the structured context.", context)


def answer_case_question(
    question: str,
    case_data: dict[str, Any],
    ai_prediction: dict[str, Any],
    explainability: dict[str, Any],
    data_quality: str,
    baseline_result: dict[str, Any] | None = None,
    **analytics: Any,
) -> str:
    question = (question or "").strip()
    if not question:
        return "Please enter a question about this ICU case, AI prediction, missing data, similar cases, cluster context, anomaly score, uncertainty, or reviewer note draft."
    if is_unsafe_question(question):
        return SAFE_REFUSAL
    if not is_supported_question(question):
        return "Please ask about AI risk explanation, why the case was flagged, missing data, similar cases, cluster context, anomaly score, uncertainty, AI vs SIRS comparison, model training/evaluation, workflow guidance, or reviewer note draft."

    if not os.getenv("LLM_API_KEY"):
        lowered = question.lower()
        if "missing" in lowered:
            return f"{OFFLINE_MODE_NOTE} {_missing_feature_text(explainability)}"
        if "baseline" in lowered or "sirs" in lowered or "compare" in lowered:
            return f"{OFFLINE_MODE_NOTE} {_baseline_text(ai_prediction, baseline_result)}"
        if "draft" in lowered or "note" in lowered:
            return (
                f"{OFFLINE_MODE_NOTE} Reviewer note draft: AI risk level {ai_prediction.get('ai_risk_level', 'not available')} "
                f"with probability {_format_probability(ai_prediction)}. Data quality: {data_quality}. "
                f"Key review signals: {_top_feature_text(explainability)}"
            )
        return build_offline_case_summary(case_data, ai_prediction, explainability, data_quality, baseline_result, **analytics)

    context = build_structured_context(case_data, ai_prediction, explainability, data_quality, baseline_result, **analytics)
    return call_llm_api(question, context)
