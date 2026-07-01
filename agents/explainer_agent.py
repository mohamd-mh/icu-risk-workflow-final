"""Explainer Agent simulation for Milestone 3."""


def run_explainer(patient_dict, risk_result, validation_result, verification_result):
    """Create a simple patient-level explanation for the workflow result."""
    risk_level = risk_result["risk_level"]
    risk_score = risk_result["risk_score"]
    main_factors = risk_result["factors"]
    icustay_id = patient_dict.get("icustay_id", "selected")

    if main_factors:
        factor_text = "; ".join(main_factors[:4])
    else:
        factor_text = "no abnormal rule-based factors"

    explanation = (
        f"For ICU stay {icustay_id}, the simulated multi-agent workflow gives a "
        f"{risk_level} risk result with a score of {risk_score}. The main reason "
        f"is: {factor_text}. Validation summary: "
        f"{validation_result['validation_summary']} Verification status: "
        f"{verification_result['verification_status']}."
    )

    return {
        "explanation": explanation,
        "main_factors": main_factors,
    }
