"""Simple unweighted baseline risk pipeline."""

from baseline.risk_rules import evaluate_rules


def get_risk_level(risk_score):
    """Convert the numeric rule score into a simple risk level."""
    if risk_score <= 2:
        return "Low"
    if risk_score == 3:
        return "Medium"
    return "High"


def run_baseline(patient_dict):
    """Run the rule-based baseline pipeline for one selected patient."""
    risk_score, factors = evaluate_rules(patient_dict)
    risk_level = get_risk_level(risk_score)

    if factors:
        explanation = (
            f"The patient is classified as {risk_level} risk because "
            f"{risk_score} abnormal feature(s) were found."
        )
    else:
        explanation = "The patient is classified as Low risk because no abnormal rule factors were found."

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "contributing_factors": factors,
        "factors": factors,
        "explanation": explanation,
    }


calculate_risk = run_baseline
