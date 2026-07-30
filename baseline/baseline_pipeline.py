"""SIRS-based baseline risk pipeline.

Baseline uses the SIRS (Systemic Inflammatory Response Syndrome) criteria
(Bone et al., 1992) - see baseline/risk_rules.py for the 4 cited criteria
and their exact thresholds. This is a 4-criteria clinical scoring rule, not
the earlier 10-rule ad-hoc threshold count.
"""

from baseline.risk_rules import evaluate_rules

SIRS_CITATION = (
    "Bone, R.C., Balk, R.A., Cerra, F.B., et al. (1992). Definitions for "
    "sepsis and organ failure and guidelines for the use of innovative "
    "therapies in sepsis. Chest, 101(6), 1644-1655."
)

# Standard clinical definition: SIRS-positive = 2 or more of the 4 criteria met.
SIRS_POSITIVE_THRESHOLD = 2


def get_risk_level(criteria_met_count):
    """Map the number of SIRS criteria met (0-4) to a risk level.

    0-1 criteria -> Low
    2 criteria   -> Medium (SIRS-positive, standard clinical threshold)
    3-4 criteria -> High (multiple criteria met, more severe)
    """
    if criteria_met_count <= 1:
        return "Low"
    if criteria_met_count == SIRS_POSITIVE_THRESHOLD:
        return "Medium"
    return "High"


def run_baseline(patient_dict):
    """Run the SIRS-based baseline (Bone et al., 1992) for one selected patient."""
    criteria_met_count, factors = evaluate_rules(patient_dict)
    risk_level = get_risk_level(criteria_met_count)
    sirs_positive = criteria_met_count >= SIRS_POSITIVE_THRESHOLD

    sirs_status = "SIRS-positive" if sirs_positive else "SIRS-negative"
    if factors:
        explanation = (
            f"SIRS-based baseline (Bone et al., 1992): {criteria_met_count} of 4 criteria met "
            f"({sirs_status}), classified as {risk_level} risk. "
            f"Criteria met: {'; '.join(factors)}."
        )
    else:
        explanation = (
            "SIRS-based baseline (Bone et al., 1992): 0 of 4 criteria met "
            "(SIRS-negative), classified as Low risk."
        )

    return {
        "risk_score": criteria_met_count,
        "risk_level": risk_level,
        "sirs_criteria_met": criteria_met_count,
        "sirs_positive": sirs_positive,
        "contributing_factors": factors,
        "factors": factors,
        "explanation": explanation,
        "citation": SIRS_CITATION,
    }


calculate_risk = run_baseline
