"""Risk Assessor Agent simulation for Milestone 3."""

from baseline.baseline_pipeline import run_baseline


def run_risk_assessor(patient_dict):
    """Reuse the baseline rule logic as the simulated agent risk assessment."""
    baseline_result = run_baseline(patient_dict)

    return {
        "risk_level": baseline_result["risk_level"],
        "risk_score": baseline_result["risk_score"],
        "factors": baseline_result["factors"],
    }
