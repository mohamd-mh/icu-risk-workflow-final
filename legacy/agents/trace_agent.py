"""Trace Agent and workflow runner simulation for Milestone 3."""

from agents.data_validator_agent import run_data_validator
from agents.explainer_agent import run_explainer
from agents.planner_agent import run_planner
from agents.risk_assessor_agent import run_risk_assessor
from agents.verifier_agent import run_verifier


def build_workflow_trace(
    planner_result,
    validation_result,
    risk_result,
    verification_result,
    explanation_result,
):
    """Create a simple numbered trace of the simulated agent workflow."""
    return [
        "Planner Agent selected required features.",
        "Data Validator Agent checked data completeness.",
        "Risk Assessor Agent calculated the risk score.",
        "Verifier Agent checked consistency.",
        "Explainer Agent generated the final explanation.",
    ]


def run_multi_agent_workflow(patient_dict):
    """Run the simulated agents in planning-driven order."""
    planner_result = run_planner(patient_dict)
    validation_result = run_data_validator(patient_dict, planner_result["required_features"])
    risk_result = run_risk_assessor(patient_dict)
    verification_result = run_verifier(patient_dict, risk_result, validation_result)
    explanation_result = run_explainer(
        patient_dict,
        risk_result,
        validation_result,
        verification_result,
    )
    trace = build_workflow_trace(
        planner_result,
        validation_result,
        risk_result,
        verification_result,
        explanation_result,
    )

    return {
        "planner": planner_result,
        "validation": validation_result,
        "risk": risk_result,
        "verification": verification_result,
        "explanation": explanation_result,
        "trace": trace,
    }
