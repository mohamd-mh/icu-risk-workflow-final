"""Evaluation runner for baseline and active multi-agent workflow."""

import json
from pathlib import Path

import pandas as pd

from agents.multi_agent_workflow import run_workflow
from baseline.baseline_pipeline import run_baseline
from evaluation.metrics import (
    calculate_accuracy,
    calculate_confusion_matrix,
    calculate_f1,
    calculate_precision,
    calculate_recall,
    risk_to_at_risk_binary,
    risk_to_binary,
)


RESULTS_PATH = Path("data") / "evaluation_results.json"


def build_metric_dict(y_true, y_pred):
    """Build the numeric metric dictionary and confusion matrix."""
    return {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "precision": calculate_precision(y_true, y_pred),
        "recall": calculate_recall(y_true, y_pred),
        "f1": calculate_f1(y_true, y_pred),
        "confusion_matrix": calculate_confusion_matrix(y_true, y_pred),
    }


def build_interpretation_metrics(y_true, baseline_levels, multi_agent_levels, mapper):
    """Evaluate both workflows using one binary risk interpretation."""
    baseline_predictions = [mapper(level) for level in baseline_levels]
    multi_agent_predictions = [mapper(level) for level in multi_agent_levels]

    return {
        "baseline": build_metric_dict(y_true, baseline_predictions),
        "multi_agent": build_metric_dict(y_true, multi_agent_predictions),
    }


def check_data_leakage(patients):
    """Check that hospital_expire_flag does not change predictions."""
    issues_count = 0

    for _, patient in patients.head(20).iterrows():
        patient_zero = patient.to_dict()
        patient_one = patient.to_dict()
        patient_zero["hospital_expire_flag"] = 0
        patient_one["hospital_expire_flag"] = 1

        baseline_zero = run_baseline(patient_zero)["risk_level"]
        baseline_one = run_baseline(patient_one)["risk_level"]
        workflow_zero = run_workflow(patient_zero)["final_risk_level"]
        workflow_one = run_workflow(patient_one)["final_risk_level"]

        if baseline_zero != baseline_one or workflow_zero != workflow_one:
            issues_count += 1

    return {
        "leakage_check": "Passed" if issues_count == 0 else "Failed",
        "leakage_issues_count": issues_count,
    }


def run_evaluation(csv_path="data/patient_features_ai.csv"):
    """Evaluate baseline and active multi-agent outputs on patient_features_ai.csv."""
    patients = pd.read_csv(csv_path)

    y_true = []
    baseline_levels = []
    multi_agent_levels = []

    for _, patient in patients.iterrows():
        patient_dict = patient.to_dict()
        y_true.append(int(patient_dict["hospital_expire_flag"]))

        baseline_result = run_baseline(patient_dict)
        multi_agent_result = run_workflow(patient_dict)

        baseline_levels.append(baseline_result["risk_level"])
        multi_agent_levels.append(multi_agent_result["final_risk_level"])

    survived_count = int(y_true.count(0))
    died_count = int(y_true.count(1))
    total_rows = len(y_true)
    mortality_rate = 0 if total_rows == 0 else died_count / total_rows

    high_risk_metrics = build_interpretation_metrics(
        y_true,
        baseline_levels,
        multi_agent_levels,
        risk_to_binary,
    )
    at_risk_metrics = build_interpretation_metrics(
        y_true,
        baseline_levels,
        multi_agent_levels,
        risk_to_at_risk_binary,
    )

    evaluation_result = {
        "high_risk_detection": high_risk_metrics,
        "at_risk_detection": at_risk_metrics,
        "baseline_metrics": high_risk_metrics["baseline"],
        "multi_agent_metrics": high_risk_metrics["multi_agent"],
        "class_distribution": {
            "total_rows": total_rows,
            "survived_count": survived_count,
            "died_count": died_count,
            "mortality_rate": mortality_rate,
        },
        "data_leakage": check_data_leakage(patients),
        "reliability_comparison": {
            "baseline": {
                "missing_data_handling": "Basic missing value handling",
                "validator": "No separate validator",
                "verifier": "No verifier",
                "workflow_trace": "No workflow trace",
            },
            "multi_agent": {
                "data_quality": "Data quality Good/Partial/Poor",
                "validator": "Validator Agent",
                "verifier": "Verifier Agent",
                "workflow_trace": "Workflow trace",
                "uncertainty_warnings": "Uncertainty warnings",
            },
        },
        "transparency_comparison": {
            "baseline": {
                "explanation": "Basic",
                "contributing_factors": "Contributing factors",
                "workflow_trace": "No workflow trace",
            },
            "multi_agent": {
                "weighted_factors": "Weighted contributing factors",
                "explanation": "Explanation",
                "validation_summary": "Validation summary",
                "verification_notes": "Verification notes",
                "workflow_trace": "Workflow trace",
            },
        },
        "summary": (
            "The multi-agent workflow mainly improves reliability and transparency. "
            "Its accuracy depends on the risk-threshold configuration and should "
            "be interpreted together with precision, recall, F1-score, and the "
            "confusion matrix."
        ),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as results_file:
        json.dump(evaluation_result, results_file, indent=2)

    return evaluation_result
