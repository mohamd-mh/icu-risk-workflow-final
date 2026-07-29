"""Planner Agent simulation for Milestone 3."""


REQUIRED_FEATURES = [
    "subject_id",
    "icustay_id",
    "age",
    "admission_type",
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


def run_planner(patient_dict):
    """Create a simple plan for the simulated multi-agent workflow."""
    plan_steps = [
        "Identify patient and ICU stay.",
        "Check required early ICU features.",
        "Calculate risk using clinical indicators.",
        "Verify result consistency.",
        "Generate explanation and workflow trace.",
    ]

    return {
        "plan_steps": plan_steps,
        "required_features": REQUIRED_FEATURES,
    }
