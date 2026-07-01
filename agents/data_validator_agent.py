"""Data Validator Agent simulation for Milestone 3."""


NUMERIC_FEATURES = [
    "subject_id",
    "icustay_id",
    "age",
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


def is_empty(value):
    return value is None or str(value).strip() == ""


def is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def run_data_validator(patient_dict, required_features):
    """Check required values and numeric clinical feature values."""
    missing_features = []
    warnings = []

    for feature in required_features:
        value = patient_dict.get(feature)

        if is_empty(value):
            missing_features.append(feature)
            continue

        if feature in NUMERIC_FEATURES and not is_number(value):
            warnings.append(f"{feature} should be numeric")

    is_valid = not missing_features and not warnings

    if is_valid:
        validation_summary = "Data validation passed. Required values are present and numeric fields look valid."
    else:
        validation_summary = "Data validation found missing or invalid feature values."

    return {
        "is_valid": is_valid,
        "missing_features": missing_features,
        "warnings": warnings,
        "validation_summary": validation_summary,
    }
