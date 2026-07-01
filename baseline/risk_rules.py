"""Simple unweighted baseline risk rules."""


BASELINE_RULES = [
    ("age", ">=", 75, "Age is 75 or older"),
    ("admission_type", "==", "EMERGENCY", "Admission type is emergency"),
    ("systolic_bp_avg", "<=", 90, "Average systolic blood pressure is low"),
    ("respiratory_rate_avg", ">=", 24, "Average respiratory rate is high"),
    ("temperature_avg", ">=", 38, "Average temperature is high"),
    ("spo2_avg", "<=", 92, "Average oxygen saturation is low"),
    ("creatinine_max", ">=", 2, "Maximum creatinine is high"),
    ("glucose_max", ">=", 200, "Maximum glucose is high"),
    ("wbc_max", ">=", 15, "Maximum white blood cell count is high"),
    ("hemoglobin_min", "<=", 9, "Minimum hemoglobin is low"),
]


def get_number(patient_dict, feature_name):
    """Convert a patient feature value to a number when possible."""
    try:
        value = float(patient_dict.get(feature_name))
    except (TypeError, ValueError):
        return None

    if value != value:
        return None
    return value


def evaluate_rules(patient_dict):
    """Apply all threshold rules and return the risk score and factor messages."""
    risk_score = 0
    factors = []

    heart_rate_avg = get_number(patient_dict, "heart_rate_avg")
    heart_rate_max = get_number(patient_dict, "heart_rate_max")
    if (
        heart_rate_avg is not None and heart_rate_avg >= 100
    ) or (
        heart_rate_max is not None and heart_rate_max >= 130
    ):
        risk_score += 1
        factors.append("Heart rate is high")

    for feature_name, operator, threshold, message in BASELINE_RULES:
        if operator == "==":
            value = str(patient_dict.get(feature_name, "")).strip().upper()
            is_abnormal = value == str(threshold).upper()
        else:
            value = get_number(patient_dict, feature_name)

            if value is None:
                continue

            is_abnormal = (
                operator == ">=" and value >= threshold
            ) or (
                operator == "<=" and value <= threshold
            )

        if is_abnormal:
            risk_score += 1
            factors.append(message)

    return risk_score, factors
