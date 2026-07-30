"""SIRS (Systemic Inflammatory Response Syndrome) baseline criteria.

Reference: Bone, R.C., Balk, R.A., Cerra, F.B., et al. (1992). "Definitions
for sepsis and organ failure and guidelines for the use of innovative
therapies in sepsis." Chest, 101(6), 1644-1655.

SIRS is defined by two or more of the four criteria below. This module
implements exactly those four cited criteria - no other clinical variables
are part of the official SIRS definition, so systolic blood pressure,
creatinine, glucose, hemoglobin, SpO2, age, and admission type are
intentionally NOT evaluated here. Those variables remain used elsewhere by
the trained ML model (all 13 input features) and by the separate weighted
risk assessor stage in agents/multi_agent_workflow.py, which is explicitly
documented there as engineering judgment, not a validated clinical score.
"""

# Named, cited thresholds (Bone et al. 1992) - not magic numbers.
SIRS_TEMPERATURE_HIGH_C = 38     # Criterion 1: temperature > 38 C
SIRS_TEMPERATURE_LOW_C = 36      # Criterion 1: temperature < 36 C
SIRS_HEART_RATE_MIN = 90         # Criterion 2: heart rate > 90/min
SIRS_RESPIRATORY_RATE_MIN = 20   # Criterion 3: respiratory rate > 20/min

# Criterion 4: WBC > 12,000/mm3 or < 4,000/mm3. This project's dataset
# stores wbc_max in K/uL (thousands of cells per microliter), so
# 12,000/mm3 = 12 K/uL and 4,000/mm3 = 4 K/uL.
SIRS_WBC_HIGH_K_PER_UL = 12
SIRS_WBC_LOW_K_PER_UL = 4

# The 4 SIRS criteria (Bone et al. 1992): (criterion_key, feature_name, message).
SIRS_CRITERIA = [
    (
        "temperature_abnormal",
        "temperature_avg",
        "Temperature above 38C or below 36C (SIRS criterion, Bone et al. 1992)",
    ),
    (
        "heart_rate_high",
        "heart_rate_avg",
        "Heart rate above 90/min (SIRS criterion, Bone et al. 1992)",
    ),
    (
        "respiratory_rate_high",
        "respiratory_rate_avg",
        "Respiratory rate above 20/min (SIRS criterion, Bone et al. 1992)",
    ),
    (
        "wbc_abnormal",
        "wbc_max",
        "White blood cell count above 12,000/mm3 or below 4,000/mm3 (SIRS criterion, Bone et al. 1992)",
    ),
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


def is_criterion_met(criterion_key, value):
    """Check a single SIRS criterion (Bone et al. 1992) against its cited threshold(s)."""
    if criterion_key == "temperature_abnormal":
        return value > SIRS_TEMPERATURE_HIGH_C or value < SIRS_TEMPERATURE_LOW_C
    if criterion_key == "heart_rate_high":
        return value > SIRS_HEART_RATE_MIN
    if criterion_key == "respiratory_rate_high":
        return value > SIRS_RESPIRATORY_RATE_MIN
    if criterion_key == "wbc_abnormal":
        return value > SIRS_WBC_HIGH_K_PER_UL or value < SIRS_WBC_LOW_K_PER_UL
    return False


def evaluate_rules(patient_dict):
    """Evaluate the 4 SIRS criteria (Bone et al. 1992) for one patient.

    Returns (criteria_met_count, factor_messages). A criterion with a
    missing/non-numeric value is skipped (not counted as met or not met),
    consistent with how missing data was already handled in this project.
    """
    criteria_met = 0
    factors = []

    for criterion_key, feature_name, message in SIRS_CRITERIA:
        value = get_number(patient_dict, feature_name)
        if value is None:
            continue
        if is_criterion_met(criterion_key, value):
            criteria_met += 1
            factors.append(message)

    return criteria_met, factors
