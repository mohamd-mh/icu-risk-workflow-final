"""Verifier Agent simulation for Milestone 3."""


def run_verifier(patient_dict, risk_result, validation_result):
    """Check the risk result for simple consistency problems."""
    verification_status = "Passed"
    verification_notes = []

    if not validation_result["is_valid"]:
        verification_status = "Failed"
        verification_notes.append("Data validation failed, so the result should be reviewed.")

    factors = risk_result.get("factors", [])
    risk_level = risk_result.get("risk_level")

    if risk_level == "High" and not factors:
        verification_status = "Warning"
        verification_notes.append("High risk was reported, but no contributing factors were found.")

    if risk_level == "Low" and len(factors) > 2:
        verification_status = "Warning"
        verification_notes.append("Low risk was reported, but several abnormal factors were found.")

    if "hospital_expire_flag" in patient_dict:
        label = patient_dict.get("hospital_expire_flag")
        verification_notes.append(
            f"Reference note: hospital_expire_flag is {label}. It was not used as an input feature."
        )

    if not verification_notes:
        verification_notes.append("Risk result is consistent with the validation and factor checks.")

    return {
        "verification_status": verification_status,
        "verification_notes": verification_notes,
    }
