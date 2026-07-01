"""Simple evaluation metric helpers."""


def risk_to_binary(risk_level):
    """Convert risk level into high-risk mortality detection."""
    if risk_level == "High":
        return 1
    return 0


def risk_to_at_risk_binary(risk_level):
    """Convert risk level into at-risk detection."""
    if risk_level in ["Medium", "High"]:
        return 1
    return 0


def calculate_accuracy(y_true, y_pred):
    """Calculate the fraction of correct predictions."""
    if not y_true:
        return 0

    correct = 0
    for true_value, pred_value in zip(y_true, y_pred):
        if true_value == pred_value:
            correct += 1

    return correct / len(y_true)


def calculate_precision(y_true, y_pred):
    """Calculate precision for the high-risk class."""
    true_positive = 0
    false_positive = 0

    for true_value, pred_value in zip(y_true, y_pred):
        if pred_value == 1 and true_value == 1:
            true_positive += 1
        elif pred_value == 1 and true_value == 0:
            false_positive += 1

    if true_positive + false_positive == 0:
        return 0

    return true_positive / (true_positive + false_positive)


def calculate_recall(y_true, y_pred):
    """Calculate recall for the high-risk class."""
    true_positive = 0
    false_negative = 0

    for true_value, pred_value in zip(y_true, y_pred):
        if true_value == 1 and pred_value == 1:
            true_positive += 1
        elif true_value == 1 and pred_value == 0:
            false_negative += 1

    if true_positive + false_negative == 0:
        return 0

    return true_positive / (true_positive + false_negative)


def calculate_f1(y_true, y_pred):
    """Calculate F1-score for the high-risk class."""
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


def calculate_confusion_matrix(y_true, y_pred):
    """Calculate binary confusion matrix counts."""
    matrix = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }

    for true_value, pred_value in zip(y_true, y_pred):
        if true_value == 1 and pred_value == 1:
            matrix["true_positive"] += 1
        elif true_value == 0 and pred_value == 1:
            matrix["false_positive"] += 1
        elif true_value == 0 and pred_value == 0:
            matrix["true_negative"] += 1
        elif true_value == 1 and pred_value == 0:
            matrix["false_negative"] += 1

    return matrix
