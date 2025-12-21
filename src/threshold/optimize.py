import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_threshold_metrics(y_true, y_proba):
    """
    Compute precision, recall, F1-score across all probability thresholds.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary containing thresholds, precision, recall, and F1 scores.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    return {
        "thresholds": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
        "f1": f1_scores[:-1],
    }


def select_best_f1_threshold(metrics):
    """
    Select threshold that maximizes F1-score.
    """
    idx = np.argmax(metrics["f1"])
    return metrics["thresholds"][idx]


def select_threshold_by_recall(metrics, min_recall: float):
    """
    Select the lowest threshold achieving at least `min_recall`.
    """
    valid = np.where(metrics["recall"] >= min_recall)[0]
    if len(valid) == 0:
        raise ValueError("No threshold satisfies recall constraint")

    return metrics["thresholds"][valid[-1]]


def select_threshold_by_precision(metrics, min_precision: float):
    """
    Select the highest threshold achieving at least `min_precision`.
    """
    valid = np.where(metrics["precision"] >= min_precision)[0]
    if len(valid) == 0:
        raise ValueError("No threshold satisfies precision constraint")

    return metrics["thresholds"][valid[0]]


def evaluate_at_threshold(y_true, y_proba, threshold: float):
    """
    Evaluate classification metrics at a fixed probability threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def build_final_model_artifact(
    model,
    threshold: float,
    val_metrics: dict,
    test_metrics: dict,
):
    """
    Bundle model, threshold, and evaluation metrics into a deployable artifact.
    """
    return {
        "model": model,
        "threshold": float(threshold),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
