import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
)


def evaluate_binary_classifier(
    model,
    X,
    y,
    return_curves: bool = False,
):
    """
    Evaluate a binary classifier using ROC-AUC and PR-AUC.

    Parameters
    ----------
    model : object
        Trained model implementing `predict` and `predict_proba`.
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix.
    y : pandas.Series or numpy.ndarray
        True labels.
    return_curves : bool, optional
        Whether to return precision-recall curve points.

    Returns
    -------
    metrics : dict
        Dictionary containing ROC-AUC, PR-AUC, and optionally
        precision-recall curve arrays.
    """
    y_proba = model.predict_proba(X)[:, 1]

    roc = roc_auc_score(y, y_proba)
    precision, recall, _ = precision_recall_curve(y, y_proba)
    pr_auc = auc(recall, precision)

    result = {
        "roc_auc": roc,
        "pr_auc": pr_auc,
    }

    if return_curves:
        result["precision"] = precision
        result["recall"] = recall

    return result
