import joblib
from pathlib import Path

import numpy as np


def save_model(model, path: Path):
    """
    Persist a trained model to disk.

    Parameters
    ----------
    model : object
        Trained model instance.
    path : pathlib.Path
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    """
    Load a persisted model from disk.
    """
    return joblib.load(path)


def predict_with_threshold(model, X, threshold: float):
    """
    Generate probability and class predictions using a fixed threshold.

    Parameters
    ----------
    model : trained classifier
        Model exposing `predict_proba`.
    X : pandas.DataFrame or numpy.ndarray
        Input feature matrix.
    threshold : float
        Probability cutoff for classification.

    Returns
    -------
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    y_pred : np.ndarray
        Binary predictions after thresholding.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return y_proba, y_pred
