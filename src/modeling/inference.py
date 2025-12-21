import joblib
from pathlib import Path

import pandas as pd


def load_final_model(model_path: Path):
    """
    Load a trained model artifact containing the estimator and decision threshold.

    Parameters
    ----------
    model_path : Path
        Path to the serialized model artifact.

    Returns
    -------
    model : object
        Trained model.
    threshold : float
        Optimized decision threshold.
    """
    artifact = joblib.load(model_path)

    return artifact["model"], artifact["threshold"]


def save_predictions(y_proba, y_pred, output_path):
    """
    Save model predictions to disk.

    Parameters
    ----------
    y_proba : array-like
        Predicted probabilities.
    y_pred : array-like
        Binary predictions.
    output_path : Path
        Destination CSV file.
    """
    df = pd.DataFrame({
        "y_proba": y_proba,
        "y_pred": y_pred,
    })

    df.to_csv(output_path, index=False)
