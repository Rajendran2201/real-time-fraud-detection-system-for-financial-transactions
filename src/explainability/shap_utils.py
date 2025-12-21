import shap
import numpy as np


def compute_shap_values(model, X):
    """
    Compute SHAP values for tree-based models.

    Parameters
    ----------
    model : trained tree-based model
        Model compatible with SHAP TreeExplainer.
    X : pandas.DataFrame
        Input data used for explanation.

    Returns
    -------
    explainer : shap.TreeExplainer
        Fitted SHAP explainer.
    shap_values : np.ndarray
        SHAP values for all samples and features.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return explainer, shap_values


def save_shap_values(shap_values, output_path):
    """
    Save SHAP values to disk for later analysis or audit.

    Parameters
    ----------
    shap_values : np.ndarray
        Computed SHAP values.
    output_path : str or Path
        Destination .npy file.
    """
    np.save(output_path, shap_values)
