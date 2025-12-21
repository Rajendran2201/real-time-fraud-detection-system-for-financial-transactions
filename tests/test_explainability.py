import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.explainability.shap_utils import compute_shap_values, save_shap_values


@pytest.fixture
def synthetic_model_and_data():
    """Create a simple RandomForest model and synthetic dataset."""
    X = pd.DataFrame({
        "feature1": np.random.rand(20),
        "feature2": np.random.rand(20),
        "feature3": np.random.rand(20),
    })
    y = np.random.randint(0, 2, size=20)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    return model, X, y


def test_compute_shap_values(synthetic_model_and_data):
    """Test that SHAP values are computed correctly."""
    model, X, _ = synthetic_model_and_data
    explainer, shap_values = compute_shap_values(model, X)

    assert explainer is not None, "SHAP explainer was not created"
    assert isinstance(shap_values, (list, np.ndarray)), "SHAP values have wrong type"
    # For tree-based classifiers, shap_values can be a list (one array per class)
    if isinstance(shap_values, list):
        assert all(arr.shape == X.shape for arr in shap_values), "SHAP array shape mismatch"
    else:
        assert shap_values.shape == X.shape, "SHAP array shape mismatch"


def test_save_shap_values(tmp_path, synthetic_model_and_data):
    """Test that SHAP values can be saved and loaded."""
    model, X, _ = synthetic_model_and_data
    _, shap_values = compute_shap_values(model, X)

    output_file = tmp_path / "shap_values.npy"
    save_shap_values(shap_values, output_file)

    loaded_values = np.load(output_file, allow_pickle=True)
    if isinstance(shap_values, list):
        # Convert list to array for comparison
        shap_values_array = np.array(shap_values, dtype=object)
        assert np.array_equal(shap_values_array, loaded_values)
    else:
        assert np.array_equal(shap_values, loaded_values)
