import pytest 
import joblib
import numpy as np
from pathlib import Path 

from src.modeling.inference import load_final_model
from src.modeling.predict import predict_with_threshold
from src.threshold.optimize import select_best_f1_threshold, compute_threshold_metrics

from sklearn.dummy import DummyClassifier

from sklearn.metrics import f1_score

@pytest.fixture 
def synthetic_model(tmp_path):
  """
  Create a summy model artifact saved to disk for testing
  """

  model = DummyClassifier(strategy="most_frequent")
  X_train = np.array([[0], [1]])
  y_train = np.array([0, 1])
  model.fit(X_train, y_train)

  # save the model artifact 
  artifact_path = tmp_path / "dummy_model.joblib"
  joblib.dump({"model": model, "threshold": 0.5}, artifact_path)

  return artifact_path

@pytest.fixture 
def synthetic_data():
  """
  Create synthetic test data
  """
  X_test = np.random.rand(10, 1)
  y_test = np.random.randint(0, 2, size=10)
  y_proba = np.random.rand(10)
  return X_test, y_test, y_proba

def test_load_final_data(synthetic_model):
  """
  Test thatthe final model artifact loads correcltly.
  """

  model, threshold = load_final_model(synthetic_model)
  assert hasattr(model, "predict_proba")
  assert isinstance(threshold, float)



def test_predict_with_threshold(synthetic_model, synthetic_data):
  """
  Test prediction with threshold functionality.
  """
  model, threshold = load_final_model(synthetic_model)
  X_test, y_test, y_proba = synthetic_data

  y_pred, y_pred_proba = predict_with_threshold(model, X_test, threshold)

  # check lengths 
  assert len(y_pred) == len(X_test)
  assert len(y_pred_proba) == len(X_test)

  # check that predictions are binary 
  assert set(np.unique(y_pred)).issubset({0,1})

  # check that probabilities are in [0,1]
  assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1))


def test_compute_threshold_metrics(synthetic_data):
    """
    Test the functionality to compute the threshold metrics.
    """
    _, y_true, y_proba = synthetic_data
    metrics = compute_threshold_metrics(y_true, y_proba)
    
    # Ensure required keys exist
    assert all(k in metrics for k in ["thresholds", "precision", "recall", "f1"])
    # Ensure lengths match
    n_thresh = len(metrics["thresholds"])
    assert len(metrics["precision"]) == n_thresh
    assert len(metrics["recall"]) == n_thresh
    assert len(metrics["f1"]) == n_thresh


def test_select_best_f1_threshold(synthetic_data):
  """
  Test the functionality to select the best F1 threshold.
  """
  _, y_true, y_proba = synthetic_data
  metrics = compute_threshold_metrics(y_true, y_proba)
  best_thresh = select_best_f1_threshold(metrics)
  

  # check that the threshold is in [0,1]
  assert 0 <= best_thresh <= 1

  # check F1 at selected threhsold 
  y_pred = (y_proba >= best_thresh).astype(int)
  f1 = f1_score(y_true, y_pred)
  
  # check f1 is in [0,1]
  assert 0 <= f1 <= 1