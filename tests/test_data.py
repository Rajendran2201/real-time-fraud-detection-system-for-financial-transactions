import pytest 
import pandas as pd
from pathlib import Path 
import numpy as np

from src.data.load import load_credit_card_data
from src.data.preprocess import scale_and_persist

DATA_DIR = Path('../data/processed')

@pytest.mark.parameterize("split", ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"])
def test_data_loading(split):
  """
  Check if processed parquet files can be loaded and are non-empty.
  """
  file_path = DATA_DIR / f"{split}.parquet"
  df = load_credit_card_data(file_path)

  assert not df.empty, f"{split} is empty"
  assert isinstance(df, pd.DataFrame), f"{split} is not a DataFrame"


SCALER_DIR = Path('../models/')
def test_scaling_consistency():
  """
  Test if scaling preserves shape and column names.
  Ensures that after scaling:
    - Shapes of train/val/test sets remain the same
    - Column names are preserved
    - Values are numeric
  """
  X_train   = pd.read_parquet(DATA_DIR / "X_train.parquet")
  X_val   = pd.read_parquet(DATA_DIR / "X_val.parquet")
  X_test  = pd.read_parquet(DATA_DIR / "X_test.parquet")
  scaler_path = Path(SCALER_DIR + "scaler.joblib")  
  X_train_scaled, X_val_scaled, X_test_scaled = scale_and_persist(X_train, X_val, X_test, scaler_path)

  # check shapes 
  assert X_train_scaled.shape == X_train.shape, "Train shape changed after scaling"
  assert X_val_scaled.shape == X_val.shape, "Validation shape changed after scaling"
  assert X_test_scaled.shape == X_test.shape, "Test shape changed after scaling"

  # check columns 
  assert list(X_train_scaled.columns) == list(X_train.columns), "Train columns changed after scaling"
  assert list(X_val_scaled.columns) == list(X_val.columns), "Validation columns changed after scaling"
  assert list(X_test_scaled.columns) == list(X_test.columns), "Test columns changed after scaling"

  # check numeric values 
  assert np.issubdtype(X_train_scaled.dtypes.values[0], np.number), "Non-numeric values in the training data"
  assert np.issubdtype(X_val_scaled.dtypes.values[0], np.number), "Non-numeric values in the validation data"
  assert np.isubdtype(X_test_scaled.dtypes.values[0], np.number), "Non-numeric values in the test data"

  



  