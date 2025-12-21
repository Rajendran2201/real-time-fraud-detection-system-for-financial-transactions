import pytest 
import pandas as pd
from pathlib import Path 
import numpy as np

from src.data.load import load_credit_card_data
from src.data.preprocess import scale_and_persist

PROJECT_ROOT = Path(__file__).resolve().parent.parent  
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SCALER_DIR = PROJECT_ROOT / "models"

@pytest.mark.parametrize("split", ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"])
def test_data_loading(split):
  """
  Check if processed parquet files can be loaded and are non-empty.
  """
  file_path = DATA_DIR / f"{split}.parquet"
  df = load_credit_card_data(file_path)

  assert not df.empty, f"{split} is empty"
  assert isinstance(df, pd.DataFrame), f"{split} is not a DataFrame"



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
  scaler_path = SCALER_DIR / "scaler.joblib"  
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
  assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train_scaled.dtypes), "Non-numeric values in training data"
  assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_val_scaled.dtypes), "Non-numeric values in validation data"
  assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_test_scaled.dtypes), "Non-numeric values in test data"

  



  