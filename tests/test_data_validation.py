# tests/test_data_validation.py
import pytest
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

@pytest.mark.data
def test_data_shapes():
    paths = [
        "data/processed/X_train.parquet",
        "data/processed/X_val.parquet",
        "data/processed/X_test.parquet",
        "data/processed/y_train.parquet",
        "data/processed/y_val.parquet",
        "data/processed/y_test.parquet",
    ]
    for rel_path in paths:
        path = ROOT / rel_path
        assert path.exists(), f"Missing file: {path}"
        df = pd.read_parquet(path)
        assert len(df) > 0, f"Empty file: {path}"


@pytest.mark.data
def test_fraud_rates():
    for split in ["train", "val", "test"]:
        path = ROOT / f"data/processed/y_{split}.parquet"
        assert path.exists(), f"Missing y_{split}.parquet"
        y = pd.read_parquet(path).squeeze()
        fraud_rate = y.mean()
        assert 0 < fraud_rate < 0.01, f"Unexpected fraud rate in {split}: {fraud_rate:.6f}"  # typical for this dataset