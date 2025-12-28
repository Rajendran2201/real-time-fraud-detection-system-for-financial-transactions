# src/api/utils.py
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent  # Project root
MODEL_PATH = ROOT / "models" / "final_xgb_with_threshold.joblib"
SCALER_PATH = ROOT / "models" / "scaler.joblib"

def load_artifacts():
    """Load model artifact and scaler, handling different saved formats."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model artifact type: {type(artifact)}")

    # Case 1: It's a dict with model and threshold
    if isinstance(artifact, dict):
        model = artifact.get("model") or artifact.get("clf") or artifact.get("estimator")
        threshold = artifact.get("threshold", 0.5)
        logger.info(f"Extracted model and custom threshold: {threshold}")
    else:
        # Case 2: It's the raw model (no threshold wrapper)
        model = artifact
        threshold = 0.5
        logger.info("Loaded raw model, using default threshold 0.5")

    # Load scaler if exists
    scaler = None
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded successfully")
    else:
        logger.info("No scaler found — assuming model doesn't need scaling")

    return model, float(threshold), scaler


# Load once at import time
try:
    MODEL, THRESHOLD, SCALER = load_artifacts()
    logger.info(f"API artifacts loaded: Model ready | Threshold = {THRESHOLD} | Scaler = {'Yes' if SCALER else 'No'}")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    raise