# src/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np  # Add this
from .schemas import Transaction, PredictionResponse, BatchPredictionResponse
from .utils import MODEL, THRESHOLD, SCALER  # Import the loaded globals
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real-Time Credit Card Fraud Detection API",
    description="Detects fraudulent transactions using XGBoost with optimized threshold",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("API startup complete - ready for predictions")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_threshold": THRESHOLD}

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    try:
        df = pd.DataFrame([transaction.dict()])

        # Apply scaling if scaler exists
        if SCALER is not None:
            df = pd.DataFrame(SCALER.transform(df), columns=df.columns)

        # Predict probability
        if hasattr(MODEL, "predict_proba"):
            prob = float(MODEL.predict_proba(df)[0][1])
        else:
            raise AttributeError("Model does not support predict_proba")

        is_fraud = prob >= THRESHOLD

        return PredictionResponse(
            fraud_probability=round(prob, 4),
            is_fraud=is_fraud
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(transactions: list[Transaction]):
    try:
        if not transactions:
            raise HTTPException(status_code=400, detail="Empty transaction list")

        # Convert list of transactions to DataFrame
        df = pd.DataFrame([t.dict() for t in transactions])

        # Apply scaling if scaler exists
        if SCALER is not None:
            df = pd.DataFrame(SCALER.transform(df), columns=df.columns)

        # Predict probabilities
        if hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(df)[:, 1]
        else:
            raise AttributeError("Model does not support predict_proba")

        # Apply threshold to get binary predictions
        predictions = probs >= THRESHOLD

        # Build response list
        results = [
            PredictionResponse(
                fraud_probability=round(float(prob), 4),
                is_fraud=bool(pred)
            )
            for prob, pred in zip(probs, predictions)
        ]

        return BatchPredictionResponse(predictions=results)

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")