# src/api/main.py

import sys
import os

sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from config import APP_TITLE, APP_VERSION
from schema import TransactionInput, PredictionOutput
from model import load_model, predict


ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading model from mlflow registry...")
    ml_model["instance"] = load_model()
    print("model ready")
    yield
    ml_model.clear()


app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "instance" in ml_model}


@app.post("/predict", response_model=PredictionOutput)
def predict_fraud(transaction: TransactionInput):
    if "instance" not in ml_model:
        raise HTTPException(status_code=503, detail="model not loaded")

    data = transaction.model_dump()
    threshold = data.pop("threshold", 0.5)

    prob = predict(ml_model["instance"], data)

    return PredictionOutput(
        fraud_probability=round(prob, 6),
        is_fraud=prob >= threshold,
        threshold=threshold,
    )