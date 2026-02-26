
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from config import MLFLOW_TRACKING_URI, MODEL_NAME


def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"no registered model found with name '{MODEL_NAME}'")

    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    model_uri = f"models:/{MODEL_NAME}/{latest.version}"
    print(f"  loading version {latest.version} from {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)
    return model


def predict(model, data: dict) -> float:
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[:, 1][0]
    return float(prob)
