import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "fraud-detector"
MODEL_STAGE = "latest"

APP_TITLE = "Fraud Detection API"
APP_VERSION = "1.0.0"