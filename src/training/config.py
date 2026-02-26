INPUT_FILE = "data/processed/features.parquet"

TARGET_COL = "isFraud"
DROP_COLS = ["TransactionID", "TransactionDT"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "fraud-detection"
MODEL_NAME = "fraud-detector"

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 500,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}