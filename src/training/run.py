
import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

sys.path.append(os.path.dirname(__file__))

from config import (
    INPUT_FILE, TARGET_COL, DROP_COLS, TEST_SIZE, RANDOM_STATE,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, MODEL_NAME, LGBM_PARAMS,
)
from utils import load_features, split_data, train_model, evaluate, get_feature_importance


def run():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("loading features...")
    X, y = load_features(INPUT_FILE, TARGET_COL, DROP_COLS)
    print(f"  X shape: {X.shape} | fraud rate: {y.mean():.4f}")

    print("splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)

    with mlflow.start_run() as run:
        print("training LightGBM...")
        model = train_model(X_train, y_train, LGBM_PARAMS)

        print("evaluating...")
        metrics = evaluate(model, X_test, y_test)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_metrics(metrics)

        importance = get_feature_importance(model, X_train.columns.tolist())
        for _, row in importance.head(10).iterrows():
            mlflow.log_param(f"top_feat_{row['feature']}", round(float(row["importance"]), 2))

        print("registering model...")
        signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        print(f"run id: {run.info.run_id}")
        print(f"model uri: {model_info.model_uri}")
        print(f"model registered as: {MODEL_NAME}")


if __name__ == "__main__":
    run()
