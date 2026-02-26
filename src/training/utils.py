import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from lightgbm import LGBMClassifier


def load_features(path: str, target: str, drop_cols: list):
    df = pd.read_parquet(path)
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop + [target])
    y = df[target]
    return X, y


def split_data(X, y, test_size: float, random_state: int):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_model(X_train, y_train, params: dict) -> LGBMClassifier:
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
        "precision_fraud": report["1"]["precision"],
        "recall_fraud": report["1"]["recall"],
        "f1_fraud": report["1"]["f1-score"],
    }


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )