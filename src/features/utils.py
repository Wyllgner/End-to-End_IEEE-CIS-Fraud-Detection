import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df["hour"] = (df[time_col] // 3600) % 24
    df["day"] = (df[time_col] // (3600 * 24)) % 7
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(np.int8)
    return df


def add_amount_features(df: pd.DataFrame, amount_col: str) -> pd.DataFrame:
    df["amt_log"] = np.log1p(df[amount_col])
    df["amt_cents"] = (df[amount_col] % 1 * 100).round().astype(np.int16)
    df["is_round_amount"] = (df[amount_col] % 1 == 0).astype(np.int8)
    return df


def add_aggregation_features(df: pd.DataFrame, group_cols: list, amount_col: str) -> pd.DataFrame:
    for col in group_cols:
        if col not in df.columns:
            continue

        group = df.groupby(col)[amount_col]

        df[f"{col}_amt_mean"] = group.transform("mean").astype(np.float32)
        df[f"{col}_amt_std"] = group.transform("std").fillna(0).astype(np.float32)
        df[f"{col}_amt_max"] = group.transform("max").astype(np.float32)
        df[f"{col}_count"] = group.transform("count").astype(np.int32)

        df[f"{col}_amt_diff"] = (
            (df[amount_col] - df[f"{col}_amt_mean"]) / (df[f"{col}_amt_std"] + 1e-5)
        ).astype(np.float32)

    return df


def add_email_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col not in df.columns:
            continue
        df[f"{col}_domain"] = df[col] % 1000

    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        df["same_email"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(np.int8)

    return df