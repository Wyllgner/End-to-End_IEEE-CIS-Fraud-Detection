
import pandas as pd
import numpy as np
from config import NULL_THRESHOLD


def load_data(transaction_path: str, identity_path: str, join_key: str) -> pd.DataFrame:
    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    df = transactions.merge(identity, on=join_key, how="left")
    return df


def drop_high_null_cols(df: pd.DataFrame, threshold: float = NULL_THRESHOLD) -> pd.DataFrame:
    null_ratio = df.isnull().mean()
    cols_to_drop = null_ratio[null_ratio > threshold].index.tolist()
    return df.drop(columns=cols_to_drop)


def fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("unknown")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes
    return df


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include="int64").columns:
        df[col] = df[col].astype(np.int32)
    return df