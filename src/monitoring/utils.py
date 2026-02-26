import pandas as pd
import numpy as np


def load_reference(path: str, sample: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)


def load_production(transaction_path: str, identity_path: str, join_key: str, sample: int) -> pd.DataFrame:
    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    df = transactions.merge(identity, on=join_key, how="left")
    return df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)


def encode_production(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.Categorical(df[col]).codes.astype(np.int16)
    return df


def fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def align_columns(reference: pd.DataFrame, production: pd.DataFrame) -> tuple:
    common_cols = [c for c in reference.columns if c in production.columns]
    return reference[common_cols], production[common_cols]