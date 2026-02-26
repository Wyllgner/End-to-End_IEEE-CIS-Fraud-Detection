
import os
import sys
import time
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from config import TRANSACTION_FILE, IDENTITY_FILE, OUTPUT_FILE, JOIN_KEY, TARGET_COL
from utils import (
    load_data,
    drop_high_null_cols,
    fill_nulls,
    encode_categoricals,
    reduce_memory,
)


def run():
    start = time.time()

    print("loading raw data...")
    df = load_data(TRANSACTION_FILE, IDENTITY_FILE, JOIN_KEY)
    print(f"  shape after merge: {df.shape}")

    print("dropping high-null columns...")
    df = drop_high_null_cols(df)
    print(f"  shape after dropping: {df.shape}")

    print("filling nulls...")
    df = fill_nulls(df)

    print("encoding categoricals...")
    df = encode_categoricals(df)

    print("reducing memory usage...")
    df = reduce_memory(df)

    fraud_rate = df[TARGET_COL].mean() * 100
    print(f"  fraud rate: {fraud_rate:.2f}%")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    elapsed = time.time() - start
    print(f"saved to {OUTPUT_FILE} ({elapsed:.1f}s)")


if __name__ == "__main__":
    run()