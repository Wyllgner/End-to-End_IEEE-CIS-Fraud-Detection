import os
import sys
import time
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from config import INPUT_FILE, OUTPUT_FILE, TARGET_COL, AGG_GROUP_COLS, AMOUNT_COL, TIME_COL
from utils import (
    add_time_features,
    add_amount_features,
    add_aggregation_features,
    add_email_features,
)


def run():
    start = time.time()

    print("loading clean data...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"  shape: {df.shape}")

    initial_cols = df.shape[1]

    print("adding time features...")
    df = add_time_features(df, TIME_COL)

    print("adding amount features...")
    df = add_amount_features(df, AMOUNT_COL)

    print("adding aggregation features...")
    df = add_aggregation_features(df, AGG_GROUP_COLS, AMOUNT_COL)

    print("adding email features...")
    df = add_email_features(df)

    new_cols = df.shape[1] - initial_cols
    print(f"  {new_cols} new features created - total: {df.shape[1]}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    elapsed = time.time() - start
    print(f"saved to {OUTPUT_FILE} ({elapsed:.1f}s)")


if __name__ == "__main__":
    run()