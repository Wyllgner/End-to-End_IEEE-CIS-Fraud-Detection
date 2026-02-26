RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

TRANSACTION_FILE = f"{RAW_DIR}/train_transaction.csv"
IDENTITY_FILE = f"{RAW_DIR}/train_identity.csv"
OUTPUT_FILE = f"{PROCESSED_DIR}/clean.parquet"

JOIN_KEY = "TransactionID"
TARGET_COL = "isFraud"

# columns with too many nulls to keep
NULL_THRESHOLD = 0.5
