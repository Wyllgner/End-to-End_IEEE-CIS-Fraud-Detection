INPUT_FILE = "data/processed/clean.parquet"
OUTPUT_FILE = "data/processed/features.parquet"

TARGET_COL = "isFraud"
JOIN_KEY = "TransactionID"

AGG_GROUP_COLS = ["card1", "card2", "addr1"]

AMOUNT_COL = "TransactionAmt"

TIME_COL = "TransactionDT"