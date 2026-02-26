
TRAIN_DATA_PATH = "data/processed/features.parquet"
PROD_DATA_PATH = "data/raw/test_transaction.csv"
PROD_IDENTITY_PATH = "data/raw/test_identity.csv"

REPORT_OUTPUT_PATH = "reports/drift_report.html"

TARGET_COL = "isFraud"
JOIN_KEY = "TransactionID"

REFERENCE_SAMPLE = 10000
PRODUCTION_SAMPLE = 10000