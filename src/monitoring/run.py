
import os
import sys

sys.path.append(os.path.dirname(__file__))

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

from config import (
    TRAIN_DATA_PATH,
    PROD_DATA_PATH,
    PROD_IDENTITY_PATH,
    REPORT_OUTPUT_PATH,
    JOIN_KEY,
    REFERENCE_SAMPLE,
    PRODUCTION_SAMPLE,
)
from utils import load_reference, load_production, encode_production, fill_nulls, align_columns


def run():
    print("loading reference data (train)...")
    reference = load_reference(TRAIN_DATA_PATH, REFERENCE_SAMPLE)
    print(f"  shape: {reference.shape}")

    print("loading production data (test)...")
    production = load_production(PROD_DATA_PATH, PROD_IDENTITY_PATH, JOIN_KEY, PRODUCTION_SAMPLE)
    production = encode_production(production)
    production = fill_nulls(production)
    print(f"  shape: {production.shape}")

    print("aligning columns...")
    reference, production = align_columns(reference, production)
    print(f"  common columns: {reference.shape[1]}")

    print("generating drift report...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(reference_data=reference, current_data=production)

    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    report.save_html(REPORT_OUTPUT_PATH)
    print(f"report saved to {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    run()
