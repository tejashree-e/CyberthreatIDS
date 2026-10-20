# src/capture/validate_on_wireshark.py
"""
Runs every flow in a wireshark_features.csv through the full Orchestrator
(Detection + Knowledge + Explanation agents) and saves the predictions —
this is how you validate the system on real VIT WiFi traffic instead of
just CICIDS-2017.

Usage:
    python -m src.capture.validate_on_wireshark \
        data/processed/wireshark_features.csv \
        --out data/processed/wireshark_predictions.csv
"""

import argparse
import pandas as pd
from src.agents.orchestrator import Orchestrator


def validate(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    orch = Orchestrator()

    results = []
    for _, row in df.iterrows():
        flow_dict = row.to_dict()
        result = orch.process_flow(flow_dict)
        results.append({
            "label": result["label"],
            "confidence": result["confidence"],
            "risk_level": result["risk_level"],
            "explanation": result["explanation"],
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)

    print(f"\nProcessed {len(out_df)} flows from {csv_path}")
    print("\nLabel distribution:")
    print(out_df["label"].value_counts())
    print(f"\nSaved predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to wireshark_features.csv")
    parser.add_argument("--out", default="data/processed/wireshark_predictions.csv")
    args = parser.parse_args()
    validate(args.csv_path, args.out)