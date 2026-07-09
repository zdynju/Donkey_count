#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate counting error from CSV files.")
    parser.add_argument("--pred", required=True, help="Predicted CSV with frame_index,count columns.")
    parser.add_argument("--truth", required=True, help="Ground-truth CSV with frame_index,count columns.")
    return parser.parse_args()


def read_counts(path: str | Path) -> dict[str, int]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return {row["frame_index"]: int(row["count"]) for row in csv.DictReader(file)}


def main() -> None:
    args = parse_args()
    pred = read_counts(args.pred)
    truth = read_counts(args.truth)
    common_keys = sorted(set(pred) & set(truth))
    if not common_keys:
        raise ValueError("No overlapping frame_index values between prediction and truth CSV files.")

    errors = [pred[key] - truth[key] for key in common_keys]
    abs_errors = [abs(error) for error in errors]
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    within_1 = sum(1 for error in abs_errors if error <= 1) / len(abs_errors)
    within_2 = sum(1 for error in abs_errors if error <= 2) / len(abs_errors)

    print(f"frames={len(common_keys)}")
    print(f"mae={mae:.4f}")
    print(f"rmse={rmse:.4f}")
    print(f"within_1={within_1:.4f}")
    print(f"within_2={within_2:.4f}")


if __name__ == "__main__":
    main()

