#!/usr/bin/env python3
import argparse
import glob
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge shard predictions and compute metrics")
    parser.add_argument("--exp", required=True, help="Experiment name, e.g. Patch16_2class")
    parser.add_argument("--outpath", required=True, help="Output directory containing shard csv files")
    args = parser.parse_args()

    pattern = os.path.join(args.outpath, f"{args.exp}_predictions_g*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No shard prediction files found: {pattern}")

    merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "image" in merged.columns:
        merged = merged.drop_duplicates(subset=["image"], keep="first")

    merged_path = os.path.join(args.outpath, f"{args.exp}_predictions.csv")
    merged.to_csv(merged_path, index=False)
    print(f"Merged shard predictions -> {merged_path} rows={len(merged)}")

    if "predicted_label" not in merged.columns or "ground_truth_label" not in merged.columns:
        print("Missing predicted_label/ground_truth_label columns; skip final metric export.")
        return 0

    y_pred = pd.to_numeric(merged["predicted_label"], errors="coerce")
    y_true = pd.to_numeric(merged["ground_truth_label"], errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    if not valid.any():
        print("No valid numeric labels found in merged predictions; skip final metric export.")
        return 0

    y_true = y_true[valid].astype(int)
    y_pred = y_pred[valid].astype(int)

    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    is_binary = len(classes) <= 2

    res = {
        "n": int(len(y_true)),
        "num_classes": int(len(classes)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Keep binary fields for backward compatibility with existing result consumers.
    if is_binary:
        res["precision"] = float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
        res["recall"] = float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
        res["f1"] = float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
    else:
        res["precision"] = float("nan")
        res["recall"] = float("nan")
        res["f1"] = float("nan")

    metrics_path = os.path.join(args.outpath, f"{args.exp}_results.csv")
    pd.DataFrame([res]).to_csv(metrics_path, index=False)
    print(f"Metrics saved -> {metrics_path}")
    print(
        "Final Metrics | "
        f"n={res['n']} "
        f"classes={res['num_classes']} "
        f"acc={res['acc']:.6f} "
        f"p_macro={res['precision_macro']:.6f} "
        f"r_macro={res['recall_macro']:.6f} "
        f"f1_macro={res['f1_macro']:.6f}"
    )

    if is_binary:
        print(
            "Binary Metrics | "
            f"p={res['precision']:.6f} "
            f"r={res['recall']:.6f} "
            f"f1={res['f1']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())