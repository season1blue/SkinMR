#!/usr/bin/env python3
"""Replay the stored SkinVL engineering recipe without a new parameter search."""
import argparse
import json
from pathlib import Path
import random
import runpy
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("base", "memvr"), required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1] / "code"
    sys.path.insert(0, str(code))
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    module = runpy.run_path(str(code / "engineering.py"))
    config = SimpleNamespace(model_path=args.model, csv=args.csv, image_root=args.images,
                             conv_mode="mistral_instruct", domain_prompt=True,
                             entropy_threshold=0.6, progress_every=100)
    if not args.output.is_dir():
        raise ValueError("Use run.py to prepare a new checked output directory")
    rows = module["load_rows"](config)
    import csv
    with Path(args.csv).open(newline="") as stream:
        expected = len(list(csv.DictReader(stream)))
    if len(rows) != expected:
        raise ValueError("Missing images; refusing partial results")
    scores = module["run_scores"](config, "first_token", args.method, 5, 12, 0.1, rows)
    scores.to_csv(args.output / "scores.csv", index=False)
    values = module["evaluate_margin"](scores, args.threshold)
    values.update(method=args.method, threshold_source="historical_train_calibration",
                  code_version="current_snapshot_not_historical_hash_verified")
    (args.output / "metrics.json").write_text(json.dumps(values, indent=2) + "\n")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
