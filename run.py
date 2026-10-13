#!/usr/bin/env python3
"""Plan or execute one isolated, explicitly selected experiment."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_sources():
    for item in read_json(ROOT / "provenance/source_snapshot.json")["files"]:
        path = ROOT / item["file"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Source snapshot changed: {item['file']}")


def validate_data(csv_path, images, spec, exploratory=False):
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "categories"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires image and categories columns")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV must not be empty")
    seen = set()
    for row in rows:
        labels = ast.literal_eval(row["categories"])
        if not isinstance(labels, list) or not labels or labels[0] not in spec["labels_observed_in_csv_order"]:
            raise ValueError("Unexpected category: keep the profile label vocabulary")
        image = row["image"]
        if not image or image in seen:
            raise ValueError("Empty or duplicate image path")
        seen.add(image)
        path = Path(image)
        path = path if path.is_absolute() else images / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")
    digest = sha256(csv_path)
    if not exploratory and (len(rows) != spec["n"] or digest != spec["csv_sha256"]):
        raise ValueError("CSV differs from the recorded evaluation list. Obtain the exact list, or use --allow-different-data for a separately labelled experiment.")
    return len(rows), digest


def commands(profile, model, data, images, output):
    family = profile["family"]
    code = ROOT / "code"
    if family == "skinir":
        cmd = [sys.executable, "-u", str(code / "ZS_classification.py"),
               "--experiment", profile["experiment"], "--model_path", str(model),
               "--model-name", "Qwen2.5-VL-7B-Instruct", "--dataframe", str(data),
               "--image-folder", str(images), "--result-path", str(output),
               "--batch-size", str(profile["batch_size"]), "--num_beams", "1", "--max-new-tokens", "32",
               "--random-seed", str(profile["seed"]), "--method", profile["method"]]
        if profile["method"] != "base":
            cmd += ["--starting-layer", str(profile["starting_layer"]), "--ending-layer", str(profile["ending_layer"]),
                    "--entropy-threshold", str(profile["entropy_threshold"]), "--retracing-ratio", str(profile["retracing_ratio"]),
                    "--injection-mode", "ffn", "--trigger-strategy", "entropy", "--retrace-delay-layers", "1"]
            if profile["target_layers"]:
                cmd += ["--retrace-target-layers", profile["target_layers"]]
        return code, [cmd]
    cmd = [sys.executable, "-u", str(ROOT / "tools/replay_engineering.py"),
           "--model", str(model), "--csv", str(data), "--images", str(images), "--output", str(output),
           "--method", profile["method"], "--threshold", str(profile["threshold"])]
    return code, [cmd]


def main():
    profiles = read_json(ROOT / "configs/profiles.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--profile", choices=sorted(profiles))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--execute", action="store_true", help="Without this flag, validate and print only")
    parser.add_argument("--allow-different-data", action="store_true", help="Smoke/new data only; no historical score claim")
    args = parser.parse_args()
    if args.list:
        print("\n".join(sorted(profiles)))
        return 0
    for name in ("profile", "model", "csv", "images", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required")
    if not args.gpu.isdigit():
        parser.error("--gpu must be one physical GPU index")
    args.model, args.csv, args.images, args.output = [p.expanduser().resolve() for p in (args.model, args.csv, args.images, args.output)]
    if not args.model.is_dir() or not args.images.is_dir():
        raise ValueError("Model and image roots must be existing local directories")
    if args.output.exists():
        raise FileExistsError("Output must be a new directory; resume/overwrite is disabled")
    if args.output.is_relative_to(ROOT) and not args.output.is_relative_to(ROOT / "runs"):
        raise ValueError("Inside this package, outputs are permitted only under ignored runs/")
    profile = profiles[args.profile]
    spec = read_json(ROOT / "configs/datasets.json")[profile["dataset"]]
    verify_sources()
    count, digest = validate_data(args.csv, args.images, spec, args.allow_different_data)
    code, calls = commands(profile, args.model, args.csv, args.images, args.output)
    plan = dict(profile=args.profile, config=profile, n=count, csv_sha256=digest,
                exploratory_data=args.allow_different_data, commands=calls,
                source_manifest_sha256=sha256(ROOT / "provenance/source_snapshot.json"),
                source_status="Current SkinIR source; historical full-score equivalence has not been established.")
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    usage = {line.split(",")[0].strip(): int(line.split(",")[1]) for line in raw.splitlines()}
    if args.gpu not in usage or usage[args.gpu] >= 1000:
        raise RuntimeError("GPU invalid or busy; no experiment started")
    args.output.mkdir(parents=True)
    (args.output / "invocation.json").write_text(json.dumps(plan, indent=2) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", FORCE_TRANSFORMERS550="0",
               PYTHONPATH=str(code) + os.pathsep + str(code / "experiments"))
    rc = 0
    try:
        with (args.output / "run.log").open("w") as log:
            for command in calls:
                rc = subprocess.call(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
                if rc:
                    break
    except BaseException:
        (args.output / "exit_code.txt").write_text("1\n")
        raise
    (args.output / "exit_code.txt").write_text(str(rc) + "\n")
    print(f"exit_code={rc}; see {args.output / 'run.log'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
