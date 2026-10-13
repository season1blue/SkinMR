#!/usr/bin/env python3
"""Patch16 enhancement experiments 1/2/3.

1. Calibrate the tumor-vs-nontumor margin on a train-derived calibration set.
2. Score the first two verbalizer tokens instead of only the first token.
3. Scan Patch16-specific MemVR settings on the existing natural verbalizer.

All outputs are isolated from the existing benchmark directories.
"""

import argparse
import ast
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.memvr_llava import apply_memvr_llava
from utils.eval_help import binary_metrics


VERBALIZERS = {
    "nontumor": "non-tumor tissue",
    "tumor": "tumor tissue",
}


def resolve_image(root, value):
    value = str(value)
    candidates = [value]
    if not os.path.isabs(value):
        norm = value.replace("\\", "/")
        parts = [x for x in norm.split("/") if x]
        candidates.extend([
            os.path.join(root, value),
            os.path.join(root, *parts[1:]) if len(parts) > 1 else "",
            os.path.join(root, os.path.basename(norm)),
        ])
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def build_prompt(tokenizer, conv_mode, domain_prompt):
    names = ", ".join(VERBALIZERS.values())
    if domain_prompt:
        text = (
            "This is a skin histopathology image tile. Determine whether the "
            "tile shows non-tumor tissue or tumor tissue. Choose exactly one "
            f"category from: {names}. The category is:"
        )
    else:
        text = (
            "This is a skin lesion image. Choose exactly one category from: "
            f"{names}. The category is:"
        )
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    return ids, text


def tokenize_variants(tokenizer, text):
    variants = []
    for value in (text, f" {text}"):
        ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if ids:
            variants.append([int(x) for x in ids])
    return variants


def configure_memvr(model, args, layer_start, layer_end, ratio):
    apply_memvr_llava(
        model=model,
        starting_layer=layer_start,
        ending_layer=layer_end,
        entropy_threshold=args.entropy_threshold,
        retracing_ratio=ratio,
        retrace_delay_layers=1,
        retrace_target_layers="",
        method="memvr",
        state_drift_threshold=0.5,
        state_drift_pooling="mean",
        skinir_plus=False,
        injection_mode="ffn",
    )


def forward_logits(model, image_processor, model_config, input_ids, image, device):
    image_inputs = process_images([image], image_processor, model_config)
    if isinstance(image_inputs, list):
        image_inputs = [x.to(device=device, dtype=torch.float16) for x in image_inputs]
    else:
        image_inputs = image_inputs.to(device=device, dtype=torch.float16)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids.to(device),
            images=image_inputs,
            use_cache=False,
        )
    return output.logits[0, -1, :].float()


def score_first_token(logprobs, token_variants):
    return max(float(logprobs[ids[0]]) for ids in token_variants if ids)


def score_two_tokens(model, image_processor, model_config, prefix_ids, image,
                     device, token_variants):
    image_inputs = process_images([image], image_processor, model_config)
    if isinstance(image_inputs, list):
        image_inputs = [x.to(device=device, dtype=torch.float16) for x in image_inputs]
    else:
        image_inputs = image_inputs.to(device=device, dtype=torch.float16)
    with torch.inference_mode():
        first = model(
            input_ids=prefix_ids.to(device), images=image_inputs, use_cache=False
        )
        lp1 = torch.log_softmax(first.logits[0, -1, :].float(), dim=-1)
        scores = {}
        for label, variants in token_variants.items():
            best = None
            for ids in variants:
                value = lp1[ids[0]]
                if len(ids) > 1:
                    appended = torch.cat([
                        prefix_ids.to(device),
                        torch.tensor([[ids[0]]], dtype=torch.long, device=device),
                    ], dim=1)
                    second = model(
                        input_ids=appended,
                        images=image_inputs,
                        use_cache=False,
                    )
                    lp2 = torch.log_softmax(second.logits[0, -1, :].float(), dim=-1)
                    value = value + lp2[ids[1]]
                best = value if best is None else torch.maximum(best, value)
            scores[label] = float(best)
    return scores


def load_model(args, method, layer_start=None, layer_end=None, ratio=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "LlavaMistralForCausalLM", False, False, device=device
    )
    if method == "memvr":
        configure_memvr(model, args, layer_start, layer_end, ratio)
    model.eval()
    return tokenizer, model, image_processor, device


def load_rows(args):
    df = pd.read_csv(args.csv)
    rows = []
    for _, row in df.iterrows():
        image_path = resolve_image(args.image_root, row["image"])
        if image_path is None:
            continue
        category = ast.literal_eval(row["categories"])[0]
        rows.append({
            "image": row["image"],
            "image_path": image_path,
            "label": int(category == "tumor"),
        })
    return rows


def load_training_rows(args):
    df = pd.read_csv(args.train_csv)
    if len(df) > args.calibration_samples:
        # Stratify by binary tumor status, retaining the concrete train split
        # class frequencies while keeping the calibration pass manageable.
        df = df.assign(
            _binary=df["categories"].map(
                lambda value: int(ast.literal_eval(value)[0].startswith("tumor"))
            )
        ).groupby("_binary", group_keys=False).apply(
            lambda group: group.sample(
                n=min(len(group), max(1, int(args.calibration_samples * len(group) / len(df)))),
                random_state=args.seed,
            ),
            include_groups=False,
        ).reset_index(drop=True)
    rows = []
    for _, row in df.iterrows():
        image_path = resolve_image(args.image_root, row["image"])
        if image_path is None:
            continue
        cls = ast.literal_eval(row["categories"])[0]
        rows.append({
            "image": row["image"],
            "image_path": image_path,
            "label": int(cls.startswith("tumor")),
        })
    return rows


def metrics_from_labels(labels, preds):
    return binary_metrics(labels, preds, label_set=[0, 1])


def run_scores(args, mode, method="base", layer_start=None, layer_end=None,
               ratio=None, rows=None):
    tokenizer, model, image_processor, device = load_model(
        args, method, layer_start, layer_end, ratio
    )
    prefix_ids, question = build_prompt(tokenizer, args.conv_mode, args.domain_prompt)
    variants = {
        label: tokenize_variants(tokenizer, text)
        for label, text in VERBALIZERS.items()
    }
    rows = rows if rows is not None else load_rows(args)
    output = []
    for index, row in enumerate(rows, 1):
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        if mode == "two_token":
            scores = score_two_tokens(
                model, image_processor, model.config, prefix_ids, image, device, variants
            )
        else:
            logits = forward_logits(
                model, image_processor, model.config, prefix_ids, image, device
            )
            logprobs = torch.log_softmax(logits, dim=-1)
            scores = {
                label: score_first_token(logprobs, values)
                for label, values in variants.items()
            }
        margin = scores["tumor"] - scores["nontumor"]
        output.append({
            "image": row["image"],
            "ground_truth_label": row["label"],
            "nontumor_score": scores["nontumor"],
            "tumor_score": scores["tumor"],
            "margin": margin,
            "question": question,
        })
        if index % args.progress_every == 0:
            print(f"[{mode}/{method}] {index}/{len(rows)}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(output)


def evaluate_margin(df, threshold):
    labels = df.ground_truth_label.to_numpy()
    preds = (df.margin.to_numpy() > threshold).astype(int)
    result = metrics_from_labels(labels, preds)
    result.update({
        "threshold": float(threshold),
        "n_samples": int(len(df)),
        "pred_tumor": int(preds.sum()),
    })
    return result


def split_calibration(rows, fraction, seed):
    rng = np.random.default_rng(seed)
    labels = np.array([r["label"] for r in rows])
    indices = np.arange(len(rows))
    calibration = []
    evaluation = []
    for label in (0, 1):
        group = indices[labels == label]
        rng.shuffle(group)
        cut = max(1, int(len(group) * fraction))
        calibration.extend(group[:cut].tolist())
        evaluation.extend(group[cut:].tolist())
    return calibration, evaluation


def run_calibration(args, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    train_score_path = outdir / "train_calibration_scores.csv"
    test_score_path = outdir / "test_margin_scores.csv"
    if train_score_path.exists():
        train_scores = pd.read_csv(train_score_path)
    else:
        train_scores = run_scores(args, "first_token", rows=load_training_rows(args))
        train_scores.to_csv(train_score_path, index=False)
    if test_score_path.exists():
        test_scores = pd.read_csv(test_score_path)
    else:
        test_scores = run_scores(args, "first_token", rows=load_rows(args))
        test_scores.to_csv(test_score_path, index=False)

    # Choose threshold using only train-derived calibration data. The fixed
    # test subset is evaluated once with the selected threshold.
    thresholds = np.unique(np.quantile(train_scores.margin, np.linspace(0, 1, 201)))
    sweep = []
    for threshold in thresholds:
        item = evaluate_margin(train_scores, threshold)
        sweep.append(item)
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(outdir / "train_margin_threshold_sweep.csv", index=False)
    chosen_row = sweep_df.sort_values(
        ["balanced_acc", "f1_macro", "accuracy"], ascending=False
    ).iloc[0]
    chosen = float(chosen_row["threshold"])
    test_result = evaluate_margin(test_scores, chosen)
    test_result.update({
        "selection": "train_split_balanced_accuracy",
        "train_calibration_n": int(len(train_scores)),
        "test_n": int(len(test_scores)),
        "train_selected_balanced_acc": float(chosen_row["balanced_acc"]),
        "train_selected_f1_macro": float(chosen_row["f1_macro"]),
    })
    with open(outdir / "calibrated_margin_result.json", "w") as handle:
        json.dump(test_result, handle, indent=2)
    return test_result


def run_memvr_scan(args, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    configs = [("base", None, None, 0.0)] + [
        (f"L{start}_{end}_rr{str(ratio).replace('.', '')}", start, end, ratio)
        for start, end, ratio in ((2, 8, 0.1), (5, 12, 0.1), (5, 16, 0.3), (8, 16, 0.3))
    ]
    rows = []
    for name, start, end, ratio in configs:
        method = "base" if name == "base" else "memvr"
        scores_path = outdir / f"{name}_scores.csv"
        if scores_path.exists():
            scores = pd.read_csv(scores_path)
            print(f"[memvr] reuse cached scores: {name}", flush=True)
        else:
            scores = run_scores(args, "first_token", method, start, end, ratio)
            scores.to_csv(scores_path, index=False)
        # Keep the default decision threshold at zero for the matched score
        # comparison; all configs use the same natural verbalizer.
        result = evaluate_margin(scores, 0.0)
        result.update({"config": name, "method": method})
        rows.append(result)
        pd.DataFrame(rows).to_csv(outdir / "memvr_scan_comparison.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--csv", default="Dataframe/test/classification/Patch16_2class_test_10pct_seed42.csv")
    parser.add_argument("--train-csv", default="Dataframe/test/classification/Patch16_train.csv")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--outdir", default="result/skinvl_patch16_enhance123_20261006")
    parser.add_argument("--conv-mode", default="mistral_instruct")
    parser.add_argument("--domain-prompt", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--calibration-samples", type=int, default=2000)
    parser.add_argument("--entropy-threshold", type=float, default=0.60)
    parser.add_argument("--only", choices=["all", "margin", "two_token", "memvr"], default="all")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.only in ("all", "margin"):
        print("[1] margin calibration", flush=True)
        run_calibration(args, outdir / "01_margin")
    if args.only in ("all", "two_token"):
        print("[2] two-token scoring", flush=True)
        two = run_scores(args, "two_token")
        two.to_csv(outdir / "02_two_token_scores.csv", index=False)
        two_result = evaluate_margin(two, 0.0)
        with open(outdir / "02_two_token_result.json", "w") as handle:
            json.dump(two_result, handle, indent=2)
    if args.only in ("all", "memvr"):
        print("[3] memvr scan", flush=True)
        run_memvr_scan(args, outdir / "03_memvr_scan")


if __name__ == "__main__":
    main()
