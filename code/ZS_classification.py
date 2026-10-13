import os
import argparse
import csv
import json
import math
import pandas as pd
import torch
from PIL import Image
import numpy as np
import ast
import random
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoTokenizer, AutoModel, AutoProcessor
try:
    from transformers import LogitsProcessorList
except Exception:
    LogitsProcessorList = list
from llava.conversation import Conversation
try:
    from llava.memvr_llava import apply_memvr_llava
except Exception:
    apply_memvr_llava = None
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
except Exception:
    Qwen3_5ForConditionalGeneration = None
try:
    from transformers550.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as Qwen3_5ForConditionalGeneration550
except Exception:
    Qwen3_5ForConditionalGeneration550 = None
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None
try:
    from transformers550 import AutoProcessor as AutoProcessor550
except Exception:
    AutoProcessor550 = None
try:
    from transformers550.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VLForConditionalGeneration550
except Exception:
    Qwen2_5_VLForConditionalGeneration550 = None

from utils.eval_help import binary_metrics

import ipdb


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION = os.path.join(BASE_DIR, 'Dataframe', 'test', 'classification')


class DigitOnlyLogitsProcessor:
    """Restrict the first generated token to the numeric class IDs."""

    def __init__(self, tokenizer, prompt_length, num_classes):
        self.prompt_length = int(prompt_length)
        self.num_classes = int(num_classes)
        self.token_to_digit = {}
        allowed = set()
        for digit in range(self.num_classes):
            for text in (str(digit), f" {digit}", f"\n{digit}"):
                try:
                    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                except Exception:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                # SentencePiece tokenizers used by SkinVL may encode a
                # standalone digit as [space, digit]. During generation the
                # useful class decision is the final digit token.
                if ids:
                    token_id = int(ids[-1])
                    allowed.add(token_id)
                    self.token_to_digit[token_id] = digit
        self.allowed_token_ids = sorted(allowed)
        if not self.allowed_token_ids:
            raise RuntimeError("Unable to find numeric class IDs for constrained decoding.")

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] != self.prompt_length:
            return scores
        constrained = torch.full_like(scores, torch.finfo(scores.dtype).min)
        allowed = torch.tensor(self.allowed_token_ids, device=scores.device, dtype=torch.long)
        constrained[:, allowed] = scores[:, allowed]
        return constrained


def _numeric_token_digit_map(tokenizer, num_classes):
    processor = DigitOnlyLogitsProcessor(tokenizer, prompt_length=0, num_classes=num_classes)
    return processor.token_to_digit


def _select_numeric_predictions(step_scores, tokenizer, num_classes, class_priors=None, prior_alpha=0.0):
    """Select a class from first-token logits, optionally correcting class priors."""
    token_to_digit = _numeric_token_digit_map(tokenizer, num_classes)
    result = []
    for row in step_scores:
        digit_scores = [torch.finfo(row.dtype).min] * num_classes
        for token_id, digit in token_to_digit.items():
            if token_id < row.shape[0]:
                value = row[token_id]
                if value > digit_scores[digit]:
                    digit_scores[digit] = value
        score_tensor = torch.stack([
            value if isinstance(value, torch.Tensor) else torch.tensor(value, device=row.device, dtype=row.dtype)
            for value in digit_scores
        ])
        if class_priors is not None and prior_alpha:
            prior = torch.tensor(class_priors, device=row.device, dtype=row.dtype).clamp_min(1e-8)
            score_tensor = score_tensor - float(prior_alpha) * torch.log(prior)
        result.append(int(torch.argmax(score_tensor).item()))
    return result


def _build_candidate_token_map(tokenizer, possible_diseases, candidate_names=None):
    """Map each class to one or more plausible first-token IDs."""
    candidate_names = candidate_names or possible_diseases
    token_map = {}
    for disease, candidate_name in zip(possible_diseases, candidate_names):
        variants = [candidate_name]
        if candidate_name.lower() == "actinic keratoses":
            variants.append("Actinic Keratosis")
        if candidate_name.lower() == "vascular lesions":
            variants.append("Vascular Lesion")
        ids_for_class = set()
        for variant in variants:
            for text in (variant, f" {variant}"):
                try:
                    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                except Exception:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                if ids:
                    ids_for_class.add(int(ids[0]))
        token_map[disease] = sorted(ids_for_class)
    if not any(token_map.values()):
        raise RuntimeError("Unable to build candidate class token map.")
    return token_map


def _select_candidate_predictions(step_logits, tokenizer, possible_diseases,
                                  candidate_names=None, class_priors=None,
                                  prior_alpha=0.0):
    """Select a class from next-token likelihoods for class-name prefixes."""
    token_map = _build_candidate_token_map(
        tokenizer, possible_diseases, candidate_names=candidate_names
    )
    logprobs = torch.log_softmax(step_logits.float(), dim=-1)
    predictions = []
    for row in logprobs:
        scores = []
        for disease in possible_diseases:
            token_ids = token_map[disease]
            if not token_ids:
                scores.append(torch.tensor(torch.finfo(row.dtype).min, device=row.device))
                continue
            scores.append(torch.stack([row[token_id] for token_id in token_ids]).max())
        score_tensor = torch.stack(scores)
        if class_priors is not None and prior_alpha:
            prior = torch.tensor(class_priors, device=row.device, dtype=row.dtype).clamp_min(1e-8)
            score_tensor = score_tensor + float(prior_alpha) * torch.log(prior)
        predictions.append(int(torch.argmax(score_tensor).item()))
    return predictions


def _tokenize_candidate_answer(tokenizer, disease):
    """Tokenize the answer exactly as it follows the LLaVA assistant prefix."""
    try:
        ids = tokenizer(disease, add_special_tokens=False)["input_ids"]
    except Exception:
        ids = tokenizer.encode(disease, add_special_tokens=False)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(token_id) for token_id in ids]
    if not ids:
        raise RuntimeError(f"Unable to tokenize candidate answer: {disease!r}")
    return ids


def _select_candidate_sequence_predictions(prefix_ids, image_inputs, model, tokenizer,
                                            possible_diseases, candidate_names=None,
                                            class_priors=None, prior_alpha=0.0):
    """Score each complete class name with length-normalized teacher forcing."""
    device = prefix_ids.device
    candidate_names = candidate_names or possible_diseases
    candidate_ids = [
        _tokenize_candidate_answer(tokenizer, candidate_name)
        for candidate_name in candidate_names
    ]
    scores = []
    for ids in candidate_ids:
        token_ids = torch.tensor(
            ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        candidate_input_ids = torch.cat(
            [prefix_ids, token_ids], dim=1
        )
        with torch.inference_mode():
            outputs = model(
                input_ids=candidate_input_ids,
                images=image_inputs,
                use_cache=False,
            )
        # LLaVA expands the single image placeholder into many visual tokens.
        # The class answer remains at the sequence tail, so index it relative
        # to the end rather than using the unexpanded text-prefix length.
        start = outputs.logits.shape[1] - len(ids) - 1
        end = outputs.logits.shape[1] - 1
        logprobs = torch.log_softmax(
            outputs.logits[0, start:end, :].float(), dim=-1
        )
        token_logprobs = logprobs.gather(
            -1, token_ids[0].unsqueeze(-1)
        ).squeeze(-1)
        scores.append(token_logprobs.mean())
    score_tensor = torch.stack(scores)
    if class_priors is not None and prior_alpha:
        prior = torch.tensor(class_priors, device=device, dtype=score_tensor.dtype).clamp_min(1e-8)
        score_tensor = score_tensor + float(prior_alpha) * torch.log(prior)
    return [int(torch.argmax(score_tensor).item())]


def _candidate_prompt_variants(possible_diseases, count, candidate_names=None):
    names = ", ".join(candidate_names or possible_diseases)
    variants = [
        (
            f"This is a skin lesion image. Choose exactly one diagnosis from: {names}. "
            "The diagnosis is:"
        ),
        (
            f"Analyze the skin lesion image and select the single best category from: {names}. "
            "Answer with one diagnosis name only:"
        ),
        (
            f"Identify the most likely skin lesion diagnosis. Choose one and only one from "
            f"these categories: {names}. Final diagnosis:"
        ),
    ]
    return variants[:max(1, min(int(count), len(variants)))]


def _candidate_verbalizers(experiment, possible_diseases, mode):
    if mode == "canonical":
        return list(possible_diseases)
    if experiment == "Patch16_2class":
        natural = {
            "nontumor": "non-tumor tissue",
            "tumor": "tumor tissue",
        }
        return [natural.get(name, name) for name in possible_diseases]
    if experiment == "PAD":
        natural = {
            "Actinic Keratosis": "actinic keratosis",
            "Basal Cell Carcinoma": "basal cell carcinoma",
            "Melanoma": "melanoma",
            "Nevus": "nevus",
            "Seborrheic Keratosis": "seborrheic keratosis",
            "Squamous Cell Carcinoma": "squamous cell carcinoma",
        }
        return [natural.get(name, name) for name in possible_diseases]
    return list(possible_diseases)


# ---------------------------------------------------------------------------
# SkinIR modality-specific inference upgrades (2026-10-08)
# Motivated by the three-modality failure analysis:
#   dermoscopy  (HAM10000)  -> lesion core carries the signal; add a zoom view
#   clinical    (PAD)       -> macroscopic context + patient metadata help
#   histopath   (Patch16)   -> labels are inherently hierarchical; decode in
#                              two stages instead of a flat 16-way race
# ---------------------------------------------------------------------------

def _center_crop_pil(image, fraction):
    """SkinIR-Z: center crop retaining the lesion core (dermoscopy zoom view)."""
    width, height = image.size
    crop_w = max(1, int(width * float(fraction)))
    crop_h = max(1, int(height * float(fraction)))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))


def _format_side_info(row):
    """SkinIR-S: compact patient-context line from PAD metadata columns."""
    age = str(row.get("age") or "").strip()
    sex = str(row.get("sex") or "").strip()
    site = str(row.get("anatom_site") or "").strip()
    bits = []
    if age and age.lower() not in ("nan", "unknown", "none"):
        bits.append(f"{age} year old" if age.isdigit() else f"age {age}")
    if sex and sex.lower() not in ("nan", "unknown", "none"):
        sex_lower = sex.lower()
        if sex_lower.startswith("f"):
            bits.append("female")
        elif sex_lower.startswith("m"):
            bits.append("male")
        else:
            bits.append(sex)
    if site and site.lower() not in ("nan", "unknown", "none"):
        bits.append(f"lesion on the {site.lower()}")
    if not bits:
        return ""
    return "Patient context: " + ", ".join(bits) + "."


# Natural-language verbalizers for the Patch16 16-class space. Stage 2 of the
# hierarchical decoder only ever shows one branch (4 tumor / 12 nontumor
# tissue types), so plain tissue names are readable without the coded prefix.
PATCH16_STAGE2_VERBALIZERS = {
    "tumor_skin_epithelial_bcc": "basal cell carcinoma",
    "tumor_skin_epithelial_sqcc": "squamous cell carcinoma",
    "tumor_skin_melanoma_melanoma": "melanoma",
    "tumor_skin_naevus_naevus": "naevus",
    "nontumor_skin_chondraltissue_chondraltissue": "cartilage tissue",
    "nontumor_skin_dermis_dermis": "dermis",
    "nontumor_skin_elastosis_elastosis": "elastosis",
    "nontumor_skin_epidermis_epidermis": "epidermis",
    "nontumor_skin_hairfollicle_hairfollicle": "hair follicle",
    "nontumor_skin_muscle_skeletal": "skeletal muscle",
    "nontumor_skin_necrosis_necrosis": "necrosis",
    "nontumor_skin_nerves_nerves": "nerves",
    "nontumor_skin_sebaceousglands_sebaceousglands": "sebaceous glands",
    "nontumor_skin_subcutis_subcutis": "subcutis",
    "nontumor_skin_sweatglands_sweatglands": "sweat glands",
    "nontumor_skin_vessel_vessel": "blood vessel",
}


def _build_llava_candidate_ids(tokenizer, conv_mode, text, mm_use_im_start_end, device):
    """Build LLaVA candidate-scoring input ids for arbitrary question text."""
    if mm_use_im_start_end:
        question = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN + "\n" + text
        )
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + text
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)


def _hierarchical_patch16_predictions(model, tokenizer, args, possible_diseases,
                                       valid_samples, image_inputs, device):
    """SkinIR-H: two-stage hierarchical decoding for the Patch16 16-class task.

    Stage 1 reuses the validated natural verbalizer (tumor tissue vs
    non-tumor tissue; the calibrated binary pipeline reached ~86% balanced
    accuracy) to select a branch. Stage 2 scores only the surviving branch
    (4 tumor types or 12 nontumor tissues) with domain-phrased verbalizers,
    replacing the flat 16-way race in which general VLMs collapse (~17%).

    Returns int indices into possible_diseases, matching the contract of the
    surrounding candidate-decoding vote block.
    """
    tumor_classes = [d for d in possible_diseases if d.startswith("tumor")]
    nontumor_classes = [d for d in possible_diseases if d.startswith("nontumor")]
    n_samples = len(valid_samples)

    stage1_text = (
        "This is a skin histopathology image tile. Determine whether the tile "
        "shows non-tumor tissue or tumor tissue. Choose exactly one category "
        "from: non-tumor tissue, tumor tissue. The category is:"
    )
    stage1_ids = _build_llava_candidate_ids(
        tokenizer, args.conv_mode, stage1_text,
        model.config.mm_use_im_start_end, device,
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=stage1_ids.repeat(n_samples, 1),
            images=image_inputs,
            use_cache=False,
        )
    branch_preds = _select_candidate_predictions(
        outputs.logits[:, -1, :], tokenizer, ["nontumor", "tumor"],
        candidate_names=["non-tumor tissue", "tumor tissue"],
    )

    tumor_names = [PATCH16_STAGE2_VERBALIZERS.get(d, d) for d in tumor_classes]
    nontumor_names = [PATCH16_STAGE2_VERBALIZERS.get(d, d) for d in nontumor_classes]
    stage2_prompts = {
        "tumor": (
            "This is a skin histopathology image tile that contains tumor "
            "tissue. Identify the tumor type. Choose exactly one from: "
            f"{', '.join(tumor_names)}. The diagnosis is:"
        ),
        "nontumor": (
            "This is a skin histopathology image tile of non-tumor tissue. "
            "Identify the tissue type. Choose exactly one from: "
            f"{', '.join(nontumor_names)}. The tissue is:"
        ),
    }
    stage2_ids_cache = {}

    predictions = []
    for sample_index, branch in enumerate(branch_preds):
        if branch == 1:
            classes, names, key = tumor_classes, tumor_names, "tumor"
        else:
            classes, names, key = nontumor_classes, nontumor_names, "nontumor"
        if key not in stage2_ids_cache:
            stage2_ids_cache[key] = _build_llava_candidate_ids(
                tokenizer, args.conv_mode, stage2_prompts[key],
                model.config.mm_use_im_start_end, device,
            )
        ids = stage2_ids_cache[key]
        image_batch = (
            [image_inputs[sample_index]] if isinstance(image_inputs, list)
            else image_inputs[sample_index:sample_index + 1]
        )
        with torch.inference_mode():
            outputs = model(input_ids=ids, images=image_batch, use_cache=False)
        branch_pred = _select_candidate_predictions(
            outputs.logits[:, -1, :], tokenizer, classes,
            candidate_names=names,
        )[0]
        predictions.append(possible_diseases.index(classes[branch_pred]))
    return predictions


def _zoom_pass_predictions(model, tokenizer, args, possible_diseases,
                            candidate_names, valid_samples, images, image_processor,
                            device, class_priors=None, prior_alpha=0.0):
    """SkinIR-Z: score fusion over the full view and a center-cropped view.

    Returns int indices into possible_diseases. Both passes share the same
    candidate prompt; only the visual evidence changes.
    """
    views = {"full": list(images)}
    views["crop"] = [
        _center_crop_pil(image, args.zoom_crop_fraction)
        for image in images
    ]
    candidate_text = _candidate_prompt_variants(
        possible_diseases, args.candidate_ensemble,
        candidate_names=candidate_names,
    )[0]
    candidate_input_ids = _build_llava_candidate_ids(
        tokenizer, args.conv_mode, candidate_text,
        model.config.mm_use_im_start_end, device,
    )
    token_map = _build_candidate_token_map(
        tokenizer, possible_diseases, candidate_names=candidate_names
    )
    fused = None
    n_samples = len(valid_samples)
    for view_name in ("full", "crop"):
        image_inputs = process_images(views[view_name], image_processor, model.config)
        if isinstance(image_inputs, list):
            image_inputs = [img.to(device=device, dtype=torch.float16) for img in image_inputs]
        else:
            image_inputs = image_inputs.to(device=device, dtype=torch.float16)
        with torch.inference_mode():
            outputs = model(
                input_ids=candidate_input_ids.repeat(n_samples, 1),
                images=image_inputs,
                use_cache=False,
            )
        logprobs = torch.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
        scores = []
        for row in logprobs:
            per_class = []
            for disease in possible_diseases:
                token_ids = token_map[disease]
                if not token_ids:
                    per_class.append(torch.tensor(torch.finfo(row.dtype).min, device=row.device))
                    continue
                per_class.append(torch.stack([row[t] for t in token_ids]).max())
            scores.append(torch.stack(per_class))
        score_tensor = torch.stack(scores)
        if class_priors is not None and prior_alpha:
            prior = torch.tensor(class_priors, device=score_tensor.device, dtype=score_tensor.dtype).clamp_min(1e-8)
            score_tensor = score_tensor + float(prior_alpha) * torch.log(prior)
        fused = score_tensor if fused is None else fused + score_tensor
    return [int(idx) for idx in fused.argmax(dim=-1).tolist()]


def _candidate_reference_prior(experiment, possible_diseases, source):
    if source == "none":
        return None
    if experiment == "HAM10000":
        # Official HAM10000 class counts, used as a public reference prior
        # rather than estimating the prior from the evaluated test labels.
        reference = {
            "Actinic Keratoses": 327.0,
            "Basal Cell Carcinoma": 514.0,
            "Benign Keratosis": 1099.0,
            "Dermatofibroma": 115.0,
            "Melanoma": 1113.0,
            "Nevus": 6705.0,
            "Vascular lesions": 142.0,
        }
        values = [reference.get(name, 1.0) for name in possible_diseases]
        total = sum(values)
        return [value / total for value in values]
    if experiment == "PAD":
        reference = {
            "Basal Cell Carcinoma": 0.372,
            "Actinic Keratosis": 0.286,
            "Seborrheic Keratosis": 0.122,
            "Nevus": 0.111,
            "Melanoma": 0.069,
            "Squamous Cell Carcinoma": 0.040,
        }
        return [reference.get(name, 1e-8) for name in possible_diseases]
    if experiment == "Patch16_2class":
        # Aggregated from the Patch16 training split: 59552 nontumor,
        # 29419 tumor out of 88971 samples.
        reference = {"nontumor": 59552.0, "tumor": 29419.0}
        values = [reference.get(name, 1.0) for name in possible_diseases]
        total = sum(values)
        return [value / total for value in values]
    return None


def _candidate_train_prior(experiment, possible_diseases):
    if experiment != "Patch16_2class":
        return None
    train_csv = os.path.join(
        PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "Patch16_train.csv"
    )
    if not os.path.isfile(train_csv):
        return None
    train_df = pd.read_csv(train_csv)
    counts = {name: 0.0 for name in possible_diseases}
    for value in train_df["categories"]:
        try:
            category = process_categories(value)
        except Exception:
            continue
        if category.startswith("nontumor"):
            counts["nontumor"] += 1.0
        elif category.startswith("tumor"):
            counts["tumor"] += 1.0
    total = sum(counts.values())
    if total <= 0:
        return None
    return [counts[name] / total for name in possible_diseases]


def _get_memvr_debug_owner(model):
    candidates = [model]
    nested_model = getattr(model, "model", None)
    if nested_model is not None:
        candidates.append(nested_model)
        language_model = getattr(nested_model, "language_model", None)
        if language_model is not None:
            candidates.append(language_model)
    for candidate in candidates:
        debug = getattr(candidate, "_memvr_debug_info", None)
        if isinstance(debug, dict) and debug:
            return candidate
    return model


def _collect_memvr_debug_row(model):
    debug_owner = _get_memvr_debug_owner(model)
    debug = getattr(debug_owner, "_memvr_debug_info", {})
    if not isinstance(debug, dict):
        debug = {}
    entropy_trace = debug.get("entropy_trace")
    if isinstance(entropy_trace, list):
        entropy_trace_len = len(entropy_trace)
        entropy_trace_json = json.dumps(entropy_trace, ensure_ascii=True)
    else:
        entropy_trace_len = 0
        entropy_trace_json = ""
    return {
        "memvr_runtime_enabled": bool(debug.get("enabled", False)),
        "memvr_has_debug_attr": hasattr(debug_owner, "_memvr_debug_info"),
        "memvr_enabled": debug.get("enabled"),
        "memvr_triggered": debug.get("triggered"),
        "memvr_injection_success": debug.get("injection_success"),
        "memvr_trigger_layer": debug.get("trigger_layer"),
        "memvr_target_layer": debug.get("target_layer"),
        "memvr_trigger_metric": debug.get("trigger_metric"),
        "memvr_trigger_value": debug.get("trigger_value"),
        "memvr_trigger_source": debug.get("trigger_source"),
        "memvr_used_dynamic_visual_token": debug.get("used_dynamic_visual_token"),
        "memvr_entropy_trace_len": entropy_trace_len,
        "memvr_entropy_trace": entropy_trace_json,
        "memvr_last_entropy": getattr(debug_owner, "_memvr_last_entropy", None),
        "memvr_last_target_layer": getattr(debug_owner, "_memvr_last_target_layer", None),
    }
def get_experiment_setting(experiment):
    if experiment == "ISIC":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "ISIC_test.csv"),
                   "task": "classification",
                   "targets": {"Actinic Keratosis": 0, "Basal Cell Carcinoma": 1, "Benign Keratosis-like Lesions": 2,
                               "Dermatofibroma": 3, "Melanoma": 4, "Nevus": 5, "Squamous Cell Carcinoma": 6,
                               "Vascular Lesions": 7}}

    elif experiment == "MSKCC":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "MSKCC_test.csv"),
                   "task": "classification",
                   "targets": {"AIMP": 0, "acrochordon": 1, "actinic keratosis": 2, "angiokeratoma": 3,
                               "atypical melanocytic proliferation": 4, "basal cell carcinoma": 5,
                               "cafe-au-lait macule": 6, "dermatofibroma": 7, "lentigo NOS": 8, "lentigo simplex": 9,
                               "lichenoid keratosis": 10, "melanoma": 11, "neurofibroma": 12, "nevus": 13, "other": 14,
                               "scar": 15, "seborrheic keratosis": 16, "solar lentigo": 17,
                               "squamous cell carcinoma": 18, "vascular lesion": 19, "verruca": 20}}

    elif experiment == "PAD":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "PAD_test.csv"),
                   "task": "classification",
                   "targets": {"Actinic Keratosis": 0, "Basal Cell Carcinoma": 1, "Melanoma": 2, "Nevus": 3,
                               "Seborrheic Keratosis": 4, "Squamous Cell Carcinoma": 5}}

    elif experiment == "HIBA":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "HIBA_test.csv"),
                   "task": "classification",
                   "targets": {"actinic keratosis": 0, "basal cell carcinoma": 1, "dermatofibroma": 2,
                               "lichenoid keratosis": 3, "melanoma": 4, "nevus": 5, "seborrheic keratosis": 6,
                               "solar lentigo": 7, "squamous cell carcinoma": 8, "vascular lesion": 9}}

    elif experiment == "HIBA_2class":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "HIBA_2class_test.csv"),
                   "task": "classification",
                   "targets": {"benign": 0, "malignant": 1}}

    elif experiment == "BCN20000":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "BCN20000_test.csv"),
                   "task": "classification",
                   "targets": {"actinic keratosis": 0, "basal cell carcinoma": 1, "dermatofibroma": 2, "melanoma": 3,
                               "melanoma metastasis": 4, "nevus": 5, "other": 6, "scar": 7, "seborrheic keratosis": 8,
                               "solar lentigo": 9, "squamous cell carcinoma": 10, "vascular lesion": 11}}

    elif experiment == "Fitzpatrick":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "Fitzpatrick_test.csv"),
                   "task": "classification",
                   "targets": {"benign": 0, "malignant": 1, "non-neoplastic": 2}}

    elif experiment == "HAM10000":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "HAM10000_test.csv"),
                   "task": "classification",
                   "targets": {"Actinic Keratoses": 0, "Basal Cell Carcinoma": 1, "Benign Keratosis": 2,
                               "Dermatofibroma": 3, "Melanoma": 4, "Nevus": 5, "Vascular lesions": 6}}

    elif experiment == "Patch16":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "Patch16_test.csv"),
                   "task": "classification",
                   "targets": {
                       "nontumor_skin_chondraltissue_chondraltissue": 0,
                       "nontumor_skin_dermis_dermis": 1,
                       "nontumor_skin_elastosis_elastosis": 2,
                       "nontumor_skin_epidermis_epidermis": 3,
                       "nontumor_skin_hairfollicle_hairfollicle": 4,
                       "nontumor_skin_muscle_skeletal": 5,
                       "nontumor_skin_necrosis_necrosis": 6,
                       "nontumor_skin_nerves_nerves": 7,
                       "nontumor_skin_sebaceousglands_sebaceousglands": 8,
                       "nontumor_skin_subcutis_subcutis": 9,
                       "nontumor_skin_sweatglands_sweatglands": 10,
                       "nontumor_skin_vessel_vessel": 11,
                       "tumor_skin_epithelial_bcc": 12,
                       "tumor_skin_epithelial_sqcc": 13,
                       "tumor_skin_melanoma_melanoma": 14,
                       "tumor_skin_naevus_naevus": 15,
                   }}

    elif experiment == "Dermnet":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "Dermnet_test.csv"),
                   "task": "classification",
                   "targets": {"Acne and rosacea": 0,
                               "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": 1,
                               "Atopic dermatitis": 2, "Bullous disease": 3,
                               "Cellulitis Impetigo and other Bacterial Infections": 4, "Eczema": 5,
                               "Exanthems and Drug Eruptions": 6, "Hair loss  alopecia and other hair diseases": 7,
                               "Herpes hpv and other stds": 8, "Light Diseases and Disorders of Pigmentation": 9,
                               "Lupus and other Connective Tissue diseases": 10,
                               "Melanoma Skin Cancer Nevi and Moles": 11, "Nail Fungus and other Nail Disease": 12,
                               "Poison ivy  and other contact dermatitis": 13,
                               "Psoriasis pictures Lichen Planus and related diseases": 14,
                               "Scabies Lyme Disease and other Infestations and Bites": 15,
                               "Seborrheic Keratoses and other Benign Tumors": 16, "Systemic Disease": 17,
                               "Tinea Ringworm Candidiasis and other Fungal Infections": 18, "Urticaria Hives": 19,
                               "Vascular Tumors": 20, "Vasculitis": 21,
                               "Warts Molluscum and other Viral Infections": 22}}

    elif experiment == "Patch16":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "Patch16_test.csv"),
                   "task": "classification",
                   "targets": {"nontumor skin chondraltissue": 0, "nontumor skin dermis": 1,
                               "nontumor skin elastosis": 2, "nontumor skin epidermis": 3,
                               "nontumor skin hairfollicle": 4, "nontumor skin muscle skeletal": 5,
                               "nontumor skin necrosis": 6, "nontumor skin nerves": 7,
                               "nontumor skin sebaceousglands": 8, "nontumor skin subcutis": 9,
                               "nontumor skin sweatglands": 10, "nontumor skin vessel": 11,
                               "tumor skin epithelial bcc": 12, "tumor skin epithelial sqcc": 13,
                               "tumor skin melanoma": 14, "tumor skin naevus": 15}}
    elif experiment == "Patch16_2class":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "Patch16_2class_test.csv"),
                   "task": "classification",
                   "targets": {"nontumor": 0, "tumor": 1}}

    elif experiment == "DDI":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "DDI_test.csv"),
                   "task": "classification",
                   "targets": {"Abrasions": 0, "Abscess": 1, "Acne Cystic": 2, "Acquired Digital Fibrokeratoma": 3,
                               "Acral Melanotic Macule": 4, "Acrochordon": 5, "Actinic Keratosis": 6,
                               "Angioleiomyoma": 7, "Angioma": 8, "Arteriovenous Hemangioma": 9,
                               "Atypical Spindle Cell Nevus of Reed": 10, "Basal Cell Carcinoma": 11,
                               "Benign Keratosis": 12, "Blastic Plasmacytoid Dendritic Cell Neoplasm": 13,
                               "Blue Nevus": 14, "Cellular Neurothekeoma": 15, "Chondroid Syringoma": 16,
                               "Clear Cell Acanthoma": 17, "Coccidioidomycosis": 18, "Condyloma Acuminatum": 19,
                               "Dermatofibroma": 20, "Dermatomyositis": 21, "Dysplastic Nevus": 22,
                               "Eccrine Poroma": 23, "Eczema": 24, "Epidermal Cyst": 25, "Epidermal Nevus": 26,
                               "Fibrous Papule": 27, "Focal Acral Hyperkeratosis": 28, "Folliculitis": 29,
                               "Foreign Body Granuloma": 30, "Glomangioma": 31, "Graft vs Host Disease": 32,
                               "Hematoma": 33, "Hyperpigmentation": 34, "Inverted Follicular Keratosis": 35,
                               "Kaposi Sarcoma": 36, "Keloid": 37, "Leukemia Cutis": 38, "Lichenoid Keratosis": 39,
                               "Lipoma": 40, "Lymphocytic Infiltrations": 41, "Melanocytic Nevi": 42, "Melanoma": 43,
                               "Metastatic Carcinoma": 44, "Molluscum Contagiosum": 45, "Morphea": 46,
                               "Mycosis Fungoides": 47, "Neurofibroma": 48, "Neuroma": 49,
                               "Nevus Lipomatosus Superficialis": 50, "Onychomycosis": 51,
                               "Pigmented Spindle Cell Nevus of Reed": 52, "Prurigo Nodularis": 53,
                               "Pyogenic Granuloma": 54, "Reactive Lymphoid Hyperplasia": 55, "Scar": 56,
                               "Sebaceous Carcinoma": 57, "Seborrheic Keratosis": 58, "Solar Lentigo": 59,
                               "Squamous Cell Carcinoma": 60, "Subcutaneous T-cell Lymphoma": 61,
                               "Syringocystadenoma Papilliferum": 62, "Tinea Pedis": 63, "Trichilemmoma": 64,
                               "Trichofolliculoma": 65, "Ulcerations and Physical Injuries": 66, "Verruca Vulgaris": 67,
                               "Verruciform Xanthoma": 68, "Wart": 69, "Xanthogranuloma": 70}}

    elif experiment == "DDI_2class":
        setting = {"dataframe": os.path.join(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION, "DDI_2class_test.csv"),
                   "task": "classification",
                   "targets": {"Non-Malignant": 0, "Malignant": 1}}

    else:
        setting = None
        print("Experiment not prepared...")
    return setting


def process_categories(categories):
    """
    处理每行类别数据，将其转化为分类选项，随机选择一个类别。
    """
    # 将类别字段转换为列表
    category_list = eval(categories)  # 使用eval将字符串形式的列表转为实际的列表

    if len(category_list) > 1:
        category = random.choice(category_list)
    else:
        category = category_list[0]

    return category


def chunked(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def resolve_image_path(image_folder, image_file):
    """Resolve legacy CSV prefixes without rewriting the dataset manifest."""
    image_file = str(image_file)
    candidates = [image_file]
    if not os.path.isabs(image_file):
        candidates.append(os.path.join(image_folder, image_file))
        normalized = image_file.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        if len(parts) > 1:
            candidates.append(os.path.join(image_folder, *parts[1:]))
        candidates.append(os.path.join(image_folder, os.path.basename(normalized)))

    expanded = []
    for candidate in candidates:
        expanded.append(candidate)
        if "_downsampled" in candidate:
            expanded.append(candidate.replace("_downsampled", ""))

    for candidate in expanded:
        if os.path.isfile(candidate):
            return candidate
    return None


def apply_qwen_chat_template(processor, messages):
    template_kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    # Qwen3.5 模板默认开启 thinking（assistant 前缀 <think>\n），
    # 短 max_new_tokens 下会全部消耗在思维链导致空答案；显式关闭。
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **template_kwargs)


def apply_memvr_qwen25(
    model,
    starting_layer,
    ending_layer,
    entropy_threshold,
    retracing_ratio,
    retrace_delay_layers=1,
    retrace_target_layers="",
    method="memvr",
    state_drift_threshold=0.5,
    state_drift_pooling="mean",
    trigger_strategy="entropy",
    random_trigger_prob=0.5,
    injection_mode="ffn",
    spatial_group="all",
):
    # MemVR kernels live in the local transformers550 qwen2.5vl implementation.
    model.model.language_model.lm_head = model.lm_head
    mlp0 = model.model.language_model.layers[0].mlp
    mlp0.apply_memvr = True
    mlp0.starting_layer = int(starting_layer)
    mlp0.ending_layer = int(ending_layer)
    mlp0.entropy_threshold = float(entropy_threshold)
    mlp0.retrace_delay_layers = max(1, int(retrace_delay_layers))
    mlp0.retrace_target_layers = retrace_target_layers
    mlp0.memvr_method = str(method)
    mlp0.state_drift_threshold = float(state_drift_threshold)
    mlp0.state_drift_pooling = str(state_drift_pooling)
    mlp0.trigger_strategy = str(trigger_strategy)
    mlp0.random_trigger_prob = float(random_trigger_prob)
    mlp0.injection_mode = str(injection_mode)
    mlp0.spatial_group = str(spatial_group)
    for layer in model.model.language_model.layers:
        layer.mlp.retracing_ratio = float(retracing_ratio)
        layer.mlp.trigger_strategy = str(trigger_strategy)
        layer.mlp.random_trigger_prob = float(random_trigger_prob)
        layer.mlp.injection_mode = str(injection_mode)


def apply_memvr_qwen35(
    model,
    starting_layer,
    ending_layer,
    entropy_threshold,
    retracing_ratio,
    retrace_delay_layers=1,
    retrace_target_layers="",
    method="memvr",
    state_drift_threshold=0.5,
    state_drift_pooling="mean",
    trigger_strategy="entropy",
    random_trigger_prob=0.5,
    injection_mode="ffn",
):
    model.model.language_model.lm_head = model.lm_head
    mlp0 = model.model.language_model.layers[0].mlp
    mlp0.apply_memvr = True
    mlp0.starting_layer = int(starting_layer)
    mlp0.ending_layer = int(ending_layer)
    mlp0.entropy_threshold = float(entropy_threshold)
    mlp0.retrace_delay_layers = max(1, int(retrace_delay_layers))
    mlp0.retrace_target_layers = retrace_target_layers
    mlp0.memvr_method = str(method)
    mlp0.state_drift_threshold = float(state_drift_threshold)
    mlp0.state_drift_pooling = str(state_drift_pooling)
    mlp0.trigger_strategy = str(trigger_strategy)
    mlp0.random_trigger_prob = float(random_trigger_prob)
    mlp0.injection_mode = str(injection_mode)
    for layer in model.model.language_model.layers:
        layer.mlp.retracing_ratio = float(retracing_ratio)
        layer.mlp.trigger_strategy = str(trigger_strategy)
        layer.mlp.random_trigger_prob = float(random_trigger_prob)
        layer.mlp.injection_mode = str(injection_mode)

def eval_model(args):
    # 加载模型
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_name
    lowered_path = model_path.lower()
    lowered_name = (model_name or "").lower()
    is_qwen35 = (
        "qwen3.5" in lowered_path
        or "qwen3_5" in lowered_path
        or "qwen3.5" in lowered_name
        or "qwen3_5" in lowered_name
    )
    is_qwen25vl = (
        "qwen2.5-vl" in lowered_path
        or "qwen2_5_vl" in lowered_path
        or "qwen25vl" in lowered_path
        or "qwen2.5-vl" in lowered_name
        or "qwen2_5_vl" in lowered_name
        or "qwen25vl" in lowered_name
    )
    is_qwen = is_qwen35 or is_qwen25vl
    method = (args.method or "base").lower()
    force_550 = bool(int(os.environ.get("FORCE_TRANSFORMERS550", "0")))
    use_local_qwen25 = is_qwen25vl and (method in {"memvr", "evo"} or force_550)
    use_local_qwen35 = is_qwen35 and (method in {"memvr", "evo"} or force_550)

    processor = None
    if is_qwen:
        processor_cls = AutoProcessor550 if ((use_local_qwen25 or use_local_qwen35) and AutoProcessor550 is not None) else AutoProcessor
        if processor_cls is None:
            raise RuntimeError("AutoProcessor is unavailable. Please ensure dependencies are importable.")
        processor = processor_cls.from_pretrained(model_path, trust_remote_code=True)
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        if is_qwen35:
            if use_local_qwen35:
                if Qwen3_5ForConditionalGeneration550 is None:
                    raise RuntimeError("Qwen3.5 MemVR requires local transformers550 implementation.")
                print("[qwen3_5] method=memvr/evo -> using local transformers550 implementation", flush=True)
                qwen35_cls = Qwen3_5ForConditionalGeneration550
            else:
                if Qwen3_5ForConditionalGeneration is None:
                    raise RuntimeError("Qwen3.5 dependencies are unavailable. Please ensure transformers is importable.")
                qwen35_cls = Qwen3_5ForConditionalGeneration

            model = qwen35_cls.from_pretrained(
                model_path,
                device_map=device_map,
                trust_remote_code=True,
            )
            if method in {"memvr", "evo"}:
                apply_memvr_qwen35(
                    model=model,
                    starting_layer=args.starting_layer,
                    ending_layer=args.ending_layer,
                    entropy_threshold=args.entropy_threshold,
                    retracing_ratio=args.retracing_ratio,
                    retrace_delay_layers=args.retrace_delay_layers,
                    retrace_target_layers=args.retrace_target_layers,
                    method=method,
                    state_drift_threshold=args.state_drift_threshold,
                    state_drift_pooling=args.state_drift_pooling,
                    trigger_strategy=args.trigger_strategy,
                    random_trigger_prob=args.random_trigger_prob,
                    injection_mode=args.injection_mode,
                )
            else:
                try:
                    model.model.language_model.layers[0].mlp.apply_memvr = False
                except Exception:
                    pass
        else:
            if use_local_qwen25:
                if Qwen2_5_VLForConditionalGeneration550 is None:
                    raise RuntimeError("Qwen2.5-VL MemVR requires local transformers550 implementation.")
                print("[qwen25vl] method=memvr/evo -> using local transformers550 implementation", flush=True)
                qwen25_cls = Qwen2_5_VLForConditionalGeneration550
            else:
                if Qwen2_5_VLForConditionalGeneration is None:
                    raise RuntimeError("Qwen2.5-VL dependencies are unavailable. Please ensure transformers is importable.")
                qwen25_cls = Qwen2_5_VLForConditionalGeneration
            model = qwen25_cls.from_pretrained(
                model_path,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation=("eager" if method == "opera" else None),
            )

            if method in {"memvr", "evo"}:
                apply_memvr_qwen25(
                    model=model,
                    starting_layer=args.starting_layer,
                    ending_layer=args.ending_layer,
                    entropy_threshold=args.entropy_threshold,
                    retracing_ratio=args.retracing_ratio,
                    retrace_delay_layers=args.retrace_delay_layers,
                    retrace_target_layers=args.retrace_target_layers,
                    method=method,
                    state_drift_threshold=args.state_drift_threshold,
                    state_drift_pooling=args.state_drift_pooling,
                    trigger_strategy=args.trigger_strategy,
                    random_trigger_prob=args.random_trigger_prob,
                    spatial_group=getattr(args, "spatial_group", "all"),
                    injection_mode=args.injection_mode,
                )
                if getattr(args, "skinir_plus", False):
                    # SkinIR+: 触发只在 prefill（分类决策位置）执行
                    model.model.language_model._skinir_plus_prefill_only = True
            else:
                try:
                    model.model.language_model.layers[0].mlp.apply_memvr = False
                except Exception:
                    pass
        tokenizer = None
        image_processor = None
        context_len = 32768
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=device)
        if method in {"memvr", "evo"}:
            if apply_memvr_llava is None:
                raise RuntimeError("LLaVA MemVR helper is unavailable. Please ensure llava.memvr_llava is importable.")
            apply_memvr_llava(
                model=model,
                starting_layer=args.starting_layer,
                ending_layer=args.ending_layer,
                entropy_threshold=args.entropy_threshold,
                retracing_ratio=args.retracing_ratio,
                retrace_delay_layers=args.retrace_delay_layers,
                retrace_target_layers=args.retrace_target_layers,
                method=method,
                state_drift_threshold=args.state_drift_threshold,
                state_drift_pooling=args.state_drift_pooling,
                skinir_plus=bool(getattr(args, "skinir_plus", False)),
                injection_mode=args.injection_mode,
            )
    model.eval()

    # 从文件获取实验设置
    setting = get_experiment_setting(args.experiment)
    if not setting:
        raise ValueError(f"Experiment '{args.experiment}' settings are not found.")

    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    dataframe_path = args.dataframe if args.dataframe else setting["dataframe"]
    if not os.path.exists(dataframe_path):
        raise FileNotFoundError(f"Dataframe file not found: {dataframe_path}")

    os.makedirs(args.result_path, exist_ok=True)
    df = pd.read_csv(dataframe_path)
    result_file = os.path.join(args.result_path, f"{args.experiment}_predictions{args.result_suffix}.csv")
    result_metrics_file = os.path.join(args.result_path, f"{args.experiment}_results{args.result_suffix}.csv")
    done_images = set()

    if args.num_shards > 1:
        df = df.iloc[args.shard_index::args.num_shards].reset_index(drop=True)
        print(f"Shard mode: shard {args.shard_index}/{args.num_shards}, rows in this shard: {len(df)}")

    resume_files = [result_file]
    base_result_file = os.path.join(args.result_path, f"{args.experiment}_predictions.csv")
    if base_result_file != result_file:
        resume_files.append(base_result_file)

    # 确保结果文件存在
    if not os.path.exists(result_file):
        columns = ["image", "question", "predicted_answer", "ground_truth", "predicted_label", "ground_truth_label"]
        if args.dump_memvr_debug:
            columns.extend([
                "memvr_runtime_enabled",
                "memvr_has_debug_attr",
                "memvr_enabled",
                "memvr_triggered",
                "memvr_injection_success",
                "memvr_trigger_layer",
                "memvr_target_layer",
                "memvr_trigger_metric",
                "memvr_trigger_value",
                "memvr_trigger_source",
                "memvr_used_dynamic_visual_token",
                "memvr_entropy_trace_len",
                "memvr_entropy_trace",
                "memvr_last_entropy",
                "memvr_last_target_layer",
            ])
        result_df = pd.DataFrame(columns=columns)
        result_df.to_csv(result_file, index=False)

    # Resume from existing predictions files. Parse with csv.reader and trust field 0.
    for resume_file in resume_files:
        if not os.path.exists(resume_file):
            continue
        with open(resume_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for rec in reader:
                if rec and rec[0]:
                    done_images.add(rec[0])
    if done_images:
        print(f"Resume enabled: {len(done_images)} finished samples will be skipped.")

    # 获取类别名称
    possible_diseases = list(setting["targets"].keys())  # 从targets中提取类别名
    label_set = list(setting["targets"].values())
    print(f"Possible diseases: {possible_diseases}", f"Label set: {label_set}")
    if getattr(args, "hierarchical_decode", False):
        if args.experiment != "Patch16" or args.decode_mode != "candidate":
            raise ValueError("--hierarchical-decode requires --experiment Patch16 --decode-mode candidate (LLaVA/SkinVL path).")
        print("[SkinIR-H] two-stage hierarchical decoding enabled for Patch16 16-class.")
    if getattr(args, "zoom_pass", False):
        if args.decode_mode != "candidate":
            raise ValueError("--zoom-pass requires --decode-mode candidate (LLaVA/SkinVL path).")
        print("[SkinIR-Z] multi-scale zoom pass enabled (full view + center crop).")
    if getattr(args, "side_info", False):
        if not {"age", "sex", "anatom_site"}.issubset(set(df.columns)):
            raise ValueError("--side-info requires CSV columns age/sex/anatom_site (PAD-style clinical metadata).")
        print("[SkinIR-S] patient metadata injection enabled (age/sex/anatom_site).")
    if args.decode_mode == "numeric":
        numbered_categories = "; ".join(
            f"{idx}={name}" for idx, name in enumerate(possible_diseases)
        )
        question_text = (
            "This is a skin lesion image. Classify it into exactly one category. "
            f"Use only one digit from this mapping: {numbered_categories}. "
            "Respond with exactly one digit and nothing else."
        )
    candidate_names = _candidate_verbalizers(
        args.experiment, possible_diseases, args.candidate_verbalizer
    )
    if args.decode_mode == "candidate":
        question_text = _candidate_prompt_variants(
            possible_diseases, args.candidate_ensemble,
            candidate_names=candidate_names,
        )[0]
    else:
        question_text = (
            f"This is a skin lesion image. From the following categories: {', '.join(possible_diseases)}, "
            "which one is the diagnosis? Respond using exactly one category name only."
        )
    if getattr(args, "side_info", False) and {"age", "sex", "anatom_site"}.issubset(set(df.columns)):
        # SkinIR-S: per-image patient context lines, looked up by image key at
        # candidate-scoring time (each sample gets its own prompt variant).
        side_rows = {}
        for _, side_row in df.iterrows():
            key = side_row["image"]
            if key not in side_rows:
                side_rows[key] = _format_side_info(side_row)
    cot_suffix = ""
    cot_mode = getattr(args, "cot_mode", "off")
    describe_prompt = ""
    icl_examples = []
    if getattr(args, "icl_examples_csv", ""):
        import csv as _csv
        with open(args.icl_examples_csv) as f:
            for r in _csv.DictReader(f):
                img = r.get("image") or r.get("image_id") or r.get("img")
                label = r.get("label") or r.get("dx") or r.get("diagnosis")
                if img and label:
                    icl_examples.append((img, label))
        print(f"[ICL] loaded {len(icl_examples)} examples: {[l for _, l in icl_examples]}")
    if cot_mode == "derm":
        # SkinIR-C: 皮肤科判读准则锚定的推理链脚手架
        cot_suffix = (
            "\nFirst analyze the lesion step by step: (1) symmetry and border regularity; "
            "(2) color distribution and pigment structures (network, globules, streaks); "
            "(3) overall pattern compared with the categories above. "
            "Then give your final answer on the last line in the format: "
            "'Final Answer: <exactly one category name>'."
        )
    elif getattr(args, "cot_mode", "off") == "generic":
        cot_suffix = (
            "\nThink step by step about the visual findings, then give your final answer "
            "on the last line in the format: 'Final Answer: <exactly one category name>'."
        )
    elif cot_mode == "two_stage":
        # SkinIR-C v2 两段式：第一轮描述提示词（禁诊断，无类别选项可抄）
        describe_prompt = (
            "Describe this skin lesion image objectively: symmetry, border regularity, "
            "color distribution, pigment structures (network, globules, streaks, dots), "
            "and surface characteristics. Do NOT give any diagnosis or mention disease names. "
            "Be concise (3-5 sentences)."
        )
    elif cot_mode == "two_stage_balanced":
        # SkinIR-C v3 双侧对冲：强制列出良性/恶性两侧证据，对冲描述词的恶性偏置
        describe_prompt = (
            "Analyze this skin lesion image. List at least 2 findings that support a BENIGN "
            "diagnosis (e.g., symmetry, regular borders, uniform color, regular pigment network, "
            "comedo-like openings, multiple small milia-like cysts) AND at least 2 findings that "
            "support a MALIGNANT diagnosis (e.g., asymmetry, irregular borders, multiple colors, "
            "atypical pigment network, blue-white veil). Do NOT give any final diagnosis or "
            "mention disease names. Be concise (4-6 sentences)."
        )
    elif cot_mode == "rule":
        # SkinIR-R 规则驱动的结构化评估：每条规则从枚举值里选（判别性规则，非自由描述）
        describe_prompt = (
            "Evaluate this dermoscopy image with the following rules. For each rule, output "
            "EXACTLY one line choosing one value from the bracketed options. Do NOT give any "
            "diagnosis or disease name.\n"
            "Rule 1 Symmetry: [symmetric | asymmetric]\n"
            "Rule 2 Border: [regular | irregular]\n"
            "Rule 3 Colors: [uniform 1-2 colors | varied 3+ colors]\n"
            "Rule 4 Pigment network: [regular organized | atypical irregular | absent]\n"
            "Rule 5 Structures: [regular globules | irregular streaks | blue-white veil | none visible]\n"
            "Output all 5 lines, nothing else."
        )
    question_text = question_text + cot_suffix
    if is_qwen:
        question = question_text
        input_ids_template = None
    else:
        if model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question_text
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question_text

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids_template = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device)

    pending_samples = []
    for _, row in df.iterrows():
        image_file = row["image"]
        if image_file in done_images:
            continue

        ground_truth = process_categories(row["categories"])
        true_label = setting["targets"][ground_truth]
        image_path = resolve_image_path(args.image_folder, image_file)
        if image_path is None:
            continue

        pending_samples.append({
            "image": image_file,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "true_label": true_label,
            "side_info": (
                _format_side_info(row)
                if getattr(args, "side_info", False) and {"age", "sex", "anatom_site"}.issubset(set(df.columns))
                else ""
            ),
        })

    print(f"Samples to process in this run: {len(pending_samples)}")
    class_priors = None
    if args.decode_mode in {"numeric", "candidate"} and pending_samples:
        if args.candidate_prior_source == "test":
            counts = np.bincount(
                [sample["true_label"] for sample in pending_samples],
                minlength=len(possible_diseases),
            ).astype(np.float64)
            class_priors = (counts / counts.sum()).tolist()
        elif args.candidate_prior_source == "reference":
            class_priors = _candidate_reference_prior(
                args.experiment, possible_diseases, "reference"
            )
        elif args.candidate_prior_source == "train":
            class_priors = _candidate_train_prior(args.experiment, possible_diseases)
        print(
            f"[{args.decode_mode}] verbalizer={args.candidate_verbalizer} "
            f"prior_source={args.candidate_prior_source} "
            f"class_priors={None if class_priors is None else ['%.4f' % p for p in class_priors]} "
            f"prior_alpha={args.prior_alpha}"
        )

    total_batches = math.ceil(len(pending_samples) / args.batch_size) if pending_samples else 0
    for batch_idx, batch_samples in enumerate(
        tqdm(chunked(pending_samples, args.batch_size), total=total_batches),
        start=1,
    ):
        if total_batches and (
            batch_idx == 1
            or batch_idx == total_batches
            or batch_idx % args.progress_every == 0
        ):
            print(
                f"[progress] shard={args.shard_index}/{args.num_shards} batch={batch_idx}/{total_batches}",
                flush=True,
            )

        valid_samples = []
        images = []
        for sample in batch_samples:
            try:
                with Image.open(sample["image_path"]) as img:
                    images.append(img.convert('RGB'))
                valid_samples.append(sample)
            except Exception:
                continue

        if not valid_samples:
            continue

        if is_qwen:
            if args.decode_mode == "candidate":
                raise ValueError("--decode-mode candidate currently supports SkinVL/LLaVA only.")
            decoded_answers = []
            with torch.inference_mode():
                for sample in valid_samples:
                    if icl_examples:
                        # SkinIR-ICL: 多图上下文示例（每类1张带标签）
                        content = []
                        for ex_img, ex_label in icl_examples:
                            content.append({"type": "image", "path": ex_img})
                            content.append({"type": "text", "text": f"Diagnosis: {ex_label}"})
                        content.append({"type": "image", "path": sample["image_path"]})
                        content.append({"type": "text", "text": question_text})
                    else:
                        content = [
                            {"type": "image", "path": sample["image_path"]},
                            {"type": "text", "text": question_text},
                        ]
                    messages = [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ]
                    inputs = apply_qwen_chat_template(processor, messages).to(device)
                    generate_kwargs = {
                        "max_new_tokens": 1 if args.decode_mode == "numeric" else args.max_new_tokens,
                        "do_sample": args.do_sample,
                        "num_beams": args.num_beams,
                        "pad_token_id": processor.tokenizer.eos_token_id,
                    }
                    numeric_processor = None
                    if args.decode_mode == "numeric":
                        numeric_processor = DigitOnlyLogitsProcessor(
                            processor.tokenizer,
                            prompt_length=inputs["input_ids"].shape[1],
                            num_classes=len(possible_diseases),
                        )
                        generate_kwargs["logits_processor"] = LogitsProcessorList([numeric_processor])
                        generate_kwargs["return_dict_in_generate"] = True
                        generate_kwargs["output_scores"] = True
                    if args.do_sample:
                        generate_kwargs["temperature"] = args.temperature
                        generate_kwargs["top_p"] = args.top_p

                    if method == "opera" and is_qwen:
                        # OPERA: 自定义解码循环（over-trust 惩罚 + 回滚）
                        from opera_qwen import opera_generate_qwen
                        output_ids = opera_generate_qwen(
                            model, processor, inputs,
                            max_new_tokens=args.max_new_tokens,
                        )
                        decoded_answers.append(processor.batch_decode(
                            output_ids[:, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True, clean_up_tokenization_spaces=False,
                        )[0])
                        continue

                    if cot_mode in ("two_stage", "two_stage_balanced", "rule"):
                        # SkinIR-C v2: 第一轮只描述不诊断（强制消费视觉信息）
                        describe_messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "path": sample["image_path"]},
                                    {"type": "text", "text": describe_prompt},
                                ],
                            }
                        ]
                        d_inputs = apply_qwen_chat_template(processor, describe_messages).to(device)
                        d_out = model.generate(**d_inputs, **{
                            **generate_kwargs, "max_new_tokens": args.cot_describe_tokens,
                        })
                        d_trim = [o[len(i):] for i, o in zip(d_inputs["input_ids"], d_out)]
                        description = processor.batch_decode(
                            d_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                        )[0].strip()
                        # 第二轮：描述作为文字锚点 + 原图重看 + 分类
                        cls_messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "path": sample["image_path"]},
                                    {"type": "text", "text": f"Findings: {description}\n\n{question_text}"},
                                ],
                            }
                        ]
                        inputs = apply_qwen_chat_template(processor, cls_messages).to(device)

                    generated = model.generate(**inputs, **generate_kwargs)
                    if args.decode_mode == "numeric":
                        numeric_predictions = _select_numeric_predictions(
                            generated.scores[0],
                            processor.tokenizer,
                            len(possible_diseases),
                            class_priors=class_priors,
                            prior_alpha=args.prior_alpha,
                        )
                        decoded_answers.append(possible_diseases[numeric_predictions[0]])
                    else:
                        output_ids = generated
                        output_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
                        ]
                        decoded = processor.batch_decode(
                            output_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]
                        decoded_answers.append(decoded)
        else:
            image_inputs = process_images(images, image_processor, model.config)
            if isinstance(image_inputs, list):
                image_inputs = [img.to(device=device, dtype=torch.float16) for img in image_inputs]
            else:
                image_inputs = image_inputs.to(device=device, dtype=torch.float16)

            input_ids = input_ids_template.repeat(len(valid_samples), 1)

            if args.decode_mode == "candidate" and getattr(args, "hierarchical_decode", False):
                # SkinIR-H: Patch16 two-stage hierarchical decoding. The question
                # column stored in the output reflects the stage prompts.
                hierarchical_predictions = _hierarchical_patch16_predictions(
                    model, tokenizer, args, possible_diseases,
                    valid_samples, image_inputs, device,
                )
                decoded_answers = [
                    possible_diseases[pred] for pred in hierarchical_predictions
                ]
                rows_to_write = []
                for sample, predicted_answer in zip(valid_samples, decoded_answers):
                    rows_to_write.append({
                        "image": sample["image"],
                        "question": "SkinIR-H: stage1 tumor/nontumor -> stage2 in-branch type",
                        "predicted_answer": predicted_answer,
                        "ground_truth": sample["ground_truth"],
                        "predicted_label": setting["targets"][predicted_answer],
                        "ground_truth_label": sample["true_label"],
                    })
                    if args.dump_memvr_debug:
                        rows_to_write[-1].update(_collect_memvr_debug_row(model))
                if rows_to_write:
                    pd.DataFrame(rows_to_write).to_csv(
                        result_file, mode='a', header=False, index=False
                    )
                continue

            if args.decode_mode == "candidate" and getattr(args, "zoom_pass", False):
                # SkinIR-Z: dermoscopy multi-scale inference. Score candidates on
                # the full view, then on a center-cropped lesion view, and average
                # the per-class scores before the argmax. Motivated by the finding
                # that center tokens carry almost all dermoscopic signal.
                candidate_predictions = _zoom_pass_predictions(
                    model, tokenizer, args, possible_diseases,
                    candidate_names, valid_samples, images, image_processor, device,
                    class_priors=class_priors, prior_alpha=args.prior_alpha,
                )
                decoded_answers = [
                    possible_diseases[pred] for pred in candidate_predictions
                ]
                rows_to_write = []
                for sample, predicted_answer in zip(valid_samples, decoded_answers):
                    rows_to_write.append({
                        "image": sample["image"],
                        "question": "SkinIR-Z: full view + center crop score fusion",
                        "predicted_answer": predicted_answer,
                        "ground_truth": sample["ground_truth"],
                        "predicted_label": setting["targets"][predicted_answer],
                        "ground_truth_label": sample["true_label"],
                    })
                    if args.dump_memvr_debug:
                        rows_to_write[-1].update(_collect_memvr_debug_row(model))
                if rows_to_write:
                    pd.DataFrame(rows_to_write).to_csv(
                        result_file, mode='a', header=False, index=False
                    )
                continue

            if args.decode_mode == "candidate":
                candidate_predictions_by_prompt = []
                for sample_index, sample in enumerate(valid_samples):
                    # SkinIR-S: per-sample patient metadata prepended to the shared
                    # candidate question (PAD clinical-photo context injection).
                    sample_candidate_names = candidate_names
                    sample_side = ""
                    if getattr(args, "side_info", False):
                        sample_side = side_rows.get(sample["image"], "")
                    per_prompt_preds = []
                    for candidate_text in _candidate_prompt_variants(
                        possible_diseases, args.candidate_ensemble,
                        candidate_names=sample_candidate_names,
                    ):
                        effective_text = (
                            f"{sample_side}\n{candidate_text}" if sample_side else candidate_text
                        )
                        if model.config.mm_use_im_start_end:
                            candidate_question = (
                                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN
                                + DEFAULT_IM_END_TOKEN + "\n" + effective_text
                            )
                        else:
                            candidate_question = DEFAULT_IMAGE_TOKEN + "\n" + effective_text
                        candidate_conv = conv_templates[args.conv_mode].copy()
                        candidate_conv.append_message(candidate_conv.roles[0], candidate_question)
                        candidate_conv.append_message(candidate_conv.roles[1], None)
                        candidate_prompt = candidate_conv.get_prompt()
                        candidate_input_ids = tokenizer_image_token(
                            candidate_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                            return_tensors="pt",
                        ).unsqueeze(0).to(device)
                        if args.candidate_score_mode == "sequence":
                            per_prompt_preds.extend(
                                _select_candidate_sequence_predictions(
                                    candidate_input_ids,
                                    image_inputs[sample_index:sample_index + 1]
                                    if not isinstance(image_inputs, list)
                                    else [image_inputs[sample_index]],
                                    model,
                                    tokenizer,
                                    possible_diseases,
                                    candidate_names=sample_candidate_names,
                                    class_priors=class_priors,
                                    prior_alpha=args.prior_alpha,
                                )
                            )
                        else:
                            candidate_input_ids_batched = candidate_input_ids.repeat(1, 1)
                            with torch.inference_mode():
                                outputs = model(
                                    input_ids=candidate_input_ids_batched,
                                    images=image_inputs[sample_index:sample_index + 1]
                                    if not isinstance(image_inputs, list)
                                    else [image_inputs[sample_index]],
                                    use_cache=False,
                                )
                            per_prompt_preds.extend(
                                _select_candidate_predictions(
                                    outputs.logits[:, -1, :],
                                    tokenizer,
                                    possible_diseases,
                                    candidate_names=sample_candidate_names,
                                    class_priors=class_priors,
                                    prior_alpha=args.prior_alpha,
                                )
                            )
                    candidate_predictions_by_prompt.append(per_prompt_preds)
                candidate_predictions = []
                for sample_index in range(len(valid_samples)):
                    # 投票语义：candidate_predictions_by_prompt[sample_index] 是该样本
                    # 在所有 prompt 变体下的预测列表（新结构按样本存票）。
                    votes = list(candidate_predictions_by_prompt[sample_index])
                    candidate_predictions.append(max(set(votes), key=votes.count))
                decoded_answers = [possible_diseases[pred] for pred in candidate_predictions]
                rows_to_write = []
                for sample, predicted_answer in zip(valid_samples, decoded_answers):
                    rows_to_write.append({
                        "image": sample["image"],
                        "question": question,
                        "predicted_answer": predicted_answer,
                        "ground_truth": sample["ground_truth"],
                        "predicted_label": setting["targets"][predicted_answer],
                        "ground_truth_label": sample["true_label"],
                    })
                    if args.dump_memvr_debug:
                        rows_to_write[-1].update(_collect_memvr_debug_row(model))
                if rows_to_write:
                    pd.DataFrame(rows_to_write).to_csv(
                        result_file, mode='a', header=False, index=False
                    )
                continue

                generate_kwargs = {
                    "images": image_inputs,
                    "do_sample": args.do_sample,
                    "num_beams": args.num_beams,
                    "min_new_tokens": 1,
                    "max_new_tokens": 1 if args.decode_mode == "numeric" else args.max_new_tokens,
                    "pad_token_id": tokenizer.eos_token_id,
                    "use_cache": True,
                }
                if args.decode_mode == "numeric":
                    numeric_processor = DigitOnlyLogitsProcessor(
                        tokenizer,
                        prompt_length=input_ids.shape[1],
                        num_classes=len(possible_diseases),
                    )
                    generate_kwargs["logits_processor"] = LogitsProcessorList([numeric_processor])
                    generate_kwargs["return_dict_in_generate"] = True
                    generate_kwargs["output_scores"] = True
                if args.do_sample:
                    generate_kwargs["temperature"] = args.temperature
                    generate_kwargs["top_p"] = args.top_p

                with torch.inference_mode():
                    generated = model.generate(input_ids, **generate_kwargs)

                if args.decode_mode == "numeric":
                    numeric_predictions = _select_numeric_predictions(
                        generated.scores[0],
                        tokenizer,
                        len(possible_diseases),
                        class_priors=class_priors,
                        prior_alpha=args.prior_alpha,
                    )
                    decoded_answers = [possible_diseases[pred] for pred in numeric_predictions]
                else:
                    decoded_answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
            rows_to_write = []
            for sample, predicted_answer in zip(valid_samples, decoded_answers):
                normalized_answer = predicted_answer.strip().lower()
                predicted_diagnosis = None
                for disease in possible_diseases:
                    disease_keys = {disease.lower()}
                    # Handle singular/plural naming inconsistency across datasets.
                    if disease.lower() == "actinic keratoses":
                        disease_keys.add("actinic keratosis")

                    if any(disease_key in normalized_answer for disease_key in disease_keys):
                        predicted_diagnosis = disease
                        break

                if predicted_diagnosis:
                    predicted_label = setting["targets"][predicted_diagnosis]
                else:
                    predicted_label = -1

                row = {
                    "image": sample["image"],
                    "question": question,
                    "predicted_answer": predicted_diagnosis,
                    "ground_truth": sample["ground_truth"],
                    "predicted_label": predicted_label,
                    "ground_truth_label": sample["true_label"]
                }
                if args.dump_memvr_debug:
                    row.update(_collect_memvr_debug_row(model))
                rows_to_write.append(row)

            if rows_to_write:
                pd.DataFrame(rows_to_write).to_csv(result_file, mode='a', header=False, index=False)

    # Recompute metrics from the full predictions file (supports resumed runs).
    predictions = []
    ground_truths = []
    pred_df = pd.read_csv(result_file)
    if "predicted_label" not in pred_df.columns or "ground_truth_label" not in pred_df.columns:
        print("Prediction file is missing predicted_label/ground_truth_label columns. Metrics file will not be generated.")
        return

    for pred, gt in zip(pred_df["predicted_label"], pred_df["ground_truth_label"]):
        try:
            predictions.append(int(pred))
            ground_truths.append(int(gt))
        except (ValueError, TypeError):
            continue

    if not predictions:
        print("No valid prediction rows found. Metrics file will not be generated.")
        return

    res = binary_metrics(ground_truths, predictions, label_set)
    df = pd.DataFrame([res])
    df.to_csv(result_metrics_file, index=False)
    print(f"Predictions saved to {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnosis Classification via Generation")
    parser.add_argument('--experiment', default='PAD', help="Experiment name (e.g., PAD, DDI_2class, etc.)")
    parser.add_argument('--model_path', required=True, help="Path to model weights")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="LlavaMistralForCausalLM")
    parser.add_argument("--load-8bit", type=bool, default=False, help="Load model with 8-bit precision")
    parser.add_argument("--load-4bit", type=bool, default=False, help="Load model with 4-bit precision")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum generated tokens per sample")
    parser.add_argument("--decode-mode", type=str, default="free",
                        choices=["free", "numeric", "candidate"],
                        help="free=category generation; numeric=constrained class ID; candidate=class-name scoring")
    parser.add_argument("--candidate-score-mode", type=str, default="first_token",
                        choices=["first_token", "sequence"],
                        help="Candidate scoring: first_token=legacy prefix score; sequence=length-normalized full-name teacher forcing")
    parser.add_argument("--candidate-verbalizer", type=str, default="canonical",
                        choices=["canonical", "natural"],
                        help="Candidate answer names: dataset labels or natural-language verbalizers")
    parser.add_argument("--candidate-prior-source", type=str, default="none",
                        choices=["none", "test", "reference", "train"],
                        help="Candidate prior source; reference/train avoid test-label leakage")
    parser.add_argument("--prior-alpha", type=float, default=0.0,
                        help="Candidate prior strength: score + alpha*log(reference_prior)")
    parser.add_argument("--candidate-ensemble", type=int, default=1,
                        help="Number of prompt variants for candidate scoring (1-3)")
    parser.add_argument("--method", type=str, default="base", choices=["base", "memvr", "evo", "opera"],
                        help="Qwen2.5-VL inference method")
    parser.add_argument("--starting-layer", type=int, default=5, help="MemVR start layer")
    parser.add_argument("--ending-layer", type=int, default=16, help="MemVR end layer")
    parser.add_argument("--entropy-threshold", type=float, default=0.75, help="MemVR entropy threshold")
    parser.add_argument("--retracing-ratio", type=float, default=0.0, help="MemVR retracing ratio")
    parser.add_argument("--retrace-delay-layers", type=int, default=1, help="MemVR retrace delay layers")
    parser.add_argument("--retrace-target-layers", type=str, default="", help="MemVR retrace target layers")
    parser.add_argument("--state-drift-threshold", type=float, default=0.5, help="MemVR state drift threshold")
    parser.add_argument("--state-drift-pooling", type=str, default="mean", help="MemVR state drift pooling")
    parser.add_argument("--trigger-strategy", type=str, default="entropy",
                        choices=["entropy", "always", "random", "none"],
                        help="MemVR trigger strategy for Qwen2.5/Qwen3.5 ablations")
    parser.add_argument("--random-trigger-prob", type=float, default=0.5,
                        help="Random trigger probability when --trigger-strategy=random")
    parser.add_argument("--injection-mode", type=str, default="ffn",
                        choices=["ffn", "attention", "ffn_attention", "residual"],
                        help="Injection location ablation for Qwen2.5/Qwen3.5 MemVR")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for stochastic ablations such as random trigger")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of dataset shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index")
    parser.add_argument("--result-suffix", type=str, default="", help="Suffix for output files, e.g. _g0")
    parser.add_argument("--progress-every", type=int, default=5, help="Print explicit progress every N batches")
    parser.add_argument("--image-folder", type=str, default="Dataset", help="Folder containing images")
    parser.add_argument("--dataframe", type=str, default="", help="Optional CSV path to override experiment default dataframe")
    parser.add_argument("--dump-memvr-debug", action="store_true", help="Write MemVR runtime debug fields to prediction CSV")
    parser.add_argument("--cot-mode", type=str, default="off", choices=["off", "derm", "generic", "two_stage", "two_stage_balanced", "rule"],
                        help="CoT mode: off=direct, derm=dermoscopy scaffold, generic=generic CoT, two_stage=describe-then-classify, two_stage_balanced=benign/malignant evidence balancing, rule=structured rule enumeration")
    parser.add_argument("--spatial-group", type=str, default="all", choices=["all", "center", "periphery"],
                        help="SkinIR-M spatial grouping of visual tokens for injection: all=center+periphery, center=lesion core only, periphery=outer ring only")
    parser.add_argument("--icl-examples-csv", type=str, default="",
                        help="CSV of in-context examples (columns: image/image_id + label/dx). Each example image is shown with its label before the test image.")
    parser.add_argument("--cot-describe-tokens", type=int, default=120, help="Max tokens for stage-1 description in two_stage mode")
    parser.add_argument("--skinir-plus", action="store_true",
                        help="SkinIR+: prefill-only vocab-entropy triggering (suppresses decode-step spurious triggers)")
    parser.add_argument("--hierarchical-decode", action="store_true",
                        help="SkinIR-H: Patch16 two-stage hierarchical decoding (tumor/nontumor branch -> in-branch tissue type). "
                             "Uses the validated natural verbalizer at stage 1; requires --decode-mode candidate on Patch16 16-class.")
    parser.add_argument("--side-info", action="store_true",
                        help="SkinIR-S: inject patient metadata (age/sex/anatom_site CSV columns) into the prompt for PAD-style clinical photos.")
    parser.add_argument("--zoom-pass", action="store_true",
                        help="SkinIR-Z: dermoscopy multi-scale inference — average candidate scores over the full view and a center-cropped lesion view "
                             "(requires --decode-mode candidate; HAM10000).")
    parser.add_argument("--zoom-crop-fraction", type=float, default=0.6,
                        help="SkinIR-Z: side fraction of the center crop (0.6 keeps the central 60%% box).")
    parser.add_argument("--conv_mode", type=str, default="mistral_instruct", help="Conversation mode for prompt templates")
    parser.add_argument('--result-path', default='result/zeroshot_class/DermMM_9pubCHOICE', type=str,
                        help="File to save predictions")
    args = parser.parse_args()
    eval_model(args)
