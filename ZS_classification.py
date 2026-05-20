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


def apply_qwen_chat_template(processor, messages):
    template_kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
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
    use_local_qwen25 = is_qwen25vl and method in {"memvr", "evo"}
    use_local_qwen35 = is_qwen35 and method in {"memvr", "evo"}

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
                    injection_mode=args.injection_mode,
                )
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
    question_text = (
        f"This is a skin lesion image. From the following categories: {', '.join(possible_diseases)}, "
        "which one is the diagnosis? Respond using exactly one category name only."
    )
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
        image_path = os.path.join(args.image_folder, image_file)
        if not os.path.exists(image_path) and "_downsampled" in image_file:
            image_path = os.path.join(args.image_folder, image_file.replace("_downsampled", ""))
        if not os.path.exists(image_path):
            continue

        pending_samples.append({
            "image": image_file,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "true_label": true_label
        })

    print(f"Samples to process in this run: {len(pending_samples)}")

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
            decoded_answers = []
            with torch.inference_mode():
                for sample in valid_samples:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "path": sample["image_path"]},
                                {"type": "text", "text": question_text},
                            ],
                        }
                    ]
                    inputs = apply_qwen_chat_template(processor, messages).to(device)
                    generate_kwargs = {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": args.do_sample,
                        "num_beams": args.num_beams,
                        "pad_token_id": processor.tokenizer.eos_token_id,
                    }
                    if args.do_sample:
                        generate_kwargs["temperature"] = args.temperature
                        generate_kwargs["top_p"] = args.top_p

                    output_ids = model.generate(**inputs, **generate_kwargs)
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

            generate_kwargs = {
                "images": image_inputs,
                "do_sample": args.do_sample,
                "num_beams": args.num_beams,
                "min_new_tokens": 1,
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            if args.do_sample:
                generate_kwargs["temperature"] = args.temperature
                generate_kwargs["top_p"] = args.top_p

            with torch.inference_mode():
                output_ids = model.generate(input_ids, **generate_kwargs)

            decoded_answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
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
    parser.add_argument('--model_path', default="/merge/SkinVL_PubMM", help="Path to model weights")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="LlavaMistralForCausalLM")
    parser.add_argument("--load-8bit", type=bool, default=False, help="Load model with 8-bit precision")
    parser.add_argument("--load-4bit", type=bool, default=False, help="Load model with 4-bit precision")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum generated tokens per sample")
    parser.add_argument("--method", type=str, default="base", choices=["base", "memvr", "evo"],
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
    parser.add_argument("--conv_mode", type=str, default="mistral_instruct", help="Conversation mode for prompt templates")
    parser.add_argument('--result-path', default='result/zeroshot_class/DermMM_9pubCHOICE', type=str,
                        help="File to save predictions")
    args = parser.parse_args()
    eval_model(args)