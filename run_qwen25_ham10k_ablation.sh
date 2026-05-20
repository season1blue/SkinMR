#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/miniconda3/envs/dualpd/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/mnt/data/ssz/Skin/skindata}"
HAM_CSV="${HAM_CSV:-${ROOT_DIR}/Dataframe/test/classification/HAM10K_ISIC2018_test.csv}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/result/ablation_qwen25_ham10k}"
SUMMARY_CSV="${SUMMARY_CSV:-${RESULT_ROOT}/summary.csv}"
GPU_ID="${GPU_ID:-0}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RESULT_ROOT}"
echo "group,name,result_path,acc,precision_macro,triggered_true,injection_success_true" > "${SUMMARY_CSV}"

join_layers() {
    local start="$1"
    local end="$2"
    local values=()
    local idx
    for ((idx=start; idx<=end; idx++)); do
        values+=("${idx}")
    done
    local IFS=,
    echo "${values[*]}"
}

record_summary() {
    local group="$1"
    local name="$2"
    local result_path="$3"
    local pred_file="$4"
    local metrics_file="$5"

    "${PYTHON_BIN}" - <<PY
import pandas as pd
from pathlib import Path

summary_path = Path(${SUMMARY_CSV@Q})
metrics_file = Path(${metrics_file@Q})
pred_file = Path(${pred_file@Q})
group = ${group@Q}
name = ${name@Q}
result_path = ${result_path@Q}

metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
pred = pd.read_csv(pred_file)
triggered_true = ""
injection_success_true = ""
if "memvr_triggered" in pred.columns:
    trig = pred["memvr_triggered"].fillna(False).astype(bool)
    inj = pred["memvr_injection_success"].fillna(False).astype(bool)
    triggered_true = int(trig.sum())
    injection_success_true = int(inj.sum())

row = pd.DataFrame([{
    "group": group,
    "name": name,
    "result_path": result_path,
    "acc": metrics.get("accuracy", metrics.get("acc", "")),
    "precision_macro": metrics.get("precision_macro", metrics.get("precision", "")),
    "triggered_true": triggered_true,
    "injection_success_true": injection_success_true,
}])
row.to_csv(summary_path, mode="a", header=False, index=False)
print(row.to_csv(index=False).strip())
PY
}

run_eval() {
    local group="$1"
    local name="$2"
    shift 2

    local result_path="${RESULT_ROOT}/${group}_${name}"
    local pred_file="${result_path}/HAM10000_predictions.csv"
    local metrics_file="${result_path}/HAM10000_results.csv"

    if [[ -f "${metrics_file}" && -f "${pred_file}" ]]; then
        echo "[skip] ${group}/${name}"
        record_summary "${group}" "${name}" "${result_path}" "${pred_file}" "${metrics_file}"
        return
    fi

    rm -rf "${result_path}"
    mkdir -p "${result_path}"

    echo "[run] ${group}/${name}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ROOT_DIR}/ZS_classification.py" \
        --experiment HAM10000 \
        --result-path "${result_path}" \
        --model_path "${MODEL_PATH}" \
        --image-folder "${IMAGE_FOLDER}" \
        --dataframe "${HAM_CSV}" \
        --batch-size 8 \
        --num_beams 1 \
        --max-new-tokens 32 \
        --random-seed 42 \
        --dump-memvr-debug \
        "$@"

    record_summary "${group}" "${name}" "${result_path}" "${pred_file}" "${metrics_file}"
}

FIRST_QUARTER="$(join_layers 0 6)"
FIRST_HALF="$(join_layers 0 13)"
SECOND_HALF="$(join_layers 14 27)"
ALL_LAYERS="$(join_layers 0 27)"

run_eval trigger no_revisit --method base
run_eval trigger always_revisit \
    --method memvr --starting-layer 5 --ending-layer 16 --entropy-threshold 0.75 \
    --retracing-ratio 0.3 --retrace-delay-layers 1 --trigger-strategy always --injection-mode ffn
run_eval trigger random_trigger \
    --method memvr --starting-layer 5 --ending-layer 16 --entropy-threshold 0.75 \
    --retracing-ratio 0.3 --retrace-delay-layers 1 --trigger-strategy random \
    --random-trigger-prob 0.5 --injection-mode ffn
run_eval trigger entropy_trigger \
    --method memvr --starting-layer 5 --ending-layer 16 --entropy-threshold 0.75 \
    --retracing-ratio 0.3 --retrace-delay-layers 1 --trigger-strategy entropy --injection-mode ffn

run_eval trigger_position first_quarter \
    --method memvr --retracing-ratio 0.3 --retrace-target-layers "${FIRST_QUARTER}" \
    --trigger-strategy entropy --injection-mode ffn
run_eval trigger_position first_half \
    --method memvr --retracing-ratio 0.3 --retrace-target-layers "${FIRST_HALF}" \
    --trigger-strategy entropy --injection-mode ffn
run_eval trigger_position second_half \
    --method memvr --retracing-ratio 0.3 --retrace-target-layers "${SECOND_HALF}" \
    --trigger-strategy entropy --injection-mode ffn
run_eval trigger_position all_layers \
    --method memvr --retracing-ratio 0.3 --retrace-target-layers "${ALL_LAYERS}" \
    --trigger-strategy entropy --injection-mode ffn

for layer in $(seq 0 27); do
    run_eval trigger_position "single_layer_${layer}" \
        --method memvr --retracing-ratio 0.3 --retrace-target-layers "${layer}" \
        --trigger-strategy entropy --injection-mode ffn
done

"${PYTHON_BIN}" - <<PY
import pandas as pd
from pathlib import Path

summary_path = Path(${SUMMARY_CSV@Q})
df = pd.read_csv(summary_path)
single = df[(df["group"] == "trigger_position") & (df["name"].str.startswith("single_layer_"))].copy()
if not single.empty:
    single["acc"] = pd.to_numeric(single["acc"], errors="coerce")
    best = single.sort_values(["acc", "name"], ascending=[False, True]).iloc[0]
    best_row = pd.DataFrame([{
        "group": "trigger_position",
        "name": "single_best_layer",
        "result_path": best["result_path"],
        "acc": best["acc"],
        "precision_macro": best["precision_macro"],
        "triggered_true": best["triggered_true"],
        "injection_success_true": best["injection_success_true"],
    }])
    existing = df[(df["group"] == "trigger_position") & (df["name"] == "single_best_layer")]
    if existing.empty:
        best_row.to_csv(summary_path, mode="a", header=False, index=False)
    print(best_row.to_csv(index=False).strip())
PY

run_eval injection ffn_injection \
    --method memvr --starting-layer 5 --ending-layer 16 --entropy-threshold 0.75 \
    --retracing-ratio 0.3 --retrace-delay-layers 1 --trigger-strategy entropy --injection-mode ffn
run_eval injection attention_injection \
    --method memvr --starting-layer 5 --ending-layer 16 --entropy-threshold 0.75 \
    --retracing-ratio 0.3 --retrace-delay-layers 1 --trigger-strategy entropy --injection-mode attention
run_eval injection ffn_attention_injection \
    --method memvr --starting-layer 5 --ending-layer 16 --entropy-threshold 0.75 \
    --retracing-ratio 0.3 --retrace-delay-layers 1 --trigger-strategy entropy --injection-mode ffn_attention
run_eval injection residual_add_only \
    --method memvr --starting-layer 5 --ending-layer 16 --entropy-threshold 0.75 \
    --retracing-ratio 0.3 --retrace-delay-layers 1 --trigger-strategy entropy --injection-mode residual

echo "Summary written to ${SUMMARY_CSV}"