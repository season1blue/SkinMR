#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/miniconda3/envs/dualpd/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/mnt/data/ssz/Skin/skindata}"
HAM_CSV="${HAM_CSV:-${ROOT_DIR}/Dataframe/test/classification/HAM10K_ISIC2018_test.csv}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/result/ablation_qwen25_ham10k_ent060_rr05_4gpu}"
SUMMARY_CSV="${SUMMARY_CSV:-${RESULT_ROOT}/summary.csv}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/qwen25_ham10k_ablation_ent060_rr05_4gpu}"
GPU_IDS=(4 5 6 7)

ENTROPY_THRESHOLD="0.60"
RETRACING_RATIO="0.5"
RETRACE_DELAY_LAYERS="1"
START_LAYER="5"
END_LAYER="16"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

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

append_summary_row() {
    local summary_path="$1"
    local group="$2"
    local name="$3"
    local result_path="$4"
    local pred_file="$5"
    local metrics_file="$6"

    "${PYTHON_BIN}" - <<PY >> "${summary_path}"
import pandas as pd
from pathlib import Path

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
print(row.to_csv(index=False, header=False).strip())
PY
}

run_eval() {
    local summary_path="$1"
    local gpu_id="$2"
    local group="$3"
    local name="$4"
    shift 4

    local result_path="${RESULT_ROOT}/${group}_${name}"
    local pred_file="${result_path}/HAM10000_predictions.csv"
    local metrics_file="${result_path}/HAM10000_results.csv"

    if [[ -f "${metrics_file}" && -f "${pred_file}" ]]; then
        echo "[skip] ${group}/${name} gpu=${gpu_id}"
        append_summary_row "${summary_path}" "${group}" "${name}" "${result_path}" "${pred_file}" "${metrics_file}"
        return
    fi

    rm -rf "${result_path}"
    mkdir -p "${result_path}"

    echo "[run] ${group}/${name} gpu=${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${ROOT_DIR}/ZS_classification.py" \
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

    append_summary_row "${summary_path}" "${group}" "${name}" "${result_path}" "${pred_file}" "${metrics_file}"
}

run_worker_gpu4() {
    local summary_path="$1"
    local gpu_id="${GPU_IDS[0]}"
    : > "${summary_path}"
    run_eval "${summary_path}" "${gpu_id}" trigger no_revisit --method base
    run_eval "${summary_path}" "${gpu_id}" trigger always_revisit \
        --method memvr --starting-layer "${START_LAYER}" --ending-layer "${END_LAYER}" \
        --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
        --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --trigger-strategy always --injection-mode ffn
    run_eval "${summary_path}" "${gpu_id}" trigger random_trigger \
        --method memvr --starting-layer "${START_LAYER}" --ending-layer "${END_LAYER}" \
        --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
        --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --trigger-strategy random \
        --random-trigger-prob 0.5 --injection-mode ffn
    run_eval "${summary_path}" "${gpu_id}" trigger entropy_trigger \
        --method memvr --starting-layer "${START_LAYER}" --ending-layer "${END_LAYER}" \
        --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
        --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --trigger-strategy entropy --injection-mode ffn
    run_eval "${summary_path}" "${gpu_id}" trigger_position first_quarter \
        --method memvr --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
        --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --retrace-target-layers "$(join_layers 0 6)" \
        --trigger-strategy entropy --injection-mode ffn
    run_eval "${summary_path}" "${gpu_id}" trigger_position first_half \
        --method memvr --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
        --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --retrace-target-layers "$(join_layers 0 13)" \
        --trigger-strategy entropy --injection-mode ffn
    for layer in 0 1 2 3; do
        run_eval "${summary_path}" "${gpu_id}" trigger_position "single_layer_${layer}" \
            --method memvr --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
            --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --retrace-target-layers "${layer}" \
            --trigger-strategy entropy --injection-mode ffn
    done
}

run_worker_gpu5() {
    local summary_path="$1"
    local gpu_id="${GPU_IDS[1]}"
    : > "${summary_path}"
    run_eval "${summary_path}" "${gpu_id}" trigger_position second_half \
        --method memvr --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
        --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --retrace-target-layers "$(join_layers 14 27)" \
        --trigger-strategy entropy --injection-mode ffn
    run_eval "${summary_path}" "${gpu_id}" trigger_position all_layers \
        --method memvr --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
        --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --retrace-target-layers "$(join_layers 0 27)" \
        --trigger-strategy entropy --injection-mode ffn
    for layer in 4 5 6 7 8 9 10 11; do
        run_eval "${summary_path}" "${gpu_id}" trigger_position "single_layer_${layer}" \
            --method memvr --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
            --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --retrace-target-layers "${layer}" \
            --trigger-strategy entropy --injection-mode ffn
    done
}

run_worker_gpu6() {
    local summary_path="$1"
    local gpu_id="${GPU_IDS[2]}"
    : > "${summary_path}"
    for layer in 12 13 14 15 16 17; do
        run_eval "${summary_path}" "${gpu_id}" trigger_position "single_layer_${layer}" \
            --method memvr --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
            --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --retrace-target-layers "${layer}" \
            --trigger-strategy entropy --injection-mode ffn
    done
    run_eval "${summary_path}" "${gpu_id}" injection ffn_injection \
        --method memvr --starting-layer "${START_LAYER}" --ending-layer "${END_LAYER}" \
        --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
        --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --trigger-strategy entropy --injection-mode ffn
    run_eval "${summary_path}" "${gpu_id}" injection attention_injection \
        --method memvr --starting-layer "${START_LAYER}" --ending-layer "${END_LAYER}" \
        --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
        --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --trigger-strategy entropy --injection-mode attention
    run_eval "${summary_path}" "${gpu_id}" injection ffn_attention_injection \
        --method memvr --starting-layer "${START_LAYER}" --ending-layer "${END_LAYER}" \
        --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
        --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --trigger-strategy entropy --injection-mode ffn_attention
    run_eval "${summary_path}" "${gpu_id}" injection residual_add_only \
        --method memvr --starting-layer "${START_LAYER}" --ending-layer "${END_LAYER}" \
        --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
        --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --trigger-strategy entropy --injection-mode residual
}

run_worker_gpu7() {
    local summary_path="$1"
    local gpu_id="${GPU_IDS[3]}"
    : > "${summary_path}"
    for layer in 18 19 20 21 22 23 24 25 26 27; do
        run_eval "${summary_path}" "${gpu_id}" trigger_position "single_layer_${layer}" \
            --method memvr --entropy-threshold "${ENTROPY_THRESHOLD}" --retracing-ratio "${RETRACING_RATIO}" \
            --retrace-delay-layers "${RETRACE_DELAY_LAYERS}" --retrace-target-layers "${layer}" \
            --trigger-strategy entropy --injection-mode ffn
    done
}

PARTIAL_SUMMARY_4="${RESULT_ROOT}/summary_gpu4.csv"
PARTIAL_SUMMARY_5="${RESULT_ROOT}/summary_gpu5.csv"
PARTIAL_SUMMARY_6="${RESULT_ROOT}/summary_gpu6.csv"
PARTIAL_SUMMARY_7="${RESULT_ROOT}/summary_gpu7.csv"

run_worker_gpu4 "${PARTIAL_SUMMARY_4}" > "${LOG_DIR}/gpu4.log" 2>&1 &
PID4=$!
run_worker_gpu5 "${PARTIAL_SUMMARY_5}" > "${LOG_DIR}/gpu5.log" 2>&1 &
PID5=$!
run_worker_gpu6 "${PARTIAL_SUMMARY_6}" > "${LOG_DIR}/gpu6.log" 2>&1 &
PID6=$!
run_worker_gpu7 "${PARTIAL_SUMMARY_7}" > "${LOG_DIR}/gpu7.log" 2>&1 &
PID7=$!

FAIL=0
for pid in "$PID4" "$PID5" "$PID6" "$PID7"; do
    if ! wait "$pid"; then
        FAIL=1
    fi
done

if [[ "$FAIL" -ne 0 ]]; then
    echo "One or more ablation workers failed. Check ${LOG_DIR}" >&2
    exit 1
fi

echo "group,name,result_path,acc,precision_macro,triggered_true,injection_success_true" > "${SUMMARY_CSV}"
cat "${PARTIAL_SUMMARY_4}" "${PARTIAL_SUMMARY_5}" "${PARTIAL_SUMMARY_6}" "${PARTIAL_SUMMARY_7}" >> "${SUMMARY_CSV}"

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
    best_row.to_csv(summary_path, mode="a", header=False, index=False)
    print(best_row.to_csv(index=False).strip())
PY

echo "Summary written to ${SUMMARY_CSV}"