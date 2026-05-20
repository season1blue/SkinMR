#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/miniconda3/envs/dualpd/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/ssz/llms/Qwen2.5-VL-7B-Instruct}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/mnt/data/ssz/Skin/skindata}"
HAM_CSV="${HAM_CSV:-${ROOT_DIR}/Dataframe/test/classification/HAM10K_ISIC2018_test.csv}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/result/ablation_qwen25_ham10k_trigger_ent001_rr10_back4_4gpu}"
SUMMARY_CSV="${SUMMARY_CSV:-${RESULT_ROOT}/summary.csv}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/qwen25_ham10k_trigger_ent001_rr10_back4_4gpu}"
GPU_IDS=(4 5 6 7)

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"
echo "group,name,result_path,acc,precision_macro,triggered_true,injection_success_true" > "${SUMMARY_CSV}"

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

run_one() {
    local gpu_id="$1"
    local name="$2"
    shift 2

    local result_path="${RESULT_ROOT}/trigger_${name}"
    local pred_file="${result_path}/HAM10000_predictions.csv"
    local metrics_file="${result_path}/HAM10000_results.csv"
    local log_file="${LOG_DIR}/${name}.log"

    rm -rf "${result_path}"
    mkdir -p "${result_path}"

    {
        echo "[run] trigger/${name} gpu=${gpu_id}"
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
        record_summary "trigger" "${name}" "${result_path}" "${pred_file}" "${metrics_file}"
    } > "${log_file}" 2>&1
}

run_one "${GPU_IDS[0]}" no_revisit --method base &
PID0=$!
run_one "${GPU_IDS[1]}" always_revisit \
    --method memvr --starting-layer 5 --ending-layer 16 --entropy-threshold 0.01 \
    --retracing-ratio 1.0 --retrace-delay-layers 4 --trigger-strategy always --injection-mode ffn &
PID1=$!
run_one "${GPU_IDS[2]}" random_trigger \
    --method memvr --starting-layer 5 --ending-layer 16 --entropy-threshold 0.01 \
    --retracing-ratio 1.0 --retrace-delay-layers 4 --trigger-strategy random \
    --random-trigger-prob 0.5 --injection-mode ffn &
PID2=$!
run_one "${GPU_IDS[3]}" entropy_trigger \
    --method memvr --starting-layer 5 --ending-layer 16 --entropy-threshold 0.01 \
    --retracing-ratio 1.0 --retrace-delay-layers 4 --trigger-strategy entropy --injection-mode ffn &
PID3=$!

FAIL=0
for pid in "$PID0" "$PID1" "$PID2" "$PID3"; do
    if ! wait "$pid"; then
        FAIL=1
    fi
done

if [[ "$FAIL" -ne 0 ]]; then
    echo "One or more trigger runs failed. Check ${LOG_DIR}" >&2
    exit 1
fi

echo "All trigger ablations finished. Summary written to ${SUMMARY_CSV}"