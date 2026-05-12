#!/usr/bin/env bash
set -euo pipefail

# Interface-only config: set model and dataset keys.
# Supported LLM_KEY: skinvl_pubmm, llavamed, qwen3_5, qwen25vl
# Supported DATASET_KEY: patch16_2class, ham10k, pad
LLM_KEY="skinvl_pubmm"
DATASET_KEY="pad"
METHOD="base"  # qwen25vl supports: base, memvr, evo

# Runtime config
CONDA_ENV="dualpd"
PYTHON_BIN="/home/public/miniconda3/envs/dualpd/bin/python"
GPU_IDS=(0 1 2 3)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_FOLDER="${IMAGE_FOLDER:-/mnt/data/ssz/Skin/skindata}"
LOG_DIR="${PROJECT_ROOT}/logs"

# Patch16 control
PATCH16_USE_SUBSET=1
PATCH16_SUBSET_CSV="${PROJECT_ROOT}/Dataframe/test/classification/Patch16_2class_test_10pct_seed42.csv"
PATCH16_FULL_CSV="${PROJECT_ROOT}/Dataframe/test/classification/Patch16_2class_test.csv"

# Throughput controls
BATCH_SIZE=96
NUM_BEAMS=1
MAX_NEW_TOKENS=64
DO_SAMPLE=0
TEMPERATURE=0.5
TOP_P=0.9
FORCE_RERUN=1

# MemVR controls (effective when METHOD=memvr/evo on qwen25vl)
STARTING_LAYER=5
ENDING_LAYER=16
ENTROPY_THRESHOLD=0.75
RETRACING_RATIO=0.0
RETRACE_DELAY_LAYERS=1
RETRACE_TARGET_LAYERS=""
STATE_DRIFT_THRESHOLD=0.5
STATE_DRIFT_POOLING="mean"


case "$LLM_KEY" in
    skinvl_pubmm)
        WEIGHTSPATH="/data/ssz/MM-Skin/merge/SkinVL_PubMM"
        ;;
    llavamed)
        WEIGHTSPATH="/data/ssz/llms/llava-med-v1.5-mistral-7b"
        ;;
    qwen3_5)
        WEIGHTSPATH="/data/ssz/llms/Qwen3.5-9B"
        ;;
    qwen25vl)
        WEIGHTSPATH="/data/ssz/llms/Qwen2.5-VL-7B-Instruct"
        ;;
    *)
        echo "Unsupported LLM_KEY: $LLM_KEY"
        echo "Supported: skinvl_pubmm, llavamed, qwen3_5, qwen25vl"
        exit 1
        ;;
esac

case "$DATASET_KEY" in
    patch16_2class)
        EXP="Patch16_2class"
        if [[ "$PATCH16_USE_SUBSET" -eq 1 ]]; then
            DATAFRAME_OVERRIDE="$PATCH16_SUBSET_CSV"
        else
            DATAFRAME_OVERRIDE="$PATCH16_FULL_CSV"
        fi
        ;;
    ham10k)
        EXP="HAM10000"
        DATAFRAME_OVERRIDE="${PROJECT_ROOT}/Dataframe/test/classification/HAM10K_ISIC2018_test.csv"
        ;;
    pad)
        EXP="PAD"
        DATAFRAME_OVERRIDE="${PROJECT_ROOT}/Dataframe/test/classification/PAD_test.csv"
        ;;
    *)
        echo "Unsupported DATASET_KEY: $DATASET_KEY"
        echo "Supported: patch16_2class, ham10k, pad"
        exit 1
        ;;
esac

OUTPATH="result/zeroshot_class/${DATASET_KEY}_${LLM_KEY}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python"
fi

# Reduce allocator fragmentation for long multimodal generation runs.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"



echo "LLM_KEY=$LLM_KEY"
echo "DATASET_KEY=$DATASET_KEY"
echo "METHOD=$METHOD"
echo "MODEL_PATH=$WEIGHTSPATH"
echo "DATAFRAME=$DATAFRAME_OVERRIDE"
echo "OUTPATH=$OUTPATH"

for EXP in "$EXP"; do
    echo "==============================="
    echo "Starting experiments for: $EXP"
    echo "==============================="

    if [[ "$FORCE_RERUN" -eq 1 ]]; then
        echo "Force rerun enabled: removing previous outputs for ${EXP}"
        rm -f "${OUTPATH}/${EXP}_predictions.csv"
        rm -f "${OUTPATH}/${EXP}_results.csv"
        rm -f "${OUTPATH}/${EXP}_predictions_g"*.csv
        rm -f "${OUTPATH}/${EXP}_results_g"*.csv
    fi

    COMMON_ARGS=(
        --experiment "$EXP"
        --result-path "$OUTPATH"
        --model_path "$WEIGHTSPATH"
        --image-folder "$IMAGE_FOLDER"
        --batch-size "$BATCH_SIZE"
        --num_beams "$NUM_BEAMS"
        --max-new-tokens "$MAX_NEW_TOKENS"
        --method "$METHOD"
        --starting-layer "$STARTING_LAYER"
        --ending-layer "$ENDING_LAYER"
        --entropy-threshold "$ENTROPY_THRESHOLD"
        --retracing-ratio "$RETRACING_RATIO"
        --retrace-delay-layers "$RETRACE_DELAY_LAYERS"
        --retrace-target-layers "$RETRACE_TARGET_LAYERS"
        --state-drift-threshold "$STATE_DRIFT_THRESHOLD"
        --state-drift-pooling "$STATE_DRIFT_POOLING"
    )

    if [[ "$DO_SAMPLE" -eq 1 ]]; then
        COMMON_ARGS+=(--do-sample --temperature "$TEMPERATURE" --top_p "$TOP_P")
    fi
    if [[ -n "$DATAFRAME_OVERRIDE" ]]; then
        COMMON_ARGS+=(--dataframe "$DATAFRAME_OVERRIDE")
    fi

    NUM_GPUS=${#GPU_IDS[@]}
    if [[ "$NUM_GPUS" -le 1 ]]; then
        LOG_FILE="${LOG_DIR}/log_${EXP}_${LLM_KEY}_g${GPU_IDS[0]}.txt"
        echo "Single-GPU mode on GPU ${GPU_IDS[0]}, log: ${LOG_FILE}"
        if [[ "$FORCE_RERUN" -eq 1 ]]; then
            rm -f "$LOG_FILE"
        fi
        CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} "$PYTHON_BIN" ZS_classification.py "${COMMON_ARGS[@]}" \
            2>&1 | sed -u "s/^/[g0] /" | tee -a "$LOG_FILE"
    else
        echo "Multi-GPU mode with ${NUM_GPUS} workers: ${GPU_IDS[*]}"
        pids=()
        fail=0
        for i in "${!GPU_IDS[@]}"; do
            gpu_id=${GPU_IDS[$i]}
            suffix="_g${i}"
            log_file="${LOG_DIR}/log_${EXP}_${LLM_KEY}_g${i}.txt"
            echo "Launching worker ${i} on GPU ${gpu_id}, suffix ${suffix}, log ${log_file}"
            if [[ "$FORCE_RERUN" -eq 1 ]]; then
                rm -f "$log_file"
            fi
            (
                set -o pipefail
                CUDA_VISIBLE_DEVICES=${gpu_id} "$PYTHON_BIN" ZS_classification.py "${COMMON_ARGS[@]}" \
                    --num-shards "$NUM_GPUS" \
                    --shard-index "$i" \
                    --result-suffix "$suffix" 2>&1 | sed -u "s/^/[g${i}] /" | tee "$log_file"
            ) &
            pids+=("$!")
        done

        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                fail=1
            fi
        done
        if [[ "$fail" -ne 0 ]]; then
            echo "One or more workers failed. Check ${LOG_DIR}/log_${EXP}_${LLM_KEY}_g*.txt"
            exit 1
        fi
        echo "All shard workers finished for ${EXP}."

        "$PYTHON_BIN" utils/merge_and_score.py --exp "$EXP" --outpath "$OUTPATH"
    fi

    echo "Completed experiments for: $EXP"
    echo ""
done