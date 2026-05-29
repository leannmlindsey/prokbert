#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 4/5 of phage replication: run the winning model for ARCH on one CSV.
# Reads <REPL_OUTPUT_DIR>/winners.json to dispatch between:
#   - type=finetune       -> inference_lambda.py
#   - type=linear_probe   -> inference_embedding_head.py --head_type linear_probe
#   - type=three_layer_nn -> inference_embedding_head.py --head_type three_layer_nn
#
# Required env:
#   REPL_OUTPUT_DIR
#   ARCH
#   INPUT_CSV            path to the CSV to predict on
#   OUTPUT_FILENAME      name for the predictions CSV (e.g. test_predictions.csv)
# Optional env:
#   BATCH_SIZE (32), MAX_LENGTH (auto), SAVE_METRICS (1|0, default 1)

set -euo pipefail

echo "=== inference ${ARCH}  input=${INPUT_CSV}  output=${OUTPUT_FILENAME} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true
conda activate prokbert 2>/dev/null || source activate prokbert 2>/dev/null || true
export PYTHONNOUSERSITE=1

# REPO_ROOT is supplied by the launcher via --export. SLURM stages this
# script to /var/spool/slurm/... so BASH_SOURCE[0] would not resolve to the
# real repo; never compute REPO_ROOT from the script's own location.
if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

if [ -z "${MAX_LENGTH:-}" ]; then
    case "${ARCH}" in
        prokbert-mini)      MAX_LENGTH=1024 ;;
        prokbert-mini-c)    MAX_LENGTH=2048 ;;
        prokbert-mini-long) MAX_LENGTH=2048 ;;
        *) echo "ERROR: unknown ARCH=${ARCH}"; exit 1 ;;
    esac
fi
BATCH_SIZE=${BATCH_SIZE:-32}
SAVE_METRICS=${SAVE_METRICS:-1}

WINNERS_JSON="${REPL_OUTPUT_DIR}/winners.json"
if [ ! -f "${WINNERS_JSON}" ]; then
    echo "ERROR: ${WINNERS_JSON} not found (stage 3 must run first)"; exit 1
fi

# Extract winner fields. The helper script emits shlex-quoted exports we eval.
eval "$(python scripts/print_winner_exports.py "${WINNERS_JSON}" "${ARCH}")"

echo "  winner type:   ${WINNER_TYPE}"
echo "  winner path:   ${WINNER_PATH}"
echo "  base model:    ${BASE_MODEL}"

OUTPUT_DIR="${REPL_OUTPUT_DIR}/inference/${ARCH}"
mkdir -p "${OUTPUT_DIR}"

SAVE_METRICS_FLAG=""
[ "${SAVE_METRICS}" = "1" ] && SAVE_METRICS_FLAG="--save_metrics"

case "${WINNER_TYPE}" in
    finetune)
        python inference_lambda.py \
            --checkpoint_path="${WINNER_PATH}" \
            --base_model="${BASE_MODEL}" \
            --dataset_file="${INPUT_CSV}" \
            --max_length=${MAX_LENGTH} \
            --batch_size=${BATCH_SIZE} \
            --output_dir="${OUTPUT_DIR}" \
            --output_file="${OUTPUT_FILENAME}" \
            ${SAVE_METRICS_FLAG}
        ;;
    linear_probe)
        python inference_embedding_head.py \
            --base_model="${BASE_MODEL}" \
            --head_type=linear_probe \
            --head_path="${WINNER_PATH}" \
            --dataset_file="${INPUT_CSV}" \
            --max_length=${MAX_LENGTH} \
            --batch_size=${BATCH_SIZE} \
            --output_dir="${OUTPUT_DIR}" \
            --output_file="${OUTPUT_FILENAME}" \
            ${SAVE_METRICS_FLAG}
        ;;
    three_layer_nn)
        python inference_embedding_head.py \
            --base_model="${BASE_MODEL}" \
            --head_type=three_layer_nn \
            --head_path="${WINNER_PATH}" \
            --scaler_path="${WINNER_SCALER}" \
            --dataset_file="${INPUT_CSV}" \
            --max_length=${MAX_LENGTH} \
            --batch_size=${BATCH_SIZE} \
            --output_dir="${OUTPUT_DIR}" \
            --output_file="${OUTPUT_FILENAME}" \
            ${SAVE_METRICS_FLAG}
        ;;
    *)
        echo "ERROR: unknown winner type '${WINNER_TYPE}'"; exit 1
        ;;
esac

echo "Done: $(date)"
