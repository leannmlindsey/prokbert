#!/bin/bash
#
# Lambda replication — STAGE 1: fire off all training jobs.
#
# Submits:
#   - one finetune sbatch job per (arch, seed) combination
#   - one embedding-analysis sbatch job per arch
# All jobs run in parallel (no --dependency chaining). Once they all complete,
# run run_lambda_inference.sh to find the best model and run inference.
#
# Usage:
#   1. Edit configs/lambda_replication.conf — set LAMBDA_DIR and OUTPUT_DIR.
#   2. bash slurm_scripts/lambda_replication/run_lambda_training.sh
#   3. Wait for jobs: squeue -u $USER
#   4. bash slurm_scripts/lambda_replication/run_lambda_inference.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}/../.."
CONFIG="${REPO_ROOT}/configs/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

# --- validate -----------------------------------------------------------------

if [[ "${LAMBDA_DIR}" == /path/to/* ]] || [[ "${OUTPUT_DIR}" == /path/to/* ]]; then
    echo "ERROR: edit ${CONFIG} — LAMBDA_DIR or OUTPUT_DIR still set to placeholder"
    exit 1
fi
for f in train.csv test.csv; do
    [ -f "${LAMBDA_DIR}/${f}" ] || { echo "ERROR: ${LAMBDA_DIR}/${f} not found"; exit 1; }
done
if [ ! -f "${LAMBDA_DIR}/dev.csv" ] && [ ! -f "${LAMBDA_DIR}/val.csv" ]; then
    echo "ERROR: ${LAMBDA_DIR} must contain dev.csv or val.csv"; exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"

# --- summary ------------------------------------------------------------------

echo "============================================================"
echo "Lambda replication — Stage 1: training + embedding"
echo "============================================================"
echo "  LAMBDA_DIR:  ${LAMBDA_DIR}"
echo "  OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "  ARCHS:       ${ARCHS}"
echo "  SEEDS:       ${SEEDS}"
echo "  FT params:   lr=${LR} batch=${BATCH_SIZE} epochs=${NUM_EPOCHS}"
echo "  Emb params:  pooling=${POOLING} nn_epochs=${NN_EPOCHS} nn_lr=${NN_LR}"
echo "============================================================"

# --- common sbatch flags ------------------------------------------------------

LOGDIR="${OUTPUT_DIR}/logs"

FT_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${FT_MEM}" --time="${FT_TIME}" --cpus-per-task=8)
EMB_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${EMB_MEM}" --time="${EMB_TIME}" --cpus-per-task=8)

FT_ENV="REPL_OUTPUT_DIR=${OUTPUT_DIR},LAMBDA_DIR=${LAMBDA_DIR},LR=${LR},BATCH_SIZE=${BATCH_SIZE},NUM_EPOCHS=${NUM_EPOCHS},EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},USE_FP16=${USE_FP16}"
EMB_ENV="REPL_OUTPUT_DIR=${OUTPUT_DIR},LAMBDA_DIR=${LAMBDA_DIR},POOLING=${POOLING},NN_EPOCHS=${NN_EPOCHS},NN_LR=${NN_LR}"

NUM_JOBS=0

# --- submit finetune jobs -----------------------------------------------------

for ARCH in ${ARCHS}; do
    for SEED in ${SEEDS}; do
        JOB="ft_${ARCH}_s${SEED}"
        echo "  submitting ${JOB}..."
        sbatch \
            --job-name="${JOB}" \
            --output="${LOGDIR}/${JOB}_%j.out" \
            --error="${LOGDIR}/${JOB}_%j.err" \
            "${FT_FLAGS[@]}" \
            --export="ALL,${FT_ENV},ARCH=${ARCH},SEED=${SEED}" \
            "${SCRIPT_DIR}/lambda_finetune_job.sh"
        NUM_JOBS=$((NUM_JOBS + 1))
    done
done

# --- submit embedding jobs ----------------------------------------------------

for ARCH in ${ARCHS}; do
    JOB="emb_${ARCH}"
    echo "  submitting ${JOB}..."
    sbatch \
        --job-name="${JOB}" \
        --output="${LOGDIR}/${JOB}_%j.out" \
        --error="${LOGDIR}/${JOB}_%j.err" \
        "${EMB_FLAGS[@]}" \
        --export="ALL,${EMB_ENV},ARCH=${ARCH}" \
        "${SCRIPT_DIR}/lambda_embedding_job.sh"
    NUM_JOBS=$((NUM_JOBS + 1))
done

echo ""
echo "Submitted ${NUM_JOBS} jobs. Monitor with: squeue -u \$USER"
echo "When all jobs are done, run:"
echo "  bash ${SCRIPT_DIR}/run_lambda_inference.sh"
