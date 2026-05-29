#!/bin/bash
#
# Lambda replication — STAGE 1: fire off all training jobs.
#
# For each segment length in SEGMENT_LENGTHS, submits:
#   - one finetune sbatch job per (arch, seed)
#   - one embedding-analysis sbatch job per arch
# All jobs run in parallel (no --dependency chaining). Once they all complete,
# run run_lambda_inference.sh to find the best model and run inference.
#
# Usage:
#   1. Edit configs/lambda_replication.conf — set LAMBDA_BASE and OUTPUT_DIR.
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

if [[ "${LAMBDA_BASE}" == /path/to/* ]] || [[ "${OUTPUT_DIR}" == /path/to/* ]]; then
    echo "ERROR: edit ${CONFIG} — LAMBDA_BASE or OUTPUT_DIR still set to placeholder"
    exit 1
fi
[ -d "${LAMBDA_BASE}/train_val_test" ] || {
    echo "ERROR: ${LAMBDA_BASE}/train_val_test not found (expected LAMBDA_v1 layout)"
    exit 1
}
if [ -z "${SEGMENT_LENGTHS}" ]; then
    echo "ERROR: SEGMENT_LENGTHS is empty"; exit 1
fi

# Validate per-length input dirs exist before submitting anything.
for LEN in ${SEGMENT_LENGTHS}; do
    LDIR="${LAMBDA_BASE}/train_val_test/${LEN}"
    [ -d "${LDIR}" ] || { echo "ERROR: ${LDIR} not found"; exit 1; }
    for f in train.csv test.csv; do
        [ -f "${LDIR}/${f}" ] || { echo "ERROR: ${LDIR}/${f} not found"; exit 1; }
    done
    if [ ! -f "${LDIR}/dev.csv" ] && [ ! -f "${LDIR}/val.csv" ]; then
        echo "ERROR: ${LDIR} must contain dev.csv or val.csv"; exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"

# --- summary ------------------------------------------------------------------

echo "============================================================"
echo "Lambda replication — Stage 1: training + embedding"
echo "============================================================"
echo "  LAMBDA_BASE:     ${LAMBDA_BASE}"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  SEGMENT_LENGTHS: ${SEGMENT_LENGTHS}"
echo "  ARCHS:           ${ARCHS}"
echo "  SEEDS:           ${SEEDS}"
echo "  FT params:       lr=${LR} batch=${BATCH_SIZE} epochs=${NUM_EPOCHS}"
echo "  Emb params:      pooling=${POOLING} nn_epochs=${NN_EPOCHS} nn_lr=${NN_LR}"
echo "============================================================"

# --- common sbatch flags ------------------------------------------------------

FT_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${FT_MEM}" --time="${FT_TIME}" --cpus-per-task=8)
EMB_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${EMB_MEM}" --time="${EMB_TIME}" --cpus-per-task=8)

# REPO_ROOT is propagated to every job so they can `cd` to the real repo —
# SLURM stages each job script to /var/spool/slurm/... where BASH_SOURCE[0]
# can't be used to recover the original location.
FT_ENV_BASE="REPO_ROOT=${REPO_ROOT},LR=${LR},BATCH_SIZE=${BATCH_SIZE},NUM_EPOCHS=${NUM_EPOCHS},EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},USE_FP16=${USE_FP16}"
EMB_ENV_BASE="REPO_ROOT=${REPO_ROOT},POOLING=${POOLING},NN_EPOCHS=${NN_EPOCHS},NN_LR=${NN_LR}"

NUM_JOBS=0

for LEN in ${SEGMENT_LENGTHS}; do
    LAMBDA_DIR="${LAMBDA_BASE}/train_val_test/${LEN}"
    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    mkdir -p "${REPL_LEN_DIR}"

    echo ""
    echo "--- length: ${LEN} ---"
    echo "    lambda dir:   ${LAMBDA_DIR}"
    echo "    output dir:   ${REPL_LEN_DIR}"

    # Finetune jobs (arch × seed)
    for ARCH in ${ARCHS}; do
        for SEED in ${SEEDS}; do
            JOB="ft_${LEN}_${ARCH}_s${SEED}"
            echo "    submitting ${JOB}..."
            sbatch \
                --job-name="${JOB}" \
                --output="${LOGDIR}/${JOB}_%j.out" \
                --error="${LOGDIR}/${JOB}_%j.err" \
                "${FT_FLAGS[@]}" \
                --export="ALL,REPL_OUTPUT_DIR=${REPL_LEN_DIR},LAMBDA_DIR=${LAMBDA_DIR},${FT_ENV_BASE},ARCH=${ARCH},SEED=${SEED}" \
                "${SCRIPT_DIR}/lambda_finetune_job.sh"
            NUM_JOBS=$((NUM_JOBS + 1))
        done
    done

    # Embedding analysis jobs (per arch)
    for ARCH in ${ARCHS}; do
        JOB="emb_${LEN}_${ARCH}"
        echo "    submitting ${JOB}..."
        sbatch \
            --job-name="${JOB}" \
            --output="${LOGDIR}/${JOB}_%j.out" \
            --error="${LOGDIR}/${JOB}_%j.err" \
            "${EMB_FLAGS[@]}" \
            --export="ALL,REPL_OUTPUT_DIR=${REPL_LEN_DIR},LAMBDA_DIR=${LAMBDA_DIR},${EMB_ENV_BASE},ARCH=${ARCH}" \
            "${SCRIPT_DIR}/lambda_embedding_job.sh"
        NUM_JOBS=$((NUM_JOBS + 1))
    done
done

echo ""
echo "Submitted ${NUM_JOBS} jobs. Monitor with: squeue -u \$USER"
echo "When all jobs are done, run:"
echo "  bash ${SCRIPT_DIR}/run_lambda_inference.sh"
