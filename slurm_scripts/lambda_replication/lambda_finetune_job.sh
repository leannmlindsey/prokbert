#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 1 of phage replication: finetune ONE (arch, seed) combination.
# Submitted by replicate_phage.py. All paths and resources are supplied via
# --export=ALL,<env vars>.
#
# Required env:
#   REPL_OUTPUT_DIR   replication root output dir
#   LAMBDA_DIR        train/dev/test CSV directory
#   ARCH              prokbert-mini | prokbert-mini-c | prokbert-mini-long
#   SEED              integer
# Optional env:
#   LR (1e-4), BATCH_SIZE (32), MAX_LENGTH (auto), NUM_EPOCHS (3),
#   EARLY_STOPPING_PATIENCE (3), USE_FP16 (1|0)

set -euo pipefail

echo "=== finetune ${ARCH} seed=${SEED} ==="
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

# Per-architecture max_length defaults (override with MAX_LENGTH env var).
if [ -z "${MAX_LENGTH:-}" ]; then
    case "${ARCH}" in
        prokbert-mini)      MAX_LENGTH=1024 ;;
        prokbert-mini-c)    MAX_LENGTH=2048 ;;
        prokbert-mini-long) MAX_LENGTH=2048 ;;
        *) echo "ERROR: unknown ARCH=${ARCH}"; exit 1 ;;
    esac
fi

LR=${LR:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_EPOCHS=${NUM_EPOCHS:-3}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-3}
USE_FP16=${USE_FP16:-1}

OUTPUT_DIR="${REPL_OUTPUT_DIR}/finetune/${ARCH}/seed-${SEED}"
mkdir -p "${OUTPUT_DIR}"

echo "  base model:     neuralbioinfo/${ARCH}"
echo "  lambda dir:     ${LAMBDA_DIR}"
echo "  output:         ${OUTPUT_DIR}"
echo "  lr=${LR}  batch=${BATCH_SIZE}  max_length=${MAX_LENGTH}  epochs=${NUM_EPOCHS}"

FP16_FLAG=""
[ "${USE_FP16}" = "1" ] && FP16_FLAG="--fp16"

python finetune_prokbert_phage.py \
    --dataset_dir="${LAMBDA_DIR}" \
    --model_name="neuralbioinfo/${ARCH}" \
    --output_dir="${OUTPUT_DIR}" \
    --learning_rate=${LR} \
    --per_device_train_batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --seed=${SEED} \
    --num_train_epochs=${NUM_EPOCHS} \
    --early_stopping_patience=${EARLY_STOPPING_PATIENCE} \
    ${FP16_FLAG}

echo "Done: $(date)"
