#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 2 of phage replication: embedding analysis for ONE architecture.
# Trains linear probe + 3-layer NN on pretrained embeddings.
#
# Required env:
#   REPL_OUTPUT_DIR, LAMBDA_DIR, ARCH
# Optional env:
#   BATCH_SIZE (32), MAX_LENGTH (auto), POOLING (mean),
#   NN_EPOCHS (100), NN_LR (0.001),
#   INCLUDE_RANDOM_BASELINE (false) — pass "true" to also evaluate a
#       randomly-initialized encoder and compute embedding power.

set -euo pipefail

echo "=== embedding analysis ${ARCH} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

source /u/llindsey1/miniconda3/etc/profile.d/conda.sh
conda activate prokbert
export PYTHONNOUSERSITE=1
export HF_HOME=/work/hdd/bfzj/llindsey1/hf_cache   # keep HF downloads off home quota

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
POOLING=${POOLING:-mean}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_LR=${NN_LR:-0.001}
INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}

# embedding_analysis_prokbert.py appends the model basename to output_dir, so we
# point it at <repl>/embedding and it writes <repl>/embedding/<arch>/...
OUTPUT_BASE="${REPL_OUTPUT_DIR}/embedding"
mkdir -p "${OUTPUT_BASE}"

RANDOM_BASELINE_FLAG=""
if [ "${INCLUDE_RANDOM_BASELINE,,}" = "true" ]; then
    RANDOM_BASELINE_FLAG="--include_random_baseline"
fi

echo "  lambda dir:  ${LAMBDA_DIR}"
echo "  output base: ${OUTPUT_BASE}"
echo "  pooling=${POOLING}  batch=${BATCH_SIZE}  max_length=${MAX_LENGTH}"
echo "  nn_epochs=${NN_EPOCHS}  nn_lr=${NN_LR}"
echo "  include_random_baseline=${INCLUDE_RANDOM_BASELINE}"

python embedding_analysis_prokbert.py \
    --csv_dir="${LAMBDA_DIR}" \
    --model_path="neuralbioinfo/${ARCH}" \
    --output_dir="${OUTPUT_BASE}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --pooling="${POOLING}" \
    --seed=${EMB_SEED:-42} \
    --nn_epochs=${NN_EPOCHS} \
    --nn_lr=${NN_LR} \
    ${RANDOM_BASELINE_FLAG}

echo "Done: $(date)"
