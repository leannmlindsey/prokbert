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
#   NN_EPOCHS (100), NN_LR (0.001)

set -euo pipefail

echo "=== embedding analysis ${ARCH} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true
conda activate prokbert 2>/dev/null || source activate prokbert 2>/dev/null || true
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}/../.."
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

# embedding_analysis_prokbert.py appends the model basename to output_dir, so we
# point it at <repl>/embedding and it writes <repl>/embedding/<arch>/...
OUTPUT_BASE="${REPL_OUTPUT_DIR}/embedding"
mkdir -p "${OUTPUT_BASE}"

echo "  lambda dir:  ${LAMBDA_DIR}"
echo "  output base: ${OUTPUT_BASE}"
echo "  pooling=${POOLING}  batch=${BATCH_SIZE}  max_length=${MAX_LENGTH}"
echo "  nn_epochs=${NN_EPOCHS}  nn_lr=${NN_LR}"

python embedding_analysis_prokbert.py \
    --csv_dir="${LAMBDA_DIR}" \
    --model_path="neuralbioinfo/${ARCH}" \
    --output_dir="${OUTPUT_BASE}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --pooling="${POOLING}" \
    --nn_epochs=${NN_EPOCHS} \
    --nn_lr=${NN_LR}

echo "Done: $(date)"
