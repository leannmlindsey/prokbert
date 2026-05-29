#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 3 of phage replication: per-architecture, pick the model with the
# highest test-set MCC across finetune seeds + linear probe + 3-layer NN.
# Writes <REPL_OUTPUT_DIR>/winners.json.
#
# Required env:
#   REPL_OUTPUT_DIR, ARCHITECTURES (space-separated)

set -euo pipefail

echo "=== select_best ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load conda 2>/dev/null || true
conda activate prokbert 2>/dev/null || source activate prokbert 2>/dev/null || true
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}/../.."
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ARCHITECTURES is colon-separated; turn it into argv for select_best_model.py.
IFS=':' read -r -a ARCH_LIST <<< "${ARCHITECTURES}"

python scripts/select_best_model.py \
    --output_dir="${REPL_OUTPUT_DIR}" \
    --architectures "${ARCH_LIST[@]}"

echo "Done: $(date)"
