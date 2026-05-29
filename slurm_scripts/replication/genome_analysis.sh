#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 6 of phage replication: threshold + clustering sweep over the
# genome-wide predictions for ONE architecture.
#
# Required env:
#   REPL_OUTPUT_DIR, ARCH

set -euo pipefail

echo "=== genome analysis ${ARCH} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load conda 2>/dev/null || true
conda activate prokbert 2>/dev/null || source activate prokbert 2>/dev/null || true
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}/../.."
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Stage 4/5 wrote predictions here; stage 6 reads them.
PREDICTIONS_DIR="${REPL_OUTPUT_DIR}/inference/${ARCH}"
ANALYSIS_DIR="${REPL_OUTPUT_DIR}/genome_wide_analysis/${ARCH}"
mkdir -p "${ANALYSIS_DIR}"

echo "  predictions dir: ${PREDICTIONS_DIR}"
echo "  analysis dir:    ${ANALYSIS_DIR}"

# Stage A: initial threshold sweep, no smoothing.
echo "--- threshold sweep (no smoothing) ---"
python analyze_genome_wide_results.py \
    -d "${PREDICTIONS_DIR}" \
    -m "${ARCH}" \
    -r "${ANALYSIS_DIR}" \
    --window-size 0 \
    --threshold-sweep "0.3,0.5,0.7,0.8,0.9,0.95,0.99"

# Stage B: clustering parameter sweep at promising thresholds.
echo "--- clustering parameter sweep ---"
python analyze_genome_wide_results.py \
    -d "${PREDICTIONS_DIR}" \
    -m "${ARCH}" \
    -r "${ANALYSIS_DIR}" \
    -o "${ARCH}_clustering_sweep" \
    --window-size 0 \
    --threshold-sweep "0.5,0.7,0.9" \
    --merge-gap-sweep "1000,3000,5000,10000" \
    --min-cluster-size-sweep "1000,3000,5000,10000"

# Stage C: smoothing comparison.
echo "--- smoothing comparison ---"
python analyze_genome_wide_results.py \
    -d "${PREDICTIONS_DIR}" \
    -m "${ARCH}" \
    -r "${ANALYSIS_DIR}" \
    -o "${ARCH}_smoothing_sweep" \
    --threshold-sweep "0.5,0.7,0.9" \
    --cluster-threshold-sweep "0.3,0.5,0.7"

echo "Done: $(date)"
