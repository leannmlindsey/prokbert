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
