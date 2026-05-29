#!/bin/bash
#SBATCH --job-name=prokbert-sweep
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# Directory containing *_metrics.json and *_predictions.csv from inference
RESULTS_DIR="/path/to/inference/output"

# Model name (used for output file prefixes)
MODEL_NAME="prokbert-mini"

# Output directory for sweep results
OUTPUT_DIR="/path/to/sweep/output"

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate
if [ "${RESULTS_DIR}" == "/path/to/inference/output" ]; then
    echo "ERROR: Please set RESULTS_DIR"; exit 1
fi
if [ "${OUTPUT_DIR}" == "/path/to/sweep/output" ]; then
    echo "ERROR: Please set OUTPUT_DIR"; exit 1
fi

module load conda 2>/dev/null || true
conda activate prokbert 2>/dev/null || source activate prokbert 2>/dev/null || true
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
export PYTHONPATH="${PWD}:${PYTHONPATH}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "ProkBERT Parameter Sweep"
echo "============================================================"
echo "Results dir:  ${RESULTS_DIR}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Model name:   ${MODEL_NAME}"
echo "Started at:   $(date)"
echo "============================================================"

# Stage 1: Sweep initial threshold (no smoothing)
echo ""
echo "=== STAGE 1: Threshold sweep (no smoothing) ==="
python analyze_genome_wide_results.py \
    -d "${RESULTS_DIR}" \
    -m "${MODEL_NAME}" \
    -r "${OUTPUT_DIR}" \
    --window-size 0 \
    --threshold-sweep "0.3,0.5,0.7,0.8,0.9,0.95,0.99"

# Stage 2: Sweep merge-gap and min-cluster-size at a few thresholds
echo ""
echo "=== STAGE 2: Clustering parameter sweep ==="
python analyze_genome_wide_results.py \
    -d "${RESULTS_DIR}" \
    -m "${MODEL_NAME}" \
    -r "${OUTPUT_DIR}" \
    -o "${MODEL_NAME}_clustering_sweep" \
    --window-size 0 \
    --threshold-sweep "0.5,0.7,0.9" \
    --merge-gap-sweep "1000,3000,5000,10000" \
    --min-cluster-size-sweep "1000,3000,5000,10000"

# Stage 3: Compare smoothing vs no smoothing at best thresholds
echo ""
echo "=== STAGE 3: Smoothing comparison ==="
python analyze_genome_wide_results.py \
    -d "${RESULTS_DIR}" \
    -m "${MODEL_NAME}" \
    -r "${OUTPUT_DIR}" \
    -o "${MODEL_NAME}_smoothing_sweep" \
    --threshold-sweep "0.5,0.7,0.9" \
    --cluster-threshold-sweep "0.3,0.5,0.7"

echo ""
echo "============================================================"
echo "Sweep complete at: $(date)"
echo "Results in: ${OUTPUT_DIR}"
echo "============================================================"
