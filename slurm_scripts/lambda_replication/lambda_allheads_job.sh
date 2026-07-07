#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# All-heads genome-wide (linear probe + 3-layer NN, ONE embedding pass) for ONE
# genome-wide CSV — ProkBERT (prokbert-mini | -mini-c | -mini-long). VARIANT is
# the architecture; BASE_MODEL is neuralbioinfo/<arch>. Reuses the saved probe
# artifacts in REPL_OUTPUT_DIR/embedding/<arch>; writes lp/ and nn/ subdirs under
# REPL_OUTPUT_DIR/genome_wide_heads/<arch>/.
#
# conda + env mirror lambda_embedding_job.sh (self-activate prokbert; HF_HOME set;
# NOT forced offline, so an uncached base model can still download).
#
# Required env: REPO_ROOT, REPL_OUTPUT_DIR, VARIANT, BASE_MODEL, INPUT_CSV, MAX_LENGTH
# Optional env: BATCH_SIZE(32), POOLING(mean), THRESHOLD(0.5), HF_HOME

echo "=== all-heads genome-wide  arch=${VARIANT}  input=${INPUT_CSV} ==="
echo "Started: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

source /u/llindsey1/miniconda3/etc/profile.d/conda.sh
conda activate prokbert
echo "  conda env: ${CONDA_DEFAULT_ENV:-none}   python: $(command -v python)"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/work/hdd/bfzj/llindsey1/hf_cache}

if [ -z "${REPO_ROOT:-}" ]; then echo "ERROR: REPO_ROOT not set"; exit 1; fi
if [ ! -f "${INPUT_CSV:-/nonexistent}" ]; then echo "ERROR: INPUT_CSV not found: ${INPUT_CSV:-<unset>}"; exit 1; fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

EMB_DIR="${REPL_OUTPUT_DIR}/embedding/${VARIANT}"
OUT_DIR="${REPL_OUTPUT_DIR}/genome_wide_heads/${VARIANT}"
GW_DIR="$(dirname "${INPUT_CSV}")"
STEM="$(basename "${INPUT_CSV}" .csv)"

for f in linear_probe_pretrained.pkl three_layer_nn_pretrained.pt three_layer_nn_pretrained_scaler.pkl; do
    [ -f "${EMB_DIR}/${f}" ] || { echo "ERROR: missing probe artifact ${EMB_DIR}/${f} — run embedding analysis first"; exit 1; }
done

python genome_wide_all_heads_prokbert.py \
    --base_model "${BASE_MODEL}" \
    --embedding_dir "${EMB_DIR}" \
    --input_dir "${GW_DIR}" --pattern "${STEM}.csv" \
    --output_dir "${OUT_DIR}" \
    --max_length "${MAX_LENGTH:-1024}" \
    --batch_size "${BATCH_SIZE:-32}" \
    --pooling "${POOLING:-mean}" \
    --threshold "${THRESHOLD:-0.5}" \
    --save_metrics

echo "Done: $(date)"
