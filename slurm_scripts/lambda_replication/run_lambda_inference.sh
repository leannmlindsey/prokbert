#!/bin/bash
#
# Lambda replication — STAGE 2: pick the best model per architecture and
# submit all inference jobs.
#
# Per architecture, the best model is the highest test-set MCC across:
#   - every finetune seed (from <OUTPUT_DIR>/finetune/<arch>/seed-N/test_results.json)
#   - the linear probe and 3-layer NN  (from <OUTPUT_DIR>/embedding/<arch>/
#                                       embedding_analysis_results.json)
#
# Submits:
#   - one inference sbatch job per (arch, diagnostic dataset)
#   - one genome-wide inference job per arch
#   - one genome-wide analysis job per arch (chained --dependency=afterok)
#
# Re-running is safe: each inference job overwrites its own predictions CSV.
#
# Usage (after run_lambda_training.sh has finished — verify with `squeue`):
#   bash slurm_scripts/lambda_replication/run_lambda_inference.sh

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

if [[ "${OUTPUT_DIR}" == /path/to/* ]]; then
    echo "ERROR: edit ${CONFIG} — OUTPUT_DIR still set to placeholder"
    exit 1
fi
if [ ! -d "${OUTPUT_DIR}/finetune" ] || [ ! -d "${OUTPUT_DIR}/embedding" ]; then
    echo "ERROR: ${OUTPUT_DIR} doesn't look like a completed training output."
    echo "       Run run_lambda_training.sh first and wait for all jobs to finish."
    exit 1
fi

# Parse colon-separated name=path entries.
declare -a DIAG_NAMES DIAG_PATHS
IFS=':' read -r -a _diag_pairs <<< "${DIAGNOSTIC_DATASETS}"
for pair in "${_diag_pairs[@]}"; do
    [ -z "${pair}" ] && continue
    name="${pair%%=*}"
    path="${pair#*=}"
    if [[ "${path}" == /path/to/* ]]; then
        echo "ERROR: edit ${CONFIG} — diagnostic '${name}' still set to placeholder"; exit 1
    fi
    [ -f "${path}" ] || { echo "ERROR: diagnostic '${name}': ${path} not found"; exit 1; }
    DIAG_NAMES+=("${name}")
    DIAG_PATHS+=("${path}")
done
if [ "${#DIAG_NAMES[@]}" -eq 0 ]; then
    echo "ERROR: DIAGNOSTIC_DATASETS is empty"; exit 1
fi
[ -f "${GENOME_WIDE_CSV}" ] || { echo "ERROR: GENOME_WIDE_CSV=${GENOME_WIDE_CSV} not found"; exit 1; }

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"

# --- pick winners (login-node, runs locally; reads JSON only) ----------------

echo "============================================================"
echo "Lambda replication — Stage 2: select winners + inference"
echo "============================================================"
echo ""
echo "Selecting best model per architecture..."
cd "${REPO_ROOT}"
python scripts/select_best_model.py \
    --output_dir "${OUTPUT_DIR}" \
    --architectures ${ARCHS}
echo ""
echo "Winners written to: ${OUTPUT_DIR}/winners.json"
echo ""

# --- submit inference jobs ----------------------------------------------------

INF_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${INF_MEM}" --time="${INF_TIME}" --cpus-per-task=8)
GA_FLAGS=(--partition="${GA_PARTITION}" --mem="${GA_MEM}" --time="${GA_TIME}" --cpus-per-task=4)

NUM_JOBS=0

for ARCH in ${ARCHS}; do
    # Diagnostic CSVs
    for i in "${!DIAG_NAMES[@]}"; do
        NAME="${DIAG_NAMES[$i]}"
        CSV="${DIAG_PATHS[$i]}"
        JOB="inf_${ARCH}_${NAME}"
        echo "  submitting ${JOB}..."
        sbatch \
            --job-name="${JOB}" \
            --output="${LOGDIR}/${JOB}_%j.out" \
            --error="${LOGDIR}/${JOB}_%j.err" \
            "${INF_FLAGS[@]}" \
            --export="ALL,REPL_OUTPUT_DIR=${OUTPUT_DIR},ARCH=${ARCH},INPUT_CSV=${CSV},OUTPUT_FILENAME=${NAME}_predictions.csv" \
            "${SCRIPT_DIR}/lambda_inference_job.sh"
        NUM_JOBS=$((NUM_JOBS + 1))
    done

    # Genome-wide inference + dependent analysis
    GW_JOB="gwinf_${ARCH}"
    echo "  submitting ${GW_JOB}..."
    GW_ID=$(sbatch --parsable \
        --job-name="${GW_JOB}" \
        --output="${LOGDIR}/${GW_JOB}_%j.out" \
        --error="${LOGDIR}/${GW_JOB}_%j.err" \
        "${INF_FLAGS[@]}" \
        --export="ALL,REPL_OUTPUT_DIR=${OUTPUT_DIR},ARCH=${ARCH},INPUT_CSV=${GENOME_WIDE_CSV},OUTPUT_FILENAME=genome_wide_predictions.csv" \
        "${SCRIPT_DIR}/lambda_inference_job.sh")
    NUM_JOBS=$((NUM_JOBS + 1))

    GA_JOB="gwana_${ARCH}"
    echo "  submitting ${GA_JOB} (depends on ${GW_ID})..."
    sbatch \
        --job-name="${GA_JOB}" \
        --output="${LOGDIR}/${GA_JOB}_%j.out" \
        --error="${LOGDIR}/${GA_JOB}_%j.err" \
        "${GA_FLAGS[@]}" \
        --dependency="afterok:${GW_ID}" \
        --export="ALL,REPL_OUTPUT_DIR=${OUTPUT_DIR},ARCH=${ARCH}" \
        "${SCRIPT_DIR}/lambda_genome_analysis_job.sh"
    NUM_JOBS=$((NUM_JOBS + 1))
done

echo ""
echo "Submitted ${NUM_JOBS} jobs. Monitor with: squeue -u \$USER"
echo "Results under: ${OUTPUT_DIR}/inference/  and  ${OUTPUT_DIR}/genome_wide_analysis/"
