#!/bin/bash
#
# Lambda replication — STAGE 2: pick the best model per architecture and
# submit all inference jobs.
#
# For each segment length in SEGMENT_LENGTHS:
#   1. Run select_best_model.py to pick the per-arch winner by test-set MCC
#      across finetune seeds + linear probe + 3-layer NN; writes winners.json.
#   2. Submit one inference job per (arch, diagnostic dataset). Diagnostic
#      datasets used (each must match the segment length):
#        - test       train_val_test/<LEN>/test.csv
#        - fpr        fpr_test/<LEN>/bacteria_segments_<LEN>.csv
#        - gc_control shuffled_controls/<LEN>/test_shuffled.csv
#        - fnr        FNR_<LEN> if set                       (optional)
#   3. If GENOME_WIDE_<LEN> is set, submit a genome-wide inference job per
#      arch per genome-wide CSV (if the var is a directory, each *.csv is one
#      job), then chain ONE threshold + clustering sweep job per (length, arch)
#      that depends on all the genome-wide inference jobs for that arch.
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

if [[ "${LAMBDA_BASE}" == /path/to/* ]] || [[ "${OUTPUT_DIR}" == /path/to/* ]]; then
    echo "ERROR: edit ${CONFIG} — LAMBDA_BASE or OUTPUT_DIR still set to placeholder"
    exit 1
fi
if [ -z "${SEGMENT_LENGTHS}" ]; then
    echo "ERROR: SEGMENT_LENGTHS is empty"; exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"

# Sanity: every length must have completed training before we run inference.
for LEN in ${SEGMENT_LENGTHS}; do
    LDIR="${OUTPUT_DIR}/${LEN}"
    if [ ! -d "${LDIR}/finetune" ] || [ ! -d "${LDIR}/embedding" ]; then
        echo "ERROR: ${LDIR} doesn't look like a completed training output for length ${LEN}."
        echo "       Run run_lambda_training.sh first and wait for all jobs to finish."
        exit 1
    fi
done

# --- common sbatch flags ------------------------------------------------------

INF_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${INF_MEM}" --time="${INF_TIME}" --cpus-per-task=8)
GA_FLAGS=(--partition="${GA_PARTITION}" --mem="${GA_MEM}" --time="${GA_TIME}" --cpus-per-task=4)

echo "============================================================"
echo "Lambda replication — Stage 2: select winners + inference"
echo "============================================================"
echo "  LAMBDA_BASE:     ${LAMBDA_BASE}"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  SEGMENT_LENGTHS: ${SEGMENT_LENGTHS}"
echo "============================================================"

NUM_JOBS=0
cd "${REPO_ROOT}"

for LEN in ${SEGMENT_LENGTHS}; do
    echo ""
    echo "--- length: ${LEN} ---"

    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"

    # --- select winners (login-node; reads JSON only) ---
    # select_best_model.py SKIPs archs with no candidates (rather than
    # aborting) so a partial training run still produces inference for the
    # archs that did complete. The exit-1 path only triggers if NO arch has
    # any data — for that length we skip downstream submission below.
    echo "  selecting best model per architecture..."
    if ! python scripts/select_best_model.py \
            --output_dir "${REPL_LEN_DIR}" \
            --architectures ${ARCHS}; then
        echo "  SKIP length ${LEN}: no archs have completed training yet"
        unset DIAG_NAMES DIAG_PATHS GW_CSVS 2>/dev/null
        continue
    fi

    # --- assemble diagnostic dataset list (name -> path) ---
    declare -a DIAG_NAMES DIAG_PATHS

    # Built-in LAMBDA_v1 diagnostics.
    DIAG_NAMES=(test fpr gc_control)
    DIAG_PATHS=(
        "${LAMBDA_BASE}/train_val_test/${LEN}/test.csv"
        "${LAMBDA_BASE}/fpr_test/${LEN}/bacteria_segments_${LEN}.csv"
        "${LAMBDA_BASE}/shuffled_controls/${LEN}/test_shuffled.csv"
    )

    # Optional FNR — indirect lookup on FNR_<LEN>.
    fnr_var="FNR_${LEN}"
    FNR_PATH="${!fnr_var:-}"
    if [ -n "${FNR_PATH}" ]; then
        if [ -f "${FNR_PATH}" ]; then
            DIAG_NAMES+=(fnr)
            DIAG_PATHS+=("${FNR_PATH}")
        else
            echo "  WARNING: ${fnr_var}=${FNR_PATH} not found — skipping fnr for ${LEN}"
        fi
    fi

    # Validate built-in diagnostics exist before submitting.
    for i in "${!DIAG_NAMES[@]}"; do
        if [ ! -f "${DIAG_PATHS[$i]}" ]; then
            echo "ERROR: diagnostic '${DIAG_NAMES[$i]}' missing: ${DIAG_PATHS[$i]}"; exit 1
        fi
    done

    # --- assemble genome-wide CSV list (may be empty, one file, or many) ---
    # Indirect lookup on GENOME_WIDE_<LEN>. Accept a file or a directory of
    # *.csv. Sorted for deterministic ordering.
    declare -a GW_CSVS=()
    gw_var="GENOME_WIDE_${LEN}"
    GW_PATH="${!gw_var:-}"
    if [ -n "${GW_PATH}" ]; then
        if [ -f "${GW_PATH}" ]; then
            GW_CSVS=("${GW_PATH}")
        elif [ -d "${GW_PATH}" ]; then
            shopt -s nullglob
            for csv in "${GW_PATH}"/*.csv; do
                GW_CSVS+=("${csv}")
            done
            shopt -u nullglob
            if [ "${#GW_CSVS[@]}" -eq 0 ]; then
                echo "  WARNING: ${gw_var}=${GW_PATH} is a directory but contains no *.csv — skipping genome-wide for ${LEN}"
            fi
        else
            echo "  WARNING: ${gw_var}=${GW_PATH} is not a file or directory — skipping genome-wide for ${LEN}"
        fi
        if [ "${#GW_CSVS[@]}" -gt 0 ]; then
            echo "  genome-wide CSVs for ${LEN}: ${#GW_CSVS[@]} file(s)"
        fi
    fi

    # Read which archs actually have winners for this length.
    WINNERS_JSON="${REPL_LEN_DIR}/winners.json"
    HAVE_ARCHS=$(python -c "import json; print(' '.join(json.load(open('${WINNERS_JSON}')).keys()))")

    # --- submit jobs per architecture ---
    for ARCH in ${ARCHS}; do
        # Skip if select_best_model.py couldn't find candidates for this arch.
        if [[ " ${HAVE_ARCHS} " != *" ${ARCH} "* ]]; then
            echo "    skip ${ARCH}: no winner (training incomplete?)"
            continue
        fi
        # Diagnostic inference
        for i in "${!DIAG_NAMES[@]}"; do
            NAME="${DIAG_NAMES[$i]}"
            CSV="${DIAG_PATHS[$i]}"
            JOB="inf_${LEN}_${ARCH}_${NAME}"
            echo "    submitting ${JOB}..."
            sbatch \
                --job-name="${JOB}" \
                --output="${LOGDIR}/${JOB}_%j.out" \
                --error="${LOGDIR}/${JOB}_%j.err" \
                "${INF_FLAGS[@]}" \
                --export="ALL,REPO_ROOT=${REPO_ROOT},REPL_OUTPUT_DIR=${REPL_LEN_DIR},ARCH=${ARCH},INPUT_CSV=${CSV},OUTPUT_FILENAME=${NAME}_predictions.csv,POOLING=${POOLING}" \
                "${SCRIPT_DIR}/lambda_inference_job.sh"
            NUM_JOBS=$((NUM_JOBS + 1))
        done

        # Genome-wide inference (one job per CSV) + a single analysis job that
        # waits for all of them. analyze_genome_wide_results.py takes a
        # directory of predictions, so aggregation happens there.
        if [ "${#GW_CSVS[@]}" -gt 0 ]; then
            GW_IDS=()
            for csv in "${GW_CSVS[@]}"; do
                stem=$(basename "${csv}" .csv)
                JOB="gwinf_${LEN}_${ARCH}_${stem}"
                echo "    submitting ${JOB}..."
                jid=$(sbatch --parsable \
                    --job-name="${JOB}" \
                    --output="${LOGDIR}/${JOB}_%j.out" \
                    --error="${LOGDIR}/${JOB}_%j.err" \
                    "${INF_FLAGS[@]}" \
                    --export="ALL,REPO_ROOT=${REPO_ROOT},REPL_OUTPUT_DIR=${REPL_LEN_DIR},ARCH=${ARCH},INPUT_CSV=${csv},OUTPUT_FILENAME=genome_wide_${stem}_predictions.csv,POOLING=${POOLING}" \
                    "${SCRIPT_DIR}/lambda_inference_job.sh")
                GW_IDS+=("${jid}")
                NUM_JOBS=$((NUM_JOBS + 1))
            done

            # Chain analysis after ALL genome-wide inference jobs for this arch.
            DEP="afterok:$(IFS=:; echo "${GW_IDS[*]}")"
            GA_JOB="gwana_${LEN}_${ARCH}"
            echo "    submitting ${GA_JOB} (depends on ${#GW_IDS[@]} job(s))..."
            sbatch \
                --job-name="${GA_JOB}" \
                --output="${LOGDIR}/${GA_JOB}_%j.out" \
                --error="${LOGDIR}/${GA_JOB}_%j.err" \
                "${GA_FLAGS[@]}" \
                --dependency="${DEP}" \
                --export="ALL,REPO_ROOT=${REPO_ROOT},REPL_OUTPUT_DIR=${REPL_LEN_DIR},ARCH=${ARCH}" \
                "${SCRIPT_DIR}/lambda_genome_analysis_job.sh"
            NUM_JOBS=$((NUM_JOBS + 1))
        fi
    done

    # Reset arrays before the next length so old entries don't bleed in.
    unset DIAG_NAMES DIAG_PATHS GW_CSVS
done

echo ""
echo "Submitted ${NUM_JOBS} jobs. Monitor with: squeue -u \$USER"
echo "Results: ${OUTPUT_DIR}/<LEN>/inference/ and ${OUTPUT_DIR}/<LEN>/genome_wide_analysis/"
