#!/bin/bash
#
# ProkBERT — genome-wide predictions for BOTH frozen-embedding probe heads
# (linear probe + 3-layer NN) across all genome-wide CSVs, for every arch in
# ARCHS. Fills the missing LP + NN heads (FT genome-wide already exists) in ONE
# embedding pass per CSV.
#
# Sources configs/lambda_replication.conf (ProkBERT keeps the conf under
# configs/, not slurm_scripts/). arch -> base model is neuralbioinfo/<arch>;
# arch -> max_length matches lambda_embedding_job.sh (mini=1024, -c/-long=2048).
# Submits one lambda_allheads_job.sh per (LEN, arch, genome CSV).
#
# Usage (login node, repo pulled; the job self-activates prokbert):
#   bash slurm_scripts/lambda_replication/run_lambda_allheads.sh [LEN ...]

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${REPO_ROOT}/configs/lambda_replication.conf"
JOB="${SCRIPT_DIR}/lambda_allheads_job.sh"
[ -f "${CONFIG}" ] || { echo "ERROR: missing ${CONFIG}"; exit 1; }
[ -f "${JOB}" ]    || { echo "ERROR: missing ${JOB}"; exit 1; }
# shellcheck disable=SC1090
source "${CONFIG}"

# arch -> max_length, as in lambda_embedding_job.sh.
maxlen_for () {
    case "$1" in
        prokbert-mini)      echo 1024 ;;
        prokbert-mini-c)    echo 2048 ;;
        prokbert-mini-long) echo 2048 ;;
        *) echo "ERROR: unknown arch $1" >&2; exit 1 ;;
    esac
}

LENS=("$@"); [ "${#LENS[@]}" -gt 0 ] || read -ra LENS <<< "${SEGMENT_LENGTHS:-2k}"
BATCH="${INF_BATCH_SIZE:-${BATCH_SIZE:-32}}"

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"
FLAGS=(--account=bfzj-dtai-gh --partition=ghx4 --gpus-per-node=1 --mem="${INF_MEM}" --time="${INF_TIME}" --cpus-per-task=8)

echo "============================================================"
echo "ProkBERT — all-heads (LP + NN) genome-wide"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}   LENGTHS: ${LENS[*]}   ARCHS: ${ARCHS}"
echo "============================================================"

NUM=0
for LEN in "${LENS[@]}"; do
    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    gw_var="GENOME_WIDE_${LEN}"; GW_PATH="${!gw_var:-}"
    if [ -z "${GW_PATH}" ] || [ ! -d "${GW_PATH}" ]; then
        echo "WARNING: no genome-wide dir for ${LEN} (${GW_PATH:-unset}) — skipping"; continue
    fi
    for ARCH in ${ARCHS}; do
        MAX_LENGTH="$(maxlen_for "${ARCH}")"
        EMB_DIR="${REPL_LEN_DIR}/embedding/${ARCH}"
        if [ ! -f "${EMB_DIR}/linear_probe_pretrained.pkl" ]; then
            echo "WARNING: no saved LP probe in ${EMB_DIR} — run embedding analysis first; skipping ${LEN}/${ARCH}"; continue
        fi
        shopt -s nullglob; gw_csvs=("${GW_PATH}"/*.csv); shopt -u nullglob
        [ "${#gw_csvs[@]}" -gt 0 ] || { echo "WARNING: ${GW_PATH} has no *.csv — skipping ${LEN}/${ARCH}"; continue; }
        echo "--- ${LEN}/${ARCH} (neuralbioinfo/${ARCH}): ${#gw_csvs[@]} genome CSV(s)  max_length=${MAX_LENGTH} ---"
        for csv in "${gw_csvs[@]}"; do
            stem="$(basename "${csv}" .csv)"; J="gwheads_${LEN}_${ARCH}_${stem}"
            sbatch --job-name="${J}" \
                --output="${LOGDIR}/${J}_%j.out" --error="${LOGDIR}/${J}_%j.err" \
                "${FLAGS[@]}" \
                --export="ALL,REPO_ROOT=${REPO_ROOT},HF_HOME=${HF_HOME:-/work/hdd/bfzj/llindsey1/hf_cache},REPL_OUTPUT_DIR=${REPL_LEN_DIR},VARIANT=${ARCH},BASE_MODEL=neuralbioinfo/${ARCH},INPUT_CSV=${csv},MAX_LENGTH=${MAX_LENGTH},BATCH_SIZE=${BATCH},POOLING=${POOLING:-mean},THRESHOLD=${THRESHOLD:-0.5}" \
                "${JOB}"
            NUM=$((NUM+1))
        done
    done
done
echo ""
echo "Submitted ${NUM} all-heads genome-wide jobs. Monitor: squeue -u \$USER"
echo "Output: ${OUTPUT_DIR}/<LEN>/genome_wide_heads/<arch>/{lp,nn}/"
