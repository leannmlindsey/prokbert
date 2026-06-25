#!/bin/bash
#
# ProkBERT LAMBDA replication — check that all STAGE 2 inference jobs finished.
#
# Reads the same configs/lambda_replication.conf as the launcher, then for every
# (length, arch) reports the outputs run_lambda_inference.sh should produce:
#   WINNER     winners.json lists this arch (best of seeds / LP / NN picked)
#   EMBED      embedding/<arch>/ has a results json
#   <diag>     inference/<arch>/<diag>_predictions.csv (+ _predictions_metrics.json)
#              for test / fpr / gc_control / fnr (fnr only if FNR_<LEN> set+exists),
#              with accuracy & mcc straight from the metrics JSON
#   PHROG      inference/<arch>/<arch>_phage_annotated_segments_<LEN>_predictions.csv
#              AND that it carries the phrog_db_category annotation column
#   GENOME     genome_wide_*_predictions.csv count vs CSVs in GENOME_WIDE_<LEN>,
#              AND each has a 'start' column (the metadata-passthrough fix),
#              AND genome_wide_analysis/<arch>/ produced sweep output
# then lists any non-empty inference/embedding/genome .err files.
#
# Usage:
#   bash slurm_scripts/lambda_replication/check_inference.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
CONFIG="${REPO_ROOT}/configs/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR is empty (check ${CONFIG})"; exit 1
fi
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR not found: ${OUTPUT_DIR}"; exit 1
fi

RUN_LENGTHS="$(echo "${SEGMENT_LENGTHS}" | xargs)"
LOGDIR="${OUTPUT_DIR}/logs"

# Print "acc / mcc" from an inference metrics JSON, or a dash if absent.
metrics_line() {
    python - "$1" 2>/dev/null <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    acc = d.get("accuracy"); mcc = d.get("mcc")
    fa = f"{acc:.4f}" if isinstance(acc, (int, float)) else "?"
    fm = f"{mcc:.4f}" if isinstance(mcc, (int, float)) else "?"
    print(f"acc={fa} mcc={fm}")
except Exception:
    print("-")
PY
}

# Does a CSV's header contain the given column? prints "yes"/"no".
has_col() {
    python - "$1" "$2" 2>/dev/null <<'PY'
import csv, sys
try:
    with open(sys.argv[1], newline="") as f:
        header = next(csv.reader(f))
    print("yes" if sys.argv[2] in header else "no")
except Exception:
    print("no")
PY
}

echo "============================================================"
echo "ProkBERT LAMBDA replication — inference check"
echo "============================================================"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  ARCHS:           ${ARCHS}"
echo "============================================================"

for LEN in ${RUN_LENGTHS}; do
    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    WINNERS_JSON="${REPL_LEN_DIR}/winners.json"

    # diagnostics expected for this length: include fnr only if FNR_<LEN> set+exists.
    DIAGS="test fpr gc_control"
    fnr_var="FNR_${LEN}"
    if [ -n "${!fnr_var:-}" ] && [ -f "${!fnr_var}" ]; then
        DIAGS="${DIAGS} fnr"
    fi

    # PHROG expected only if PHROG_<LEN> set+exists.
    phrog_var="PHROG_${LEN}"
    PHROG_EXPECTED=0
    if [ -n "${!phrog_var:-}" ] && [ -f "${!phrog_var}" ]; then
        PHROG_EXPECTED=1
    fi

    # genome-wide expected count.
    gw_var="GENOME_WIDE_${LEN}"
    GW_PATH="${!gw_var:-}"
    GW_EXPECTED=0
    if [ -n "${GW_PATH}" ]; then
        if [ -f "${GW_PATH}" ]; then
            GW_EXPECTED=1
        elif [ -d "${GW_PATH}" ]; then
            shopt -s nullglob
            gw_files=("${GW_PATH}"/*.csv)
            shopt -u nullglob
            GW_EXPECTED="${#gw_files[@]}"
        fi
    fi

    echo ""
    echo "######## length: ${LEN} ########"
    if [ -f "${WINNERS_JSON}" ]; then
        HAVE_ARCHS=$(python -c "import json;print(' '.join(json.load(open('${WINNERS_JSON}')).keys()))" 2>/dev/null)
    else
        HAVE_ARCHS=""
        echo "  WARNING: winners.json MISSING — run_lambda_inference.sh may not have run"
    fi

    for ARCH in ${ARCHS}; do
        echo ""
        echo "  --- arch: ${ARCH} ---"
        INF_DIR="${REPL_LEN_DIR}/inference/${ARCH}"
        EMB_DIR="${REPL_LEN_DIR}/embedding/${ARCH}"
        GA_DIR="${REPL_LEN_DIR}/genome_wide_analysis/${ARCH}"

        # winner?
        if [[ " ${HAVE_ARCHS} " == *" ${ARCH} "* ]]; then
            echo "    WINNER   ok"
        else
            echo "    WINNER   MISSING (no winning seed — predictions skipped)"
        fi

        # embedding — accept any results json the analysis writes.
        shopt -s nullglob
        emb_json=("${EMB_DIR}"/*results*.json "${EMB_DIR}"/*.json)
        shopt -u nullglob
        if [ "${#emb_json[@]}" -gt 0 ]; then
            echo "    EMBED    ok"
        else
            echo "    EMBED    MISSING"
        fi

        # diagnostics — canonical names (test/fpr/gc_control/fnr).
        for NAME in ${DIAGS}; do
            CSV="${INF_DIR}/${NAME}_predictions.csv"
            MJSON="${INF_DIR}/${NAME}_predictions_metrics.json"
            if [ -f "${CSV}" ]; then
                if [ -f "${MJSON}" ]; then
                    printf "    %-12s ok   %s\n" "${NAME}" "$(metrics_line "${MJSON}")"
                else
                    printf "    %-12s ok   (no _metrics.json — labels absent?)\n" "${NAME}"
                fi
            else
                printf "    %-12s MISSING\n" "${NAME}"
            fi
        done

        # PHROG — annotated-set predictions + the phrog_db_category passthrough.
        if [ "${PHROG_EXPECTED}" -eq 1 ]; then
            PHROG_CSV="${INF_DIR}/${ARCH}_phage_annotated_segments_${LEN}_predictions.csv"
            if [ -f "${PHROG_CSV}" ]; then
                if [ "$(has_col "${PHROG_CSV}" phrog_db_category)" = "yes" ]; then
                    printf "    %-12s ok   (phrog_db_category present)\n" "phrog"
                else
                    printf "    %-12s PRESENT but MISSING phrog_db_category column!\n" "phrog"
                fi
            else
                printf "    %-12s MISSING\n" "phrog"
            fi
        fi

        # genome-wide — count predictions vs expected, verify start col + analysis output.
        if [ "${GW_EXPECTED}" -gt 0 ]; then
            shopt -s nullglob
            gw_pred=("${INF_DIR}"/genome_wide_*_predictions.csv)
            shopt -u nullglob
            GW_GOT="${#gw_pred[@]}"
            if [ "${GW_GOT}" -eq "${GW_EXPECTED}" ]; then GWS=ok; else GWS=INCOMPLETE; fi
            printf "    %-12s %s  predictions=%s/%s\n" "genome" "${GWS}" "${GW_GOT}" "${GW_EXPECTED}"

            # the metadata-passthrough fix: first genome-wide CSV must carry 'start'.
            if [ "${GW_GOT}" -gt 0 ]; then
                if [ "$(has_col "${gw_pred[0]}" start)" = "yes" ]; then
                    printf "    %-12s ok   (start/end columns present)\n" "gw-coords"
                else
                    printf "    %-12s MISSING 'start' column — analysis will KeyError!\n" "gw-coords"
                fi
            fi

            # analysis sweep output present?
            shopt -s nullglob
            ga_out=("${GA_DIR}"/*.csv)
            shopt -u nullglob
            if [ "${#ga_out[@]}" -gt 0 ]; then
                printf "    %-12s ok   (%s file(s))\n" "gw-analysis" "${#ga_out[@]}"
            else
                printf "    %-12s MISSING\n" "gw-analysis"
            fi
        fi
    done
done

echo ""
echo "=== non-empty .err files (potential failures) ==="
ERRS=$(find "${LOGDIR}" \( -name "inf_*.err" -o -name "gwinf_*.err" -o -name "gwana_*.err" -o -name "emb_*.err" -o -name "phrog_*.err" \) -size +0c -printf "%s  %p\n" 2>/dev/null | sort -rn)
if [ -n "${ERRS}" ]; then
    echo "${ERRS}"
else
    echo "  (none)"
fi
