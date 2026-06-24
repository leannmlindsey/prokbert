#!/bin/bash
#
# Verify the random-init embedding baseline actually ran for every cell, not
# just that INCLUDE_RANDOM_BASELINE=true was set. For each (length, arch) under
# OUTPUT_DIR/<LEN>/embedding/<arch>/ it checks:
#   - embeddings_random.npz exists, AND
#   - embedding_analysis_results.json contains random_baseline_linear,
#     random_baseline_nn, and embedding_power.
# It also prints the pretrained vs random linear-probe MCC so you can eyeball
# that random sits well BELOW pretrained (the weak baseline). Prints an
# N/total complete count at the end and exits non-zero if any cell is bad.
#
# Usage (after run_lambda_training.sh has finished):
#   bash slurm_scripts/lambda_replication/check_random_baseline.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}/../.."
CONFIG="${REPO_ROOT}/configs/lambda_replication.conf"
[ -f "${CONFIG}" ] || { echo "ERROR: missing ${CONFIG}"; exit 1; }
# shellcheck disable=SC1090
source "${CONFIG}"

echo "============================================================"
echo "Random-baseline check"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  SEGMENT_LENGTHS: ${SEGMENT_LENGTHS}"
echo "  ARCHS:           ${ARCHS}"
echo "============================================================"

total=0
complete=0

for LEN in ${SEGMENT_LENGTHS}; do
    for ARCH in ${ARCHS}; do
        total=$((total + 1))
        DIR="${OUTPUT_DIR}/${LEN}/embedding/${ARCH}"
        JSON="${DIR}/embedding_analysis_results.json"
        NPZ="${DIR}/embeddings_random.npz"

        if [ ! -d "${DIR}" ]; then
            printf "  [MISSING] %-3s %-18s  no embedding dir\n" "${LEN}" "${ARCH}"
            continue
        fi
        if [ ! -f "${JSON}" ]; then
            printf "  [MISSING] %-3s %-18s  no embedding_analysis_results.json\n" "${LEN}" "${ARCH}"
            continue
        fi

        LINE=$(python - "${JSON}" "${NPZ}" <<'PY'
import json, os, sys
json_path, npz_path = sys.argv[1], sys.argv[2]
try:
    j = json.load(open(json_path))
except Exception as e:
    print(f"FAIL  unreadable json ({e})"); sys.exit(0)
need = ["random_baseline_linear", "random_baseline_nn", "embedding_power"]
miss = [k for k in need if k not in j]
npz_ok = os.path.isfile(npz_path)
def mcc(block):
    d = j.get(block) or {}
    return d.get("mcc")
pre, rnd = mcc("linear_probe"), mcc("random_baseline_linear")
def fmt(x): return f"{x:.4f}" if isinstance(x, (int, float)) else "NA"
if not npz_ok:
    print("FAIL  embeddings_random.npz missing")
elif miss:
    print("FAIL  missing keys: " + ",".join(miss))
else:
    print(f"OK    pretrained_lp_mcc={fmt(pre)}  random_lp_mcc={fmt(rnd)}")
PY
)
        STATUS="${LINE%% *}"
        REST="${LINE#* }"
        printf "  [%-7s] %-3s %-18s  %s\n" "${STATUS}" "${LEN}" "${ARCH}" "${REST# }"
        [ "${STATUS}" = "OK" ] && complete=$((complete + 1))
    done
done

echo ""
echo "Random baseline complete: ${complete} / ${total} cells"
[ "${complete}" -eq "${total}" ] || { echo "Some cells incomplete — re-run the embedding job(s) above."; exit 1; }
