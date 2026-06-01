#!/bin/bash
#
# temp_summarize_inference_metrics.sh
#
# Throwaway script: print a side-by-side table of all diagnostic
# inference metrics for the 3 ProkBERT architectures at 2k.
#
# Reads <OUTPUT_DIR>/2k/inference/<arch>/<dataset>_predictions_metrics.json
# for arch ∈ {prokbert-mini, prokbert-mini-c, prokbert-mini-long}
# and dataset ∈ {test, fpr, gc_control, fnr}.
#
# Usage on Biowulf:
#   cd /vf/users/lindseylm/.../ProkBERT_generic_sequence_classification
#   bash temp_summarize_inference_metrics.sh
#
# Delete this file after use; it's a one-time diagnostic.

set -euo pipefail

OUTPUT_DIR=/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/PROKBERT/ProkBERT_generic_sequence_classification/outputs
LEN=2k

python - <<'PY'
import json
import os
import sys

OUTPUT_DIR = "/data/lindseylm/GLM_EVALUATIONS/NAR_GENOMICS_LAMBDA_REPO/PROKBERT/ProkBERT_generic_sequence_classification/outputs"
LEN = "2k"
ARCHS = ["prokbert-mini", "prokbert-mini-c", "prokbert-mini-long"]
DATASETS = ["test", "fpr", "gc_control", "fnr"]
METRICS = ["mcc", "f1", "accuracy", "precision", "recall", "auc", "sensitivity", "specificity"]

root = os.path.join(OUTPUT_DIR, LEN, "inference")

# Header
hdr = f"{'arch':<22} {'dataset':<12} " + " ".join(f"{m:>9}" for m in METRICS)
print(hdr)
print("-" * len(hdr))

for arch in ARCHS:
    for ds in DATASETS:
        path = os.path.join(root, arch, f"{ds}_predictions_metrics.json")
        if not os.path.isfile(path):
            print(f"{arch:<22} {ds:<12} MISSING ({path})")
            continue
        with open(path) as f:
            m = json.load(f)
        vals = " ".join(f"{m.get(k, float('nan')):>9.4f}" for k in METRICS)
        print(f"{arch:<22} {ds:<12} {vals}")
    print()  # blank line between archs

# Also dump the winners.json contents for context
winners_path = os.path.join(OUTPUT_DIR, LEN, "winners.json")
if os.path.isfile(winners_path):
    print("=" * 80)
    print(f"winners.json ({winners_path}):")
    print("=" * 80)
    with open(winners_path) as f:
        w = json.load(f)
    for arch, info in w.items():
        seed_str = f" seed-{info['seed']}" if "seed" in info else ""
        print(f"  {arch}: {info['type']}{seed_str}  test_mcc={info['test_mcc']:.4f}")
PY
