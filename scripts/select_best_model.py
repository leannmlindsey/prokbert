#!/usr/bin/env python3
"""
Per-architecture, pick the model with the highest test-set MCC across all
candidates: every finetune seed, the linear probe, and the 3-layer NN.

Writes <output_dir>/winners.json:
    {
      "prokbert-mini": {
        "type": "finetune" | "linear_probe" | "three_layer_nn",
        "test_mcc": 0.85,
        "path": "<absolute path the inference job needs>",
        "scaler_path": "<for LP/NN: scaler pickle>",  # finetune omits this
        "base_model": "neuralbioinfo/prokbert-mini",
        "all_candidates": [{type, seed?, test_mcc}, ...]
      },
      ...
    }

Reads:
  <output_dir>/finetune/<arch>/seed-<N>/test_results.json   (eval_mcc)
  <output_dir>/embedding/<arch>/embedding_analysis_results.json
                                                          (linear_probe.mcc,
                                                           three_layer_nn.mcc)

Used by replicate_phage.py as the stage-3 job (depends on all stage-1+2 jobs).
"""

import argparse
import glob
import json
import os
import sys


ARCH_TO_HF = {
    "prokbert-mini": "neuralbioinfo/prokbert-mini",
    "prokbert-mini-c": "neuralbioinfo/prokbert-mini-c",
    "prokbert-mini-long": "neuralbioinfo/prokbert-mini-long",
}


def collect_finetune_candidates(arch_dir):
    out = []
    for seed_dir in sorted(glob.glob(os.path.join(arch_dir, "seed-*"))):
        results_path = os.path.join(seed_dir, "test_results.json")
        if not os.path.isfile(results_path):
            print(f"  WARN: missing {results_path}, skipping", file=sys.stderr)
            continue
        with open(results_path) as f:
            metrics = json.load(f)
        mcc = metrics.get("eval_mcc")
        if mcc is None:
            print(f"  WARN: no eval_mcc in {results_path}, skipping", file=sys.stderr)
            continue
        seed = int(os.path.basename(seed_dir).split("-")[1])
        # finetune_prokbert_phage.py writes the trained model into the seed dir
        # itself (load_best_model_at_end=True moves it there at the end).
        out.append({
            "type": "finetune",
            "seed": seed,
            "test_mcc": float(mcc),
            "path": os.path.abspath(seed_dir),
        })
    return out


def collect_embedding_candidates(embedding_arch_dir):
    results_path = os.path.join(embedding_arch_dir, "embedding_analysis_results.json")
    if not os.path.isfile(results_path):
        print(f"  WARN: missing {results_path}", file=sys.stderr)
        return []
    with open(results_path) as f:
        results = json.load(f)
    out = []
    lp = results.get("linear_probe", {})
    if "mcc" in lp:
        out.append({
            "type": "linear_probe",
            "test_mcc": float(lp["mcc"]),
            "path": os.path.abspath(
                os.path.join(embedding_arch_dir, "linear_probe_pretrained.pkl")
            ),
        })
    nn = results.get("three_layer_nn", {})
    if "mcc" in nn:
        out.append({
            "type": "three_layer_nn",
            "test_mcc": float(nn["mcc"]),
            "path": os.path.abspath(
                os.path.join(embedding_arch_dir, "three_layer_nn_pretrained.pt")
            ),
            "scaler_path": os.path.abspath(
                os.path.join(embedding_arch_dir, "three_layer_nn_pretrained_scaler.pkl")
            ),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True,
                        help="Replication output directory (contains finetune/ and embedding/)")
    parser.add_argument("--architectures", nargs="+", required=True,
                        help="Architectures to select for (e.g. prokbert-mini prokbert-mini-c)")
    args = parser.parse_args()

    winners = {}
    for arch in args.architectures:
        print(f"\n=== {arch} ===")
        finetune_dir = os.path.join(args.output_dir, "finetune", arch)
        embedding_dir = os.path.join(args.output_dir, "embedding", arch)

        candidates = (
            collect_finetune_candidates(finetune_dir)
            + collect_embedding_candidates(embedding_dir)
        )
        if not candidates:
            print(f"  ERROR: no candidates found for {arch}", file=sys.stderr)
            sys.exit(1)

        for c in sorted(candidates, key=lambda c: c["test_mcc"], reverse=True):
            tag = f"{c['type']}" + (f"/seed-{c['seed']}" if "seed" in c else "")
            print(f"  test_mcc={c['test_mcc']:.4f}  {tag}")

        winner = max(candidates, key=lambda c: c["test_mcc"])
        winner["base_model"] = ARCH_TO_HF.get(arch, arch)
        winner["all_candidates"] = [
            {k: v for k, v in c.items() if k in ("type", "seed", "test_mcc")}
            for c in candidates
        ]
        winners[arch] = winner
        print(f"  WINNER: {winner['type']} (test_mcc={winner['test_mcc']:.4f})")

    out_path = os.path.join(args.output_dir, "winners.json")
    with open(out_path, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
