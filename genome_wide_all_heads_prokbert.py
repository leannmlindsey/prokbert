#!/usr/bin/env python3
"""
Collect GENOME-WIDE predictions for BOTH probe heads (linear probe + 3-layer NN)
in a single embedding pass — ProkBERT, for the fragment-vs-genome-wide transfer
analysis (best-head in the main table, full per-head in the appendix).

Run ONCE PER ARCHITECTURE (base model + embedding dir differ per variant):
    prokbert-mini       neuralbioinfo/prokbert-mini
    prokbert-mini-c     neuralbioinfo/prokbert-mini-c
    prokbert-mini-long  neuralbioinfo/prokbert-mini-long

For each genome-wide CSV: extract ProkBERT embeddings ONCE, then apply the saved
linear probe AND 3-layer NN (from embedding_analysis_prokbert.py). Writes:
    <output_dir>/lp/genome_wide_<stem>_predictions.csv   (+ _metrics.json)
    <output_dir>/nn/genome_wide_<stem>_predictions.csv   (+ _metrics.json)

Usage:
    python genome_wide_all_heads_prokbert.py \
        --base_model neuralbioinfo/prokbert-mini \
        --embedding_dir results/embedding/prokbert-mini \
        --input_dir <LAMBDA_BASE>/genome_wide/2k \
        --output_dir results/inference/prokbert-mini/genome_wide_heads \
        --max_length 1024 --save_metrics
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from prokbert.training_utils import get_default_pretrained_model_parameters
from embedding_analysis_prokbert import ThreeLayerNN, extract_embeddings

LP_FILE = "linear_probe_pretrained.pkl"
NN_FILE = "three_layer_nn_pretrained.pt"
NN_SCALER_FILE = "three_layer_nn_pretrained_scaler.pkl"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", required=True, help="HF name of the pretrained ProkBERT model")
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pattern", default="*.csv")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--pooling", default="mean", choices=["mean", "max", "cls"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save_metrics", action="store_true")
    return p.parse_args()


def load_heads(embedding_dir, device):
    with open(os.path.join(embedding_dir, LP_FILE), "rb") as f:
        bundle = pickle.load(f)
    lp = (bundle["classifier"], bundle["scaler"])
    ckpt = torch.load(os.path.join(embedding_dir, NN_FILE), map_location=device)
    nn_model = ThreeLayerNN(ckpt["input_dim"], ckpt["hidden_dim"]).to(device)
    nn_model.load_state_dict(ckpt["model_state_dict"]); nn_model.eval()
    with open(os.path.join(embedding_dir, NN_SCALER_FILE), "rb") as f:
        nn_scaler = pickle.load(f)
    return lp, (nn_model, nn_scaler)


def apply_lp(lp, emb):
    clf, scaler = lp
    return clf.predict_proba(scaler.transform(emb))


def apply_nn(nn, emb, device):
    model, scaler = nn
    X = torch.FloatTensor(scaler.transform(emb)).to(device)
    with torch.no_grad():
        return torch.softmax(model(X), dim=1).cpu().numpy()


def write_out(df, probs, threshold, out_csv, save_metrics):
    preds = (probs[:, 1] >= threshold).astype(int)
    out = df.copy()
    out["prob_0"] = probs[:, 0]; out["prob_1"] = probs[:, 1]; out["pred_label"] = preds
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    if save_metrics and "label" in df.columns:
        from sklearn.metrics import matthews_corrcoef
        y = df["label"].astype(int).values
        m = {"mcc": float(matthews_corrcoef(y, preds)) if len(np.unique(y)) > 1 else 0.0}
        with open(out_csv.replace(".csv", "_metrics.json"), "w") as f:
            json.dump(m, f)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  embedding_dir={args.embedding_dir}")
    model, tokenizer = get_default_pretrained_model_parameters(
        model_name=args.base_model, model_class="MegatronBertModel",
        output_hidden_states=True, output_attentions=False,
        move_to_gpu=torch.cuda.is_available())
    model = model.to(device); model.eval()
    lp, nn = load_heads(args.embedding_dir, device)

    csvs = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    print(f"{len(csvs)} genome CSV(s)")
    for i, csv in enumerate(csvs):
        df = pd.read_csv(csv)
        if "sequence" not in df.columns:
            print(f"  [skip] {os.path.basename(csv)} (no sequence)"); continue
        labels = df["label"].astype(int).tolist() if "label" in df.columns else [0] * len(df)
        # ProkBERT extract_embeddings returns embeddings only.
        emb = extract_embeddings(model, tokenizer, df["sequence"].astype(str).tolist(),
                                 labels, args.batch_size, args.max_length, args.pooling, device)
        stem = os.path.basename(csv).replace(".csv", "")
        name = f"genome_wide_{stem}_predictions.csv" if not stem.startswith("genome_wide_") else f"{stem}_predictions.csv"
        write_out(df, apply_lp(lp, emb), args.threshold,
                  os.path.join(args.output_dir, "lp", name), args.save_metrics)
        write_out(df, apply_nn(nn, emb, device), args.threshold,
                  os.path.join(args.output_dir, "nn", name), args.save_metrics)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(csvs)} done")
    print(f"Done -> {args.output_dir}/{{lp,nn}}/")


if __name__ == "__main__":
    main()
