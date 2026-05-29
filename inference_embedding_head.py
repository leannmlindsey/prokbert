#!/usr/bin/env python3
"""
Inference with a classifier head trained on top of frozen ProkBERT embeddings.

Loads the pretrained ProkBERT base model, extracts embeddings on the input CSV,
then applies a saved linear probe or 3-layer NN classifier (the artifacts
written by embedding_analysis_prokbert.py).

Output format matches inference_lambda.py / inference_hf.py:
    segment_id, label?, prob_class_0, prob_1, pred_label, correct?

Usage:
    # Linear probe
    python inference_embedding_head.py \\
        --base_model neuralbioinfo/prokbert-mini \\
        --head_type linear_probe \\
        --head_path results/embedding/prokbert-mini/linear_probe_pretrained.pkl \\
        --dataset_file data/test.csv \\
        --output_dir results/inference/prokbert-mini \\
        --output_file test_predictions.csv \\
        --save_metrics

    # 3-layer NN (needs scaler too)
    python inference_embedding_head.py \\
        --base_model neuralbioinfo/prokbert-mini \\
        --head_type three_layer_nn \\
        --head_path results/embedding/prokbert-mini/three_layer_nn_pretrained.pt \\
        --scaler_path results/embedding/prokbert-mini/three_layer_nn_pretrained_scaler.pkl \\
        --dataset_file data/test.csv \\
        --output_dir results/inference/prokbert-mini \\
        --output_file test_predictions.csv \\
        --save_metrics
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from prokbert.training_utils import get_default_pretrained_model_parameters

# Reuse helpers from the embedding analysis script so the embedding pipeline
# stays identical (same pooling, same shift-handling, same tokenization path).
from embedding_analysis_prokbert import (
    ThreeLayerNN,
    extract_embeddings,
)


def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except ValueError:
            metrics["auc"] = 0.0
    else:
        metrics["auc"] = 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    return metrics


def load_linear_probe(path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["classifier"], bundle["scaler"]


def load_three_layer_nn(path, scaler_path, device):
    ckpt = torch.load(path, map_location=device)
    model = ThreeLayerNN(ckpt["input_dim"], ckpt["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_linear_probe(clf, scaler, embeddings):
    scaled = scaler.transform(embeddings)
    probs = clf.predict_proba(scaled)
    preds = clf.predict(scaled)
    return preds, probs


def predict_three_layer_nn(model, scaler, embeddings, device):
    scaled = scaler.transform(embeddings)
    X = torch.FloatTensor(scaled).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
    return preds, probs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_model", required=True,
                        help="HF name of the pretrained ProkBERT model used to extract embeddings")
    parser.add_argument("--head_type", required=True, choices=["linear_probe", "three_layer_nn"])
    parser.add_argument("--head_path", required=True,
                        help="Path to .pkl (linear_probe) or .pt (three_layer_nn) classifier")
    parser.add_argument("--scaler_path", default=None,
                        help="Path to scaler .pkl (required for three_layer_nn)")
    parser.add_argument("--dataset_file", required=True, help="Input CSV with `sequence` column")
    parser.add_argument("--output_dir", default="inference_results")
    parser.add_argument("--output_file", default=None,
                        help="Output CSV filename. Default: predictions_<input_basename>.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--pooling", default="mean", choices=["mean", "max", "cls"])
    parser.add_argument("--no_labels", action="store_true",
                        help="Input has no `label` column; skip metric computation")
    parser.add_argument("--save_metrics", action="store_true",
                        help="Write a sibling _metrics.json (only meaningful with labels)")
    parser.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    args = parser.parse_args()

    if args.head_type == "three_layer_nn" and not args.scaler_path:
        parser.error("--scaler_path is required when --head_type=three_layer_nn")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    print(f"Loading input CSV: {args.dataset_file}")
    df = pd.read_csv(args.dataset_file)
    if "sequence" not in df.columns:
        raise ValueError(f"{args.dataset_file} has no `sequence` column")
    sequences = df["sequence"].tolist()
    has_labels = (not args.no_labels) and ("label" in df.columns)
    labels = df["label"].values if has_labels else np.zeros(len(df), dtype=int)
    print(f"  {len(sequences)} sequences, labels={'yes' if has_labels else 'no'}")

    print(f"Loading pretrained ProkBERT: {args.base_model}")
    model, tokenizer = get_default_pretrained_model_parameters(
        model_name=args.base_model,
        model_class="MegatronBertModel",
        output_hidden_states=True,
        output_attentions=False,
        move_to_gpu=torch.cuda.is_available(),
    )
    model = model.to(device)
    model.eval()

    max_pos = model.config.max_position_embeddings
    if args.max_length > max_pos:
        print(f"  WARNING: --max_length ({args.max_length}) > model max ({max_pos}); clamping")
        args.max_length = max_pos

    print(f"Extracting embeddings (pooling={args.pooling})...")
    t0 = time.time()
    embeddings = extract_embeddings(
        model, tokenizer, sequences, labels.tolist(),
        args.batch_size, args.max_length, args.pooling, device,
    )
    print(f"  shape={embeddings.shape}  elapsed={time.time()-t0:.1f}s")

    print(f"Loading classifier head: {args.head_path}")
    if args.head_type == "linear_probe":
        clf, scaler = load_linear_probe(args.head_path)
        preds, probs = predict_linear_probe(clf, scaler, embeddings)
    else:
        nn_model, scaler = load_three_layer_nn(args.head_path, args.scaler_path, device)
        preds, probs = predict_three_layer_nn(nn_model, scaler, embeddings, device)

    # Build output. Match column names of inference_lambda.py / inference_hf.py
    # exactly so downstream analysis scripts (analyze_genome_wide_results.py,
    # analyze_threshold.py, reeval_with_threshold.py) work transparently.
    out = pd.DataFrame({
        "segment_id": [f"seq_{i}" for i in range(len(sequences))],
    })
    if has_labels:
        out["label"] = labels
    out["prob_class_0"] = probs[:, 0]
    out["prob_1"] = probs[:, 1]
    out["pred_label"] = preds
    if has_labels:
        out["correct"] = (preds == labels).astype(int)

    # Preserve metadata columns from the input (e.g. start/end for genome-wide).
    metadata_cols = [c for c in df.columns
                     if c not in ("sequence", "label") and c not in out.columns]
    for col in metadata_cols:
        out[col] = df[col].values

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_file:
        out_path = os.path.join(args.output_dir, args.output_file)
    else:
        stem = os.path.splitext(os.path.basename(args.dataset_file))[0]
        out_path = os.path.join(args.output_dir, f"predictions_{stem}.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote predictions: {out_path}")

    if has_labels and args.save_metrics:
        metrics = calculate_metrics(labels, preds, probs)
        metrics_path = out_path.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote metrics: {metrics_path}")
        print(f"  MCC={metrics['mcc']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
