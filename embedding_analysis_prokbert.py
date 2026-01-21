#!/usr/bin/env python3
"""
Embedding Analysis Script for ProkBERT

This script extracts embeddings from ProkBERT and evaluates them using:
1. Linear probe (logistic regression)
2. 3-layer neural network
3. Silhouette score for embedding quality
4. PCA visualization

ProkBERT uses LCA tokenization and outputs embeddings from a MegatronBERT model.

Input: Directory containing train.csv, dev.csv (or val.csv), test.csv
       Each CSV should have 'sequence' and 'label' columns

Output: Metrics, embeddings, visualizations, and trained classifiers
"""

import argparse
import json
import os
import pickle
import random
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from prokbert.training_utils import get_default_pretrained_model_parameters


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from ProkBERT and evaluate for binary classification"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Directory containing train.csv, dev.csv (or val.csv), test.csv",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="neuralbioinfo/prokbert-mini",
        help="Path to ProkBERT model (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/embedding_analysis",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length in base pairs",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "cls"],
        help="Pooling strategy for embeddings",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--nn_epochs",
        type=int,
        default=100,
        help="Number of epochs for 3-layer NN training",
    )
    parser.add_argument(
        "--nn_hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for 3-layer NN",
    )
    parser.add_argument(
        "--nn_lr",
        type=float,
        default=0.001,
        help="Learning rate for 3-layer NN",
    )
    parser.add_argument(
        "--include_random_baseline",
        action="store_true",
        help="Also evaluate randomly initialized model as baseline",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv_data(csv_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test CSV files."""
    train_path = os.path.join(csv_dir, "train.csv")
    test_path = os.path.join(csv_dir, "test.csv")

    # Check for dev.csv or val.csv
    val_path = os.path.join(csv_dir, "dev.csv")
    if not os.path.exists(val_path):
        val_path = os.path.join(csv_dir, "val.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found in {csv_dir}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Neither dev.csv nor val.csv found in {csv_dir}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found in {csv_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


class ThreeLayerNN(nn.Module):
    """Simple 3-layer neural network for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.network(x)


def extract_embeddings(
    model,
    tokenizer,
    sequences: List[str],
    batch_size: int,
    max_length: int,
    pooling: str,
    device: torch.device,
) -> np.ndarray:
    """Extract embeddings from ProkBERT for given sequences."""
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_sequences = sequences[i : i + batch_size]

        # Tokenize each sequence using ProkBERT's tokenizer
        input_ids_list = []
        attention_mask_list = []

        for seq in batch_sequences:
            # Truncate sequence to max_length
            seq = seq[:max_length] if len(seq) > max_length else seq

            # Tokenize
            tokens = tokenizer.tokenize(seq)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Add special tokens: [CLS] ... [SEP]
            # Limit to max tokens (1024 - 2 for special tokens)
            token_ids = [tokenizer.cls_token_id] + token_ids[:1022] + [tokenizer.sep_token_id]
            attention_mask = [1] * len(token_ids)

            input_ids_list.append(token_ids)
            attention_mask_list.append(attention_mask)

        # Pad all sequences to same length
        max_len = max(len(ids) for ids in input_ids_list)
        for idx in range(len(input_ids_list)):
            padding_length = max_len - len(input_ids_list[idx])
            input_ids_list[idx] = input_ids_list[idx] + [tokenizer.pad_token_id] * padding_length
            attention_mask_list[idx] = attention_mask_list[idx] + [0] * padding_length

        # Convert to tensors
        input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(device)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # Get the last hidden state
            last_hidden_state = outputs.hidden_states[-1]

            # Apply pooling
            if pooling == "mean":
                # Mean pooling (considering attention mask)
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, 1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            elif pooling == "max":
                # Max pooling
                embeddings = last_hidden_state.max(dim=1)[0]
            elif pooling == "cls":
                # CLS token embedding (first token)
                embeddings = last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }

    # AUC if probabilities provided
    if probabilities is not None:
        try:
            metrics["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
        except ValueError:
            metrics["auc"] = 0.0

    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return metrics


def train_linear_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[Dict[str, float], LogisticRegression, StandardScaler]:
    """Train and evaluate logistic regression on embeddings."""
    # Standardize embeddings
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    val_scaled = scaler.transform(val_embeddings)
    test_scaled = scaler.transform(test_embeddings)

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_scaled, train_labels)

    # Evaluate on test set
    test_preds = clf.predict(test_scaled)
    test_probs = clf.predict_proba(test_scaled)

    metrics = calculate_metrics(test_labels, test_preds, test_probs)
    return metrics, clf, scaler


def train_three_layer_nn(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[Dict[str, float], ThreeLayerNN, StandardScaler]:
    """Train and evaluate 3-layer neural network on embeddings."""
    # Standardize embeddings
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    val_scaled = scaler.transform(val_embeddings)
    test_scaled = scaler.transform(test_embeddings)

    # Convert to tensors
    X_train = torch.FloatTensor(train_scaled).to(device)
    y_train = torch.LongTensor(train_labels).to(device)
    X_val = torch.FloatTensor(val_scaled).to(device)
    y_val = torch.LongTensor(val_labels).to(device)
    X_test = torch.FloatTensor(test_scaled).to(device)

    # Initialize model
    input_dim = train_embeddings.shape[1]
    model = ThreeLayerNN(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_probs = torch.softmax(test_outputs, dim=-1).cpu().numpy()
        test_preds = torch.argmax(test_outputs, dim=-1).cpu().numpy()

    metrics = calculate_metrics(test_labels, test_preds, test_probs)
    return metrics, model, scaler


def create_pca_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "PCA of Embeddings",
) -> Tuple[float, float]:
    """Create PCA visualization and return variance explained."""
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.6,
        s=30,
    )
    plt.colorbar(scatter, label="Class")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]


def main():
    """Main function to run embedding analysis."""
    args = parse_arguments()
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("ProkBERT Embedding Analysis")
    print("=" * 60)

    start_time = time.time()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from: {args.csv_dir}")
    train_df, val_df, test_df = load_csv_data(args.csv_dir)
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")

    # Load model and tokenizer
    print(f"\nLoading ProkBERT model from: {args.model_path}")
    model, tokenizer = get_default_pretrained_model_parameters(
        model_name=args.model_path,
        model_class='MegatronBertModel',
        output_hidden_states=True,
        output_attentions=False,
        move_to_gpu=torch.cuda.is_available()
    )
    model = model.to(device)
    model.eval()

    # Get embedding dimension from model config
    embedding_dim = model.config.hidden_size
    print(f"  Embedding dimension: {embedding_dim}")

    # Extract embeddings
    print(f"\nExtracting embeddings (pooling={args.pooling})...")

    train_embeddings = extract_embeddings(
        model, tokenizer, train_df["sequence"].tolist(),
        args.batch_size, args.max_length, args.pooling, device,
    )
    val_embeddings = extract_embeddings(
        model, tokenizer, val_df["sequence"].tolist(),
        args.batch_size, args.max_length, args.pooling, device,
    )
    test_embeddings = extract_embeddings(
        model, tokenizer, test_df["sequence"].tolist(),
        args.batch_size, args.max_length, args.pooling, device,
    )

    train_labels = train_df["label"].values
    val_labels = val_df["label"].values
    test_labels = test_df["label"].values

    print(f"  Train embeddings shape: {train_embeddings.shape}")
    print(f"  Val embeddings shape: {val_embeddings.shape}")
    print(f"  Test embeddings shape: {test_embeddings.shape}")

    # Save embeddings
    embeddings_path = os.path.join(args.output_dir, "embeddings_pretrained.npz")
    np.savez(
        embeddings_path,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
    )
    print(f"\nSaved embeddings to: {embeddings_path}")

    # Results dictionary
    results = {
        "model_path": args.model_path,
        "csv_dir": args.csv_dir,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "embedding_dim": embedding_dim,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
    }

    # Linear probe
    print("\n" + "-" * 40)
    print("Training Linear Probe (Logistic Regression)")
    print("-" * 40)
    linear_metrics, linear_clf, linear_scaler = train_linear_probe(
        train_embeddings, train_labels,
        val_embeddings, val_labels,
        test_embeddings, test_labels,
    )
    results["linear_probe"] = linear_metrics
    print(f"  Accuracy: {linear_metrics['accuracy']:.4f}")
    print(f"  F1: {linear_metrics['f1']:.4f}")
    print(f"  MCC: {linear_metrics['mcc']:.4f}")
    print(f"  AUC: {linear_metrics['auc']:.4f}")

    # Save linear probe
    linear_path = os.path.join(args.output_dir, "linear_probe_pretrained.pkl")
    with open(linear_path, "wb") as f:
        pickle.dump({"classifier": linear_clf, "scaler": linear_scaler}, f)

    # 3-layer NN
    print("\n" + "-" * 40)
    print("Training 3-Layer Neural Network")
    print("-" * 40)
    nn_metrics, nn_model, nn_scaler = train_three_layer_nn(
        train_embeddings, train_labels,
        val_embeddings, val_labels,
        test_embeddings, test_labels,
        args.nn_hidden_dim, args.nn_epochs, args.nn_lr, device,
    )
    results["three_layer_nn"] = nn_metrics
    print(f"  Accuracy: {nn_metrics['accuracy']:.4f}")
    print(f"  F1: {nn_metrics['f1']:.4f}")
    print(f"  MCC: {nn_metrics['mcc']:.4f}")
    print(f"  AUC: {nn_metrics['auc']:.4f}")

    # Save NN model
    nn_path = os.path.join(args.output_dir, "three_layer_nn_pretrained.pt")
    torch.save({
        "model_state_dict": nn_model.state_dict(),
        "input_dim": train_embeddings.shape[1],
        "hidden_dim": args.nn_hidden_dim,
    }, nn_path)

    # Save NN scaler
    nn_scaler_path = os.path.join(args.output_dir, "three_layer_nn_pretrained_scaler.pkl")
    with open(nn_scaler_path, "wb") as f:
        pickle.dump(nn_scaler, f)

    # Embedding quality metrics
    print("\n" + "-" * 40)
    print("Embedding Quality Metrics")
    print("-" * 40)

    # Silhouette score on test set
    try:
        sil_score = silhouette_score(test_embeddings, test_labels)
        results["silhouette_score"] = float(sil_score)
        print(f"  Silhouette Score: {sil_score:.4f}")
    except Exception as e:
        print(f"  Silhouette Score: Could not compute ({e})")
        results["silhouette_score"] = None

    # PCA visualization
    pca_path = os.path.join(args.output_dir, "pca_visualization_pretrained.png")
    pc1_var, pc2_var = create_pca_visualization(
        test_embeddings, test_labels, pca_path,
        title=f"ProkBERT Embeddings PCA (Test Set)"
    )
    results["pca_variance_explained"] = {"pc1": float(pc1_var), "pc2": float(pc2_var)}
    print(f"  PCA Variance Explained: PC1={pc1_var:.2%}, PC2={pc2_var:.2%}")
    print(f"  Saved PCA plot to: {pca_path}")

    # Save test predictions
    test_preds_probs = linear_clf.predict_proba(linear_scaler.transform(test_embeddings))
    test_preds = linear_clf.predict(linear_scaler.transform(test_embeddings))
    predictions_df = pd.DataFrame({
        "sequence": test_df["sequence"],
        "label": test_labels,
        "predicted": test_preds,
        "prob_0": test_preds_probs[:, 0],
        "prob_1": test_preds_probs[:, 1],
    })
    predictions_path = os.path.join(args.output_dir, "test_predictions_pretrained.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nSaved predictions to: {predictions_path}")

    # Random baseline if requested
    if args.include_random_baseline:
        print("\n" + "=" * 60)
        print("Random Baseline (Randomly Initialized Model)")
        print("=" * 60)

        # Create random model with same architecture
        from transformers import AutoConfig, MegatronBertModel

        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        random_model = MegatronBertModel(config)
        random_model.to(device)
        random_model.eval()

        # Extract random embeddings
        print("Extracting random embeddings...")
        train_random = extract_embeddings(
            random_model, tokenizer, train_df["sequence"].tolist(),
            args.batch_size, args.max_length, args.pooling, device,
        )
        val_random = extract_embeddings(
            random_model, tokenizer, val_df["sequence"].tolist(),
            args.batch_size, args.max_length, args.pooling, device,
        )
        test_random = extract_embeddings(
            random_model, tokenizer, test_df["sequence"].tolist(),
            args.batch_size, args.max_length, args.pooling, device,
        )

        # Linear probe on random
        random_linear_metrics, _, _ = train_linear_probe(
            train_random, train_labels,
            val_random, val_labels,
            test_random, test_labels,
        )
        results["random_baseline_linear"] = random_linear_metrics
        print(f"\nRandom Linear Probe:")
        print(f"  Accuracy: {random_linear_metrics['accuracy']:.4f}")
        print(f"  F1: {random_linear_metrics['f1']:.4f}")
        print(f"  MCC: {random_linear_metrics['mcc']:.4f}")

        # 3-layer NN on random
        random_nn_metrics, _, _ = train_three_layer_nn(
            train_random, train_labels,
            val_random, val_labels,
            test_random, test_labels,
            args.nn_hidden_dim, args.nn_epochs, args.nn_lr, device,
        )
        results["random_baseline_nn"] = random_nn_metrics
        print(f"\nRandom 3-Layer NN:")
        print(f"  Accuracy: {random_nn_metrics['accuracy']:.4f}")
        print(f"  F1: {random_nn_metrics['f1']:.4f}")
        print(f"  MCC: {random_nn_metrics['mcc']:.4f}")

        # PCA for random
        random_pca_path = os.path.join(args.output_dir, "pca_visualization_random.png")
        create_pca_visualization(
            test_random, test_labels, random_pca_path,
            title="Random ProkBERT Embeddings PCA (Test Set)"
        )

        # Clean up
        del random_model
        torch.cuda.empty_cache()

    # Save all results
    results_path = os.path.join(args.output_dir, "embedding_analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Pooling: {args.pooling}")
    print(f"\nLinear Probe:     Acc={linear_metrics['accuracy']:.4f}, F1={linear_metrics['f1']:.4f}, MCC={linear_metrics['mcc']:.4f}, AUC={linear_metrics['auc']:.4f}")
    print(f"3-Layer NN:       Acc={nn_metrics['accuracy']:.4f}, F1={nn_metrics['f1']:.4f}, MCC={nn_metrics['mcc']:.4f}, AUC={nn_metrics['auc']:.4f}")
    if results.get("silhouette_score"):
        print(f"Silhouette Score: {results['silhouette_score']:.4f}")
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
