#!/usr/bin/env python
"""
Finetuning script for ProkBERT on phage/bacteria classification
Adapted for leannmlindsey/PD-GB dataset with three subsets:
- phage_fragment_inphared
- phage_fragment_inphared_shuffled
- phage_fragment_phaster

Binary classification: 0 = bacteria, 1 = phage
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)
from prokbert.prokbert_tokenizer import ProkBERTTokenizer

# Fallback tokenization parameters for ProkBERT models when AutoTokenizer fails
# (e.g. prokbert-mini-long has a broken tokenizer_config.json on HuggingFace)
PROKBERT_TOKENIZATION_PARAMS = {
    'prokbert-mini': {'kmer': 6, 'shift': 1},
    'prokbert-mini-long': {'kmer': 6, 'shift': 2},
    'prokbert-mini-c': {'kmer': 1, 'shift': 1},
}


def get_prokbert_tokenizer(model_name):
    """
    Load tokenizer for ProkBERT models.
    Uses ProkBERTTokenizer directly for known models (avoids broken vocab_size
    in HuggingFace remote tokenizer classes that causes save_pretrained to fail).
    Falls back to AutoTokenizer for unknown models.
    """
    import re

    normalized = model_name.replace('neuralbioinfo/', '')

    # For known ProkBERT models, use ProkBERTTokenizer directly
    # (AutoTokenizer may return a remote class with broken vocab_size property)
    if normalized in PROKBERT_TOKENIZATION_PARAMS:
        params = PROKBERT_TOKENIZATION_PARAMS[normalized]
        print(f"  Using ProkBERTTokenizer with params: {params}")
        return ProkBERTTokenizer(tokenization_params=params, operation_space='sequence')

    # Try to extract params from model name pattern (e.g. k6s2)
    match = re.search(r'k(\d+)s(\d+)', normalized)
    if match:
        kmer, shift = map(int, match.groups())
        params = {'kmer': kmer, 'shift': shift}
        print(f"  Using ProkBERTTokenizer with params: {params}")
        return ProkBERTTokenizer(tokenization_params=params, operation_space='sequence')

    # Unknown model — try AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        raise ValueError(
            f"AutoTokenizer failed for '{model_name}': {e}. "
            f"Add this model to PROKBERT_TOKENIZATION_PARAMS or provide a known ProkBERT model name."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ProkBERT for phage detection")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="neuralbioinfo/prokbert-mini",
        help="Pretrained model name or path (neuralbioinfo/prokbert-mini, prokbert-mini-long, prokbert-mini-c)"
    )
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Use random initialization instead of pre-trained weights (to test if pre-training helps)"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="leannmlindsey/PD-GB",
        help="Hugging Face dataset name (ignored if --dataset_dir is provided)"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="phage_fragment_inphared",
        choices=["phage_fragment_inphared", "phage_fragment_inphared_shuffled", "phage_fragment_phaster"],
        help="Which dataset configuration to use (ignored if --dataset_dir is provided)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to directory containing train.csv, test.csv, dev.csv files (if provided, overrides --dataset_name)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length (256, 512, 1024, or 2048 for mini-long)"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./prokbert_phage_finetuned",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size per GPU for training"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Batch size per GPU for evaluation"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before backward pass"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep"
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Save strategy"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps"
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        default=True,
        help="Load best model at the end of training"
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_mcc",
        help="Metric to use for selecting best model"
    )
    
    return parser.parse_args()

def compute_metrics(eval_pred):
    """
    Compute comprehensive metrics for binary classification
    Following ProkBERT paper: accuracy, MCC, sensitivity, specificity, F1, AUC
    """
    predictions, labels = eval_pred
    
    # Get predicted probabilities and classes
    if len(predictions.shape) > 1:
        probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
        preds = np.argmax(predictions, axis=1)
    else:
        preds = predictions
        probs = None
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    mcc = matthews_corrcoef(labels, preds)
    
    # Confusion matrix for sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC AUC if probabilities available
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }
    
    if probs is not None and len(probs.shape) > 1:
        try:
            auc = roc_auc_score(labels, probs[:, 1])
            metrics['auc'] = auc
        except:
            pass
    
    return metrics


def preprocess_function(examples, tokenizer, max_length):
    """
    Tokenize sequences for ProkBERT.
    Uses ProkBERTTokenizer's encode_plus with manual truncation/padding,
    since it doesn't support standard HuggingFace truncation/padding kwargs.
    Falls back to standard __call__ for non-ProkBERT tokenizers.
    """
    if isinstance(tokenizer, ProkBERTTokenizer):
        all_input_ids = []
        all_attention_masks = []

        for seq in examples["sequence"]:
            # Truncate raw sequence to max_length BP *before* tokenization.
            # The ProkBERT tokenizer's internal lca_tokenize_segment rejects
            # sequences longer than ~2050 bp with ValueError, so we can't
            # rely on the token-level truncation below — it never gets to
            # run on long inputs.
            if len(seq) > max_length:
                seq = seq[:max_length]
            encoded = tokenizer.encode_plus(seq)
            input_ids = list(encoded["input_ids"])
            attention_mask = list(encoded["attention_mask"])

            # Truncate: keep [CLS] at start, truncate middle, keep [SEP] at end
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length - 1] + [input_ids[-1]]
                attention_mask = attention_mask[:max_length - 1] + [attention_mask[-1]]

            # Pad to max_length
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [0] * pad_len
                attention_mask = attention_mask + [0] * pad_len

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": examples["label"],
        }
    else:
        # Standard HuggingFace tokenizer path
        tokenized = tokenizer(
            examples["sequence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Load tokenizer
    # Use ProkBERTTokenizer for ProkBERT models (avoids broken tokenizer_config.json
    # on HuggingFace for some variants like prokbert-mini-long)
    print(f"\nLoading tokenizer from {args.model_name}...")
    tokenizer = get_prokbert_tokenizer(args.model_name)
    
    # Load dataset
    if args.dataset_dir is not None:
        # Load from local CSV files
        print(f"\nLoading dataset from local directory: {args.dataset_dir}")
        
        # Check if directory exists
        if not os.path.isdir(args.dataset_dir):
            raise ValueError(f"Dataset directory not found: {args.dataset_dir}")
        
        # Check for required files. Accept val.csv as a fallback for dev.csv
        # (LAMBDA_v1 ships val.csv; older internal datasets ship dev.csv).
        train_file = os.path.join(args.dataset_dir, "train.csv")
        test_file = os.path.join(args.dataset_dir, "test.csv")
        dev_file = os.path.join(args.dataset_dir, "dev.csv")
        if not os.path.isfile(dev_file):
            dev_file = os.path.join(args.dataset_dir, "val.csv")

        missing_files = []
        if not os.path.isfile(train_file):
            missing_files.append("train.csv")
        if not os.path.isfile(test_file):
            missing_files.append("test.csv")
        if not os.path.isfile(dev_file):
            missing_files.append("dev.csv (or val.csv)")

        if missing_files:
            raise ValueError(f"Missing required files in {args.dataset_dir}: {', '.join(missing_files)}")

        # Load CSV files
        print(f"  Loading train.csv...")
        train_df = pd.read_csv(train_file)
        print(f"  Loading test.csv...")
        test_df = pd.read_csv(test_file)
        print(f"  Loading {os.path.basename(dev_file)}...")
        dev_df = pd.read_csv(dev_file)
        
        # Verify required columns
        required_columns = ["sequence", "label"]
        for df_name, df in [("train", train_df), ("test", test_df), ("dev", dev_df)]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"{df_name}.csv missing required columns: {', '.join(missing_cols)}")
        
        # Convert to datasets
        dataset = {
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df),
            "validation": Dataset.from_pandas(dev_df)
        }
        
        print(f"\n✓ Loaded local dataset from: {args.dataset_dir}")
        
    else:
        # Load from Hugging Face
        print(f"\nLoading dataset: {args.dataset_name} ({args.dataset_config})...")
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            trust_remote_code=True
        )
    
    print(f"Dataset structure: {dataset}")
    
    # If dataset doesn't have train/validation/test splits, create them
    if "train" not in dataset:
        print("Creating train/val/test splits...")
        # Split: 80% train, 10% validation, 10% test
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=args.seed)
        test_val = dataset["test"].train_test_split(test_size=0.5, seed=args.seed)
        dataset = {
            "train": dataset["train"],
            "validation": test_val["train"],
            "test": test_val["test"]
        }
    
    print(f"\nDataset sizes:")
    for split, data in dataset.items():
        print(f"  {split}: {len(data)} examples")
    
    # Check class balance
    print(f"\nClass distribution (0=bacteria, 1=phage):")
    for split, data in dataset.items():
        labels = data["label"]
        n_bacteria = sum(1 for l in labels if l == 0)
        n_phage = sum(1 for l in labels if l == 1)
        print(f"  {split}: {n_bacteria} bacteria, {n_phage} phage ({n_phage/(n_bacteria+n_phage)*100:.1f}% phage)")
    
    # Tokenize datasets
    print(f"\nTokenizing sequences (max_length={args.max_length})...")
    tokenized_dataset = {}
    for split in dataset.keys():
        tokenized_dataset[split] = dataset[split].map(
            lambda x: preprocess_function(x, tokenizer, args.max_length),
            batched=True,
            remove_columns=dataset[split].column_names,
            desc=f"Tokenizing {split}"
        )
    
    # Load model
    print(f"\nLoading model: {args.model_name}...")
    
    if args.random_init:
        print("  Using RANDOM INITIALIZATION (no pre-trained weights)")
        # Load config but initialize with random weights
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            args.model_name,
            num_labels=2,
            trust_remote_code=True,
            problem_type="single_label_classification"
        )
        model = AutoModelForSequenceClassification.from_config(
            config,
            trust_remote_code=True
        )
    else:
        print("  Using PRE-TRAINED weights")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            trust_remote_code=True,
            problem_type="single_label_classification"
        )
    
    # Move model to device
    model = model.to(device)
    
    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True if args.metric_for_best_model in ["eval_mcc", "eval_accuracy", "eval_f1", "eval_auc"] else False,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to="none",  # no tensorboard dep for headless SLURM runs; metrics come from test_results.json
        logging_dir=os.path.join(args.output_dir, "logs"),
        push_to_hub=False,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation", tokenized_dataset.get("test")),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    
    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    try:
        tokenizer.save_pretrained(args.output_dir)
    except NotImplementedError:
        print("  tokenizer.save_pretrained() failed (vocab_size not implemented).")
        print("  Saving vocabulary file directly...")
        tokenizer.save_vocabulary(args.output_dir)
    
    # Training metrics
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    print(f"\nTraining metrics:")
    for key, value in train_result.metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on test set
    if "test" in tokenized_dataset:
        print("\n" + "="*80)
        print("Evaluating on test set...")
        print("="*80 + "\n")
        
        test_results = trainer.evaluate(tokenized_dataset["test"])
        
        print("\nTest set metrics:")
        print(f"  Accuracy:    {test_results['eval_accuracy']:.4f}")
        print(f"  Precision:   {test_results['eval_precision']:.4f}")
        print(f"  Recall:      {test_results['eval_recall']:.4f}")
        print(f"  F1:          {test_results['eval_f1']:.4f}")
        print(f"  MCC:         {test_results['eval_mcc']:.4f}")
        print(f"  Sensitivity: {test_results['eval_sensitivity']:.4f}")
        print(f"  Specificity: {test_results['eval_specificity']:.4f}")
        if 'eval_auc' in test_results:
            print(f"  AUC:         {test_results['eval_auc']:.4f}")
        
        # Save test results
        import json
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Model and results saved to: {args.output_dir}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

