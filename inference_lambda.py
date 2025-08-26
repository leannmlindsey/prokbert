#!/usr/bin/env python3
"""
Inference script for lambda dataset using fine-tuned ProkBERT model.
This script performs inference on a specified dataset using a fine-tuned checkpoint.

Usage:
    python inference_lambda.py --checkpoint_path <path_to_checkpoint> --dataset <dataset_name_or_path> [options]
    
Examples:
    # Using HuggingFace dataset
    python inference_lambda.py --checkpoint_path finetuning_outputs/lambda_finetuned --dataset leannmlindsey/lambda --split test
    
    # Using local CSV file
    python inference_lambda.py --checkpoint_path finetuning_outputs/lambda_finetuned --dataset_file data/test.csv
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

from prokbert.prokbert_tokenizer import ProkBERTTokenizer
from prokbert.training_utils import get_default_pretrained_model_parameters, get_torch_data_from_segmentdb_classification
from prokbert.models import BertForBinaryClassificationWithPooling
from prokbert.prok_datasets import ProkBERTTrainingDatasetPT
from torch.utils.data import DataLoader


def prepare_dataframe_from_dataset(dataset_split, max_length=1024, preserve_metadata=True):
    """
    Convert dataset split to the format expected by ProkBERT.
    
    Args:
        dataset_split: HuggingFace dataset split or pandas DataFrame
        max_length: Maximum sequence length
        preserve_metadata: Whether to preserve additional metadata columns
    
    Returns:
        pd.DataFrame: Prepared dataframe with required columns
    """
    if hasattr(dataset_split, 'to_pandas'):
        df = dataset_split.to_pandas()
    else:
        df = dataset_split.copy()
    
    # Store original metadata columns if present
    metadata_cols = []
    if preserve_metadata:
        # Identify metadata columns (everything except sequence and label)
        core_cols = {'sequence', 'label', 'segment', 'segment_id', 'y'}
        metadata_cols = [col for col in df.columns if col not in core_cols]
        if metadata_cols:
            print(f"  Preserving metadata columns: {metadata_cols}")
    
    # Truncate sequences that are too long
    df['sequence'] = df['sequence'].apply(lambda x: x[:max_length] if len(x) > max_length else x)
    
    # Rename 'sequence' to 'segment' if needed
    if 'sequence' in df.columns and 'segment' not in df.columns:
        df['segment'] = df['sequence']
    
    # Create segment_id as a unique identifier (use seq_id if available, otherwise generate)
    if 'segment_id' not in df.columns:
        if 'seq_id' in df.columns:
            df['segment_id'] = df['seq_id'].astype(str)
        else:
            df['segment_id'] = [f"seq_{i}" for i in range(len(df))]
    
    # Create 'y' column (same as label for binary classification)
    if 'label' in df.columns and 'y' not in df.columns:
        df['y'] = df['label']
    
    # Print truncation statistics
    original_lengths = df['sequence'].apply(len) if 'sequence' in df.columns else df['segment'].apply(len)
    truncated_count = (original_lengths > max_length).sum()
    if truncated_count > 0:
        print(f"  Truncated {truncated_count} sequences from max length {original_lengths.max()} to {max_length}")
    
    return df


def load_inference_dataset(args):
    """
    Load dataset for inference from either HuggingFace or local file.
    
    Args:
        args: Argument namespace with dataset parameters
    
    Returns:
        pd.DataFrame: Prepared dataset
    """
    if args.dataset_file:
        # Load from local file
        print(f"Loading dataset from local file: {args.dataset_file}")
        if args.dataset_file.endswith('.csv'):
            df = pd.read_csv(args.dataset_file)
        elif args.dataset_file.endswith('.tsv'):
            df = pd.read_csv(args.dataset_file, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {args.dataset_file}")
        
        # Check required columns
        if 'sequence' not in df.columns:
            raise ValueError("Dataset must have a 'sequence' column")
        if 'label' not in df.columns and not args.no_labels:
            print("Warning: No 'label' column found. Running in prediction-only mode.")
            args.no_labels = True
            df['label'] = 0  # Dummy labels
        
        return prepare_dataframe_from_dataset(df, max_length=args.max_length)
    
    else:
        # Load from HuggingFace
        print(f"Loading dataset from HuggingFace: {args.dataset}")
        dataset = load_dataset(args.dataset)
        
        if args.split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(f"Split '{args.split}' not found. Available splits: {available_splits}")
        
        dataset_split = dataset[args.split]
        return prepare_dataframe_from_dataset(dataset_split, max_length=args.max_length)


def perform_inference(model, dataloader, device, show_progress=True):
    """
    Perform inference on the dataset.
    
    Args:
        model: Fine-tuned model
        dataloader: DataLoader for inference
        device: Device to run on
        show_progress: Whether to show progress bar
    
    Returns:
        tuple: (predictions, probabilities)
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    iterator = tqdm(dataloader, desc="Running inference") if show_progress else dataloader
    
    with torch.no_grad():
        for batch in iterator:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle both dict and object outputs
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs.logits
            
            # Get probabilities and predictions
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    metrics['true_negatives'] = cm[0, 0]
    metrics['false_positives'] = cm[0, 1]
    metrics['false_negatives'] = cm[1, 0]
    metrics['true_positives'] = cm[1, 1]
    
    # AUC if probabilities are available
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except:
            metrics['auc'] = None
    
    return metrics


def save_results(predictions, probabilities, labels, segment_ids, output_path, metadata_df=None):
    """
    Save inference results to file.
    
    Args:
        predictions: Predicted labels
        probabilities: Prediction probabilities
        labels: True labels (if available)
        segment_ids: Sequence identifiers
        output_path: Path to save results
        metadata_df: Optional dataframe with metadata columns to preserve
    """
    results = pd.DataFrame({
        'segment_id': segment_ids,
        'predicted_label': predictions,
        'prob_class_0': probabilities[:, 0],
        'prob_class_1': probabilities[:, 1]
    })
    
    if labels is not None:
        results['true_label'] = labels
        results['correct'] = (predictions == labels).astype(int)
    
    # Merge with metadata if provided
    if metadata_df is not None and not metadata_df.empty:
        # Ensure segment_id columns match type
        results['segment_id'] = results['segment_id'].astype(str)
        if 'segment_id' in metadata_df.columns:
            metadata_df['segment_id'] = metadata_df['segment_id'].astype(str)
            results = pd.merge(results, metadata_df, on='segment_id', how='left')
        elif 'seq_id' in metadata_df.columns:
            # Use seq_id if segment_id not available
            metadata_df['segment_id'] = metadata_df['seq_id'].astype(str)
            # Drop duplicate columns except segment_id
            metadata_cols = [col for col in metadata_df.columns if col not in results.columns or col == 'segment_id']
            results = pd.merge(results, metadata_df[metadata_cols], on='segment_id', how='left')
    
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on lambda dataset with fine-tuned ProkBERT model')
    
    # Model arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to fine-tuned model checkpoint')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='leannmlindsey/lambda',
                        help='HuggingFace dataset name or path (default: leannmlindsey/lambda)')
    parser.add_argument('--dataset_file', type=str, default=None,
                        help='Path to local dataset file (CSV or TSV format). Overrides --dataset if provided')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use (default: test)')
    parser.add_argument('--no_labels', action='store_true',
                        help='Run inference without labels (prediction only mode)')
    
    # Processing arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length (default: 1024)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detected if not specified')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save results (default: inference_results)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output filename (default: auto-generated based on dataset and checkpoint)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("PROKBERT LAMBDA INFERENCE")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset_file if args.dataset_file else args.dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print("="*60)
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path '{args.checkpoint_path}' does not exist")
        sys.exit(1)
    
    # Load tokenizer (using base model's tokenizer)
    print("\n1. Loading tokenizer...")
    _, tokenizer = get_default_pretrained_model_parameters(
        model_name="neuralbioinfo/prokbert-mini",
        model_class='MegatronBertModel',
        output_hidden_states=False,
        output_attentions=False,
        move_to_gpu=False
    )
    
    # Load fine-tuned model
    print(f"\n2. Loading fine-tuned model from {args.checkpoint_path}...")
    try:
        model = BertForBinaryClassificationWithPooling.from_pretrained(args.checkpoint_path)
        model = model.to(device)
        model.eval()
        print("   Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load dataset
    print(f"\n3. Loading dataset...")
    test_df = load_inference_dataset(args)
    print(f"   Loaded {len(test_df)} sequences")
    
    # Store metadata columns for later
    core_cols = {'sequence', 'label', 'segment', 'segment_id', 'y'}
    metadata_cols = [col for col in test_df.columns if col not in core_cols]
    metadata_df = test_df[metadata_cols + ['segment_id']] if metadata_cols else pd.DataFrame()
    
    # Prepare data for ProkBERT
    print(f"\n4. Preparing data for inference...")
    [X_test, y_test, torchdb_test] = get_torch_data_from_segmentdb_classification(tokenizer, test_df)
    test_ds = ProkBERTTrainingDatasetPT(X_test, y_test, AddAttentionMask=True)
    
    # Create dataloader
    test_dataloader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Perform inference
    print(f"\n5. Running inference on {len(test_ds)} sequences...")
    predictions, probabilities = perform_inference(model, test_dataloader, device)
    
    # Calculate metrics if labels are available
    if not args.no_labels:
        print(f"\n6. Calculating metrics...")
        metrics = calculate_metrics(y_test, predictions, probabilities)
        
        print("\nRESULTS:")
        print("-"*40)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        if metrics.get('auc'):
            print(f"AUC:       {metrics['auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              0     1")
        print(f"Actual 0  {metrics['true_negatives']:5d} {metrics['false_positives']:5d}")
        print(f"       1  {metrics['false_negatives']:5d} {metrics['true_positives']:5d}")
    
    # Save results
    print(f"\n7. Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.output_file:
        output_path = os.path.join(args.output_dir, args.output_file)
    else:
        checkpoint_name = os.path.basename(args.checkpoint_path)
        dataset_name = args.dataset.replace('/', '_') if not args.dataset_file else os.path.basename(args.dataset_file).split('.')[0]
        output_path = os.path.join(args.output_dir, f"predictions_{dataset_name}_{checkpoint_name}.csv")
    
    segment_ids = test_df['segment_id'].values if 'segment_id' in test_df.columns else [f"seq_{i}" for i in range(len(predictions))]
    save_results(predictions, probabilities, y_test if not args.no_labels else None, segment_ids, output_path, metadata_df)
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()