import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd
from prokbert.training_utils import get_default_pretrained_model_parameters, get_torch_data_from_segmentdb_classification
from prokbert.prok_datasets import ProkBERTTrainingDatasetPT
from prokbert.models import BertForBinaryClassificationWithPooling
import time
import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_lambda_dataframe(df, max_length=1024):
    """
    Convert lambda dataset dataframe to the format expected by ProkBERT.
    """
    # df is already a pandas DataFrame
    df = df.copy()  # Make a copy to avoid modifying the original
    
    # Truncate sequences that are too long
    df['sequence'] = df['sequence'].apply(lambda x: x[:max_length] if len(x) > max_length else x)
    
    # Rename 'sequence' to 'segment'
    df['segment'] = df['sequence']
    
    # Create segment_id as a unique identifier
    df['segment_id'] = [f"seq_{i}" for i in range(len(df))]
    
    # Create 'y' column (same as label for binary classification)
    df['y'] = df['label']
    
    return df

def get_embeddings_from_dataset(model, dataset, batch_size=32, model_type='pretrained'):
    """
    Extract embeddings using ProkBERT's dataset format.
    """
    model.eval()
    embeddings = []
    device = next(model.parameters()).device
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    print(f"Processing {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if model_type == 'finetuned':
                # For fine-tuned model, get the pooled output before classification
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_pooled_output=True
                )
                # Use the pooled output from the fine-tuned model
                if 'pooled_output' in outputs:
                    batch_embeddings = outputs['pooled_output']
                else:
                    # Fallback to last hidden state mean pooling
                    last_hidden_state = outputs['hidden_states'][-1]
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, 1)
                    sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
            else:
                # For pretrained model, use mean pooling over last hidden state
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Use the last hidden state
                last_hidden_state = outputs.hidden_states[-1]
                
                # Mean pooling (considering attention mask)
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, 1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
            
            embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)

def visualize_embeddings(embeddings, labels, output_dir, model_name, n_points=1000):
    """
    Create visualization of embeddings using PCA projection to 2D.
    
    Args:
        embeddings: numpy array of embeddings
        labels: numpy array of labels (0 or 1)
        output_dir: directory to save the plot
        model_name: name of the model for the plot title
        n_points: number of points to visualize (default 1000)
    """
    # Sample if we have more points than requested
    if len(embeddings) > n_points:
        indices = np.random.choice(len(embeddings), n_points, replace=False)
        embeddings_vis = embeddings[indices]
        labels_vis = labels[indices]
    else:
        embeddings_vis = embeddings
        labels_vis = labels
        n_points = len(embeddings)
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_vis)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot each class with different colors
    # Class 0: pink, Class 1: blue
    colors = ['#FF69B4', '#4169E1']  # Hot pink and Royal blue
    class_names = ['Non-promoter (0)', 'Promoter (1)']
    
    for i in range(2):
        mask = labels_vis == i
        plt.scatter(embeddings_2d[mask, 0], 
                   embeddings_2d[mask, 1], 
                   c=colors[i], 
                   label=class_names[i],
                   alpha=0.6, 
                   edgecolors='white',
                   linewidth=0.5)
    
    plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title(f'PCA Visualization of Embeddings\nModel: {model_name}\n({n_points} points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'embedding_visualization.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save as PDF
    print(f"Visualization saved to {plot_path}")
    
    # Create a density plot as well
    plt.figure(figsize=(10, 8))
    
    # Create hexbin plot for density
    plt.hexbin(embeddings_2d[:, 0], embeddings_2d[:, 1], 
               C=labels_vis, 
               gridsize=30, 
               cmap='coolwarm', 
               alpha=0.8)
    
    plt.colorbar(label='Average label value')
    plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title(f'Density Plot of Embeddings\nModel: {model_name}\n({n_points} points)')
    plt.grid(True, alpha=0.3)
    
    density_path = os.path.join(output_dir, 'embedding_density.png')
    plt.tight_layout()
    plt.savefig(density_path, dpi=150, bbox_inches='tight')
    print(f"Density plot saved to {density_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Extract and analyze embeddings from ProkBERT models')
    parser.add_argument('--model', type=str, default='neuralbioinfo/prokbert-mini',
                       help='Model name or path (default: neuralbioinfo/prokbert-mini)')
    parser.add_argument('--model_type', type=str, choices=['pretrained', 'finetuned'], 
                       default='pretrained',
                       help='Type of model: pretrained or finetuned (default: pretrained)')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of sequences to analyze (default: 5000)')
    parser.add_argument('--vis_points', type=int, default=1000,
                       help='Number of points to visualize (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing (default: 64)')
    parser.add_argument('--output_base_dir', type=str, default='embedding_analysis',
                       help='Base directory for output (default: embedding_analysis)')
    
    args = parser.parse_args()
    
    # Create output directory based on model name
    model_short_name = args.model.replace('/', '_').replace('\\', '_')
    if args.model_type == 'finetuned':
        model_short_name += '_finetuned'
    output_dir = os.path.join(args.output_base_dir, model_short_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print(f"\nLoading model: {args.model}")
    print(f"Model type: {args.model_type}")
    start_time = time.time()
    
    # Load the model based on type
    if args.model_type == 'finetuned':
        # Load fine-tuned model
        print("Loading fine-tuned model...")
        # First load the base model
        base_model, tokenizer = get_default_pretrained_model_parameters(
            model_name="neuralbioinfo/prokbert-mini",  # Base model
            model_class='MegatronBertModel',
            output_hidden_states=True,
            output_attentions=False,
            move_to_gpu=torch.cuda.is_available()
        )
        # Then load the fine-tuned weights
        model = BertForBinaryClassificationWithPooling.from_pretrained(args.model)
    else:
        # Load pretrained model
        print("Loading pretrained model...")
        model, tokenizer = get_default_pretrained_model_parameters(
            model_name=args.model,
            model_class='MegatronBertModel',
            output_hidden_states=True,
            output_attentions=False,
            move_to_gpu=torch.cuda.is_available()
        )
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    print(f"Model loading time: {time.time() - start_time:.2f} seconds")
    
    print("\nLoading lambda dataset...")
    dataset = load_dataset("leannmlindsey/lambda")
    
    # Process train set
    train_set = dataset["train"]
    train_df = train_set.to_pandas()
    
    # Sample if dataset is large
    if len(train_df) > args.num_samples:
        train_df = train_df.sample(n=args.num_samples, random_state=42)
        print(f"Sampled {args.num_samples} sequences from training set")
    
    # Prepare dataframe in ProkBERT format
    train_db = prepare_lambda_dataframe(train_df, max_length=1024)
    labels = train_db['y'].values
    
    print(f"Processing {len(train_db)} sequences...")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Use ProkBERT's tokenization pipeline
    print("\nTokenizing sequences...")
    tokenize_start = time.time()
    [X_train, y_train, torchdb_train] = get_torch_data_from_segmentdb_classification(tokenizer, train_db)
    train_ds = ProkBERTTrainingDatasetPT(X_train, y_train, AddAttentionMask=True)
    print(f"Tokenization time: {time.time() - tokenize_start:.2f} seconds")
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    embed_start = time.time()
    embeddings = get_embeddings_from_dataset(model, train_ds, 
                                            batch_size=args.batch_size,
                                            model_type=args.model_type)
    print(f"Embedding extraction time: {time.time() - embed_start:.2f} seconds")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Calculate silhouette score on full embeddings
    print("\nCalculating silhouette score on full embeddings...")
    silhouette_full = silhouette_score(embeddings, labels)
    print(f"Silhouette score (full embeddings): {silhouette_full:.4f}")
    
    # Apply PCA for dimensionality reduction and analysis
    print("\nApplying PCA for dimensionality reduction...")
    pca_results = {}
    pca_dims = [2, 10, 50, 100]
    
    for n_components in pca_dims:
        if n_components < embeddings.shape[1]:
            pca = PCA(n_components=n_components)
            embeddings_pca = pca.fit_transform(embeddings)
            silhouette_pca = silhouette_score(embeddings_pca, labels)
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"Silhouette score (PCA-{n_components}): {silhouette_pca:.4f} (explained variance: {explained_var:.2%})")
            
            pca_results[f'pca_{n_components}'] = {
                'silhouette_score': float(silhouette_pca),
                'explained_variance': float(explained_var)
            }
    
    # Additional analysis: calculate average distance between classes
    print("\nClass separation analysis:")
    embeddings_class_0 = embeddings[labels == 0]
    embeddings_class_1 = embeddings[labels == 1]
    
    # Calculate centroids
    centroid_0 = np.mean(embeddings_class_0, axis=0)
    centroid_1 = np.mean(embeddings_class_1, axis=0)
    
    # Calculate inter-class distance
    inter_class_distance = np.linalg.norm(centroid_1 - centroid_0)
    
    # Calculate average intra-class distances
    intra_class_0 = np.mean([np.linalg.norm(e - centroid_0) for e in embeddings_class_0])
    intra_class_1 = np.mean([np.linalg.norm(e - centroid_1) for e in embeddings_class_1])
    avg_intra_class = (intra_class_0 + intra_class_1) / 2
    
    print(f"Inter-class distance: {inter_class_distance:.4f}")
    print(f"Average intra-class distance: {avg_intra_class:.4f}")
    print(f"Separation ratio (inter/intra): {inter_class_distance/avg_intra_class:.4f}")
    
    # Create visualization
    print(f"\nCreating visualization with {args.vis_points} points...")
    visualize_embeddings(embeddings, labels, output_dir, args.model, args.vis_points)
    
    # Save results
    results = {
        'model': args.model,
        'model_type': args.model_type,
        'silhouette_score_full': float(silhouette_full),
        'pca_results': pca_results,
        'inter_class_distance': float(inter_class_distance),
        'intra_class_distance_0': float(intra_class_0),
        'intra_class_distance_1': float(intra_class_1),
        'avg_intra_class_distance': float(avg_intra_class),
        'separation_ratio': float(inter_class_distance/avg_intra_class),
        'num_samples': len(train_db),
        'embedding_dim': embeddings.shape[1],
        'label_distribution': {
            'class_0': int(np.sum(labels == 0)),
            'class_1': int(np.sum(labels == 1))
        }
    }
    
    # Save results to file
    results_path = os.path.join(output_dir, 'embedding_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    # Save embeddings for future analysis
    embeddings_path = os.path.join(output_dir, 'embeddings.npy')
    np.save(embeddings_path, embeddings)
    labels_path = os.path.join(output_dir, 'labels.npy')
    np.save(labels_path, labels)
    print(f"Embeddings saved to {embeddings_path}")
    print(f"Labels saved to {labels_path}")
    
    # Interpretation guide
    print("\n" + "="*50)
    print("INTERPRETATION GUIDE:")
    print("Silhouette Score: Ranges from -1 to 1")
    print("  - Close to 1: Good separation between classes")
    print("  - Close to 0: Overlapping classes")  
    print("  - Negative: Misclassified samples")
    print(f"\nYour score of {silhouette_full:.4f} indicates: ", end="")
    
    if silhouette_full > 0.5:
        print("GOOD separation - the model learned meaningful representations")
    elif silhouette_full > 0.25:
        print("MODERATE separation - some discriminative features learned")
    elif silhouette_full > 0:
        print("WEAK separation - limited discriminative power")
    else:
        print("POOR separation - classes are not well distinguished")

if __name__ == "__main__":
    main()