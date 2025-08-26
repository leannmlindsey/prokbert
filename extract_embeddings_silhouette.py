import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd
from prokbert.training_utils import get_default_pretrained_model_parameters

def truncate_sequence(sequence, max_length=1024):
    """Truncate sequence to max_length"""
    return sequence[:max_length] if len(sequence) > max_length else sequence

def get_embeddings(model, tokenizer, sequences, batch_size=32):
    """
    Extract embeddings from ProkBERT model for a list of sequences.
    
    Args:
        model: ProkBERT model
        tokenizer: ProkBERT tokenizer
        sequences: List of DNA sequences
        batch_size: Batch size for processing
        
    Returns:
        numpy array of embeddings
    """
    model.eval()
    embeddings = []
    device = next(model.parameters()).device  # Get the device the model is on
    
    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_sequences = sequences[i:i+batch_size]
        
        # Tokenize batch
        encoded = tokenizer(
            batch_sequences,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        )
        
        # Move tensors to the same device as the model
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                output_hidden_states=True
            )
            
            # Use the last hidden state and apply mean pooling
            last_hidden_state = outputs.hidden_states[-1]
            
            # Mean pooling (considering attention mask)
            attention_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, 1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            embeddings.append(mean_pooled.cpu().numpy())
    
    return np.vstack(embeddings)

def main():
    print("Loading ProkBERT model...")
    # Load the pretrained ProkBERT model
    pretrained_model, tokenizer = get_default_pretrained_model_parameters(
        model_name="neuralbioinfo/prokbert-mini",
        model_class='MegatronBertModel',
        output_hidden_states=True,
        output_attentions=False,
        move_to_gpu=torch.cuda.is_available()
    )
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = pretrained_model.to(device)
    print(f"Using device: {device}")
    
    print("Loading lambda dataset...")
    dataset = load_dataset("leannmlindsey/lambda")
    
    # Sample a subset for faster computation (you can remove this for full dataset)
    num_samples = 5000  # Adjust as needed
    
    # Process train set
    train_set = dataset["train"]
    train_df = train_set.to_pandas()
    
    # Sample if dataset is large
    if len(train_df) > num_samples:
        train_df = train_df.sample(n=num_samples, random_state=42)
        print(f"Sampled {num_samples} sequences from training set")
    
    # Truncate sequences
    train_df['sequence'] = train_df['sequence'].apply(lambda x: truncate_sequence(x, 1024))
    
    sequences = train_df['sequence'].tolist()
    labels = train_df['label'].values
    
    print(f"Processing {len(sequences)} sequences...")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Extract embeddings
    embeddings = get_embeddings(pretrained_model, tokenizer, sequences, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Calculate silhouette score on full embeddings
    print("\nCalculating silhouette score on full embeddings...")
    silhouette_full = silhouette_score(embeddings, labels)
    print(f"Silhouette score (full embeddings): {silhouette_full:.4f}")
    
    # Apply PCA for dimensionality reduction and visualization
    print("\nApplying PCA for dimensionality reduction...")
    pca_dims = [2, 10, 50, 100]
    
    for n_components in pca_dims:
        if n_components < embeddings.shape[1]:
            pca = PCA(n_components=n_components)
            embeddings_pca = pca.fit_transform(embeddings)
            silhouette_pca = silhouette_score(embeddings_pca, labels)
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"Silhouette score (PCA-{n_components}): {silhouette_pca:.4f} (explained variance: {explained_var:.2%})")
    
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
    
    # Save results
    results = {
        'silhouette_score_full': silhouette_full,
        'inter_class_distance': inter_class_distance,
        'avg_intra_class_distance': avg_intra_class,
        'separation_ratio': inter_class_distance/avg_intra_class,
        'num_samples': len(sequences),
        'embedding_dim': embeddings.shape[1]
    }
    
    # Save results to file
    import json
    with open('embedding_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to embedding_analysis_results.json")
    
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