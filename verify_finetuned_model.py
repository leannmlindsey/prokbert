"""
Script to verify that the fine-tuned model has been properly saved and has different weights than the base model.
"""

import torch
import numpy as np
from prokbert.training_utils import get_default_pretrained_model_parameters
from prokbert.models import BertForBinaryClassificationWithPooling
import os
import sys

def compare_models(model1, model2):
    """Compare two models and return statistics about weight differences."""
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    total_params = 0
    changed_params = 0
    max_diff = 0
    avg_diff = []
    
    for key in state_dict1.keys():
        if key in state_dict2:
            param1 = state_dict1[key].cpu().numpy()
            param2 = state_dict2[key].cpu().numpy()
            
            if param1.shape == param2.shape:
                diff = np.abs(param1 - param2)
                max_diff = max(max_diff, np.max(diff))
                avg_diff.append(np.mean(diff))
                
                # Count parameters that changed
                n_params = param1.size
                n_changed = np.sum(diff > 1e-7)
                
                total_params += n_params
                changed_params += n_changed
                
                if np.mean(diff) > 0.01:
                    print(f"  Layer '{key}': avg diff = {np.mean(diff):.6f}, max diff = {np.max(diff):.6f}")
    
    return {
        'total_params': total_params,
        'changed_params': changed_params,
        'percent_changed': (changed_params / total_params) * 100 if total_params > 0 else 0,
        'max_diff': max_diff,
        'avg_diff': np.mean(avg_diff) if avg_diff else 0
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_finetuned_model.py <path_to_finetuned_model>")
        print("Example: python verify_finetuned_model.py finetuning_outputs/lambda_finetuned")
        sys.exit(1)
    
    finetuned_path = sys.argv[1]
    
    if not os.path.exists(finetuned_path):
        print(f"Error: Path {finetuned_path} does not exist")
        sys.exit(1)
    
    print("="*60)
    print("VERIFYING FINE-TUNED MODEL")
    print("="*60)
    
    # Check what files exist in the fine-tuned model directory
    print(f"\nFiles in {finetuned_path}:")
    for file in os.listdir(finetuned_path):
        file_path = os.path.join(finetuned_path, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({size_mb:.2f} MB)")
    
    # Load base model
    print("\n1. Loading base ProkBERT model...")
    base_model, tokenizer = get_default_pretrained_model_parameters(
        model_name="neuralbioinfo/prokbert-mini",
        model_class='MegatronBertModel',
        output_hidden_states=True,
        output_attentions=False,
        move_to_gpu=False
    )
    base_classifier = BertForBinaryClassificationWithPooling(base_model)
    
    # Load fine-tuned model
    print(f"\n2. Loading fine-tuned model from {finetuned_path}...")
    try:
        finetuned_model = BertForBinaryClassificationWithPooling.from_pretrained(finetuned_path)
        print("   Fine-tuned model loaded successfully!")
    except Exception as e:
        print(f"   Error loading fine-tuned model: {e}")
        sys.exit(1)
    
    # Compare model architectures
    print("\n3. Model Architecture Comparison:")
    base_params = sum(p.numel() for p in base_classifier.parameters())
    finetuned_params = sum(p.numel() for p in finetuned_model.parameters())
    print(f"   Base model parameters: {base_params:,}")
    print(f"   Fine-tuned model parameters: {finetuned_params:,}")
    
    if base_params != finetuned_params:
        print("   WARNING: Models have different architectures!")
    
    # Compare weights
    print("\n4. Weight Comparison:")
    print("   Comparing base model weights with fine-tuned weights...")
    
    stats = compare_models(base_classifier, finetuned_model)
    
    print(f"\n   Summary:")
    print(f"   - Total parameters: {stats['total_params']:,}")
    print(f"   - Changed parameters: {stats['changed_params']:,}")
    print(f"   - Percent changed: {stats['percent_changed']:.2f}%")
    print(f"   - Maximum weight difference: {stats['max_diff']:.6f}")
    print(f"   - Average weight difference: {stats['avg_diff']:.6f}")
    
    # Check classification head specifically
    print("\n5. Classification Head Analysis:")
    if hasattr(finetuned_model, 'classifier'):
        classifier_weight = finetuned_model.classifier.weight.data
        print(f"   Classifier output shape: {classifier_weight.shape}")
        print(f"   Classifier weight norm: {torch.norm(classifier_weight).item():.4f}")
        
        if hasattr(base_classifier, 'classifier'):
            base_classifier_weight = base_classifier.classifier.weight.data
            diff = torch.norm(classifier_weight - base_classifier_weight).item()
            print(f"   Difference from base classifier: {diff:.6f}")
    
    # Verdict
    print("\n" + "="*60)
    print("VERDICT:")
    if stats['percent_changed'] > 5:
        print("✅ Model appears to be properly fine-tuned!")
        print(f"   {stats['percent_changed']:.1f}% of parameters have changed significantly")
    elif stats['percent_changed'] > 0.1:
        print("⚠️  Model has some changes but fewer than expected")
        print(f"   Only {stats['percent_changed']:.1f}% of parameters have changed")
    else:
        print("❌ Model appears to be identical to base model!")
        print("   The fine-tuning may not have been saved correctly")
    print("="*60)

if __name__ == "__main__":
    main()