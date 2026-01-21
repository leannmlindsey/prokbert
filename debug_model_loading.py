#!/usr/bin/env python3
"""
Debug script to verify ProkBERT model loading and embedding extraction.
Run this to diagnose why embeddings might be performing poorly.
"""

import torch
import numpy as np
from transformers import MegatronBertModel, AutoConfig

def main():
    model_name = "neuralbioinfo/prokbert-mini"

    print("=" * 60)
    print("ProkBERT Model Loading Diagnostic")
    print("=" * 60)

    # Step 1: Check what's in the model on HuggingFace
    print(f"\n1. Loading config from: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"   Config type: {type(config).__name__}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Num layers: {config.num_hidden_layers}")
    print(f"   Num attention heads: {config.num_attention_heads}")

    # Step 2: Load the model
    print(f"\n2. Loading MegatronBertModel from: {model_name}")
    model = MegatronBertModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        trust_remote_code=True
    )
    print(f"   Model type: {type(model).__name__}")
    print(f"   Number of parameters: {model.num_parameters():,}")

    # Step 3: Check model weights are not random
    print("\n3. Checking model weights (should NOT be near zero mean if pretrained):")
    for name, param in list(model.named_parameters())[:5]:
        print(f"   {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")

    # Step 4: Check embedding layer specifically
    print("\n4. Checking word embeddings:")
    word_emb = model.embeddings.word_embeddings.weight
    print(f"   Shape: {word_emb.shape}")
    print(f"   Mean: {word_emb.mean().item():.6f}")
    print(f"   Std: {word_emb.std().item():.6f}")
    print(f"   Min: {word_emb.min().item():.6f}")
    print(f"   Max: {word_emb.max().item():.6f}")

    # Step 5: Create a random model for comparison
    print("\n5. Creating randomly initialized model for comparison:")
    random_model = MegatronBertModel(config)
    random_word_emb = random_model.embeddings.word_embeddings.weight
    print(f"   Random word emb mean: {random_word_emb.mean().item():.6f}")
    print(f"   Random word emb std: {random_word_emb.std().item():.6f}")

    # Step 6: Compare embeddings on sample input
    print("\n6. Comparing outputs on sample input:")
    model.eval()
    random_model.eval()

    # Create dummy input (batch_size=2, seq_len=10)
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        pretrained_out = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        random_out = random_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    pretrained_hidden = pretrained_out.hidden_states[-1]
    random_hidden = random_out.hidden_states[-1]

    print(f"   Pretrained last hidden state - mean: {pretrained_hidden.mean().item():.6f}, std: {pretrained_hidden.std().item():.6f}")
    print(f"   Random last hidden state - mean: {random_hidden.mean().item():.6f}, std: {random_hidden.std().item():.6f}")

    # Step 7: Check if outputs are different
    print("\n7. Are pretrained and random outputs different?")
    diff = (pretrained_hidden - random_hidden).abs().mean().item()
    print(f"   Mean absolute difference: {diff:.6f}")
    if diff < 0.01:
        print("   WARNING: Outputs are very similar! Model may not have loaded pretrained weights!")
    else:
        print("   OK: Outputs are different, pretrained weights loaded successfully.")

    # Step 8: Test with actual ProkBERT loading function
    print("\n8. Testing with ProkBERT's get_default_pretrained_model_parameters:")
    try:
        from prokbert.training_utils import get_default_pretrained_model_parameters
        prokbert_model, tokenizer = get_default_pretrained_model_parameters(
            model_name=model_name,
            model_class='MegatronBertModel',
            output_hidden_states=True,
            output_attentions=False,
            move_to_gpu=False
        )

        # Handle DataParallel wrapper
        if hasattr(prokbert_model, 'module'):
            actual_model = prokbert_model.module
        else:
            actual_model = prokbert_model

        print(f"   Loaded model type: {type(actual_model).__name__}")
        prokbert_word_emb = actual_model.embeddings.word_embeddings.weight
        print(f"   Word emb mean: {prokbert_word_emb.mean().item():.6f}")
        print(f"   Word emb std: {prokbert_word_emb.std().item():.6f}")

        # Check if weights match the direct load
        weight_diff = (word_emb - prokbert_word_emb.cpu()).abs().mean().item()
        print(f"   Weight difference from direct load: {weight_diff:.6f}")
        if weight_diff > 0.001:
            print("   WARNING: Weights differ! Something may be wrong with ProkBERT loading.")
        else:
            print("   OK: Weights match.")

    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n" + "=" * 60)
    print("Diagnostic complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
