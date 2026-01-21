#!/usr/bin/env python3
"""
Comprehensive diagnostic script for ProkBERT embedding analysis.
This will help identify why pretrained model embeddings might perform poorly.
"""

import torch
import numpy as np
import pandas as pd
from transformers import MegatronBertModel, AutoConfig

def main():
    print("=" * 70)
    print("ProkBERT Embedding Analysis Diagnostic")
    print("=" * 70)

    model_name = "neuralbioinfo/prokbert-mini"

    # =========================================================================
    # STEP 1: Verify model loading
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Verify Model Loading")
    print("=" * 70)

    # Load directly from HuggingFace
    print(f"\n1a. Loading model directly from HuggingFace: {model_name}")
    direct_model = MegatronBertModel.from_pretrained(model_name, output_hidden_states=True, trust_remote_code=True)
    direct_model.eval()

    # Create random model for comparison
    print("1b. Creating randomly initialized model for comparison")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    random_model = MegatronBertModel(config)
    random_model.eval()

    # Compare weights
    print("\n1c. Comparing word embedding weights:")
    direct_emb = direct_model.embeddings.word_embeddings.weight
    random_emb = random_model.embeddings.word_embeddings.weight

    print(f"   Direct model - mean: {direct_emb.mean().item():.6f}, std: {direct_emb.std().item():.6f}")
    print(f"   Random model - mean: {random_emb.mean().item():.6f}, std: {random_emb.std().item():.6f}")

    weight_diff = (direct_emb - random_emb).abs().mean().item()
    print(f"   Weight difference: {weight_diff:.6f}")

    if weight_diff < 0.01:
        print("   *** WARNING: Weights are nearly identical! Model may not have loaded correctly! ***")
    else:
        print("   OK: Weights are different (model loaded pretrained weights)")

    # =========================================================================
    # STEP 2: Verify ProkBERT loading function
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Verify ProkBERT Loading Function")
    print("=" * 70)

    from prokbert.training_utils import get_default_pretrained_model_parameters

    print(f"\n2a. Loading via get_default_pretrained_model_parameters")
    prokbert_model, tokenizer = get_default_pretrained_model_parameters(
        model_name=model_name,
        model_class='MegatronBertModel',
        output_hidden_states=True,
        output_attentions=False,
        move_to_gpu=False  # Keep on CPU for comparison
    )

    # Handle DataParallel wrapper
    if hasattr(prokbert_model, 'module'):
        prokbert_model = prokbert_model.module
        print("   Note: Model was wrapped in DataParallel, unwrapped for comparison")

    prokbert_model.eval()

    # Compare weights
    print("\n2b. Comparing ProkBERT-loaded weights with direct HuggingFace load:")
    prokbert_emb = prokbert_model.embeddings.word_embeddings.weight

    weight_diff_prokbert = (direct_emb - prokbert_emb).abs().mean().item()
    print(f"   Weight difference: {weight_diff_prokbert:.6f}")

    if weight_diff_prokbert > 0.001:
        print("   *** WARNING: ProkBERT loading produced different weights than direct load! ***")
    else:
        print("   OK: ProkBERT loading matches direct HuggingFace load")

    # =========================================================================
    # STEP 3: Verify tokenization
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Verify Tokenization")
    print("=" * 70)

    test_sequence = "ATCGATCGATCGATCGATCG"
    print(f"\n3a. Test sequence: {test_sequence}")
    print(f"   Tokenizer params: kmer={tokenizer.tokenization_params['kmer']}, shift={tokenizer.tokenization_params['shift']}")

    # Tokenize using ProkBERT tokenizer
    encoded = tokenizer.encode(test_sequence, all=True)
    print(f"\n3b. Tokenized output (all shifts):")
    for i, tokens in enumerate(encoded):
        print(f"   Shift {i}: {tokens[:10]}... (length: {len(tokens)})")

    # Check token IDs
    print(f"\n3c. Special token IDs:")
    print(f"   [PAD] ID: {tokenizer.vocab.get('[PAD]', 'N/A')}")
    print(f"   [UNK] ID: {tokenizer.vocab.get('[UNK]', 'N/A')}")
    print(f"   [CLS] ID: {tokenizer.vocab.get('[CLS]', 'N/A')}")
    print(f"   [SEP] ID: {tokenizer.vocab.get('[SEP]', 'N/A')}")
    print(f"   [MASK] ID: {tokenizer.vocab.get('[MASK]', 'N/A')}")

    # =========================================================================
    # STEP 4: Verify embedding extraction
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Verify Embedding Extraction")
    print("=" * 70)

    # Create simple input
    tokens = encoded[0]  # First shift
    input_ids = torch.tensor([tokens], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == 0] = 0  # Mask padding

    print(f"\n4a. Input shape: {input_ids.shape}")
    print(f"   Input IDs: {input_ids[0, :15].tolist()}...")
    print(f"   Attention mask: {attention_mask[0, :15].tolist()}...")

    # Get embeddings from both models
    with torch.no_grad():
        direct_out = direct_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        random_out = random_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        prokbert_out = prokbert_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    direct_hidden = direct_out.hidden_states[-1]
    random_hidden = random_out.hidden_states[-1]
    prokbert_hidden = prokbert_out.hidden_states[-1]

    print(f"\n4b. Last hidden state statistics:")
    print(f"   Direct model - mean: {direct_hidden.mean().item():.6f}, std: {direct_hidden.std().item():.6f}")
    print(f"   Random model - mean: {random_hidden.mean().item():.6f}, std: {random_hidden.std().item():.6f}")
    print(f"   ProkBERT model - mean: {prokbert_hidden.mean().item():.6f}, std: {prokbert_hidden.std().item():.6f}")

    # Compare outputs
    direct_random_diff = (direct_hidden - random_hidden).abs().mean().item()
    direct_prokbert_diff = (direct_hidden - prokbert_hidden).abs().mean().item()

    print(f"\n4c. Output differences:")
    print(f"   Direct vs Random: {direct_random_diff:.6f}")
    print(f"   Direct vs ProkBERT: {direct_prokbert_diff:.6f}")

    if direct_random_diff < 0.01:
        print("   *** WARNING: Pretrained and random outputs are nearly identical! ***")
    else:
        print("   OK: Pretrained outputs differ from random")

    if direct_prokbert_diff > 0.001:
        print("   *** WARNING: ProkBERT loading produces different outputs! ***")
    else:
        print("   OK: ProkBERT outputs match direct load")

    # =========================================================================
    # STEP 5: Test with ProkBERT's data pipeline
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Test with ProkBERT's Data Pipeline")
    print("=" * 70)

    from prokbert.training_utils import get_torch_data_from_segmentdb_classification
    from prokbert.prok_datasets import ProkBERTTrainingDatasetPT

    # Create test data
    test_data = pd.DataFrame({
        'segment': [test_sequence, "GCTAGCTAGCTAGCTAGCTA"],
        'segment_id': ['seq_0', 'seq_1'],
        'y': [0, 1],
        'label': [0, 1]
    })

    print(f"\n5a. Test data shape: {test_data.shape}")

    # Tokenize using ProkBERT's pipeline
    X, y, torchdb = get_torch_data_from_segmentdb_classification(tokenizer, test_data)

    print(f"\n5b. Tokenized data:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   X[0, :15]: {X[0, :15].tolist()}")
    print(f"   y: {y.tolist()}")

    # Create dataset
    dataset = ProkBERTTrainingDatasetPT(X, y, AddAttentionMask=True)
    sample = dataset[0]

    print(f"\n5c. Dataset sample:")
    print(f"   input_ids: {sample['input_ids'][:15].tolist()}...")
    print(f"   attention_mask: {sample['attention_mask'][:15].tolist()}...")
    print(f"   label: {sample['labels'].item()}")

    # Check attention mask logic
    print(f"\n5d. Attention mask analysis:")
    input_ids_sample = sample['input_ids']
    attn_mask_sample = sample['attention_mask']

    token_0_masked = (attn_mask_sample[input_ids_sample == 0] == 0).all().item() if (input_ids_sample == 0).any() else "N/A"
    token_1_masked = (attn_mask_sample[input_ids_sample == 1] == 1).all().item() if (input_ids_sample == 1).any() else "N/A"
    token_2_masked = (attn_mask_sample[input_ids_sample == 2] == 1).all().item() if (input_ids_sample == 2).any() else "N/A"
    token_3_masked = (attn_mask_sample[input_ids_sample == 3] == 0).all().item() if (input_ids_sample == 3).any() else "N/A"

    print(f"   Token 0 ([PAD]) masked out: {token_0_masked}")
    print(f"   Token 1 ([UNK]) included: {token_1_masked}")
    print(f"   Token 2 ([CLS]) included: {token_2_masked}")
    print(f"   Token 3 ([SEP]) masked out: {token_3_masked}")

    if token_3_masked == True:
        print("   Note: [SEP] token is being masked out - this is unusual but may be intentional")

    # =========================================================================
    # STEP 6: Full embedding extraction test
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Full Embedding Extraction Test")
    print("=" * 70)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    prokbert_model.eval()
    random_model.eval()

    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        with torch.no_grad():
            prokbert_outputs = prokbert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            random_outputs = random_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        prokbert_last_hidden = prokbert_outputs.hidden_states[-1]
        random_last_hidden = random_outputs.hidden_states[-1]

        # Mean pooling
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(prokbert_last_hidden.size()).float()
        sum_embeddings_prokbert = torch.sum(prokbert_last_hidden * attention_mask_expanded, 1)
        sum_embeddings_random = torch.sum(random_last_hidden * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)

        prokbert_embeddings = (sum_embeddings_prokbert / sum_mask).numpy()
        random_embeddings = (sum_embeddings_random / sum_mask).numpy()

        print(f"\n6a. Embeddings shape: {prokbert_embeddings.shape}")
        print(f"\n6b. ProkBERT embeddings stats:")
        print(f"   Sample 0 - mean: {prokbert_embeddings[0].mean():.6f}, std: {prokbert_embeddings[0].std():.6f}")
        print(f"   Sample 1 - mean: {prokbert_embeddings[1].mean():.6f}, std: {prokbert_embeddings[1].std():.6f}")

        print(f"\n6c. Random embeddings stats:")
        print(f"   Sample 0 - mean: {random_embeddings[0].mean():.6f}, std: {random_embeddings[0].std():.6f}")
        print(f"   Sample 1 - mean: {random_embeddings[1].mean():.6f}, std: {random_embeddings[1].std():.6f}")

        # Check cosine similarity between embeddings
        from numpy.linalg import norm
        cos_sim_prokbert = np.dot(prokbert_embeddings[0], prokbert_embeddings[1]) / (norm(prokbert_embeddings[0]) * norm(prokbert_embeddings[1]))
        cos_sim_random = np.dot(random_embeddings[0], random_embeddings[1]) / (norm(random_embeddings[0]) * norm(random_embeddings[1]))

        print(f"\n6d. Cosine similarity between two samples:")
        print(f"   ProkBERT: {cos_sim_prokbert:.6f}")
        print(f"   Random: {cos_sim_random:.6f}")

        break

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
If the diagnostics above show:
1. Pretrained weights loading correctly (different from random)
2. ProkBERT loading matches direct HuggingFace load
3. Outputs from pretrained model differ from random model

Then the model IS loading correctly, and the issue might be:
- The dataset/task is not suitable for pretrained features
- The evaluation method has issues
- Label alignment problems

If pretrained weights are similar to random, check:
- Network connectivity to HuggingFace
- Model cache corruption (try clearing ~/.cache/huggingface/)
""")

if __name__ == "__main__":
    main()
