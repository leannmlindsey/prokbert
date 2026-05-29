# ProkBERT Generic Sequence Classification

> **Note:** This is a fork of [nbrg-ppcu/prokbert](https://github.com/nbrg-ppcu/prokbert)
> extended with scripts for **generic CSV-based binary classification**, suitable for
> benchmarking ProkBERT on the [LAMBDA prophage-detection benchmark](https://github.com/leannmlindsey/LAMBDA)
> or any other binary DNA sequence classification task. The original ProkBERT
> documentation is preserved verbatim in [`UPSTREAM_README.md`](./UPSTREAM_README.md).

---

## Relationship to the upstream training code

The fine-tune script in this fork (`finetune_prokbert_phage.py`) is a thin
wrapper around `transformers.Trainer` with `AutoModelForSequenceClassification`
— **the same machinery** the upstream ProkBERT reference path uses
([`examples/finetuning.py`](./examples/finetuning.py)). Upstream reads its
hyperparameters from
[`src/prokbert/configs/pretraining.yaml`](./src/prokbert/configs/pretraining.yaml);
this fork chooses defaults better-suited to short-sequence binary
classification, while preserving every CLI flag so users can override them:

| Parameter | Default (this fork) | Source / rationale |
|-----------|---------------------|--------------------|
| `learning_rate` | 1e-4 | this fork — lower than upstream's pretraining 5e-4; standard for classification head tuning |
| `weight_decay` | 0.01 | this fork — HF default; upstream pretraining default is 0.1 |
| `warmup_ratio` | 0.1 | this fork — HF default |
| `num_train_epochs` | 3 | this fork — upstream pretraining default is 1; 3 is more typical for finetune-from-pretrained |
| `per_device_train_batch_size` | 32 | this fork |
| `per_device_eval_batch_size` | 64 | this fork |
| `gradient_accumulation_steps` | 1 | HF default |
| `max_length` | 1024 | this fork (matches `prokbert-mini`; use 2048 for `mini-c` / `mini-long`) |
| `metric_for_best_model` | `eval_mcc` | **LAMBDA-specific** (the LAMBDA paper reports MCC) |
| `load_best_model_at_end` | True | this fork |
| `early_stopping_patience` | 3 epochs | this fork |
| `save_total_limit` | 2 | this fork |
| `fp16` | opt-in flag | this fork — A100 efficiency |
| `seed` | 42 | HF convention |

The intentional deviations from upstream pretraining defaults are tuned for
short-sequence classification (lower LR, fewer epochs, MCC-driven model
selection). The upstream Colab notebooks and `examples/finetuning.py` remain
unchanged in this fork; use them if you want their original training loop.

## What this fork adds

| File | Purpose |
|------|---------|
| `finetune_prokbert_phage.py` | Fine-tune any ProkBERT checkpoint on a binary CSV dataset (`train.csv` / `dev.csv` (or `val.csv`) / `test.csv` with `sequence,label` columns). |
| `inference_lambda.py` | Inference using a locally-stored fine-tuned checkpoint produced by the script above. |
| `inference_hf.py` | Inference using a fine-tuned ProkBERT model hosted on HuggingFace Hub (e.g. `neuralbioinfo/prokbert-mini-c-phage`). |
| `inference_embedding_head.py` | Inference using a saved linear-probe or 3-layer-NN head over pretrained ProkBERT embeddings (used when those beat the full finetune). |
| `embedding_analysis_prokbert.py` | Extract pretrained embeddings; train a linear probe + 3-layer NN; compute silhouette score, PCA, and (optionally) a random-init baseline. |
| `analyze_threshold.py` | ROC + PR curves; find the optimal classification threshold from a predictions CSV. |
| `reeval_with_threshold.py` | Re-apply a chosen threshold to existing predictions and recompute metrics. |
| `analyze_predictions.py` | Compare prediction CSVs across multiple tools; taxonomy-aware plots. |
| `analyze_genome_wide_results.py` | Threshold + clustering parameter sweep for genome-wide windowed predictions. |
| `find_best_seed.py` | Pick the highest-scoring seed across multiple finetune runs. |
| `scripts/select_best_model.py` | Pick the best of `{finetune seeds, linear probe, 3-layer NN}` per architecture by test-set MCC. |
| `configs/lambda_replication.conf` | Config file for the LAMBDA-replication pipeline. |
| `slurm_scripts/wrapper_run_*.sh` | SLURM submission wrappers for generic CSV tasks (finetune, embedding analysis, batch inference). |
| `slurm_scripts/lambda_replication/` | SLURM submission scripts for the end-to-end LAMBDA replication pipeline. |

## Installation

Install the upstream ProkBERT package (recommended in editable mode for local
development of the scripts above):

```bash
pip install git+https://github.com/nbrg-ppcu/prokbert.git
# or, for editing this fork's scripts:
git clone https://github.com/leannmlindsey/ProkBERT_generic_sequence_classification.git
cd ProkBERT_generic_sequence_classification
pip install -e .
```

Install a CUDA-enabled PyTorch first for training on GPU:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Docker / Singularity images and the Bioconda package are detailed in
[`UPSTREAM_README.md`](./UPSTREAM_README.md#installation).

## Using the fork

Two supported workflows:

| If you want to... | Go to |
|---|---|
| Use ProkBERT on **your own** binary classification CSV (finetune, evaluate embeddings, predict) | [Generic classification](#generic-classification) |
| **Replicate** the LAMBDA phage paper — train all three ProkBERT variants on the LAMBDA dataset, pick the best per architecture, run all diagnostic + genome-wide inference | [LAMBDA replication](#lambda-replication) |

### Generic classification

**Inputs:** a directory containing `train.csv`, `dev.csv` (or `val.csv`),
`test.csv`. Each CSV must have a `sequence` column and a `label` column (0/1).

Three sub-steps, each a separate SLURM submission:

```bash
# 1. Embedding analysis — linear probe + 3-layer NN on pretrained embeddings
#    (edit the CSV_DIR / MODEL_PATH config block at the top, then run)
bash slurm_scripts/wrapper_run_embedding_analysis.sh

# 2. Fine-tuning — full encoder fine-tune
#    (edit CSV_DIR / MODEL_NAME / hyperparams, then run)
#    Set NUM_REPLICATES > 1 to submit multiple seeds as separate jobs.
bash slurm_scripts/wrapper_run_prokbert_csv.sh

# 3. Inference — either path:
bash slurm_scripts/wrapper_run_batch_inference.sh       # local fine-tuned checkpoint
bash slurm_scripts/wrapper_run_batch_inference_hf.sh    # HuggingFace-hosted model
```

`INPUT_LIST` in the batch-inference wrappers is a text file with one CSV path
per line; one SLURM job per input.

For running directly with Python (no SLURM) or the full flag list for each
script, see [Reference: script flags](#reference-script-flags).

### LAMBDA replication

A two-step workflow over a single config file. The pipeline loops over the
LAMBDA_v1 segment lengths (2k / 4k / 8k by default) and for each length
submits: finetune × 3 architectures × N seeds, embedding analysis × 3
architectures, automatic best-model selection (by test-set MCC), inference on
the matching-length diagnostic CSVs, and genome-wide inference + threshold +
clustering sweep.

```bash
# 1. Edit the config — LAMBDA_BASE and OUTPUT_DIR are required;
#    SEEDS, ARCHS, FNR_<LEN>, GENOME_WIDE_<LEN> are optional.
$EDITOR configs/lambda_replication.conf

# 2. Launch all training (finetune × N seeds + embedding analysis × 3 archs,
#    per segment length, in parallel — no dependency chaining)
bash slurm_scripts/lambda_replication/run_lambda_training.sh

# 3. Wait — squeue -u $USER

# 4. Launch all inference (per length: pick winner by test-MCC; run inference
#    on test, fpr, gc_control, fnr; run genome-wide inference + sweep)
bash slurm_scripts/lambda_replication/run_lambda_inference.sh
```

**Expected LAMBDA_v1 layout** (auto-derived from `LAMBDA_BASE`):

```
LAMBDA_BASE/
├── train_val_test/<LEN>/{train,val,test}.csv     finetune + embedding + test diagnostic
├── fpr_test/<LEN>/bacteria_segments_<LEN>.csv    fpr diagnostic
└── shuffled_controls/<LEN>/test_shuffled.csv     gc_control diagnostic
```

FNR and genome-wide inputs are not part of LAMBDA_v1; provide them via the
optional `FNR_<LEN>` and `GENOME_WIDE_<LEN>` config variables (`GENOME_WIDE_<LEN>`
can be a single CSV or a directory of CSVs — each becomes its own inference job).

**Output layout:**

```
<OUTPUT_DIR>/
├── <LEN>/                              one subdir per SEGMENT_LENGTHS entry
│   ├── finetune/<arch>/seed-<N>/       test_results.json, best_model/
│   ├── embedding/<arch>/               embedding_analysis_results.json, classifiers
│   ├── winners.json                    picked by run_lambda_inference.sh
│   ├── inference/<arch>/               <dataset>_predictions.csv (+ _metrics.json)
│   └── genome_wide_analysis/<arch>/    threshold + clustering sweep CSVs
└── logs/                               SLURM stdout/stderr per job (shared)
```

## Reference: script flags

### `embedding_analysis_prokbert.py`

| Argument | Default | Description |
| --- | --- | --- |
| `--csv_dir` | (required) | Directory with train/dev/test CSVs |
| `--model_path` | `neuralbioinfo/prokbert-mini` | HuggingFace name or local path |
| `--output_dir` | `./results/embedding_analysis` | Model name appended automatically |
| `--batch_size` | 32 | |
| `--max_length` | 1024 | Clamped to model max |
| `--pooling` | `mean` | `mean` / `max` / `cls` |
| `--nn_epochs` | 100 | 3-layer NN training epochs |
| `--nn_hidden_dim` | auto | Defaults to model embedding dim |
| `--nn_lr` | 0.001 | |
| `--seed` | 42 | |
| `--include_random_baseline` | off | Also evaluate a randomly initialized encoder |

### `finetune_prokbert_phage.py`

See the [hyperparameter table above](#relationship-to-the-upstream-training-code)
for the most-tuned defaults. Other useful flags:

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset_dir` | (required) | Directory with train/dev/test CSVs |
| `--model_name` | `neuralbioinfo/prokbert-mini` | HuggingFace name or local path |
| `--output_dir` | `./prokbert_phage_finetuned` | |
| `--eval_strategy` | `epoch` | `no` / `steps` / `epoch` |
| `--save_strategy` | `epoch` | `no` / `steps` / `epoch` |
| `--logging_steps` | 100 | |
| `--random_init` | off | Random initialization instead of pretrained weights |

### `inference_lambda.py` (local checkpoint)

| Argument | Default | Description |
| --- | --- | --- |
| `--checkpoint_path` | (required) | Fine-tuned checkpoint dir |
| `--base_model` | `neuralbioinfo/prokbert-mini` | Base the checkpoint was finetuned from |
| `--dataset` | `leannmlindsey/lambda` | HuggingFace dataset name |
| `--dataset_file` | none | Local CSV/TSV (overrides `--dataset`) |
| `--split` | `test` | |
| `--batch_size` | 32 | |
| `--max_length` | 1024 | |
| `--output_dir` | `inference_results` | |
| `--output_file` | auto | |
| `--no_labels` | off | Prediction-only mode |
| `--save_metrics` | off | Write a sibling `_metrics.json` |
| `--device` | auto | Force `cuda` or `cpu` |

### `inference_hf.py` (HuggingFace-hosted model)

| Argument | Default | Description |
| --- | --- | --- |
| `--model_name` | `neuralbioinfo/prokbert-mini-c-phage` | HuggingFace model name |
| `--kmer` | auto | Auto-detected from model name |
| `--shift` | auto | Auto-detected from model name |
| `--dataset` | none | HuggingFace dataset name |
| `--dataset_file` | none | Local CSV/TSV (overrides `--dataset`) |
| `--split` | `test` | |
| `--batch_size` | 32 | |
| `--max_length` | 1024 | |
| `--output_dir` | `inference_results` | |
| `--output_file` | auto | |
| `--no_labels` | off | |
| `--save_metrics` | off | |
| `--device` | auto | |

### `inference_embedding_head.py` (LP / 3-layer NN head)

| Argument | Default | Description |
| --- | --- | --- |
| `--base_model` | (required) | Pretrained ProkBERT (for embeddings) |
| `--head_type` | (required) | `linear_probe` or `three_layer_nn` |
| `--head_path` | (required) | `.pkl` (LP) or `.pt` (NN) classifier |
| `--scaler_path` | (NN only) | Scaler `.pkl` for the NN head |
| `--dataset_file` | (required) | Input CSV with `sequence` column |
| `--output_dir` | `inference_results` | |
| `--output_file` | auto | |
| `--batch_size` | 32 | |
| `--max_length` | 1024 | |
| `--pooling` | `mean` | Must match the pooling used in embedding analysis |
| `--no_labels` | off | |
| `--save_metrics` | off | |
| `--device` | auto | |

## Available models

| Model | k-mer | Shift | Max position embeddings | HuggingFace |
| --- | --- | --- | --- | --- |
| ProkBERT-mini | 6 | 1 | 1024 | [neuralbioinfo/prokbert-mini](https://huggingface.co/neuralbioinfo/prokbert-mini) |
| ProkBERT-mini-c | 1 | 1 | 2048 | [neuralbioinfo/prokbert-mini-c](https://huggingface.co/neuralbioinfo/prokbert-mini-c) |
| ProkBERT-mini-long | 6 | 2 | 2048 | [neuralbioinfo/prokbert-mini-long](https://huggingface.co/neuralbioinfo/prokbert-mini-long) |

Published phage finetunes from the original paper:
`neuralbioinfo/prokbert-mini-phage`, `-mini-long-phage`, `-mini-c-phage`.

## Citation

If you use ProkBERT itself, cite the original paper:

```bibtex
@Article{ProkBERT2024,
  author  = {Ligeti, Balázs and Szepesi-Nagy, István and Bodnár, Babett and Ligeti-Nagy, Noémi and Juhász, János},
  journal = {Frontiers in Microbiology},
  title   = {{ProkBERT} family: genomic language models for microbiome applications},
  year    = {2024},
  volume  = {14},
  URL     = {https://www.frontiersin.org/articles/10.3389/fmicb.2023.1331233},
  DOI     = {10.3389/fmicb.2023.1331233},
  ISSN    = {1664-302X}
}
```
