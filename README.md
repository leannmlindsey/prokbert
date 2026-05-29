# ProkBERT for sequence classification

This is a fork of [nbrg-ppcu/prokbert](https://github.com/nbrg-ppcu/prokbert) — the
[ProkBERT family](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2023.1331233/full)
of genomic language models — extended with scripts for end-to-end binary
classification of DNA sequences (e.g. phage vs. bacteria).

## Which path are you on?

Two supported workflows. Pick one and skip the other.

| If you want to... | Go to |
| --- | --- |
| Use ProkBERT on **your own** binary classification CSV (finetune, evaluate embeddings, predict) | [Path 1](#path-1-use-prokbert-on-your-own-task) |
| **Replicate** the phage paper results — train all three ProkBERT variants on the lambda dataset, pick the best, run all diagnostic + genome-wide inference | [Path 2](#path-2-replicate-the-phage-paper-results) |

The upstream ProkBERT documentation (installation, model details, pretraining,
citation) is preserved [further down](#prokbert-package--upstream-documentation).
For installation and environment setup, see [Installation](#installation).

---

## Path 1: Use ProkBERT on your own task

**Inputs you need:** a directory containing `train.csv`, `dev.csv` (or `val.csv`),
`test.csv`. Each CSV must have a `sequence` column and a `label` column (0/1
for binary classification).

**Three sub-steps** — each is a separate SLURM submission you can run independently:

### 1.1 Embedding analysis

Extract pretrained ProkBERT embeddings and evaluate them with a linear probe,
a 3-layer NN, silhouette score, and PCA. Useful as a fast first look without
training the encoder.

```bash
# Edit the config section, then submit:
bash slurm_scripts/wrapper_run_embedding_analysis.sh
```

Outputs land in `./results/embedding_analysis/<dataset>/<model>/` and include
classifiers, metrics JSON, and PCA plots.

### 1.2 Fine-tuning

Fine-tune a ProkBERT variant on your dataset. Supports early stopping,
mixed-precision training, and multiple replicates with different seeds.

```bash
# Edit the config section (CSV_DIR, MODEL_NAME, hyperparams), then submit:
bash slurm_scripts/wrapper_run_prokbert_csv.sh
```

Outputs land in `./results/csv_binary/<dataset>/lr-<lr>_batch-<batch>/seed-<N>/`.
Set `NUM_REPLICATES > 1` in the wrapper to submit multiple seeds as separate jobs.

### 1.3 Inference (batch)

Two flavors depending on where your fine-tuned model lives:

```bash
# Local fine-tuned checkpoint (from step 1.2):
bash slurm_scripts/wrapper_run_batch_inference.sh

# HuggingFace-hosted model:
bash slurm_scripts/wrapper_run_batch_inference_hf.sh
```

Both take an `INPUT_LIST` text file with one CSV path per line; one SLURM job
is submitted per input CSV. Predictions are saved as
`<input_basename>_predictions.csv` in the output directory.

For running directly with Python (no SLURM) and the full flag list, see
[Reference: script flags](#reference-script-flags).

---

## Path 2: Replicate the phage paper results

A two-step workflow over a single config file. Submits the full pipeline:
finetune × 3 architectures × N seeds, embedding analysis × 3 architectures,
automatic best-model selection (by test-set MCC), inference on diagnostic
CSVs, genome-wide inference, and a threshold + clustering sweep.

### Step 1: Edit the config

Open `configs/lambda_replication.conf` and fill in:

- `LAMBDA_DIR` — directory with `train.csv`, `dev.csv` (or `val.csv`), `test.csv`
- `OUTPUT_DIR` — where all results land
- `DIAGNOSTIC_DATASETS` — colon-separated `name=path` entries for the diagnostic
  CSVs (test set, GC control, FPR set, FNR set, etc.)
- `GENOME_WIDE_CSV` — windowed CSV for the genome-wide inference + sweep
- (Optional) `ARCHS`, `SEEDS`, hyperparameters, SLURM resources

The launcher refuses to submit if any `/path/to/...` placeholders remain.

### Step 2: Launch training

```bash
bash slurm_scripts/lambda_replication/run_lambda_training.sh
```

Submits all finetune jobs (one per `(architecture, seed)`) and all embedding
analysis jobs (one per architecture) in parallel.

**Wait for them to finish.** Monitor with `squeue -u $USER`.

### Step 3: Launch inference

```bash
bash slurm_scripts/lambda_replication/run_lambda_inference.sh
```

On the login node, this:

1. Reads every `test_results.json` and `embedding_analysis_results.json` under
   `OUTPUT_DIR` and writes `winners.json` — per architecture, the candidate
   with the highest test-set MCC across all finetune seeds + linear probe +
   3-layer NN.
2. Submits one inference job per `(architecture, diagnostic dataset)` and one
   per architecture for the genome-wide CSV.
3. Chains a threshold + clustering sweep job after each genome-wide inference
   job via `--dependency=afterok`.

The inference job script (`lambda_inference_job.sh`) auto-dispatches based on
winner type — full finetune uses `inference_lambda.py`, linear probe and
3-layer NN use `inference_embedding_head.py`.

### Output layout

```
<OUTPUT_DIR>/
├── finetune/<arch>/seed-<N>/         test_results.json, best_model/
├── embedding/<arch>/                 embedding_analysis_results.json, classifiers
├── winners.json                      picked by step 3
├── inference/<arch>/                 <dataset>_predictions.csv (+ _metrics.json)
├── genome_wide_analysis/<arch>/      threshold + clustering sweep CSVs
└── logs/                             SLURM stdout/stderr per job
```

---

## Diagnostic + analysis tools

Standalone scripts not tied to either path; useful for inspecting results:

| Script | Purpose |
| --- | --- |
| `analyze_threshold.py` | ROC + PR curves, find an optimal threshold on a predictions CSV |
| `reeval_with_threshold.py` | Apply a chosen threshold to existing predictions; recompute metrics |
| `analyze_predictions.py` | Compare prediction CSVs across multiple tools; taxonomy-aware plots |
| `analyze_genome_wide_results.py` | Threshold + clustering parameter sweep for genome-wide windowed predictions |
| `find_best_seed.py` | Pick the highest-scoring seed across multiple finetune runs |
| `scripts/select_best_model.py` | Pick best of `{finetune seeds, linear probe, 3-layer NN}` per architecture (used by Path 2 step 3) |

Each takes `--help` for arguments.

---

## Installation

### pip (recommended for the package)

```bash
pip install git+https://github.com/nbrg-ppcu/prokbert.git
```

Install a CUDA-enabled PyTorch first if you'll be training:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### conda (Bioconda)

```bash
conda install prokbert -c bioconda
```

### Docker / Singularity (Apptainer)

```bash
docker pull obalasz/prokbert
docker run --gpus all -it --rm -v $(pwd):/app obalasz/prokbert bash

singularity pull prokbert.sif docker://obalasz/prokbert
singularity run --nv prokbert.sif bash
```

For local development of the scripts in this repo (Path 1 and Path 2),
`pip install -e .` from the repo root.

---

## Available models

| Model | k-mer | Shift | Max position embeddings | HuggingFace |
| --- | --- | --- | --- | --- |
| ProkBERT-mini | 6 | 1 | 1024 | [neuralbioinfo/prokbert-mini](https://huggingface.co/neuralbioinfo/prokbert-mini) |
| ProkBERT-mini-c | 1 | 1 | 2048 | [neuralbioinfo/prokbert-mini-c](https://huggingface.co/neuralbioinfo/prokbert-mini-c) |
| ProkBERT-mini-long | 6 | 2 | 2048 | [neuralbioinfo/prokbert-mini-long](https://huggingface.co/neuralbioinfo/prokbert-mini-long) |

Published fine-tunes from the original ProkBERT paper:
- Promoter: `neuralbioinfo/prokbert-mini-promoter`, `-long-promoter`, `-c-promoter`
- Phage: `neuralbioinfo/prokbert-mini-phage`, `-long-phage`, `-c-phage`

---

## Reference: script flags

### `embedding_analysis_prokbert.py`

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--csv_dir` | (required) | Directory with train/dev/test CSVs |
| `--model_path` | `neuralbioinfo/prokbert-mini` | HuggingFace name or local path |
| `--output_dir` | `./results/embedding_analysis` | Model name is appended automatically |
| `--batch_size` | 32 | |
| `--max_length` | 1024 | Clamped to model max |
| `--pooling` | `mean` | `mean` / `max` / `cls` |
| `--nn_epochs` | 100 | 3-layer NN training epochs |
| `--nn_hidden_dim` | auto | Defaults to model embedding dim |
| `--nn_lr` | 0.001 | |
| `--seed` | 42 | |
| `--include_random_baseline` | off | Also evaluate a randomly initialized encoder |

### `finetune_prokbert_phage.py`

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--dataset_dir` | (required) | Directory with train/dev/test CSVs |
| `--model_name` | `neuralbioinfo/prokbert-mini` | HuggingFace name or local path |
| `--max_length` | 1024 | 1024 for mini, 2048 for mini-c / mini-long |
| `--output_dir` | `./prokbert_phage_finetuned` | |
| `--num_train_epochs` | 3 | |
| `--per_device_train_batch_size` | 32 | |
| `--per_device_eval_batch_size` | 64 | |
| `--learning_rate` | 1e-4 | |
| `--weight_decay` | 0.01 | |
| `--warmup_ratio` | 0.1 | |
| `--gradient_accumulation_steps` | 1 | |
| `--seed` | 42 | |
| `--fp16` | off | Mixed precision |
| `--early_stopping_patience` | 3 | Epochs without improvement |
| `--save_total_limit` | 2 | |
| `--eval_strategy` | `epoch` | `no` / `steps` / `epoch` |
| `--save_strategy` | `epoch` | `no` / `steps` / `epoch` |
| `--metric_for_best_model` | `eval_mcc` | |
| `--random_init` | off | Random init instead of pretrained weights |

### `inference_lambda.py` (local checkpoint)

| Argument | Default | Description |
| -------- | ------- | ----------- |
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
| -------- | ------- | ----------- |
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
| -------- | ------- | ----------- |
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

---

## ProkBERT package — upstream documentation

The sections below are from the upstream [nbrg-ppcu/prokbert](https://github.com/nbrg-ppcu/prokbert)
README and document the underlying ProkBERT package, models, and pretraining.

### Introduction
The ProkBERT model family is a transformer-based, encoder-only architecture
based on [BERT](https://github.com/google-research/bert). Built on transfer
learning and self-supervised methodologies, ProkBERT models capitalize on
the abundant available data, demonstrating adaptability across diverse
scenarios. The models' learned representations align with established
biological understanding, shedding light on phylogenetic relationships. With
the novel Local Context-Aware (LCA) tokenization, the ProkBERT family
overcomes the context size limitations of traditional transformer models
without sacrificing performance or the information-rich local context. In
bioinformatics tasks like promoter prediction and phage identification,
ProkBERT models excel. For promoter predictions, the best-performing model
achieved an MCC of 0.74 for E. coli and 0.62 in mixed-species contexts. In
phage identification, they all consistently outperformed tools like
VirSorter2 and DeepVirFinder, registering an MCC of 0.85.

### Features
- Tailored to microbes.
- Local Context-Aware (LCA) tokenization for better genomic sequence understanding.
- Pre-trained models available for immediate use and fine-tuning.
- High performance in various bioinformatics tasks.
- Facilitation of both supervised and unsupervised learning.

### TLDR example — load a model from HuggingFace

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("neuralbioinfo/prokbert-mini", trust_remote_code=True)
model = AutoModel.from_pretrained("neuralbioinfo/prokbert-mini", trust_remote_code=True)

segment = "TATGTAACATAATGCGACCAATAATCGTAATGAATATGAGAAGTGTGATATTATAACATTTCATGACTACTGCAAGACTAA"
inputs = tokenizer(segment, return_tensors="pt")
outputs = model(**inputs)
```

### Notebook tutorials (upstream)

- [Embedding visualization](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Embedding_visualization.ipynb)
- [Finetuning for promoter identification](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Finetuning.ipynb)
- [Segmentation](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Segmentation.ipynb)
- [Tokenization](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Tokenization.ipynb)
- [Inference / evaluation](https://colab.research.google.com/github/nbrg-ppcu/prokbert/blob/main/examples/Inference.ipynb)

### Pretraining from scratch

Preprocess fasta files, then pretrain. See `examples/prokbert_seqpreprocess.py`
and `examples/prokbert_pretrain.py`.

```bash
python examples/prokbert_seqpreprocess.py \
  --kmer 6 --shift 1 \
  --fasta_file_dir src/prokbert/data/pretraining \
  --out src/prokbert/data/preprocessed/pretraining_k6s1.h5

python examples/prokbert_pretrain.py \
  --kmer 6 --shift 1 \
  --dataset_path src/prokbert/data/preprocessed/pretraining_k6s1.h5 \
  --model_name prokbert_k6s1 \
  --output_dir ./tmppretraining \
  --model_outputpath ./tmppretraining
```

### Paper results

![UMAP embeddings of genomic segment representations](assets/Figure5_umaps.jpg)
*Figure 1: UMAP embeddings of genomic segment representations.*

![Promoter prediction performance metrics](assets/Figure6_prom_res.png)
*Figure 2: Performance metrics of ProkBERT in promoter prediction.*

![Comparative analysis of ProkBERT's phage prediction performance](assets/Figure7_phag_res.png)
*Figure 3: Comparative analysis showcasing ProkBERT's performance in phage prediction.*

### Datasets

| Dataset | HuggingFace |
| --- | --- |
| `neuralbioinfo/ESKAPE-genomic-features` | [Link](https://huggingface.co/datasets/neuralbioinfo/ESKAPE-genomic-features) |
| `neuralbioinfo/phage-test-10k` | [Link](https://huggingface.co/datasets/neuralbioinfo/phage-test-10k) |
| `neuralbioinfo/bacterial_promoters` | [Link](https://huggingface.co/datasets/neuralbioinfo/bacterial_promoters) |
| `neuralbioinfo/ESKAPE-masking` | [Link](https://huggingface.co/datasets/neuralbioinfo/ESKAPE-masking) |

---

## Citing this work

If you use the code or data in this package, please cite the original
ProkBERT paper:

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
