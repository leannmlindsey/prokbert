#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=pbml_ft_gw
#SBATCH --output=pbml_ft_gw_%j.log
#SBATCH --error=pbml_ft_gw_%j.log
#
# ProkBERT prokbert-mini-long — FT genome-wide (2k), the one missing FT head.
# Its NN probe always won winner-selection, so the fine-tuned checkpoint was never
# deployed genome-wide. This deploys the BEST fine-tuned seed (seed 2, test MCC
# 0.897) across the 2k genome-wide CSVs via the existing inference_lambda.py, so
# best-of can cover {LP, NN, FT} for mini-long too. Writes to a dedicated dir so
# it does NOT overwrite the deployed NN genome-wide.
#
# Submit FROM the ProkBERT repo root on Delta:
#   cd <prokbert repo>; sbatch slurm_scripts/lambda_replication/run_prokbert_minilong_ft_gw.sbatch
set -uo pipefail

source /u/llindsey1/miniconda3/etc/profile.d/conda.sh
conda activate prokbert

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
echo "node $(hostname)  repo=$REPO_ROOT  conda=${CONDA_DEFAULT_ENV:-none}  start $(date)"

OUTROOT=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/ProkBERT_generic_sequence_classification/outputs
CKPT="$OUTROOT/2k/finetune/prokbert-mini-long/seed-2"
BASE="neuralbioinfo/prokbert-mini-long"
GW=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/LAMBDA_v1/genome_wide/2k
OUT="$OUTROOT/2k/genome_wide_ft/prokbert-mini-long"
mkdir -p "$OUT"

[ -d "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[ -f "$REPO_ROOT/inference_lambda.py" ] || { echo "ERROR: inference_lambda.py not in $REPO_ROOT"; exit 1; }

n=0
for csv in "$GW"/*.csv; do
  stem=$(basename "$csv" .csv)
  echo "  [$((n+1))] $stem"
  python inference_lambda.py \
    --checkpoint_path="$CKPT" \
    --base_model="$BASE" \
    --dataset_file="$csv" \
    --max_length=2048 \
    --batch_size=32 \
    --output_dir="$OUT" \
    --output_file="genome_wide_${stem}_predictions.csv" \
    --keep_output_dir \
    --save_metrics
  n=$((n+1))
done
echo "DONE $n genome(s) -> $OUT  $(date)"
