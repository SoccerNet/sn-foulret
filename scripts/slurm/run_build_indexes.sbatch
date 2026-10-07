#!/usr/bin/env bash
# Build full and label-restricted indexes for one retrieval model.
# Usage: sbatch scripts/slurm/run_build_indexes.sbatch <model_key> [nframes]
#SBATCH --job-name=buildwin
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/buildwin-%j.out
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"; mkdir -p slurm_outputs
MODEL="${1:?model key required}"; NFRAMES="${2:-16}"
case "$MODEL" in
  qwen3vl-*)        CONDA_ENV=foulret-qwen ;;
  vlm2vec-*|lamra)  CONDA_ENV=foulret-vlm2vec ;;
  internvideo2-*)   CONDA_ENV=foulret-internvideo2 ;;
  *) echo "unknown model: $MODEL"; exit 1 ;;
esac
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
unset PYTHONPATH; export PYTHONNOUSERSITE=1
echo ">> build_full_index $MODEL  nframes=$NFRAMES  conda_env=$CONDA_ENV"
python -m retrieval.build_full_index --model "$MODEL" --data_dir data/mvfouls_window \
    --nframes "$NFRAMES" --index_dir "retrieval/indexes/${MODEL}" --force_rebuild
python -m retrieval.derive_label_complete_indexes \
    --index_dir "retrieval/indexes/${MODEL}"
echo "=== DONE ==="
