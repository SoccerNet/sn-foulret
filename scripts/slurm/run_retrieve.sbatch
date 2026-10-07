#!/usr/bin/env bash
# Top-100 retrieval over full / action+offence / action+offence+severity gallery views.
# Requires the full index and its label-complete evaluation indexes
# (retrieval.derive_label_complete_indexes).
# Usage: sbatch scripts/slurm/run_retrieve.sbatch <model_key>
#SBATCH --job-name=retrieve
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=slurm_outputs/multicorp-%j.out
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"; mkdir -p slurm_outputs
MODEL="${1:?model key required}"
NFRAMES=16
INSTR="Retrieve videos similar to the query video."          # generic (headline) instruction
case "$MODEL" in
  qwen3vl-*)        VENV=.venv ;;
  vlm2vec-*|lamra)  VENV=.venv_vlm2vec ;;
  internvideo2-*)   VENV=.venv_internvideo2 ;;
  *) echo "unknown model: $MODEL"; exit 1 ;;
esac
module load python/3.11.0
source "$REPO/$VENV/bin/activate"
INSTR_ARG=(--instruction "$INSTR")
case "$MODEL" in internvideo2-*) INSTR_ARG=() ;; esac        # video-native: no instruction
echo ">> retrieve $MODEL  nframes=$NFRAMES  venv=$VENV"
python -m retrieval.retrieve --model "$MODEL" \
    --index_base "retrieval/indexes/${MODEL}" \
    --data_dir data/mvfouls_window --nframes "$NFRAMES" \
    "${INSTR_ARG[@]}" --out_dir "retrieval/rankings/${MODEL}"
echo "=== DONE ==="
