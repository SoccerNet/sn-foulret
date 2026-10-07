#!/usr/bin/env bash
# Reproduce the paper's task-specific Qwen3-VL-2B LoRA retrieval baseline.
#
# All paths are relative to the repository root. The job performs training,
# indexing, and retrieval sequentially so each stage receives complete inputs.
#
# Usage: sbatch scripts/slurm/run_finetune_qwen3vl.sbatch
#SBATCH --job-name=foulret-ft
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
# The 36 h limit includes headroom for training, indexing, and query projection.
#SBATCH --mem=32GB
#SBATCH --time=36:00:00
#SBATCH --output=slurm_outputs/finetune-%j.out

set -euo pipefail

REPO="${FOULRET_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO/evaluation/metrics.py" || ! -f "$REPO/train_qwen3vl.py" ]]; then
    echo "Submit this job from the SoccerNet-FoulRet repository root" >&2
    exit 2
fi
cd "$REPO"
mkdir -p slurm_outputs finetuning_outputs

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
eval "$(conda shell.bash hook)"
conda activate foulret-qwen
unset PYTHONPATH; export PYTHONNOUSERSITE=1

MODEL=qwen3vl-2b-finetuned
RUN_DIR="$REPO/finetuning_outputs/qwen3vl-2b-labelsup"
ADAPTER="$RUN_DIR/best_model"
INDEX_BASE="$REPO/retrieval/indexes/${MODEL}"
PRED_DIR="$REPO/retrieval/rankings/$MODEL"

echo ">> Validate the Train-only label split"
python train_qwen3vl.py \
    --data_dir data/mvfouls_window \
    --label_fields action offence severity \
    --data_only

echo ">> Fine-tune Qwen3-VL-Embedding-2B with label-supervised InfoNCE"
python train_qwen3vl.py \
    --data_dir data/mvfouls_window \
    --output_dir "$RUN_DIR" \
    --label_fields action offence severity \
    --clip_mode multi_clip \
    --epochs 10 --patience 3 \
    --batch_size 12 --gradient_accumulation_steps 8 \
    --lr 1e-5 --weight_decay 0.05 --temperature 0.05 \
    --scheduler cosine \
    --tier_weights 1 1 1 \
    --lora_rank 16 --lora_alpha 16 --lora_dropout 0.10 \
    --gradient_checkpointing

test -f "$ADAPTER/adapter_config.json"

echo ">> Build the fine-tuned Train gallery index"
python -m retrieval.build_full_index \
    --model "$MODEL" \
    --lora_path "$ADAPTER" \
    --data_dir data/mvfouls_window \
    --nframes 16 --clip_mode multi_clip \
    --index_dir "$INDEX_BASE" --force_rebuild

echo ">> Derive condensed category indexes"
python -m retrieval.derive_label_complete_indexes --index_dir "$INDEX_BASE"

echo ">> Retrieve Test and Valid queries"
python -m retrieval.retrieve \
    --model "$MODEL" \
    --lora_path "$ADAPTER" \
    --index_base "$INDEX_BASE" \
    --data_dir data/mvfouls_window \
    --nframes 16 --clip_mode multi_clip \
    --instruction "Retrieve videos similar to the query video." \
    --out_dir "$PRED_DIR"

echo ">> Fine-tuned predictions are ready"
echo ">> Evaluate reproduced predictions"
python -m evaluation.precedent \
    --rankings-dir "$PRED_DIR" \
    --method-name "Fine-tuned Qwen3-VL-Embedding-2B" \
    | tee "$RUN_DIR/evaluate_precedent.txt"
python -m evaluation.category \
    --rankings-dir "$PRED_DIR" \
    --method-name "Fine-tuned Qwen3-VL-Embedding-2B" \
    | tee "$RUN_DIR/evaluate_category.txt"
echo "=== DONE ==="
