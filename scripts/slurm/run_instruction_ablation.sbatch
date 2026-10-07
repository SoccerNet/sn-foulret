#!/usr/bin/env bash
#SBATCH --job-name=instr_ablation
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128GB
#SBATCH --time=5:59:59
#SBATCH --output=slurm_outputs/slurm-%j.out
# Instruction-sensitivity sweep for one model using generic, domain, and detailed
# query instructions against the model's full corpus index. Usage:
#   sbatch scripts/slurm/run_instruction_ablation.sbatch <model_key> [nframes=16]
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"; mkdir -p slurm_outputs

MODEL="${1:?model key required}"; NFRAMES="${2:-16}"
case "$MODEL" in
  qwen3vl-*)       VENV=.venv ;;
  vlm2vec-*|lamra) VENV=.venv_vlm2vec ;;
  internvideo2-*)  VENV=.venv_internvideo2 ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
module load python/3.11.0
source "$REPO/$VENV/bin/activate"

IDX="retrieval/indexes/${MODEL}"
declare -A INSTRUCTIONS=(
  [generic]="Retrieve videos similar to the query video."
  [domain]="Retrieve soccer foul videos similar to the query."
  [detailed]="Retrieve soccer foul videos of the same action type, offence, and severity as the query."
)
for LEVEL in generic domain detailed; do
  OUT="retrieval/rankings/instruction_ablation/${MODEL}/${LEVEL}"
  echo ">> [$MODEL] $LEVEL  ::  ${INSTRUCTIONS[$LEVEL]}"
  python -m retrieval.retrieve --model "$MODEL" \
      --data_dir data/mvfouls_window --index_base "$IDX" \
      --nframes "$NFRAMES" --clip_mode multi_clip \
      --instruction "${INSTRUCTIONS[$LEVEL]}" --out_dir "$OUT" \
      --views full --top_k 20 --no_query_cache --plain_filenames
done
echo ">> SWEEP DONE for $MODEL"
