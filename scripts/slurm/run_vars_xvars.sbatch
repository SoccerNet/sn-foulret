#!/usr/bin/env bash
# Generate the VARS+XVARS text-mediated baseline rankings
# Reads the released artifacts under retrieval/vars_xvars/ and writes its rankings/.
# Usage: sbatch scripts/slurm/run_vars_xvars.sbatch
#SBATCH --job-name=baseline
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=slurm_outputs/baseline-%j.out
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO"; mkdir -p slurm_outputs
module load python/3.11.0
source "$REPO/.venv/bin/activate"
echo ">> generate the paper VARS+XVARS rankings for Test and Valid"
for SP in Test Valid; do
  python -m retrieval.vars_xvars.retrieve --split "$SP"
done
echo ">> baseline rankings -> retrieval/vars_xvars/rankings/"
echo "=== DONE ==="
