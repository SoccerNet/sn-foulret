#!/usr/bin/env bash
# Create the dedicated InternVideo2-Stage2 Conda environment. This repository
# uses InternVideo2's standard PyTorch attention path, so FlashAttention is not
# required.
# Run: bash scripts/setup/internvideo2.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"

echo ">> Initializing Clones/InternVideo"
git submodule update --init Clones/InternVideo

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Install Miniconda or Anaconda first." >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
ENV_NAME=foulret-internvideo2
unset PYTHONPATH; export PYTHONNOUSERSITE=1

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">> Reusing Conda environment $ENV_NAME"
else
  echo ">> Creating Conda environment $ENV_NAME"
  conda create -n "$ENV_NAME" python=3.11 pip -y
fi
conda activate "$ENV_NAME"
unset PYTHONPATH; export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip setuptools wheel

echo ">> Installing torch 2.4.1 + torchvision 0.19.1 (cu121)"
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

echo ">> Installing InternVideo2 and demo requirements"
python -m pip install -r requirements/internvideo2.txt -r requirements/demo.txt

echo ">> Verifying InternVideo2 dependencies"
python - <<'PY'
import importlib
import torch; print("torch", torch.__version__)
for m in ["decord", "av", "timm", "transformers", "faiss", "numpy", "pandas", "einops", "flask"]:
    importlib.import_module(m)
print("OK: InternVideo2 dependencies import cleanly")
PY
echo ">> Done. Activate with: conda activate $ENV_NAME"
