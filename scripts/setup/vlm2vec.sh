#!/usr/bin/env bash
# Create the dedicated VLM2Vec-V2/LamRA Conda environment.
# Run: bash scripts/setup/vlm2vec.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"

echo ">> Initializing Clones/VLM2Vec"
git submodule update --init Clones/VLM2Vec

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Install Miniconda or Anaconda first." >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
ENV_NAME=foulret-vlm2vec
unset PYTHONPATH; export PYTHONNOUSERSITE=1

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">> Reusing Conda environment $ENV_NAME"
else
  echo ">> Creating Conda environment $ENV_NAME"
  conda create -n "$ENV_NAME" python=3.11 pip -y
fi
conda activate "$ENV_NAME"
unset PYTHONPATH; export PYTHONNOUSERSITE=1

echo ">> Installing VLM2Vec and demo requirements"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/vlm2vec.txt -r requirements/demo.txt

echo ">> Verifying"
python - <<'PY'
import importlib, sys, os
for m in ["torch","transformers","faiss","decord","pandas","numpy","peft","flask"]:
    importlib.import_module(m)
sys.path.insert(0, os.path.join(os.getcwd(), "Clones", "VLM2Vec"))
from src.model.model import MMEBModel            # clone imports resolve
from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS
import transformers, numpy
print("OK: transformers", transformers.__version__, "numpy", numpy.__version__,
      "- VLM2Vec clone + engine deps import cleanly")
PY
echo ">> Done. Activate with: conda activate $ENV_NAME"
