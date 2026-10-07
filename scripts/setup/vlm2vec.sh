#!/usr/bin/env bash
# Create the dedicated VLM2Vec-V2/LamRA environment.
# Run: bash scripts/setup/vlm2vec.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"


command -v module >/dev/null && module load python/3.11.0 || true
unset PYTHONPATH; export PYTHONNOUSERSITE=1

echo ">> Creating venv at $REPO/.venv_vlm2vec"
rm -rf .venv_vlm2vec
python -m venv .venv_vlm2vec
source .venv_vlm2vec/bin/activate
unset PYTHONPATH; export PYTHONNOUSERSITE=1

echo ">> Installing requirements/vlm2vec.txt"
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements/vlm2vec.txt

# Preserve cluster PYTHONPATH isolation whenever the environment is activated.
ACT="$REPO/.venv_vlm2vec/bin/activate"
if ! grep -q "FoulRet venv isolation" "$ACT"; then
  cat >> "$ACT" <<'ISO'

# --- FoulRet venv isolation (appended by scripts/setup/vlm2vec.sh) ---
unset PYTHONPATH
export PYTHONNOUSERSITE=1
ISO
fi

echo ">> Verifying"
python - <<'PY'
import importlib, sys, os
for m in ["torch","transformers","faiss","decord","pandas","numpy","peft"]:
    importlib.import_module(m)
sys.path.insert(0, os.path.join(os.getcwd(), "Clones", "VLM2Vec"))
from src.model.model import MMEBModel            # clone imports resolve
from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS
import transformers, numpy
print("OK: transformers", transformers.__version__, "numpy", numpy.__version__,
      "- VLM2Vec clone + engine deps import cleanly")
PY
echo ">> Done. Activate with:  source .venv_vlm2vec/bin/activate"
