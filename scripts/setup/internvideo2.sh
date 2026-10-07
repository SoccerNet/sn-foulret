#!/usr/bin/env bash
# Create the dedicated InternVideo2-Stage2 environment. FlashAttention is
# installed from a prebuilt wheel, so no local CUDA compilation is required.
# Run: bash scripts/setup/internvideo2.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"


command -v module >/dev/null && module load python/3.11.0 || true
unset PYTHONPATH; export PYTHONNOUSERSITE=1

echo ">> Creating venv at $REPO/.venv_internvideo2"
rm -rf .venv_internvideo2
python -m venv .venv_internvideo2
source .venv_internvideo2/bin/activate
unset PYTHONPATH; export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip setuptools wheel

echo ">> Installing torch 2.4.1 + torchvision 0.19.1 (cu121)"
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

echo ">> Installing CORE flash-attn from prebuilt wheel (cu12 torch2.4 cp311 abiFALSE; no compile)"
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

echo ">> Installing requirements/internvideo2.txt"
pip install -r requirements/internvideo2.txt

# Preserve cluster PYTHONPATH isolation whenever the environment is activated.
ACT="$REPO/.venv_internvideo2/bin/activate"
if ! grep -q "FoulRet venv isolation" "$ACT"; then
  cat >> "$ACT" <<'ISO'

# --- FoulRet venv isolation (appended by scripts/setup/internvideo2.sh) ---
unset PYTHONPATH
export PYTHONNOUSERSITE=1
ISO
fi

echo ">> Verifying (core flash-attn + deps; the backbone import needs the adapter's fused-op shims)"
python - <<'PY'
import importlib
import torch; print("torch", torch.__version__)
import flash_attn; print("flash_attn", flash_attn.__version__)
# the exact core pieces the InternVideo2 FlashAttention class needs:
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
for m in ["decord", "av", "timm", "transformers", "faiss", "numpy", "pandas", "einops"]:
    importlib.import_module(m)
print("OK: core flash-attn (interface+bert_padding) + InternVideo2 deps import cleanly")
PY
echo ">> Done. Activate with:  source .venv_internvideo2/bin/activate"
