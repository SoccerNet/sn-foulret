#!/usr/bin/env bash
#
# Create the dedicated Qwen3-VL environment used for retrieval and fine-tuning.
#
#   bash scripts/setup/qwen.sh
#
# Then run the pipeline with:
#   sbatch scripts/slurm/run_retrieve.sbatch <model>  (optional; see README.md)
#
# Some cluster Python modules export a shared PYTHONPATH that can shadow virtual
# environment packages. This script clears it during installation and activation.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

command -v module >/dev/null && module load python/3.11.0 || true

# Isolate from the module's leaked PYTHONPATH so pip installs the FULL closure.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

echo ">> Creating venv at $REPO/.venv"
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
unset PYTHONPATH            # the activation hook is appended later in this script
export PYTHONNOUSERSITE=1

echo ">> Installing requirements/qwen.txt"
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements/qwen.txt

# Make the isolation a permanent property of THIS venv: appended (idempotently) so
# every future `source .venv/bin/activate` neutralizes the module's PYTHONPATH leak.
ACTIVATE="$REPO/.venv/bin/activate"
if ! grep -q "FoulRet venv isolation" "$ACTIVATE"; then
  cat >> "$ACTIVATE" <<'ISO'

# --- FoulRet venv isolation (appended by scripts/setup/qwen.sh) ---
# `module load python/3.11.0` exports PYTHONPATH at the shared site-packages, which
# would shadow this venv. Neutralize it whenever the venv is active.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
ISO
  echo ">> Patched $ACTIVATE with cluster PYTHONPATH isolation"
fi

echo ">> Verifying the environment"
python - <<'PY'
import importlib
need = ["torch","transformers","faiss","numpy","scipy","pandas","dateutil",
        "typing_extensions","qwen_vl_utils","decord","PIL","accelerate","peft"]
bad = 0
for m in need:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad += 1; print(f"  MISSING {m}: {e}")
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel  # noqa
import numpy
assert numpy.__version__.startswith("2."), f"numpy shadowed: {numpy.__version__}"
# Verify the local backend and retrieval interfaces:
from retrieval.engine import RetrievalEngine  # noqa
from retrieval.models import get_model  # noqa
from retrieval.models.qwen3vl_backend import Qwen3VLEmbedder  # noqa
assert bad == 0, f"{bad} missing dependencies"
print("  OK: deps complete, numpy", numpy.__version__, ", vendored backend imports clean")
PY

echo ">> Done. Activate with:  source .venv/bin/activate"
