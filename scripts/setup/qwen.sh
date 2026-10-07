#!/usr/bin/env bash
#
# Create the dedicated Qwen3-VL Conda environment used for retrieval and fine-tuning.
#
#   bash scripts/setup/qwen.sh
#
# Then run the pipeline with:
#   sbatch scripts/slurm/run_retrieve.sbatch <model>  (optional; see README.md)
#
# Some systems export a shared PYTHONPATH or expose user-site packages. This
# script clears both sources of package leakage during installation.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Install Miniconda or Anaconda first." >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
ENV_NAME=foulret-qwen

# Isolate pip from packages outside this environment.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">> Reusing Conda environment $ENV_NAME"
else
  echo ">> Creating Conda environment $ENV_NAME"
  conda create -n "$ENV_NAME" python=3.11 pip -y
fi
conda activate "$ENV_NAME"

echo ">> Installing requirements/qwen.txt"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/qwen.txt

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

echo ">> Done. Activate with: conda activate $ENV_NAME"
