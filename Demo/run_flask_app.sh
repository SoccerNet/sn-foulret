#!/usr/bin/env bash
#
# Launch the SoccerNet-FoulRet demo from the active Python environment.
# Configuration is supplied through environment variables; see README.md.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-qwen3vl-2b}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/mvfouls_window}"
INDEX_DIR="${INDEX_DIR:-${PROJECT_ROOT}/retrieval/indexes/${MODEL}}"
CLIP_MODE="${CLIP_MODE:-multi_clip}"
NFRAMES="${NFRAMES:-16}"
TOP_K="${TOP_K:-5}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"
LORA_PATH="${LORA_PATH:-}"

case "${MODEL}" in
    qwen3vl-*)
        EXPECTED_ENV="foulret-qwen"
        ;;
    vlm2vec-v2|lamra)
        EXPECTED_ENV="foulret-vlm2vec"
        ;;
    internvideo2-*)
        EXPECTED_ENV="foulret-internvideo2"
        ;;
    *)
        echo "Unknown model: ${MODEL}" >&2
        exit 2
        ;;
esac

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="${PYTHON_BIN}"
elif [[ -n "${VENV_PATH:-}" ]]; then
    # Backward-compatible override for existing local installations.
    PYTHON="${VENV_PATH}/bin/python"
else
    PYTHON="$(command -v python || true)"
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "Python was not found." >&2
    echo "Activate the ${EXPECTED_ENV} Conda environment described in README.md, then retry." >&2
    exit 1
fi

if ! "${PYTHON}" -c "import flask" >/dev/null 2>&1; then
    echo "Flask is not installed in the active Python environment (${PYTHON})." >&2
    echo "Activate ${EXPECTED_ENV}, then run:" >&2
    echo "  ${PYTHON} -m pip install -r ${PROJECT_ROOT}/requirements/demo.txt" >&2
    exit 1
fi

if [[ ! -d "${DATA_DIR}/Train" ]]; then
    echo "Prepared Train data not found at ${DATA_DIR}/Train." >&2
    echo "Run python prepare_dataset.py as described in README.md." >&2
    exit 1
fi

if [[ "${MODEL}" == "qwen3vl-2b-finetuned" && -z "${LORA_PATH}" ]]; then
    echo "LORA_PATH is required for ${MODEL}." >&2
    exit 1
fi

EMBEDDINGS="${INDEX_DIR}/embeddings_train.json"
FAISS_INDEX="${INDEX_DIR}/faiss_index_train.bin"
if [[ ! -f "${EMBEDDINGS}" || ! -f "${FAISS_INDEX}" ]]; then
    echo "WARNING: The Train index required by the demo is missing." >&2
    echo "Expected files:" >&2
    echo "  ${EMBEDDINGS}" >&2
    echo "  ${FAISS_INDEX}" >&2
    echo "" >&2
    echo "Build the index first from the repository root:" >&2
    printf "  %q -m retrieval.build_full_index" "${PYTHON}" >&2
    printf " --model %q" "${MODEL}" >&2
    printf " --data_dir %q" "${DATA_DIR}" >&2
    printf " --index_dir %q" "${INDEX_DIR}" >&2
    printf " --clip_mode %q" "${CLIP_MODE}" >&2
    printf " --nframes %q" "${NFRAMES}" >&2
    if [[ -n "${LORA_PATH}" ]]; then
        printf " --lora_path %q" "${LORA_PATH}" >&2
    fi
    printf "\n" >&2
    exit 1
fi

app_args=(
    "${PROJECT_ROOT}/Demo/flask_app.py"
    --model "${MODEL}"
    --data_dir "${DATA_DIR}"
    --index_dir "${INDEX_DIR}"
    --clip_mode "${CLIP_MODE}"
    --nframes "${NFRAMES}"
    --top_k "${TOP_K}"
    --host "${HOST}"
    --port "${PORT}"
)
if [[ -n "${LORA_PATH}" ]]; then
    app_args+=(--lora_path "${LORA_PATH}")
fi

echo "Starting SoccerNet-FoulRet with ${MODEL}."
echo "Python: ${PYTHON}"
echo "Open http://localhost:${PORT}"
cd "${PROJECT_ROOT}"
exec "${PYTHON}" "${app_args[@]}"
