"""Embedding-model registry for the retrieval models reported in the paper."""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "vlm2vec-v2": {
        "module": "retrieval.models.vlm2vec_v2_embedder",
        "class_name": "VLM2VecV2EmbeddingModel",
        "hf_name": "VLM2Vec/VLM2Vec-V2.0",
        "dim": 1536,  # Qwen2-VL-2B hidden size (verified at runtime; informational here)
        "data_subdir": "mvfouls",
    },
    "qwen3vl-2b": {
        "module": "retrieval.models.qwen3vl_embedder",
        "class_name": "Qwen3VLEmbeddingModel",
        "hf_name": "Qwen/Qwen3-VL-Embedding-2B",
        "dim": 2048,
        "data_subdir": "mvfouls",  # reads .mp4 files
    },
    "qwen3vl-8b": {
        "module": "retrieval.models.qwen3vl_embedder",
        "class_name": "Qwen3VLEmbeddingModel",
        "hf_name": "Qwen/Qwen3-VL-Embedding-8B",
        "dim": 4096,  # MRL default 4096 (verified at runtime); reuses the Qwen .venv stack
        "data_subdir": "mvfouls",
    },
    "qwen3vl-2b-finetuned": {
        "module": "retrieval.models.qwen3vl_embedder",
        "class_name": "Qwen3VLEmbeddingModel",
        "hf_name": "Qwen/Qwen3-VL-Embedding-2B",
        "dim": 2048,
        "data_subdir": "mvfouls",
        "lora_adapter": True,  # Requires lora_path kwarg
    },
    "lamra": {
        "module": "retrieval.models.lamra_embedder",
        "class_name": "LamRAEmbeddingModel",
        "hf_name": "code-kunkun/LamRA-Ret",
        "dim": 3584,  # Qwen2-VL-7B hidden size (verified at runtime; native-video, runs in .venv_vlm2vec)
        "data_subdir": "mvfouls",
    },
    "internvideo2-1b": {
        "module": "retrieval.models.internvideo2_embedder",
        "class_name": "InternVideo2EmbeddingModel",
        "hf_name": "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
        "dim": 512,  # vision_proj -> embed_dim (true-video ViT, native 4 frames@224, runs in .venv_internvideo2)
        "data_subdir": "mvfouls",
        "init_kwargs": {"model_size": "1b", "ckpt_filename": "InternVideo2-stage2_1b-224p-f4.pt"},
    },
    "internvideo2-6b": {
        "module": "retrieval.models.internvideo2_embedder",
        "class_name": "InternVideo2EmbeddingModel",
        "hf_name": "OpenGVLab/InternVideo2-Stage2_6B-224p-f4",  # separate gated repo (confirmed via list_repo_files)
        "dim": 768,  # 6B scales the retrieval projection to 768 (vs 1B's 512); confirmed from ckpt vision_proj
        "data_subdir": "mvfouls",
        "init_kwargs": {"model_size": "6b", "ckpt_filename": "internvideo2-s2_6b-224p-f4_with_audio_encoder.pt"},
    },
}

def model_requires_lora(model_key: str) -> bool:
    """Return whether a model key needs a LoRA adapter."""
    return bool(MODEL_REGISTRY.get(model_key, {}).get("lora_adapter"))


def get_model(model_key: str, **kwargs):
    """
    Instantiate an embedding model by its registry key.

    Args:
        model_key: One of the keys in MODEL_REGISTRY
            (e.g., "qwen3vl-2b", "qwen3vl-2b-finetuned").
        **kwargs: Extra keyword arguments forwarded to the model constructor.
            For "qwen3vl-2b-finetuned", pass lora_path=<path_to_adapter>.

    Returns:
        An instance of BaseEmbeddingModel.
    """
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model key '{model_key}'. Available: {available}")

    entry = MODEL_REGISTRY[model_key]

    # Lazy import to avoid pulling in unnecessary deps
    import importlib
    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class_name"])

    # Set default model_name from registry if not overridden
    if "model_name" not in kwargs:
        kwargs["model_name"] = entry["hf_name"]

    # Forward the registry's declared embedding dim (the Qwen3-VL adapter uses it to set the
    # output dim per size; the VLM2Vec/LamRA adapters derive dim at runtime and ignore it via **kwargs).
    if "embedding_dim" not in kwargs and "dim" in entry:
        kwargs["embedding_dim"] = entry["dim"]

    # Forward any extra per-entry constructor kwargs (e.g. InternVideo2 model_size / ckpt_filename).
    for _k, _v in entry.get("init_kwargs", {}).items():
        kwargs.setdefault(_k, _v)

    logger.info(f"Instantiating embedding model '{model_key}' ({entry['class_name']})")
    return cls(**kwargs)
