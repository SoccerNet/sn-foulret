"""Qwen3-VL-Embedding adapter for video-to-video retrieval.

Wraps the vendored Qwen3-VL backend behind `BaseEmbeddingModel` and reads
prepared MP4 clips from the configured video root. Install its dependencies
with `bash scripts/setup/qwen.sh`.
"""

import os
import sys
import logging
import numpy as np
import torch
from glob import glob
from typing import Union

# ---------------------------------------------------------------------------
# Monkey-patch: route CUDA int64 reduction ops through CPU.
# Workaround for "CUDA driver error: invalid argument" triggered by
# torch.prod / .max / .sum / repeat_interleave on int64 CUDA tensors
# when the NVIDIA driver (e.g. 570.x / CUDA 12.4) is older than the
# torch CUDA runtime (cu126 / cu128).
# ---------------------------------------------------------------------------
def _patch_torch_int64_ops():
    _INT_DTYPES = (torch.int64, torch.int32)

    def _make_safe_func(orig):
        def _safe(*args, **kwargs):
            t = args[0] if args else kwargs.get('input')
            if t is not None and hasattr(t, 'is_cuda') and t.is_cuda and t.dtype in _INT_DTYPES:
                args = (t.cpu(),) + args[1:]
                result = orig(*args, **kwargs)
                return result.to(t.device) if hasattr(result, 'to') else result
            return orig(*args, **kwargs)
        return _safe

    def _make_safe_method(orig):
        def _safe(self, *args, **kwargs):
            if self.is_cuda and self.dtype in _INT_DTYPES:
                result = orig(self.cpu(), *args, **kwargs)
                return result.to(self.device) if hasattr(result, 'to') else result
            return orig(self, *args, **kwargs)
        return _safe

    for name in ('prod', 'sum', 'max', 'min'):
        if hasattr(torch, name):
            setattr(torch, name, _make_safe_func(getattr(torch, name)))
        if hasattr(torch.Tensor, name):
            setattr(torch.Tensor, name, _make_safe_method(getattr(torch.Tensor, name)))

    _orig_repeat = torch.repeat_interleave
    def _safe_repeat(*args, **kwargs):
        tensors = [a for a in args if isinstance(a, torch.Tensor)]
        if tensors and tensors[0].is_cuda and tensors[0].dtype in _INT_DTYPES:
            args = tuple(a.cpu() if isinstance(a, torch.Tensor) else a for a in args)
            kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            result = _orig_repeat(*args, **kwargs)
            return result.to(tensors[0].device) if hasattr(result, 'to') else result
        return _orig_repeat(*args, **kwargs)
    torch.repeat_interleave = _safe_repeat

_patch_torch_int64_ops()

from retrieval.models.embedder_base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class Qwen3VLEmbeddingModel(BaseEmbeddingModel):
    """Qwen3-VL-Embedding model for text and video (.mp4) inputs."""

    EMBEDDING_DIM = 2048  # Qwen3-VL-Embedding-2B output dimension (text_config.hidden_size)

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str = None,
        fps: float = 1.0,
        max_frames: int = 16,
        # Aligned to the precedent benchmark's independent relevance axes. Relevant pairs
        # match on Action class (dominant) and Offence; contact/bodypart are ~determined
        # by action class (P(same|same action)~98%) and severity is a weak discriminator,
        # so naming them adds no independent signal and bodypart was never a VARS/XVARS
        # concept. Action + Offence are exactly the axes the VARS/XVARS pipeline filters
        # on. See the released data/foulret judgments.
        video_instruction: str = "Retrieve soccer foul videos of the same action type and offence decision as the query.",
        text_instruction: str = "Given a text description of a soccer foul, retrieve videos of the same action type and offence decision.",
        document_instruction: str = None,
        torch_dtype=None,
        attn_implementation: str = None,
        lora_path: str = None,
        video_root: str = None,
        clip_mode: str = "multi_clip",  # "single_clip" or "multi_clip"
        embedding_dim: int = None,  # output dim from registry (2B=2048, 8B=4096 MRL default); None -> EMBEDDING_DIM
        nframes: int = None,  # if set, sample EXACTLY this many frames (windowed regime: nframes=16)
    ):
        """
        Args:
            model_name: HuggingFace model identifier or local path.
            device: Device to use. Auto-detected if None.
            fps: Frame sampling rate for video files.
            max_frames: Maximum number of frames to sample from a video.
            video_instruction: Instruction for embedding video queries (video-to-video retrieval).
            text_instruction: Instruction for embedding text queries (text-to-video retrieval).
            document_instruction: Instruction for embedding documents (index building). If None,
                                  omitted so the model falls back to its default "Represent the user's input.".
            torch_dtype: Torch dtype (e.g., torch.bfloat16). None for default.
            attn_implementation: Attention implementation (e.g., "flash_attention_2").
            lora_path: Path to LoRA adapter weights (for fine-tuned models).
            video_root: Root directory containing video data (e.g., .../mvfouls with Split/action_N/ subdirs).
            clip_mode: How to handle multiple clips per action. "single_clip" uses first clip only, 
                      "multi_clip" averages across all clips (default).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Windowed regime: nframes set -> take EXACTLY nframes (high fps so the count caps at max_frames).
        if nframes is not None:
            fps = max(float(fps), float(nframes) * 4)
            max_frames = nframes
        self.fps = fps
        self.max_frames = max_frames
        self.nframes = nframes
        self.video_instruction = video_instruction
        self.text_instruction = text_instruction
        self.document_instruction = document_instruction
        self.video_root = video_root
        self.clip_mode = clip_mode
        self._embedding_dim = embedding_dim if embedding_dim else self.EMBEDDING_DIM

        # Import the vendored Qwen3-VL embedding model definition
        try:
            from retrieval.models.qwen3vl_backend import Qwen3VLEmbedder
        except ImportError as e:
            raise ImportError(
                "Could not import the vendored Qwen3VLEmbedder from "
                "retrieval.models.qwen3vl_backend. Ensure the project "
                "venv has torch, transformers (with qwen3_vl) and qwen-vl-utils "
                "installed (see requirements/qwen.txt)."
            ) from e

        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        # Use flash_attention_2 if flash-attn is installed, otherwise fall back to sdpa
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        else:
            try:
                import flash_attn  # noqa: F401
                kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                kwargs["attn_implementation"] = "sdpa"

        logger.info(f"Loading Qwen3-VL-Embedding model from: {model_name}")
        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_name,
            fps=self.fps,
            max_frames=self.max_frames,
            **kwargs,
        )
        
        # Load LoRA adapter if specified
        if lora_path is not None:
            logger.info(f"Loading LoRA adapter from: {lora_path}")
            try:
                from peft import PeftModel
                self.embedder.model = PeftModel.from_pretrained(
                    self.embedder.model,
                    lora_path,
                )
                logger.info("LoRA adapter loaded successfully")
            except ImportError:
                raise ImportError(
                    "peft is required for loading LoRA adapters. "
                    "Install with: pip install peft"
                )
        
        logger.info(f"Qwen3-VL-Embedding model loaded on {self.device}")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @torch.no_grad()
    def embed_text(self, text: str, is_query: bool = True) -> np.ndarray:
        """Generates a normalized embedding for a text string."""
        instruction = self.text_instruction if is_query else self.document_instruction
        input_dict = {"text": text}
        if instruction is not None:
            input_dict["instruction"] = instruction
        inputs = [input_dict]
        embeddings = self.embedder.process(inputs, normalize=True)
        # Convert to float32 for FAISS compatibility
        if embeddings.dtype == torch.bfloat16:
            embeddings = embeddings.float()
        result = embeddings.cpu().numpy().astype("float32")
        torch.cuda.empty_cache()
        return result

    @torch.no_grad()
    def embed_video_from_path(self, video_path: str, is_query: bool = True) -> Union[np.ndarray, None]:
        """
        Generates a normalized embedding for a video from its action directory.

        Expects video_path to be a relative path like "Test/action_0".
        This method looks for the actual .mp4 files in the video_root directory:
        {video_root}/{split}/{action}/clip_*.mp4

        Args:
            video_path: Relative path to the action directory (e.g., "Test/action_0").

        Returns:
            Normalized embedding vector (2048-dim) or None if no clips found.
        """
        # Remap to the video root directory.
        # video_path may be absolute or relative like Test/action_0
        if os.path.isabs(video_path):
            # If already under video_root, use directly (avoid double-remap)
            norm_root = os.path.normpath(self.video_root)
            norm_path = os.path.normpath(video_path)
            if norm_path.startswith(norm_root + os.sep):
                action_dir = norm_path
            elif os.path.isdir(video_path) and glob(os.path.join(video_path, "clip_*.mp4")):
                # Path exists on disk and contains clip_*.mp4 (e.g., user upload)
                action_dir = video_path
            else:
                # Extract last two components (Split/action_N) to remap
                parts = video_path.rstrip("/").split("/")
                rel = "/".join(parts[-2:])
                action_dir = os.path.join(self.video_root, rel)
        else:
            # Strip 'data/' prefix if present
            rel = video_path
            if rel.startswith('data/'):
                rel = rel[5:]
            action_dir = os.path.join(self.video_root, rel)
        
        clip_paths = sorted(glob(os.path.join(action_dir, "clip_*.mp4")))
        if not clip_paths:
            logger.warning(f"No 'clip_*.mp4' files found in {action_dir}")
            return None

        # Handle single-clip mode: use only the first clip
        if self.clip_mode == "single_clip":
            clip_paths = [clip_paths[0]]
            logger.info(f"Single-clip mode: using first clip {clip_paths[0]}")
        elif self.clip_mode == "multi_clip":
            logger.info(f"Multi-clip mode: processing {len(clip_paths)} clips")
        else:
            raise ValueError(f"Invalid clip_mode: {self.clip_mode}. Must be 'single_clip' or 'multi_clip'")

        clip_embeddings = []
        for clip_path in clip_paths:
            try:
                # Qwen3-VL expects direct video file paths
                instruction = self.video_instruction if is_query else self.document_instruction
                input_dict = {"video": clip_path}
                if instruction is not None:
                    input_dict["instruction"] = instruction
                inputs = [input_dict]
                embeddings = self.embedder.process(inputs, normalize=True)
                # Convert to float32 for FAISS compatibility
                if embeddings.dtype == torch.bfloat16:
                    embeddings = embeddings.float()
                clip_embeddings.append(embeddings.cpu())
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to embed clip at {clip_path}: {e}")
                torch.cuda.empty_cache()

        if not clip_embeddings:
            logger.warning(f"Could not generate any clip embeddings for {video_path}")
            return None

        # For single-clip mode, return directly; for multi-clip, average across clips
        if self.clip_mode == "single_clip":
            result_emb = clip_embeddings[0]
        else:  # multi_clip
            # Average across clips and re-normalize
            avg_emb = torch.mean(torch.cat(clip_embeddings, dim=0), dim=0, keepdim=True)
            avg_emb = avg_emb / avg_emb.norm(dim=-1, keepdim=True)
            result_emb = avg_emb
        
        # Convert to float32 for FAISS compatibility
        if result_emb.dtype == torch.bfloat16:
            result_emb = result_emb.float()
        return result_emb.cpu().numpy().astype("float32")
