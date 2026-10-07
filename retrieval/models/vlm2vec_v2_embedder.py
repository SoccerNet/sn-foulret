"""VLM2Vec-V2 adapter for video-to-video retrieval.

Loads the Qwen2-VL-2B retrieval model from the pinned `Clones/VLM2Vec`
submodule, uses last-token pooling with L2 normalization, and applies the same
video sampling budget as the Qwen3-VL models. The upstream submodule is imported
without modification.
"""

import os
import sys
import logging
import numpy as np
import torch
from glob import glob
from typing import Union

from retrieval.models.embedder_base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

# Repo root -> Clones/VLM2Vec (read-only reference; added to sys.path so `import src...` resolves)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VLM2VEC_ROOT = os.path.join(_REPO_ROOT, "Clones", "VLM2Vec")

# One fixed instruction for ALL clips, byte-identical for query and corpus (per task spec).
_INSTRUCTION = "Represent this football foul video clip for retrieving videos of the same type of foul."


class VLM2VecV2EmbeddingModel(BaseEmbeddingModel):
    """VLM2Vec-V2 (Qwen2-VL-2B) video embedder. Embedding-only; no reranker."""

    def __init__(
        self,
        model_name: str = "VLM2Vec/VLM2Vec-V2.0",
        device: str = None,
        fps: float = 1.0,
        max_frames: int = 16,
        max_pixels: int = 360 * 420,
        instruction: str = _INSTRUCTION,
        video_root: str = None,
        clip_mode: str = "multi_clip",
        nframes: int = None,  # if set, sample EXACTLY this many frames (windowed regime: nframes=16)
        **kwargs,  # tolerate unused factory kwargs
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fps = fps
        self.max_frames = max_frames
        self.max_pixels = max_pixels
        self.nframes = nframes
        self.instruction = instruction
        self.video_root = video_root
        self.clip_mode = clip_mode
        self._dim = None  # filled from the first real forward pass

        if _VLM2VEC_ROOT not in sys.path:
            sys.path.insert(0, _VLM2VEC_ROOT)
        try:
            from src.arguments import ModelArguments, DataArguments
            from src.model.model import MMEBModel
            from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS, backbone2model
            from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                f"Could not import the VLM2Vec clone from {_VLM2VEC_ROOT}. Ensure the "
                f"foulret-vlm2vec environment has its requirements installed (requirements/vlm2vec.txt) "
                f"and Clones/VLM2Vec exists."
            ) from e

        self._process_vision_info = process_vision_info
        self._video_token = VLM_VIDEO_TOKENS[QWEN2_VL]

        model_args = ModelArguments(
            model_name=model_name,
            pooling="last",        # VLM2Vec-V2 uses last-token pooling
            normalize=True,        # output already L2-normalized by the model
            model_backbone="qwen2_vl",
            lora=True,             # V2.0 ships as a LoRA adapter over Qwen2-VL-2B
        )
        data_args = DataArguments()

        # VLM2Vec hardcodes FlashAttention 2 for this backbone. The repository
        # uses SDPA for both VLM2Vec and Qwen3-VL so their attention kernels are
        # consistent without requiring flash-attn in this environment.
        _backbone_cls = backbone2model[QWEN2_VL]
        if not getattr(_backbone_cls, "_sdpa_forced", False):
            _orig_from_pretrained = _backbone_cls.from_pretrained

            def _from_pretrained_sdpa(*a, **kw):
                cfg = kw.get("config")
                if cfg is not None:
                    cfg._attn_implementation = "sdpa"
                    if hasattr(cfg, "vision_config"):
                        cfg.vision_config._attn_implementation = "sdpa"
                else:
                    kw.setdefault("attn_implementation", "sdpa")
                return _orig_from_pretrained(*a, **kw)

            _backbone_cls.from_pretrained = staticmethod(_from_pretrained_sdpa)
            _backbone_cls._sdpa_forced = True

        logger.info(f"Loading VLM2Vec-V2 processor + model from: {model_name}")
        self.processor = load_processor(model_args, data_args)
        self.model = MMEBModel.load(model_args, is_trainable=False)
        self.model = self.model.to(self.device, dtype=torch.bfloat16)
        self.model.eval()
        logger.info(f"VLM2Vec-V2 loaded on {self.device}")

    @property
    def embedding_dim(self) -> int:
        return self._dim if self._dim is not None else 1536  # Qwen2-VL-2B hidden size

    def embed_text(self, text: str, is_query: bool = True) -> np.ndarray:
        raise NotImplementedError("text embedding not used for video-to-video retrieval")

    @torch.no_grad()
    def _embed_clip(self, clip_path: str, is_query: bool = True) -> Union[np.ndarray, None]:
        """Embed one .mp4 clip -> (1, dim) float32 (already L2-normalized by the model).

        Query-only instruction (VLM2Vec convention, arXiv:2410.05160): the task instruction
        is prepended for queries; corpus/target clips are embedded as bare content."""
        video_item = {"type": "video", "video": clip_path, "max_pixels": self.max_pixels}
        if self.nframes is not None:                 # windowed regime: take EXACTLY N frames
            video_item["nframes"] = self.nframes
        else:
            video_item["fps"] = self.fps
            video_item["max_frames"] = self.max_frames
        messages = [{"role": "user", "content": [video_item]}]
        _, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=(f"{self._video_token} {self.instruction}" if is_query else self._video_token),
            videos=video_inputs,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # demo_v2_video.py adds the batch dim for video tensors:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].unsqueeze(0)
        inputs["video_grid_thw"] = inputs["video_grid_thw"].unsqueeze(0)
        reps = self.model(qry=inputs)["qry_reps"]          # (1, dim), bf16, L2-normalized
        if reps.dtype == torch.bfloat16:
            reps = reps.float()
        out = reps.cpu().numpy().astype("float32")
        if self._dim is None:
            self._dim = out.shape[1]
        torch.cuda.empty_cache()
        return out

    @torch.no_grad()
    def embed_video_from_path(self, video_path: str, is_query: bool = True) -> Union[np.ndarray, None]:
        """Glob clip_*.mp4 in the action dir, embed each, pool per clip_mode, L2-normalize."""
        action_dir = video_path if os.path.isabs(video_path) else os.path.join(self.video_root or "", video_path)
        clip_paths = sorted(glob(os.path.join(action_dir, "clip_*.mp4")))
        if not clip_paths:
            logger.warning(f"No 'clip_*.mp4' found in {action_dir}")
            return None
        if self.clip_mode == "single_clip":
            clip_paths = clip_paths[:1]
        elif self.clip_mode != "multi_clip":
            raise ValueError(f"Invalid clip_mode: {self.clip_mode}")

        vecs = []
        for cp in clip_paths:
            try:
                v = self._embed_clip(cp, is_query=is_query)
                if v is not None:
                    vecs.append(v)
            except Exception as e:
                logger.warning(f"Failed to embed clip {cp}: {e}")
                torch.cuda.empty_cache()
        if not vecs:
            logger.warning(f"Could not embed any clip for {video_path}")
            return None

        emb = np.vstack(vecs).mean(axis=0, keepdims=True)          # average across clips
        n = np.linalg.norm(emb, axis=1, keepdims=True); n[n == 0] = 1.0
        return (emb / n).astype("float32")                         # (1, dim), unit norm
