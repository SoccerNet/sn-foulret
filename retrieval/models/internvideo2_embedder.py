"""InternVideo2-Stage2 adapter for video-to-video retrieval.

Loads the vision encoder and contrastive projection from the pinned
`Clones/InternVideo` submodule and returns normalized video embeddings. The
official checkpoints use four-frame positional embeddings; other configured
frame counts are supported through positional interpolation. Compatibility
shims keep the inference path vision-only without modifying the upstream code.
"""

import os
import sys
import types
import logging
import numpy as np
import torch
from glob import glob
from typing import Union

from retrieval.models.embedder_base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MM_ROOT = os.path.join(_REPO_ROOT, "Clones", "InternVideo", "InternVideo2", "multi_modality")


def _apply_iv2_import_stubs():
    """Inject the shims/stubs needed to import ONLY the stage2 vision path from the read-only clone."""
    # (a) fused-op shim: core flash-attn wheel lacks DropoutAddRMSNorm (FusedMLP is present).
    #     Unused at runtime (use_fused_rmsnorm=False) but the CLIP-vision siblings import it.
    if "flash_attn.ops.rms_norm" not in sys.modules:
        try:
            import flash_attn.ops.rms_norm as _r  # noqa: F401
            if not hasattr(_r, "DropoutAddRMSNorm"):
                raise ImportError
        except Exception:
            m = types.ModuleType("flash_attn.ops.rms_norm")
            m.DropoutAddRMSNorm = type("DropoutAddRMSNorm", (object,), {})
            sys.modules["flash_attn.ops.rms_norm"] = m

    # (b) stub the 4 unused top-level model variants models/__init__.py eagerly imports.
    for name, cls in [
        ("models.internvideo2_clip", "InternVideo2_CLIP"),
        ("models.internvideo2_clip_small", "InternVideo2_CLIP_small"),
        ("models.internvideo2_stage2_visual", "InternVideo2_Stage2_visual"),
        ("models.internvideo2_stage2_audiovisual", "InternVideo2_Stage2_audiovisual"),
    ]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            setattr(m, cls, type(cls, (object,), {}))
            sys.modules[name] = m

    # (c) models/criterions.py uses a broken `from ..utils...` relative import; get_sim unused here.
    if "models.criterions" not in sys.modules:
        m = types.ModuleType("models.criterions")
        m.get_sim = lambda *a, **k: None
        sys.modules["models.criterions"] = m

    # (d) bypass the unused BERT text stack (video-to-video only): no bert download, no xbert.
    import torch.nn as nn
    if "models.backbones.bert.builder" not in sys.modules:
        b = types.ModuleType("models.backbones.bert.builder")
        b.build_bert = lambda *a, **k: nn.Module()        # dummy text encoder (never called)
        b.build_bert_decoder = lambda *a, **k: nn.Module()
        sys.modules["models.backbones.bert.builder"] = b
    if "models.backbones.bert.tokenization_bert" not in sys.modules:
        t = types.ModuleType("models.backbones.bert.tokenization_bert")

        class _BertTokStub:
            @classmethod
            def from_pretrained(cls, *a, **k):
                return cls()
        t.BertTokenizer = _BertTokStub
        sys.modules["models.backbones.bert.tokenization_bert"] = t


class InternVideo2EmbeddingModel(BaseEmbeddingModel):
    """InternVideo2-Stage2 1B video embedder (512-d). Embedding-only; no reranker."""

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
        ckpt_filename: str = "InternVideo2-stage2_1b-224p-f4.pt",
        model_size: str = "1b",          # "1b" or "6b" (same ~512-d retrieval space; 6b = bigger encoder d_model 3200)
        device: str = None,
        num_frames: int = 4,            # NATIVE = 4 (ckpt is -f4); >4 triggers pos-embed interpolation
        size_t: int = 224,
        video_root: str = None,
        clip_mode: str = "multi_clip",  # "single_clip" or "multi_clip"
        embedding_dim: int = 512,
        **kwargs,  # tolerate unused factory kwargs
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.size_t = size_t
        self.video_root = video_root
        self.clip_mode = clip_mode
        self._dim = embedding_dim

        _apply_iv2_import_stubs()
        if _MM_ROOT not in sys.path:
            sys.path.insert(0, _MM_ROOT)
        try:
            from utils.config import Config, eval_dict_leaf
            from demo.utils import setup_internvideo2, frames2tensor
        except ImportError as e:
            raise ImportError(
                f"Could not import InternVideo2 from {_MM_ROOT}. Ensure .venv_internvideo2 deps "
                f"(core flash-attn, transformers==4.46.3, peft==0.13.2, easydict, open_clip_torch...) "
                f"are installed and the clone exists."
            ) from e
        self._frames2tensor = frames2tensor

        # demo.utils.InternVideo2_Stage2.build_vision_encoder hardcodes the 1B encoder and raises
        # for 6B; monkeypatch it to dispatch 1b/6b (read-only clone untouched). Idempotent; the
        # 1B path is byte-identical to the original. Replicates the original post-build attr setup.
        import demo.utils as _iv2_demo
        from models.backbones.internvideo2 import (
            pretrain_internvideo2_1b_patch14_224 as _iv2_1b,
            pretrain_internvideo2_6b_patch14_224 as _iv2_6b,
        )
        if not getattr(_iv2_demo.InternVideo2_Stage2, "_size_dispatch", False):
            _enc = {
                "pretrain_internvideo2_1b_patch14_224": _iv2_1b,
                "pretrain_internvideo2_6b_patch14_224": _iv2_6b,
            }

            def _build_vision_encoder(self):
                ve = self.config.model.vision_encoder
                if ve.name not in _enc:
                    raise ValueError(f"Unsupported vision_encoder: {ve.name}")
                vision_encoder = _enc[ve.name](self.config.model)
                img_size, num_frames = ve.img_size, ve.num_frames
                tubelet_size, patch_size = ve.tubelet_size, ve.patch_size
                self.clip_img_size = ve.clip_input_resolution
                self.video_mask_type = ve.video_mask_type
                self.video_window_size = (num_frames // tubelet_size, img_size // patch_size, img_size // patch_size)
                self.video_mask_ratio = ve.video_mask_ratio
                self.image_mask_type = ve.image_mask_type
                self.image_window_size = (1, img_size // patch_size, img_size // patch_size)
                self.image_mask_ratio = ve.image_mask_ratio
                return vision_encoder

            _iv2_demo.InternVideo2_Stage2.build_vision_encoder = _build_vision_encoder
            _iv2_demo.InternVideo2_Stage2._size_dispatch = True

        # Resolve the demo config (${...} interpolation) and point it at the downloaded checkpoint.
        cfg = Config.from_file(os.path.join(_MM_ROOT, "demo", "internvideo2_stage2_config.py"))
        cfg = eval_dict_leaf(cfg)
        from huggingface_hub import hf_hub_download
        logger.info(f"Fetching InternVideo2 checkpoint {model_name}/{ckpt_filename} ...")
        ckpt_path = hf_hub_download(repo_id=model_name, filename=ckpt_filename)

        cfg.pretrained_path = ckpt_path
        cfg.device = self.device
        cfg.num_frames = self.num_frames
        cfg.num_frames_test = self.num_frames
        cfg.model.vision_encoder.num_frames = self.num_frames
        cfg.origin_num_frames = 4                       # ckpt native (4) -> interpolate pos-embed when num_frames != 4
        cfg.model.embed_dim = self._dim                 # retrieval projection dim: 1B=512, 6B=768 (must match ckpt vision_proj)
        if model_size == "6b":                          # bigger encoder; clip_embed_dim stays 768 -> same ~512-d output
            cfg.model.vision_encoder.name = "pretrain_internvideo2_6b_patch14_224"
            cfg.model.vision_encoder.checkpoint_num = 48
        # MUST be None (not ""): the encoder fn does `if pretrained is not None: torch.load(pretrained)`,
        # so "" would crash. Full stage2 weights are loaded by setup_internvideo2 via cfg.pretrained_path.
        cfg.model.vision_encoder.pretrained = None
        cfg.use_half_precision = False                  # keep naive attention (no flash/fused ops on this venv)
        cfg.use_bf16 = True                             # bf16 weights (memory; consistent with other models)

        logger.info(f"Building InternVideo2-Stage2 1B (num_frames={self.num_frames}) ...")
        self.model, self.tokenizer = setup_internvideo2(cfg)
        self.model.eval()
        logger.info(f"InternVideo2 loaded on {self.device}")

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed_text(self, text: str, is_query: bool = True) -> np.ndarray:
        raise NotImplementedError("InternVideo2 is video-to-video only (text encoder stubbed out)")

    @torch.no_grad()
    def _embed_clip(self, clip_path: str) -> Union[np.ndarray, None]:
        """Embed one .mp4 clip -> (1, 512) float32, L2-normalized. Frames read BGR via cv2 to
        match the demo's frames2tensor preprocessing (which does x[:, :, ::-1] BGR->RGB)."""
        import cv2
        cap = cv2.VideoCapture(clip_path)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            logger.warning(f"No frames decoded from {clip_path}")
            return None
        while len(frames) < self.num_frames:          # frames2tensor asserts len >= fnum
            frames.append(frames[-1])

        tensor = self._frames2tensor(
            frames, fnum=self.num_frames, target_size=(self.size_t, self.size_t), device=self.device
        )
        vfeat = self.model.get_vid_feat(tensor)        # (1, 512), L2-normalized by the model
        out = vfeat.float().cpu().numpy().astype("float32")
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
                v = self._embed_clip(cp)
                if v is not None:
                    vecs.append(v)
            except Exception as e:
                logger.warning(f"Failed to embed clip {cp}: {e}")
                torch.cuda.empty_cache()
        if not vecs:
            logger.warning(f"Could not embed any clip for {video_path}")
            return None

        emb = np.vstack(vecs).mean(axis=0, keepdims=True)
        n = np.linalg.norm(emb, axis=1, keepdims=True); n[n == 0] = 1.0
        return (emb / n).astype("float32")
