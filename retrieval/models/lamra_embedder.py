"""LamRA-Ret adapter for native-video retrieval.

Loads the merged Qwen2-VL-7B retrieval model and its processor from the pinned
`Clones/VLM2Vec` submodule. The local wrapper enables the model's video input
path without modifying the upstream submodule and uses the same sampling
options as the other vision-language encoders.
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

# Repo root -> Clones/VLM2Vec (read-only reference; LamRA lives under its baseline_backbone)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VLM2VEC_ROOT = os.path.join(_REPO_ROOT, "Clones", "VLM2Vec")

# Fixed instruction for ALL clips, byte-identical for query and corpus (matches VLM2Vec).
_INSTRUCTION = "Represent this football foul video clip for retrieving videos of the same type of foul."


def _make_video_lamra(base_cls):
    """Subclass LamRAQwen2VL re-enabling the (clone-commented) native-video forward branch.

    Faithful copy of the clone's forward() with the video block uncommented; pooling
    (left-padded last token = the <emb> position) and L2-norm are unchanged.
    """

    class _LamRAVideo(base_cls):
        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            pixel_values=None,
            image_grid_thw=None,
            pixel_values_videos=None,
            video_grid_thw=None,
            pooling_mask=None,
            **kwargs,
        ):
            if inputs_embeds is None:
                inputs_embeds = self.base.model.embed_tokens(input_ids)

                has_image = pixel_values is not None and (
                    any(pv is not None for pv in pixel_values)
                    if isinstance(pixel_values, list) else True
                )
                if has_image:
                    pv, thw = pixel_values, image_grid_thw
                    if isinstance(pv, list):
                        pv = torch.cat([torch.from_numpy(x) for x in pv]).to(input_ids.device)
                        thw = torch.cat([torch.from_numpy(x) for x in thw]).to(input_ids.device)
                    pv = pv.type(self.base.visual.get_dtype())
                    image_embeds = self.base.visual(pv, grid_thw=thw).to(inputs_embeds.device)
                    image_mask = input_ids == self.base.config.image_token_id
                    inputs_embeds[image_mask] = image_embeds

                # --- native-video branch (commented out in the clone; re-enabled here) ---
                if pixel_values_videos is not None:
                    pvv, vthw = pixel_values_videos, video_grid_thw
                    if isinstance(pvv, list):
                        pvv = torch.cat([torch.from_numpy(x) for x in pvv]).to(input_ids.device)
                        vthw = torch.cat([torch.from_numpy(x) for x in vthw]).to(input_ids.device)
                    pvv = pvv.type(self.base.visual.get_dtype())
                    video_embeds = self.base.visual(pvv, grid_thw=vthw).to(inputs_embeds.device)
                    video_mask = input_ids == self.base.config.video_token_id
                    inputs_embeds[video_mask] = video_embeds

                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

            outputs = self.base.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
            )

            pooling_mask = attention_mask if pooling_mask is None else pooling_mask
            left_padding = (pooling_mask[:, -1].sum() == pooling_mask.shape[0])
            if left_padding:
                embeddings = outputs.last_hidden_state[:, -1]
            else:
                seq_len = pooling_mask.sum(dim=1) - 1
                bs = outputs.last_hidden_state.shape[0]
                embeddings = outputs.last_hidden_state[
                    torch.arange(bs, device=outputs.last_hidden_state.device), seq_len
                ]
            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.contiguous()

    return _LamRAVideo


class LamRAEmbeddingModel(BaseEmbeddingModel):
    """LamRA-Ret (Qwen2-VL-7B) native-video embedder. Embedding-only; no reranker."""

    def __init__(
        self,
        model_name: str = "code-kunkun/LamRA-Ret",
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
            from src.model.baseline_backbone.lamra.lamra_inference import LamRAQwen2VL
            from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
            from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                f"Could not import LamRA / Qwen2-VL from {_VLM2VEC_ROOT}. Ensure the vlm2vec "
                f"foulret-vlm2vec environment dependencies are installed and Clones/VLM2Vec exists."
            ) from e

        self._process_vision_info = process_vision_info

        # LamRAQwen2VL hardcodes attn_implementation="flash_attention_2" (lamra_inference.py:32);
        # flash_attn is not installed and the other models run sdpa, so force sdpa on the backbone
        # load (FA2 == sdpa = exact attention; keeps all models consistent). Patch is idempotent.
        if not getattr(Qwen2VLForConditionalGeneration, "_sdpa_forced", False):
            _orig_from_pretrained = Qwen2VLForConditionalGeneration.from_pretrained

            def _from_pretrained_sdpa(*a, **kw):
                cfg = kw.get("config")
                if cfg is not None:
                    cfg._attn_implementation = "sdpa"
                    if hasattr(cfg, "vision_config"):
                        cfg.vision_config._attn_implementation = "sdpa"
                else:
                    kw.setdefault("attn_implementation", "sdpa")
                return _orig_from_pretrained(*a, **kw)

            Qwen2VLForConditionalGeneration.from_pretrained = staticmethod(_from_pretrained_sdpa)
            Qwen2VLForConditionalGeneration._sdpa_forced = True

        logger.info(f"Loading LamRA-Ret (native-video) from: {model_name}")
        VideoLamRA = _make_video_lamra(LamRAQwen2VL)
        self.model = VideoLamRA(model_name=model_name, device=self.device)
        self.model.eval()
        self.processor = self.model.processor
        logger.info(f"LamRA-Ret loaded on {self.device}")

    @property
    def embedding_dim(self) -> int:
        return self._dim if self._dim is not None else 3584  # Qwen2-VL-7B hidden size

    def embed_text(self, text: str, is_query: bool = True) -> np.ndarray:
        raise NotImplementedError("text embedding not used for video-to-video retrieval")

    @torch.no_grad()
    def _embed_clip(self, clip_path: str, is_query: bool = True) -> Union[np.ndarray, None]:
        """Embed one .mp4 clip -> (1, dim) float32 (L2-normalized by the model).

        Query-only instruction: queries carry the task instruction; corpus/target clips use
        LamRA's neutral EOL prompt (arXiv:2412.01720 -- task instructions are query-side only).

        Mirrors the clone's LamRAQwen2VL.embed() usage exactly (same <emb> assistant turn
        + apply_chat_template(add_generation_prompt=True) + last-token pooling), swapping
        the image content for a native-video clip.
        """
        video_item = {"type": "video", "video": clip_path, "max_pixels": self.max_pixels}
        if self.nframes is not None:                 # windowed regime: take EXACTLY N frames
            video_item["nframes"] = self.nframes
        else:
            video_item["fps"] = self.fps
            video_item["max_frames"] = self.max_frames
        messages = [
            {
                "role": "user",
                "content": [
                    video_item,
                    {"type": "text", "text": (self.instruction if is_query else "Summarize the video in one word.")},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=getattr(self.model, "max_length", None),
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        reps = self.model(**inputs)               # (1, dim), bf16, L2-normalized
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
