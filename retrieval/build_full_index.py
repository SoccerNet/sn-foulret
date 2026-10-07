"""Build a FAISS index for the Train gallery.

This prerequisite for `retrieval.retrieve` embeds every Train action that
contains at least one video clip. It does not require annotations.

Run from the repository root:
    python -m retrieval.build_full_index \
        --data_dir data/mvfouls_window \
        --index_dir retrieval/indexes/qwen3vl-2b \
        --clip_mode multi_clip
"""

import os
import argparse
from glob import glob

from retrieval.models import get_model, model_requires_lora
from retrieval.engine import RetrievalEngine


def gather_video_actions(data_dir: str, split: str):
    """Return entries for every action folder in `split` that contains clips.

    Each entry: {"type": "video", "path": "<split>/action_<N>", "labels": "",
                 "action_id": "<split>_action_<N>", "video_url": "", "description": ""}
    """
    base = os.path.join(data_dir, split)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Split directory not found: {base}")

    entries = []
    for action_folder in sorted(os.listdir(base)):
        action_dir = os.path.join(base, action_folder)
        if not os.path.isdir(action_dir):
            continue
        if not glob(os.path.join(action_dir, "clip_*.mp4")):
            continue
        entries.append({
            "type": "video",
            "path": f"{split}/{action_folder}",          # e.g. "Train/action_0"
            "labels": "",                                  # unused for video2video
            "action_id": f"{split}_{action_folder}",      # e.g. "Train_action_0"
            "video_url": "",
            "description": "",
        })
    return entries


def main():
    parser = argparse.ArgumentParser(description="Build a Train-only video-embedding FAISS index")
    parser.add_argument("--model", default="qwen3vl-2b",
                        help="Embedding model registry key (see retrieval/models/__init__.py)")
    parser.add_argument("--data_dir", default="data/mvfouls_window",
                        help="Root of the extracted 16-frame clips containing "
                             "Split/action_N/clip_*.mp4 (default: data/mvfouls_window)")
    parser.add_argument("--index_dir", default=None,
                        help="Where to write embeddings_train.json + faiss_index_train.bin "
                             "(default: retrieval/indexes/<model>)")
    parser.add_argument("--split", default="Train", help="Corpus split (default: Train)")
    parser.add_argument("--clip_mode", default="multi_clip",
                        choices=["single_clip", "multi_clip"])
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--nframes", type=int, default=None,
                        help="Sample EXACTLY this many frames per clip (windowed regime: 16). "
                             "Mapped to num_frames for InternVideo2.")
    parser.add_argument("--lora_path", default=None,
                        help="LoRA adapter directory for a fine-tuned Qwen model")
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    # Derive the index dir from the model key unless given explicitly, so different
    # models can never silently reuse another model's FAISS index.
    if args.index_dir is None:
        args.index_dir = f"retrieval/indexes/{args.model}"

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.index_dir)
    os.makedirs(out_dir, exist_ok=True)
    embedding_path = os.path.join(out_dir, "embeddings_train.json")
    index_path = os.path.join(out_dir, "faiss_index_train.bin")

    if os.path.exists(embedding_path) and os.path.exists(index_path) and not args.force_rebuild:
        print(f"Index already exists at {index_path}. Use --force_rebuild to rebuild.")
        return

    entries = gather_video_actions(data_dir, args.split)
    print(f"Found {len(entries)} {args.split} action folders with clips in {data_dir}")
    if not entries:
        raise SystemExit("No video actions found — check --data_dir.")

    model_kwargs = {"video_root": data_dir, "clip_mode": args.clip_mode}
    if args.fps is not None:
        model_kwargs["fps"] = args.fps
    if args.max_frames is not None:
        model_kwargs["max_frames"] = args.max_frames
    if args.nframes is not None:
        # InternVideo2 uses `num_frames`; the Qwen-family adapters use `nframes`.
        model_kwargs["num_frames" if args.model.startswith("internvideo2") else "nframes"] = args.nframes
    if args.lora_path is not None:
        model_kwargs["lora_path"] = os.path.abspath(args.lora_path)
    elif model_requires_lora(args.model):
        parser.error(f"--lora_path is required for model {args.model}")
    model = get_model(args.model, **model_kwargs)

    # project_root == data_dir so engine's relative-path join resolves directly
    # under video_root (clean branch in embed_video_from_path).
    engine = RetrievalEngine(
        model=model,
        project_root=data_dir,
        embedding_path=embedding_path,
        index_path=index_path,
    )

    ok = engine.build_index_from_data(entries)
    if not ok:
        raise SystemExit("Index build failed — no embeddings were generated.")
    print(f"\nDone. Wrote:\n  {embedding_path}\n  {index_path}")


if __name__ == "__main__":
    main()
