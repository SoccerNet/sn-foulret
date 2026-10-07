"""Run video-embedding retrieval for the benchmark gallery views.

Each query is embedded once and searched against three gallery views that share
the same stored vectors:

  * full                     -- full 2916-clip gallery (precedent relevance)
  * action_offence          -- action+offence labelled 2905 (category relevance)
  * action_offence_severity -- action+offence+severity labelled 2561 (category relevance)

Because the corpus index is exact (`IndexFlatIP`), searching a label-restricted
subset is equivalent to removing unlabelled clips from the full ranking. The
runner saves the top 100 results per view and caches query embeddings for later
re-scoring.

Normal three-view retrieval from the repository root:
    python -m retrieval.retrieve --model qwen3vl-2b \
        --index_base retrieval/indexes/qwen3vl-2b \
        --data_dir data/mvfouls_window --nframes 16 \
        --instruction "Retrieve videos similar to the query video." \
        --out_dir retrieval/rankings/qwen3vl-2b

Instruction-ablation retrieval over only the full gallery:
    python -m retrieval.retrieve --model qwen3vl-2b \
        --index_base retrieval/indexes/qwen3vl-2b \
        --data_dir data/mvfouls_window --nframes 16 \
        --views full --top_k 20 --no_query_cache --plain_filenames \
        --out_dir retrieval/rankings/instruction_ablation/qwen3vl-2b/generic
"""

import os
import json
import argparse
from glob import glob

import numpy as np
from tqdm import tqdm

from retrieval.models import get_model, model_requires_lora
from retrieval.engine import RetrievalEngine

VIEWS = ["full", "action_offence", "action_offence_severity"]


def path_to_id(path: str) -> str:
    return os.path.basename(str(path).rstrip("/")).replace("action_", "")


def list_query_ids(data_dir: str, split: str):
    base = os.path.join(data_dir, split)
    ids = []
    for folder in os.listdir(base):
        if folder.startswith("action_") and glob(os.path.join(base, folder, "clip_*.mp4")):
            ids.append(folder.replace("action_", ""))
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def main():
    ap = argparse.ArgumentParser(description="Video retrieval over one or more gallery views")
    ap.add_argument("--model", default="qwen3vl-2b")
    ap.add_argument("--index_base", required=True,
                    help="Full-corpus index dir; -action-offence and "
                         "-action-offence-severity siblings are label-complete indexes")
    ap.add_argument("--data_dir", default="data/mvfouls_window")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["Test", "Valid"])
    ap.add_argument("--views", nargs="+", choices=VIEWS, default=VIEWS,
                    help="Gallery views to search (default: all three paper views)")
    ap.add_argument("--top_k", type=int, default=100, help="Ranking depth saved per view")
    ap.add_argument("--clip_mode", default="multi_clip", choices=["single_clip", "multi_clip"])
    ap.add_argument("--nframes", type=int, default=None)
    ap.add_argument("--instruction", default=None,
                    help="Query-side task instruction; ignored for InternVideo2 (no instruction).")
    ap.add_argument("--lora_path", default=None,
                    help="LoRA adapter directory for a fine-tuned Qwen model")
    ap.add_argument("--no_query_cache", action="store_true",
                    help="Do not save query embeddings (used by instruction-ablation runs)")
    ap.add_argument("--plain_filenames", action="store_true",
                    help="Write <split>.json instead of <split>_<view>.json; requires one view")
    ap.add_argument("--limit", type=int, default=None, help="First N queries per split (smoke/gate)")
    args = ap.parse_args()

    # Preserve the requested order while ignoring accidental duplicate view names.
    selected_views = list(dict.fromkeys(args.views))
    if args.plain_filenames and len(selected_views) != 1:
        ap.error("--plain_filenames requires exactly one --views value")

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Resolve the requested view index dirs and confirm they exist.
    base = args.index_base.rstrip("/")
    view_dirs = {
        "full": base,
        "action_offence": f"{base}-action-offence",
        "action_offence_severity": f"{base}-action-offence-severity",
    }
    for v in selected_views:
        d = view_dirs[v]
        for f in ("embeddings_train.json", "faiss_index_train.bin"):
            p = os.path.join(d, f)
            if not os.path.exists(p):
                command = ("retrieval.build_full_index" if v == "full"
                           else "retrieval.derive_label_complete_indexes")
                raise SystemExit(f"[{v}] missing {p} — create it with {command}.")

    # Model (GPU) built once; used only to embed queries.
    model_kwargs = {"video_root": data_dir, "clip_mode": args.clip_mode}
    if args.nframes is not None:
        model_kwargs["num_frames" if args.model.startswith("internvideo2") else "nframes"] = args.nframes
    if args.instruction is not None and not args.model.startswith("internvideo2"):
        model_kwargs["video_instruction" if args.model.startswith("qwen3vl") else "instruction"] = args.instruction
    if args.lora_path is not None:
        model_kwargs["lora_path"] = os.path.abspath(args.lora_path)
    elif model_requires_lora(args.model):
        ap.error(f"--lora_path is required for model {args.model}")
    model = get_model(args.model, **model_kwargs)

    # One engine per selected view. Only the first holds the model used to embed queries;
    # any remaining engines are search-only over their own indexes.
    engines = {}
    embedding_view = selected_views[0]
    for v in selected_views:
        engines[v] = RetrievalEngine(
            model=model if v == embedding_view else None,
            project_root=data_dir,
            embedding_path=os.path.join(view_dirs[v], "embeddings_train.json"),
            index_path=os.path.join(view_dirs[v], "faiss_index_train.bin"),
        )

    for split in args.splits:
        query_ids = list_query_ids(data_dir, split)
        if args.limit is not None:
            query_ids = query_ids[:args.limit]
        preds = {v: {} for v in selected_views}
        cache_ids, cache_vecs = [], []
        print(f"[{split}] {len(query_ids)} queries")

        for qid in tqdm(query_ids, desc=f"{split} retrieval"):
            key = f"action_{qid}"
            for v in selected_views:
                preds[v][key] = []
            q = engines[embedding_view].embed_video_from_path(
                f"{split}/action_{qid}"
            )  # is_query=True
            if q is None:
                continue
            qv = np.asarray(q, dtype="float32").reshape(-1)
            if not args.no_query_cache:
                cache_ids.append(int(qid) if str(qid).isdigit() else qid)
                nrm = np.linalg.norm(qv)
                # Unit norm makes later re-search equivalent to inner product.
                cache_vecs.append(qv / nrm if nrm > 0 else qv)
            for v in selected_views:
                results = engines[v].search(q.copy(), top_k=args.top_k)
                preds[v][key] = [path_to_id(c["path"]) for c in results]

        split_l = split.lower()
        for v in selected_views:
            filename = f"{split_l}.json" if args.plain_filenames else f"{split_l}_{v}.json"
            p = os.path.join(out_dir, filename)
            with open(p, "w") as f:
                json.dump(preds[v], f, indent=2)
            print(f"  wrote {p}")
        if cache_vecs and not args.no_query_cache:
            cp = os.path.join(out_dir, f"query_emb_{split_l}.npz")
            np.savez(cp, ids=np.array(cache_ids), embs=np.vstack(cache_vecs))
            print(f"  cached {len(cache_vecs)} query embeddings -> {cp}")


if __name__ == "__main__":
    main()
