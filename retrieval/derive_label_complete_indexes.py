"""Derive label-complete evaluation indexes from a full corpus index.

The category-relevance evaluation (Table 4) searches only clips carrying the
labels required by each tier. This script reconstructs the stored vectors from
the full `IndexFlatIP` and builds the two label-complete subsets without
re-embedding videos.

Given a base index dir  retrieval/indexes/<model>/  writes:
    retrieval/indexes/<model>-action-offence/           (~2905)
    retrieval/indexes/<model>-action-offence-severity/  (~2561)

Each output dir mirrors the source layout (embeddings_train.json + faiss_index_train.bin)
so the retrieval runner reads them with no code change. CPU-only, seconds per model.

Run from the repository root with the model's Conda environment active:
    python -m retrieval.derive_label_complete_indexes \
        --index_dir retrieval/indexes/qwen3vl-2b
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import faiss


def nid(x):
    return int(os.path.basename(str(x).rstrip("/")).replace("action_", "").strip())


def load_train_labels(data_dir="data/mvfouls_window"):
    d = json.load(open(os.path.join(data_dir, "Train", "annotations.json")))
    A = d["Actions"] if isinstance(d, dict) and "Actions" in d else d
    out = {}
    for k, v in A.items():
        def f(name):
            s = str(v.get(name, "")).strip()
            return s if s else None
        out[nid(k)] = {"action": f("Action class"), "offence": f("Offence"), "severity": f("Severity")}
    return out


TIERS = {
    # tier -> predicate on a label dict; corpus clip kept iff predicate true
    "action_offence": lambda L: L["action"] is not None and L["offence"] is not None,
    "action_offence_severity": lambda L: (
        L["action"] is not None and L["offence"] is not None and L["severity"] is not None
    ),
}


def build_subset(base_dir, tier, labels, verify=True):
    src_emb = os.path.join(base_dir, "embeddings_train.json")
    src_bin = os.path.join(base_dir, "faiss_index_train.bin")
    meta = json.load(open(src_emb))
    index = faiss.read_index(src_bin)
    n, d = index.ntotal, index.d
    assert n == len(meta), f"metadata/index row mismatch: {len(meta)} vs {n}"

    vecs = index.reconstruct_n(0, n).astype("float32")   # lossless: raw stored vectors

    keep_meta, keep_rows = [], []
    missing = 0
    for j, rec in enumerate(meta):
        cid = nid(rec["path"])
        L = labels.get(cid)
        if L is None:
            missing += 1
            continue
        if TIERS[tier](L):
            keep_meta.append(rec)
            keep_rows.append(j)

    sub = vecs[keep_rows]
    out_index = faiss.IndexFlatIP(d)
    out_index.add(sub)

    out_dir = f"{base_dir.rstrip('/')}-{tier.replace('_', '-')}"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(keep_meta).to_json(os.path.join(out_dir, "embeddings_train.json"),
                                    orient="records", indent=2)
    faiss.write_index(out_index, os.path.join(out_dir, "faiss_index_train.bin"))

    if verify:
        # Reconstruction fidelity: a full index rebuilt from the reconstructed vectors
        # must return the identical nearest neighbours as the on-disk source index.
        full_rebuild = faiss.IndexFlatIP(d)
        full_rebuild.add(vecs)
        probe = vecs[:16].copy()
        D0, I0 = index.search(probe, 10)
        D1, I1 = full_rebuild.search(probe, 10)
        assert np.array_equal(I0, I1) and np.allclose(D0, D1, atol=1e-6), \
            "reconstruction is NOT lossless — aborting"

    print(f"  [{tier:8}] kept {out_index.ntotal:5}/{n}  (skipped: {n - out_index.ntotal}, "
          f"unindexed-ids: {missing})  -> {out_dir}")
    return out_index.ntotal


def main():
    ap = argparse.ArgumentParser(
        description="Derive label-complete evaluation indexes without re-embedding")
    ap.add_argument("--index_dir", required=True,
                    help="Base full-corpus index dir, e.g. retrieval/indexes/qwen3vl-2b")
    ap.add_argument("--data_dir", default="data/mvfouls_window",
                    help="Prepared dataset containing clips and annotations")
    ap.add_argument("--tiers", nargs="+",
                    default=["action_offence", "action_offence_severity"],
                    choices=list(TIERS))
    args = ap.parse_args()

    labels = load_train_labels(args.data_dir)
    print(f"[{args.index_dir}]")
    for tier in args.tiers:
        build_subset(args.index_dir, tier, labels)


if __name__ == "__main__":
    main()
