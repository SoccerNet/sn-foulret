"""Generate the VARS+XVARS baseline rankings reported in the paper.

Filter Train candidates by VARS (Action class, Offence); rank by mean BGE cosine over the
query's four XVARS answers versus the released corpus answer embeddings. Output schema matches
the visual models: {"action_<qid>": ["action_<trainid>", ...]}.
"""
import os, json, argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BGE = "Represent this sentence for searching relevant passages: "
NUM_SIMILAR = 20


def load_meta(p):
    d = json.load(open(p, encoding="utf-8"))
    return d["Actions"] if "Actions" in d else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="Test or Valid")
    ap.add_argument("--out_dir", default=os.path.join(PIPELINE_DIR, "rankings"))
    a = ap.parse_args()
    sl = a.split.lower()

    # corpus (Train) VARS label buckets
    train_meta = load_meta(os.path.join(PIPELINE_DIR, "vars_attributes", "train.json"))
    buckets = {}
    for nid, info in train_meta.items():
        buckets.setdefault((info.get("Action class", ""), info.get("Offence", "")), []).append(f"action_{nid}")

    # precomputed corpus answer-embeddings (q0..q3 per requested qset)
    with open(os.path.join(PIPELINE_DIR, "corpus_embeddings", "action_ids.json"),
              encoding="utf-8") as handle:
        corpus_ids = json.load(handle)
    corpus_index = {action_id: index for index, action_id in enumerate(corpus_ids)}

    embedding_files = (
        "card.npy",
        "ball_touch.npy",
        "play_intent.npy",
        "contact_point.npy",
    )
    ref_embs = [np.load(os.path.join(PIPELINE_DIR, "corpus_embeddings", name))
                for name in embedding_files]
    N = len(ref_embs)

    qmeta = load_meta(os.path.join(PIPELINE_DIR, "vars_attributes", f"{sl}.json"))
    qtext = json.load(open(os.path.join(PIPELINE_DIR, "xvars_answers", f"{sl}.json"),
                           encoding="utf-8"))

    model = SentenceTransformer(MODEL_NAME)
    out = {}
    pids = sorted([k for k in qmeta if k.isdigit()], key=int)
    for pid in pids:
        ak = f"action_{pid}"
        cands = buckets.get((qmeta[pid].get("Action class", ""), qmeta[pid].get("Offence", "")), [])
        top = []
        if cands and ak in qtext:
            ans = list(qtext[ak].values())[:N]
            while len(ans) < N:
                ans.append("")
            vecs = model.encode([BGE + str(x) for x in ans], normalize_embeddings=True)
            idxs = [corpus_index[c] for c in cands if c in corpus_index]
            if idxs:
                sims = [cosine_similarity(vecs[i].reshape(1, -1), ref_embs[i][idxs]) for i in range(N)]
                avg = np.mean(sims, axis=0)[0]
                cids = [c for c in cands if c in corpus_index]
                top = [c for c, _ in sorted(zip(cids, avg), key=lambda x: x[1], reverse=True)[:NUM_SIMILAR]]
        out[ak] = top

    os.makedirs(a.out_dir, exist_ok=True)
    op = os.path.join(a.out_dir, f"{sl}.json")
    json.dump(out, open(op, "w"), indent=2)
    print(f"wrote {op}  ({len(out)} queries, {sum(1 for v in out.values() if v)} non-empty)")


if __name__ == "__main__":
    main()
