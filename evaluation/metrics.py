"""Shared scoring for SoccerNet-FoulRet evaluation.

Two relevance signals, one metric module:
  * precedent relevance -- ranking metrics (HitRate/Recall/MRR/nDCG) over the ~4
    human-verified precedents per query, on the full gallery.
  * category relevance  -- mean Average Precision over the dense MVFoul attribute labels, per
    nested tier (Action / +Offence / +Severity).

Metrics are matched to label density: rank metrics for the sparse precedent judgments, mAP for the dense
category labels. All scorers take a `pred_of(split, key) -> [train_id, ...]` callable so the
caller wires which prediction file / corpus view is scored (full gallery, labelled subset, or a
single ablation run).
"""
import json
import math
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_GALLERY = 2916                 # Train gallery size
GTc = [("test",  "Test",  "data/foulret/test.json",  "id_test"),
       ("valid", "Valid", "data/foulret/valid.json", "id_valid")]


def strip(x):
    return str(x).replace("action_", "").strip()


def load_ranking_views(directory, views, splits=("test", "valid")):
    """Load and validate ``<split>_<view>.json`` retrieval rankings.

    Query keys must use ``action_<id>``. Candidate values may be bare IDs or
    ``action_<id>`` strings and are normalized to bare IDs. Rankings must not
    contain duplicate candidates for a query.
    """
    paths = {
        (view, split): os.path.join(directory, f"{split}_{view}.json")
        for view in views
        for split in splits
    }
    missing = [path for path in paths.values() if not os.path.isfile(path)]
    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing required ranking file(s):\n  {formatted}")

    loaded = {}
    for view in views:
        loaded[view] = {}
        for split in splits:
            path = paths[(view, split)]
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict):
                raise ValueError(f"Ranking file must contain a JSON object: {path}")

            clean = {}
            for query, ranked in raw.items():
                if not isinstance(query, str) or not query.startswith("action_"):
                    raise ValueError(
                        f"Query keys must use action_<id> in {path}: {query!r}"
                    )
                if not isinstance(ranked, list):
                    raise ValueError(f"Ranking for {query} must be a list in {path}")
                candidates = [strip(candidate) for candidate in ranked]
                if any(not candidate for candidate in candidates):
                    raise ValueError(f"Ranking for {query} contains an empty ID in {path}")
                if len(candidates) != len(set(candidates)):
                    raise ValueError(
                        f"Ranking for {query} contains duplicate candidates in {path}"
                    )
                clean[query] = candidates
            loaded[view][split] = clean
    return loaded


# ------------------------------------------------------------------ loaders
def load_precedent_relevance():
    """{split_lower: {'action_<qid>': set(approved train ids)}}."""
    G = {}
    for lo, cap, f, key in GTc:
        G[lo] = {f"action_{q[key]}": set(strip(x) for x in q.get("final_ranking", []))
                 for q in json.load(open(os.path.join(REPO, f)))}
    return G


def load_ann(cap):
    """{train/query id: {'action','offence','severity'}} from the MVFoul annotations."""
    d = json.load(open(os.path.join(REPO, f"data/mvfouls_window/{cap}/annotations.json")))
    A = d.get("Actions", d)
    out = {}
    for k, v in A.items():
        def fld(name):
            s = str(v.get(name, "")).strip()
            return s or None
        out[strip(k)] = {"action": fld("Action class"), "offence": fld("Offence"),
                         "severity": fld("Severity")}
    return out


def ap_at_k(ranked, rel, K):
    R = len(rel)
    if R == 0:
        return 0.0
    hits = 0
    acc = 0.0
    for i, x in enumerate(ranked[:K], 1):
        if x in rel:
            hits += 1
            acc += hits / i
    return acc / min(K, R)


def ndcg_at_k(ranked, rel, K):
    """Binary nDCG@K, normalized by the best ordering of the known relevant items.

    Retrieval rankings are expected to contain unique gallery IDs.  ``seen`` nevertheless
    prevents a repeated ID from receiving relevance gain more than once.
    """
    if not rel or K <= 0:
        return 0.0
    dcg = 0.0
    seen = set()
    for rank, item in enumerate(ranked[:K], 1):
        if item in rel and item not in seen:
            dcg += 1.0 / math.log2(rank + 1)
        seen.add(item)
    idcg = sum(1.0 / math.log2(rank + 1)
               for rank in range(1, min(K, len(rel)) + 1))
    return dcg / idcg


# ------------------------------------------------------------------ precedent relevance
def score_precedent(pred_of, relevance, ks=(1, 5, 10), mrr_cap=10):
    """Precedent metrics over queries with at least one approved precedent.

    MRR is truncated at ``mrr_cap``. nDCG uses binary relevance and the known approved set;
    unjudged gallery clips receive no gain but are not asserted to be true negatives.
    """
    hr = {K: 0.0 for K in ks}
    rec = {K: 0.0 for K in ks}
    ndcg = {K: 0.0 for K in ks}
    mrr = 0.0
    nq = 0
    for lo, cap, f, key in GTc:
        for k, rel in relevance[lo].items():
            if not rel:
                continue
            nq += 1
            r = pred_of(lo, k)
            for K in ks:
                h = len(set(r[:K]) & rel)
                hr[K] += 1.0 if h else 0.0
                rec[K] += h / len(rel)
                ndcg[K] += ndcg_at_k(r, rel, K)
            for rank, x in enumerate(r[:mrr_cap], 1):
                if x in rel:
                    mrr += 1.0 / rank
                    break
    return dict(nq=nq, hr={K: 100 * hr[K] / nq for K in ks},
                rec={K: 100 * rec[K] / nq for K in ks},
                ndcg={K: 100 * ndcg[K] / nq for K in ks}, mrr=100 * mrr / nq)


def score_precedent_random(relevance, ks=(1, 5, 10), mrr_cap=10, N=N_GALLERY):
    """Expected precedent metrics under uniform random retrieval of the gallery."""
    hr = {K: 0.0 for K in ks}
    rec = {K: 0.0 for K in ks}
    ndcg = {K: 0.0 for K in ks}
    mrr = 0.0
    nq = 0
    for lo, cap, f, key in GTc:
        for k, rel in relevance[lo].items():
            if not rel:
                continue
            nq += 1
            g = len(rel)
            for K in ks:
                pno = 1.0
                for i in range(K):
                    pno *= (N - g - i) / (N - i)
                hr[K] += 1 - pno
                rec[K] += K / N
                discounts = sum(1.0 / math.log2(rank + 1) for rank in range(1, K + 1))
                ideal = sum(1.0 / math.log2(rank + 1)
                            for rank in range(1, min(K, g) + 1))
                ndcg[K] += (g / N) * discounts / ideal
            for rr in range(1, mrr_cap + 1):
                p = g / (N - (rr - 1))
                for i in range(rr - 1):
                    p *= (N - g - i) / (N - i)
                mrr += p / rr
    return dict(nq=nq, hr={K: 100 * hr[K] / nq for K in ks},
                rec={K: 100 * rec[K] / nq for K in ks},
                ndcg={K: 100 * ndcg[K] / nq for K in ks}, mrr=100 * mrr / nq)


# ------------------------------------------------------------------ category relevance
# tier name, required labels, key function, and the corpus view it condenses over
TIERS = [("action",   ["action"],                       lambda v: (v["action"],),                           "action_offence"),
         ("offence",   ["action", "offence"],            lambda v: (v["action"], v["offence"]),              "action_offence"),
         ("severity",  ["action", "offence", "severity"], lambda v: (v["action"], v["offence"], v["severity"]), "action_offence_severity")]


def buckets(ann_train, need, keyfn):
    """Relevant-set index for a tier: {category-key: set(train ids)}; N = labelled corpus size."""
    c = {}
    for cid, v in ann_train.items():
        if any(v[fl] is None for fl in need):
            continue
        c.setdefault(keyfn(v), set()).add(cid)
    return c, sum(len(s) for s in c.values())


def category_query_contributions(
    pred_of, ann_train, qann, need, keyfn, ks=(1, 5, 10)
):
    """Return per-query AP@K contributions grouped by query split."""
    c2, Nc = buckets(ann_train, need, keyfn)
    contributions = {}
    for lo, cap, f, key in GTc:
        rows = []
        for qid, v in qann[cap].items():
            if any(v[fl] is None for fl in need):
                continue
            rel = c2.get(keyfn(v))
            if not rel:
                continue
            r = pred_of(lo, f"action_{qid}")
            rows.append([ap_at_k(r, rel, K) for K in ks])
        contributions[lo] = rows
    return contributions, Nc


def score_category(pred_of, ann_train, qann, need, keyfn, ks=(1, 5, 10)):
    """mAP@K for one tier. `pred_of(split, key)` returns the ranked list from whichever corpus
    view the caller wires (labelled subset = condensed; full gallery = uncondensed)."""
    contributions, Nc = category_query_contributions(
        pred_of, ann_train, qann, need, keyfn, ks
    )
    nq = sum(len(rows) for rows in contributions.values())
    totals = [0.0] * len(ks)
    for rows in contributions.values():
        for row in rows:
            totals = [total + value for total, value in zip(totals, row)]
    return {K: 100 * total / nq for K, total in zip(ks, totals)}, Nc, nq


def score_category_random(ann_train, qann, need, keyfn, ks=(1, 5, 10)):
    """Closed-form expected mAP@K under uniform random retrieval of the tier's labelled corpus."""
    c2, Nc = buckets(ann_train, need, keyfn)
    H = {K: sum(1.0 / i for i in range(1, K + 1)) for K in ks}
    mAP = {K: 0.0 for K in ks}
    nq = 0
    for lo, cap, f, key in GTc:
        for qid, v in qann[cap].items():
            if any(v[fl] is None for fl in need):
                continue
            rel = c2.get(keyfn(v))
            if not rel:
                continue
            nq += 1
            R = len(rel)
            p = R / Nc
            p2 = R * (R - 1) / (Nc * (Nc - 1))
            for K in ks:
                mAP[K] += (p * H[K] + p2 * (K - H[K])) / min(K, R)
    return {K: 100 * mAP[K] / nq for K in ks}, Nc, nq


def stratified_bootstrap_interval(
    contributions,
    resamples=20_000,
    seed=20_260_829,
    confidence=0.95,
):
    """Bootstrap query-level metric contributions while preserving split sizes.

    ``contributions`` maps each split to an ``(queries, metrics)`` array on a
    0--1 scale. The returned point estimate and interval bounds are percentages.
    """
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    arrays = {
        split: np.asarray(values, dtype=np.float64)
        for split, values in contributions.items()
    }
    total_queries = sum(len(values) for values in arrays.values())
    if total_queries == 0:
        raise ValueError("cannot bootstrap an empty query set")

    metric_count = next(iter(arrays.values())).shape[1]
    rng = np.random.default_rng(seed)
    draws = np.zeros((resamples, metric_count), dtype=np.float64)
    for values in arrays.values():
        # Batch the resampling to keep peak memory small for multi-column tables.
        for start in range(0, resamples, 1_000):
            stop = min(start + 1_000, resamples)
            indices = rng.integers(
                0, len(values), size=(stop - start, len(values)), dtype=np.int32
            )
            draws[start:stop] += values[indices].sum(axis=1)
    draws *= 100.0 / total_queries
    point = sum(values.sum(axis=0) for values in arrays.values()) * 100.0 / total_queries
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, (tail, 1.0 - tail), axis=0)
    return point, lower, upper


# ------------------------------------------------------------------ shared config
# Display order = ascending parameter count; baseline + random pinned above via the scripts.
MODELS = [("internvideo2-1b", "InternVideo2-1B", "InternVideo2", "1B"),
        ("qwen3vl-2b",      "Qwen3-VL-Emb-2B", "Qwen3-VL-Emb", "2B"),
        ("vlm2vec-v2",      "VLM2Vec-V2",      "VLM2Vec-V2",   "2B"),
        ("internvideo2-6b", "InternVideo2-6B", "InternVideo2", "6B"),
        ("lamra",           "LamRA",           "LamRA",        "7B"),
        ("qwen3vl-8b",      "Qwen3-VL-Emb-8B", "Qwen3-VL-Emb", "8B")]

# Task-specific model used in the paper.
FINETUNED_MODELS = [
    ("qwen3vl-2b-finetuned", "Qwen3-VL-Emb-2B-FT", "Qwen3-VL-Emb-FT", "2B"),
]


def format_table(headers, rows, left_columns=(0,)):
    """Return a dependency-free, aligned table suitable for terminal output."""
    headers = [str(value) for value in headers]
    rows = [[str(value) for value in row] for row in rows]
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("every table row must have the same length as the header")

    widths = [
        max([len(headers[index])] + [len(row[index]) for row in rows])
        for index in range(len(headers))
    ]
    left_columns = set(left_columns)

    def render(row):
        return " | ".join(
            value.ljust(widths[index])
            if index in left_columns
            else value.rjust(widths[index])
            for index, value in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])
