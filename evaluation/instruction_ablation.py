"""Evaluate the instruction-sensitivity ablation in paper Table 5.

For each instruction-following embedder, scores three query-side instruction levels
(Generic / Domain / Detailed) for precedent-relevance MRR@10 and category-relevance Action mAP
level. Reads the runs under retrieval/rankings/instruction_ablation/<model>/<instruction>/.
InternVideo2 is video-native (no instruction) and is excluded.

    python -m evaluation.instruction_ablation
"""
import argparse
import json
import os

from . import metrics
from .metrics import strip

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTION = metrics.TIERS[0]

ABLATION_MODELS = [("qwen3vl-2b", "Qwen3-VL-Emb-2B"),
                   ("qwen3vl-8b", "Qwen3-VL-Emb-8B"),
                   ("vlm2vec-v2", "VLM2Vec-V2"), ("lamra", "LamRA")]
LEVELS = [("Generic", "generic"), ("Domain", "domain"), ("Detailed", "detailed")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the instruction-sensitivity ablation in paper Table 5"
    )
    return parser.parse_args()


def reader(model, ldir):
    c = {lo: json.load(open(os.path.join(
        REPO, "retrieval", "rankings", "instruction_ablation", model, ldir, f"{lo}.json")))
         for lo, *_ in metrics.GTc}
    return lambda lo, key: [strip(x) for x in c[lo].get(key, [])]


def main():
    parse_args()
    precedent_relevance = metrics.load_precedent_relevance()
    ann_train = metrics.load_ann("Train")
    query_annotations = {
        split_name: metrics.load_ann(split_name)
        for _, split_name, _, _ in metrics.GTc
    }

    # Score every (model, level): precedent MRR@10 + Action mAP@{1,5,10}.
    results = {}
    for model, _ in ABLATION_MODELS:
        results[model] = {}
        for level, level_dir in LEVELS:
            predictions = reader(model, level_dir)
            mrr = metrics.score_precedent(predictions, precedent_relevance)["mrr"]
            mean_ap = metrics.score_category(
                predictions,
                ann_train,
                query_annotations,
                ACTION[1],
                ACTION[2],
            )[0]
            results[model][level] = {"mrr": mrr, "mAP": mean_ap}

    rows = []
    for model, display_name in ABLATION_MODELS:
        best = {"mrr": max(results[model][level]["mrr"] for level, _ in LEVELS)}
        for cutoff in (1, 5, 10):
            best[cutoff] = max(
                results[model][level]["mAP"][cutoff] for level, _ in LEVELS
            )
        for level, _ in LEVELS:
            result = results[model][level]
            values = [
                result["mrr"],
                result["mAP"][1],
                result["mAP"][5],
                result["mAP"][10],
            ]
            maxima = [best["mrr"], best[1], best[5], best[10]]
            cells = [
                f"{value:.{2 if index == 0 else 1}f}"
                + ("*" if abs(value - maxima[index]) < 1e-9 else "")
                for index, value in enumerate(values)
            ]
            rows.append([display_name, level, *cells])

    print("Instruction ablation")
    print("Metric values are percentages.")
    print()
    print(
        metrics.format_table(
            (
                "Model", "Instruction", "Precedent MRR@10", "Action mAP@1",
                "Action mAP@5", "Action mAP@10",
            ),
            rows,
            left_columns=(0, 1),
        )
    )
    print("* Best instruction for this model and metric.")


if __name__ == "__main__":
    main()
