"""Evaluate precedent relevance or reproduce paper Table 3.

With no custom ranking directory, all paper methods are evaluated. Pass
``--rankings-dir`` to score only one external retrieval method. The default
20,000-resample split-stratified bootstrap reproduces the paper's 95% confidence
intervals; ``--bootstrap-resamples 0`` prints point estimates only.

    python -m evaluation.precedent
    python -m evaluation.precedent --rankings-dir outputs/my_model
"""
import argparse
import json
import os

from . import metrics
from .metrics import MODELS, FINETUNED_MODELS, strip

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KS = (1, 5, 10)
METRIC_KEYS = (
    "hr1", "hr5", "hr10", "r1", "r5", "r10", "ndcg5", "ndcg10", "mrr"
)
EVAL_MODELS = MODELS + FINETUNED_MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate precedent relevance or reproduce paper Table 3"
    )
    parser.add_argument(
        "--rankings-dir",
        help="evaluate only this method; requires test_full.json and valid_full.json",
    )
    parser.add_argument(
        "--method-name",
        help="display name for --rankings-dir (default: directory name)",
    )
    parser.add_argument(
        "--pred-root",
        default="retrieval/rankings",
        help="paper mode: directory containing <model>/<split>_full.json",
    )
    parser.add_argument(
        "--vars-xvars-dir",
        default="retrieval/vars_xvars/rankings",
        help="paper mode: directory containing <split>.json VARS+XVARS rankings",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=20_000,
        help="query-bootstrap replicates; 0 disables confidence intervals",
    )
    parser.add_argument("--seed", type=int, default=20_260_829)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    if args.bootstrap_resamples < 0:
        parser.error("--bootstrap-resamples cannot be negative")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")
    if args.method_name and not args.rankings_dir:
        parser.error("--method-name requires --rankings-dir")
    return args


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def ranking_reader(directory, view="full"):
    loaded = metrics.load_ranking_views(resolve(directory), [view])[view]
    return lambda split, query: loaded[split].get(query, [])


def baseline_reader(directory):
    loaded = {}
    for split, *_ in metrics.GTc:
        path = os.path.join(resolve(directory), f"{split}.json")
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        loaded[split] = {
            query: [strip(candidate) for candidate in ranked]
            for query, ranked in raw.items()
        }
    return lambda split, query: loaded[split].get(query, [])


def query_contributions(predictions, relevance):
    """Per-query Table 3 contributions in the paper's column order."""
    contributions = {}
    for split, *_ in metrics.GTc:
        rows = []
        for query, relevant in relevance[split].items():
            if not relevant:
                continue
            ranked = predictions(split, query)
            hits = [len(set(ranked[:cutoff]) & relevant) for cutoff in KS]
            reciprocal_rank = next(
                (1.0 / rank for rank, item in enumerate(ranked[:10], 1)
                 if item in relevant),
                0.0,
            )
            rows.append(
                [float(hit > 0) for hit in hits]
                + [hit / len(relevant) for hit in hits]
                + [metrics.ndcg_at_k(ranked, relevant, cutoff) for cutoff in (5, 10)]
                + [reciprocal_rank]
            )
        contributions[split] = rows
    return contributions


def confidence_interval(predictions, relevance, args):
    if args.bootstrap_resamples == 0:
        return None
    _, lower, upper = metrics.stratified_bootstrap_interval(
        query_contributions(predictions, relevance),
        resamples=args.bootstrap_resamples,
        seed=args.seed,
        confidence=args.confidence,
    )
    return lower, upper


def metric_values(result):
    return (
        result["hr"][1], result["hr"][5], result["hr"][10],
        result["rec"][1], result["rec"][5], result["rec"][10],
        result["ndcg"][5], result["ndcg"][10], result["mrr"],
    )


HEADERS = (
    "Method", "HR@1", "HR@5", "HR@10", "Recall@1",
    "Recall@5", "Recall@10", "nDCG@5", "nDCG@10", "MRR@10",
)


def point_row(name, result, maxima=None):
    cells = []
    for key, value in zip(METRIC_KEYS, metric_values(result)):
        marker = (
            "*"
            if maxima is not None and abs(value - maxima[key]) < 1e-9
            else ""
        )
        cells.append(f"{value:.2f}{marker}")
    return [name, *cells]


def interval_row(name, interval):
    lower, upper = interval
    return [
        name,
        *[f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(lower, upper)],
    ]


def print_results(title, query_count, point_rows, interval_rows, args, best=False):
    print(title)
    print(f"Answerable queries: {query_count}")
    print("Metric values are percentages.")
    print()
    print(metrics.format_table(HEADERS, point_rows, left_columns=(0,)))
    if best:
        print("* Best evaluated non-random method in this column.")
    if interval_rows:
        confidence = 100 * args.confidence
        print()
        print(
            f"{confidence:g}% confidence intervals "
            f"({args.bootstrap_resamples:,} split-stratified bootstrap resamples)"
        )
        print(
            metrics.format_table(
                ("Method", *HEADERS[1:]), interval_rows, left_columns=(0,)
            )
        )


def evaluate_custom(args, relevance):
    predictions = ranking_reader(args.rankings_dir)
    result = metrics.score_precedent(predictions, relevance)
    interval = confidence_interval(predictions, relevance, args)
    name = args.method_name or os.path.basename(os.path.normpath(args.rankings_dir))
    print_results(
        "Precedent relevance — custom method",
        result["nq"],
        [point_row(name, result)],
        [interval_row(name, interval)] if interval is not None else [],
        args,
    )


def reproduce_paper(args, relevance):
    baseline = baseline_reader(args.vars_xvars_dir)
    predictions = {
        key: ranking_reader(os.path.join(args.pred_root, key))
        for key, *_ in EVAL_MODELS
    }
    results = {
        "Random": metrics.score_precedent_random(relevance),
        "VARS+XVARS": metrics.score_precedent(baseline, relevance),
    }
    results.update({
        key: metrics.score_precedent(predictions[key], relevance)
        for key, *_ in EVAL_MODELS
    })
    intervals = {
        "VARS+XVARS": confidence_interval(baseline, relevance, args),
        **{
            key: confidence_interval(predictions[key], relevance, args)
            for key, *_ in EVAL_MODELS
        },
    }
    candidates = ["VARS+XVARS"] + [key for key, *_ in EVAL_MODELS]
    maxima = {
        metric_key: max(metric_values(results[method])[index] for method in candidates)
        for index, metric_key in enumerate(METRIC_KEYS)
    }

    point_rows = [
        point_row("Random", results["Random"]),
        point_row("VARS+XVARS", results["VARS+XVARS"], maxima),
        *[
            point_row(display, results[key], maxima)
            for key, display, _short, _params in EVAL_MODELS
        ],
    ]
    interval_rows = []
    if args.bootstrap_resamples:
        interval_rows = [
            interval_row("VARS+XVARS", intervals["VARS+XVARS"]),
            *[
                interval_row(display, intervals[key])
                for key, display, _short, _params in EVAL_MODELS
            ],
        ]
    print_results(
        "Precedent relevance — all paper methods",
        results["Random"]["nq"],
        point_rows,
        interval_rows,
        args,
        best=True,
    )


def main():
    args = parse_args()
    relevance = metrics.load_precedent_relevance()
    if args.rankings_dir:
        evaluate_custom(args, relevance)
    else:
        reproduce_paper(args, relevance)


if __name__ == "__main__":
    main()
