"""Evaluate category relevance or reproduce paper Table 4.

Scores retrieval against dense MVFoul labels at Action, Action+Offence, and
Action+Offence+Severity, reporting mAP@K for K in {1,5,10}. With no custom
ranking directory, all paper methods are evaluated. Pass ``--rankings-dir`` to
score only one external retrieval method. The default 20,000-resample
split-stratified bootstrap computes 95% confidence intervals;
``--bootstrap-resamples 0`` prints point estimates only.

    python -m evaluation.category
    python -m evaluation.category --rankings-dir outputs/my_model
"""
import argparse
import json
import os

from . import metrics
from .metrics import MODELS, FINETUNED_MODELS, TIERS, strip

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_MODELS = MODELS + FINETUNED_MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate category relevance or reproduce paper Table 4"
    )
    parser.add_argument(
        "--rankings-dir",
        help="evaluate only this method; see README for required view files",
    )
    parser.add_argument(
        "--method-name",
        help="display name for --rankings-dir (default: directory name)",
    )
    parser.add_argument(
        "--pred-root",
        default="retrieval/rankings",
        help="paper mode: directory containing one subdirectory per model",
    )
    parser.add_argument(
        "--vars-xvars-dir",
        default="retrieval/vars_xvars/rankings",
        help="paper mode: directory containing <split>.json VARS+XVARS rankings",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="score every tier over the full gallery using <split>_full.json",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=20_000,
        help="query-bootstrap replicates; 0 disables confidence intervals",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20_260_829,
        help="NumPy bootstrap seed (default: 20260829)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="two-sided percentile confidence level (default: 0.95)",
    )
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


def ranking_reader(directory, views):
    loaded = metrics.load_ranking_views(resolve(directory), views)
    return lambda view: (
        lambda split, query: loaded[view][split].get(query, [])
    )


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
    return loaded


def scored(pred_for_view, ann_train, query_annotations, full=False):
    scores = {}
    for tier_name, required, key_function, view in TIERS:
        selected_view = "full" if full else view
        scores[tier_name] = metrics.score_category(
            pred_for_view(selected_view),
            ann_train,
            query_annotations,
            required,
            key_function,
        )[0]
    return scores


def baseline_scores(rankings, ann_train, query_annotations):
    scores = {}
    for tier_name, required, key_function, _view in TIERS:
        labelled = {
            clip_id
            for clip_id, labels in ann_train.items()
            if all(labels[field] is not None for field in required)
        }

        def predictions(split, query, labelled_ids=labelled):
            return [
                candidate
                for candidate in rankings[split].get(query, [])
                if candidate in labelled_ids
            ]

        scores[tier_name] = metrics.score_category(
            predictions,
            ann_train,
            query_annotations,
            required,
            key_function,
        )[0]
    return scores


def confidence_intervals(
    pred_for_view, ann_train, query_annotations, args, full=False
):
    if args.bootstrap_resamples == 0:
        return None
    intervals = {}
    for tier_name, required, key_function, view in TIERS:
        selected_view = "full" if full else view
        contributions, _ = metrics.category_query_contributions(
            pred_for_view(selected_view),
            ann_train,
            query_annotations,
            required,
            key_function,
        )
        _, lower, upper = metrics.stratified_bootstrap_interval(
            contributions,
            resamples=args.bootstrap_resamples,
            seed=args.seed,
            confidence=args.confidence,
        )
        intervals[tier_name] = (lower, upper)
    return intervals


def baseline_confidence_intervals(rankings, ann_train, query_annotations, args):
    if args.bootstrap_resamples == 0:
        return None
    intervals = {}
    for tier_name, required, key_function, _view in TIERS:
        labelled = {
            clip_id
            for clip_id, labels in ann_train.items()
            if all(labels[field] is not None for field in required)
        }

        def predictions(split, query, labelled_ids=labelled):
            return [
                candidate
                for candidate in rankings[split].get(query, [])
                if candidate in labelled_ids
            ]

        contributions, _ = metrics.category_query_contributions(
            predictions,
            ann_train,
            query_annotations,
            required,
            key_function,
        )
        _, lower, upper = metrics.stratified_bootstrap_interval(
            contributions,
            resamples=args.bootstrap_resamples,
            seed=args.seed,
            confidence=args.confidence,
        )
        intervals[tier_name] = (lower, upper)
    return intervals


def table_row(name, by_tier, maxima=None):
    cells = []
    for tier_name, *_ in TIERS:
        for cutoff in (1, 5, 10):
            value = by_tier[tier_name][cutoff]
            marker = (
                "*"
                if maxima is not None
                and abs(value - maxima[(tier_name, cutoff)]) < 1e-9
                else ""
            )
            cells.append(f"{value:.1f}{marker}")
    return [name, *cells]


def interval_row(name, by_tier):
    cells = []
    for tier_name, *_ in TIERS:
        lower, upper = by_tier[tier_name]
        cells.extend(
            f"[{lo:.1f}, {hi:.1f}]" for lo, hi in zip(lower, upper)
        )
    return [name, *cells]


HEADERS = (
    "Method", "Action@1", "Action@5", "Action@10",
    "A+O@1", "A+O@5", "A+O@10", "A+O+S@1", "A+O+S@5", "A+O+S@10",
)


def print_results(
    title, gallery, query_total, rows, interval_rows, args, best=False
):
    print(title)
    print(f"Gallery protocol: {gallery}")
    print(f"Action/Action+Offence queries: {query_total}")
    print("Metric values are mAP@K percentages.")
    print()
    print(metrics.format_table(HEADERS, rows, left_columns=(0,)))
    if best:
        print("* Best evaluated visual model in this column.")
    if interval_rows:
        confidence = 100 * args.confidence
        print()
        print(
            f"{confidence:g}% confidence intervals "
            f"({args.bootstrap_resamples:,} split-stratified bootstrap resamples)"
        )
        print(metrics.format_table(HEADERS, interval_rows, left_columns=(0,)))


def query_count(ann_train, query_annotations):
    _scores, _gallery_size, count = metrics.score_category_random(
        ann_train, query_annotations, ["action"], TIERS[0][2]
    )
    return count


def evaluate_custom(args, ann_train, query_annotations):
    views = ["full"] if args.full else ["action_offence", "action_offence_severity"]
    predictions = ranking_reader(args.rankings_dir, views)
    result = scored(predictions, ann_train, query_annotations, full=args.full)
    intervals = confidence_intervals(
        predictions,
        ann_train,
        query_annotations,
        args,
        full=args.full,
    )
    name = args.method_name or os.path.basename(os.path.normpath(args.rankings_dir))
    gallery = "full gallery" if args.full else "label-complete galleries"
    print_results(
        "Category relevance — custom method",
        gallery,
        query_count(ann_train, query_annotations),
        [table_row(name, result)],
        [interval_row(name, intervals)] if intervals is not None else [],
        args,
    )


def reproduce_paper(args, ann_train, query_annotations):
    views = ["full"] if args.full else ["action_offence", "action_offence_severity"]
    results = {
        key: scored(
            ranking_reader(os.path.join(args.pred_root, key), views),
            ann_train,
            query_annotations,
            full=args.full,
        )
        for key, *_ in EVAL_MODELS
    }
    random_scores = {
        tier_name: metrics.score_category_random(
            ann_train, query_annotations, required, key_function
        )[0]
        for tier_name, required, key_function, _view in TIERS
    }
    maxima = {
        (tier_name, cutoff): max(
            results[key][tier_name][cutoff] for key, *_ in EVAL_MODELS
        )
        for tier_name, *_ in TIERS
        for cutoff in (1, 5, 10)
    }

    vars_scores = None
    vars_rankings = None
    if not args.full:
        vars_rankings = baseline_reader(args.vars_xvars_dir)
        vars_scores = baseline_scores(vars_rankings, ann_train, query_annotations)

    intervals = {}
    if args.bootstrap_resamples:
        intervals = {
            key: confidence_intervals(
                ranking_reader(os.path.join(args.pred_root, key), views),
                ann_train,
                query_annotations,
                args,
                full=args.full,
            )
            for key, *_ in EVAL_MODELS
        }
        if vars_rankings is not None:
            intervals["VARS+XVARS"] = baseline_confidence_intervals(
                vars_rankings, ann_train, query_annotations, args
            )

    gallery = "full gallery" if args.full else "label-complete galleries"
    rows = [table_row("Random", random_scores)]
    if vars_scores is not None:
        rows.append(table_row("VARS+XVARS", vars_scores))
    rows.extend(
        table_row(display, results[key], maxima)
        for key, display, _short, _params in EVAL_MODELS
    )
    interval_rows = []
    if args.bootstrap_resamples:
        if vars_scores is not None:
            interval_rows.append(
                interval_row("VARS+XVARS", intervals["VARS+XVARS"])
            )
        interval_rows.extend(
            interval_row(display, intervals[key])
            for key, display, _short, _params in EVAL_MODELS
        )
    print_results(
        "Category relevance — all paper methods",
        gallery,
        query_count(ann_train, query_annotations),
        rows,
        interval_rows,
        args,
        best=True,
    )


def main():
    args = parse_args()
    ann_train = metrics.load_ann("Train")
    query_annotations = {
        split_name: metrics.load_ann(split_name)
        for _, split_name, *_ in metrics.GTc
    }
    if args.rankings_dir:
        evaluate_custom(args, ann_train, query_annotations)
    else:
        reproduce_paper(args, ann_train, query_annotations)


if __name__ == "__main__":
    main()
