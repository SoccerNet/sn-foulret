import math
import json
import os
import tempfile
import unittest

from evaluation import metrics


class PrecedentMetricTests(unittest.TestCase):
    def test_binary_ndcg(self):
        ranked = ["a", "x", "b", "c"]
        relevant = {"a", "b", "c"}
        actual = metrics.ndcg_at_k(ranked, relevant, 3)
        expected = (1.0 + 1.0 / math.log2(4)) / (
            1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)
        )
        self.assertAlmostEqual(actual, expected)

    def test_repeated_result_receives_gain_once(self):
        self.assertAlmostEqual(metrics.ndcg_at_k(["a", "a"], {"a", "b"}, 2),
                               1.0 / (1.0 + 1.0 / math.log2(3)))

    def test_score_precedent_purposes(self):
        relevance = {"test": {"action_1": {"a", "b"}}, "valid": {}}
        predictions = {"test": {"action_1": ["x", "b", "a"]}, "valid": {}}
        pred_of = lambda split, query: predictions[split].get(query, [])
        score = metrics.score_precedent(pred_of, relevance, ks=(1, 2, 3), mrr_cap=3)

        self.assertEqual(score["nq"], 1)
        self.assertEqual(score["hr"], {1: 0.0, 2: 100.0, 3: 100.0})
        self.assertEqual(score["rec"], {1: 0.0, 2: 50.0, 3: 100.0})
        self.assertEqual(score["mrr"], 50.0)
        self.assertEqual(score["ndcg"][1], score["hr"][1])
        self.assertAlmostEqual(score["ndcg"][2],
                               100.0 / math.log2(3) / (1.0 + 1.0 / math.log2(3)))

    def test_random_ndcg_at_one_equals_random_hit_rate_at_one(self):
        relevance = {"test": {"action_1": {"a", "b"}}, "valid": {}}
        score = metrics.score_precedent_random(
            relevance, ks=(1, 5), mrr_cap=5, N=20
        )
        self.assertAlmostEqual(score["ndcg"][1], score["hr"][1])


class RankingLoaderTests(unittest.TestCase):
    def test_loads_and_normalizes_custom_rankings(self):
        with tempfile.TemporaryDirectory() as directory:
            for split in ("test", "valid"):
                with open(os.path.join(directory, f"{split}_full.json"), "w",
                          encoding="utf-8") as handle:
                    json.dump({"action_0": ["action_2", "3"]}, handle)

            loaded = metrics.load_ranking_views(directory, ["full"])

            self.assertEqual(loaded["full"]["test"]["action_0"], ["2", "3"])

    def test_rejects_duplicate_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            for split in ("test", "valid"):
                with open(os.path.join(directory, f"{split}_full.json"), "w",
                          encoding="utf-8") as handle:
                    json.dump({"action_0": ["2", "action_2"]}, handle)

            with self.assertRaisesRegex(ValueError, "duplicate candidates"):
                metrics.load_ranking_views(directory, ["full"])


class CategoryMetricTests(unittest.TestCase):
    def test_query_contributions_match_category_score(self):
        ann_train = {
            "a": {"action": "A", "offence": "Yes", "severity": "1"},
            "b": {"action": "A", "offence": "No", "severity": "2"},
        }
        query_annotations = {
            "Test": {"1": {"action": "A", "offence": "Yes", "severity": "1"}},
            "Valid": {"2": {"action": "A", "offence": "No", "severity": "2"}},
        }
        predictions = {
            "test": {"action_1": ["a", "b"]},
            "valid": {"action_2": ["a", "b"]},
        }
        pred_of = lambda split, query: predictions[split][query]

        contributions, gallery_size = metrics.category_query_contributions(
            pred_of,
            ann_train,
            query_annotations,
            ["action", "offence"],
            lambda labels: (labels["action"], labels["offence"]),
            ks=(1, 2),
        )
        score, scored_gallery_size, query_count = metrics.score_category(
            pred_of,
            ann_train,
            query_annotations,
            ["action", "offence"],
            lambda labels: (labels["action"], labels["offence"]),
            ks=(1, 2),
        )

        self.assertEqual(gallery_size, scored_gallery_size)
        self.assertEqual(query_count, 2)
        self.assertEqual(contributions["test"], [[1.0, 1.0]])
        self.assertEqual(contributions["valid"], [[0.0, 0.5]])
        self.assertEqual(score, {1: 50.0, 2: 75.0})


class TerminalFormattingTests(unittest.TestCase):
    def test_plain_aligned_table(self):
        rendered = metrics.format_table(
            ("Method", "Score"),
            (("Short", "1.2"), ("Long method", "12.3")),
        )

        self.assertEqual(
            rendered,
            "Method      | Score\n"
            "------------+------\n"
            "Short       |   1.2\n"
            "Long method |  12.3",
        )


if __name__ == "__main__":
    unittest.main()
