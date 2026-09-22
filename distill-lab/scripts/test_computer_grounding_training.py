"""保护完整候选评测的分母、平局处理和开发页预选规则。"""

import copy
import unittest

from train_computer_grounding import rank_metrics, select_development


class RankingProtocolTests(unittest.TestCase):
    def test_missing_gold_is_failure_at_every_cutoff(self):
        result = rank_metrics([2.0, 1.0], ["a", "b"], ["absent"])
        self.assertFalse(result["gold_in_pool"])
        self.assertEqual(result["reciprocal_rank"], 0)
        for k in (1, 8, 24, 64):
            self.assertFalse(result[f"recall_at_{k}"])

    def test_ties_do_not_prefer_gold(self):
        result = rank_metrics([0.0, 0.0], ["b", "a"], ["b"])
        self.assertEqual(result["target_rank"], 2)
        self.assertFalse(result["recall_at_1"])
        self.assertTrue(result["recall_at_8"])

    def test_development_selection_ignores_labels_and_candidate_counts(self):
        frames = [{"id": str(i), "website": f"site-{i%2}", "gold_node_ids": [str(i)], "candidates": [str(i)]}
                  for i in range(12)]
        before = [f["id"] for f in select_development(frames, 1, 42)]
        changed = copy.deepcopy(frames)
        for i, frame in enumerate(changed):
            frame["gold_node_ids"] = []
            frame["candidates"] = ["irrelevant"]*i
        after = [f["id"] for f in select_development(list(reversed(changed)), 1, 42)]
        self.assertEqual(before, after)
        self.assertEqual(len(before), 2)


if __name__ == "__main__":
    unittest.main()
