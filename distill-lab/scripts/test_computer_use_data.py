"""检查未来动作、答案标注和网站分组不会污染模型输入或数据划分。"""

import copy
import json
import unittest

from prepare_computer_use import convert_step, observed_candidates, retrieve, split_tasks


def fixture():
    def candidate(node_id):
        return {"tag": "button", "backend_node_id": str(node_id), "attributes": "{}"}
    action = {"action_uid": "step-1", "cleaned_html": '<html><button backend_node_id="1">Search</button><button backend_node_id="2">Cancel</button></html>',
              "operation": {"original_op": "CLICK", "op": "CLICK", "value": "SECRET_GOLD_ARGUMENT"},
              "pos_candidates": [candidate(1)], "neg_candidates": [candidate(2)]}
    return {"annotation_id": "task-1", "website": "example", "confirmed_task": "Search", "_source_file": "fixture",
            "action_reprs": ["CURRENT_GOLD_ACTION", "FUTURE_ACTION"], "actions": [action]}


class DataBoundaryTests(unittest.TestCase):
    def test_only_past_history_and_no_gold_argument(self):
        row, reason = convert_step(fixture(), 0, 2, 42)
        self.assertIsNone(reason)
        model_input = json.dumps([row["state"], row["question"]])
        for hidden in ("CURRENT_GOLD_ACTION", "FUTURE_ACTION", "SECRET_GOLD_ARGUMENT", "pos_candidates", "is_original_target"):
            self.assertNotIn(hidden, model_input)
        self.assertEqual(row["state"]["previous_actions"], [])

    def test_retrieval_does_not_depend_on_gold_partition(self):
        task = fixture()
        action = task["actions"][0]
        first = retrieve(observed_candidates(action)[0], "Search", 1)
        action["pos_candidates"], action["neg_candidates"] = action["neg_candidates"], action["pos_candidates"]
        second = retrieve(observed_candidates(action)[0], "Search", 1)
        self.assertEqual(first, second)

    def test_missing_target_is_not_inserted(self):
        task = fixture()
        action = task["actions"][0]
        action["pos_candidates"], action["neg_candidates"] = action["neg_candidates"], action["pos_candidates"]
        row, reason = convert_step(task, 0, 1, 42)
        self.assertIsNone(reason)
        self.assertFalse(row["candidate_recalled"])
        self.assertEqual(row["gold_choices"], ["DEFER"])
        self.assertEqual({v["node_id"] for v in row["candidate_references"].values()}, {"1"})

    def test_split_keeps_websites_and_identical_goals_together(self):
        tasks = [{"annotation_id": f"t{i}-{j}", "website": f"site-{i}", "confirmed_task": f"unique task {i} variant {j}"}
                 for i in range(12) for j in range(2)]
        tasks[0]["confirmed_task"] = tasks[2]["confirmed_task"] = "identical shared task"
        mapping = split_tasks(tasks, 42)
        for i in range(12):
            self.assertEqual(mapping[f"t{i}-0"], mapping[f"t{i}-1"])
        self.assertEqual(mapping["t0-0"], mapping["t1-0"])
        self.assertEqual({v[0] for v in mapping.values()}, {"train", "development", "calibration", "pilot_holdout"})


if __name__ == "__main__":
    unittest.main()
