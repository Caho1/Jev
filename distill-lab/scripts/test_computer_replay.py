"""验证回放边界、候选筛选、缺失目标计分与训练导出隔离。"""

import copy
import json
from pathlib import Path
import unittest

from computer_replay import freeze, model_input, normalize_answer, summarize, training_exports
from import_typesafe_replay import import_step, source_instructions
from prepare_computer_replay import convert_frame


class TinyTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 9
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(10, 10 + len(text.split())))}


def frame():
    return {"id": "frame", "split": "train", "group_id": "site", "trajectory_id": "task", "website": "example",
            "step_index": 1, "state": {"task": "Search", "page_text": "Search Cancel", "previous_actions": ["PAST_ACTION"]},
            "candidates": [{"node_id": str(i), "element": {"tag": "button", "text": word, "attributes": {}}, "operations": ["CLICK"]}
                           for i, word in enumerate(["Search", "Cancel"])],
            "gold_node_ids": ["0"], "gold_operation": "CLICK"}


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.frame = frame()
        self.row = convert_frame(self.frame, TinyTokenizer(), 2)

    def test_gold_changes_do_not_change_request_or_retrieval(self):
        changed = copy.deepcopy(self.frame)
        changed["gold_node_ids"] = ["1"]
        changed["gold_operation"] = "TYPE"
        other = convert_frame(changed, TinyTokenizer(), 2)
        self.assertEqual(model_input(self.row), model_input(other))

    def test_envelope_is_not_forwarded(self):
        self.row["teacher"] = {"answer": "TEACHER_SECRET"}
        self.row["reference"]["private_note"] = "GOLD_SECRET"
        packet = json.dumps(model_input(self.row))
        for forbidden in ("TEACHER_SECRET", "GOLD_SECRET", "gold_node_ids", "split", "trajectory_id"):
            self.assertNotIn(forbidden, packet)
        self.assertIn("PAST_ACTION", packet)

    def test_missing_target_is_not_inserted(self):
        self.frame["gold_node_ids"] = ["1"]
        row = convert_frame(self.frame, TinyTokenizer(), 1)
        self.assertFalse(row["reference"]["target_recalled"])
        self.assertEqual(row["reference"]["gold_choices"]["item"], ["NONE"])
        self.assertEqual(list(row["reference"]["candidate_references"].values()), ["0"])

    def test_request_mutation_rejected(self):
        self.row["request"]["state"]["goal"] = "changed"
        with self.assertRaises(ValueError):
            model_input(self.row)

    def test_future_or_missing_history_rejected(self):
        self.frame["state"]["previous_actions"].append("CURRENT_GOLD_ACTION")
        with self.assertRaises(ValueError):
            convert_frame(self.frame, TinyTokenizer(), 2)

    def test_error_and_unattempted_count_in_denominator(self):
        result = summarize([self.row], [])
        self.assertEqual(result["questions"]["item"]["accuracy"], 0)
        self.assertEqual(result["joint_accuracy"], 0)

    def test_none_does_not_count_as_grounded_success(self):
        self.frame["gold_node_ids"] = ["1"]
        row = convert_frame(self.frame, TinyTokenizer(), 1)
        prediction = {"id": row["id"], "request_sha256": row["request_sha256"], "ok": True, "elapsed_ms": 1,
                      "answers": {"operation": {"choice": "CLICK"}, "item": {"choice": "NONE"}}}
        result = summarize([row], [prediction])
        self.assertEqual(result["joint_accuracy"], 1)
        self.assertEqual(result["grounded_joint_accuracy"], 0)

    def test_export_rejects_development(self):
        self.row["metadata"]["split"] = "development"
        with self.assertRaises(ValueError):
            training_exports([self.row], [])

    def test_legacy_teacher_cannot_be_exported_as_gold(self):
        self.row["reference"] = {}
        with self.assertRaises(ValueError):
            training_exports([self.row], [])

    def test_bad_probability_distribution_rejected(self):
        q = self.row["request"]["questions"]["operation"]
        for probabilities in ({"CLICK": True, "TYPE": 0, "SELECT": 0}, {"CLICK": 0.2, "TYPE": 0.2, "SELECT": 0.2}):
            with self.assertRaises(ValueError):
                normalize_answer({"type": "choice", "choice": "CLICK", "probabilities": probabilities}, q)

    def test_wrong_response_hash_rejected(self):
        with self.assertRaises(ValueError):
            summarize([self.row], [{"id": self.row["id"], "request_sha256": "wrong"}])

    def test_saved_log_keeps_focus_history_and_geometry(self):
        path = Path(__file__).resolve().parents[1] / "reference/typesafe-computer-use/typesafe_computer_use/decide.py"
        instructions = source_instructions(path.read_text())
        state = {"goal": "Open search", "now": "frozen", "focused_field": {"label": "Search"}, "previous_actions": ["Opened app"]}
        parts = []
        for title, value in [("STATE  (sent as `state`)", state), ("QUESTION kind  (Choice criteria)", {"click_item": "Click", "done": "Done"}),
                             ("QUESTION item  (Choice criteria)", {"0": "Search", "1": "Cancel"})]:
            parts.append(title + "\n" + "=" * 78 + "\n" + json.dumps(value))
        answers = {"items": [{"index": 0, "text": "Search", "x1": 10}], "field": state["focused_field"]}
        row = import_step("\n".join(parts), answers, instructions, "fixture", {"split": "development"})
        self.assertEqual(row["request"]["state"], state)
        self.assertEqual(row["observations"]["items"][0]["x1"], 10)
        self.assertFalse(row["teacher"]["eligible_for_distillation"])
        self.assertFalse(row["metadata"]["legacy_teacher_request_verified"])


if __name__ == "__main__":
    unittest.main()
