"""检查样本审计的划分保护及答案与输入隔离。"""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_browser_source_pilot as pilot


class PilotTests(unittest.TestCase):
    def test_nontrain_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with gzip.open(root / "train_frames.jsonl.gz", "wt") as stream:
                stream.write(json.dumps({"id": "x", "split": "train", "trajectory_id": "heldout"}) + "\n")
            with patch.object(pilot, "GROUND", root), patch.object(pilot, "OUT", root):
                with self.assertRaisesRegex(ValueError, "非训练轨迹"):
                    pilot.inspect_mind2web([{"trajectory_id": "heldout", "split": "pilot_holdout"}])

    def test_gold_change_does_not_change_observation(self):
        row = {"id": "x", "split": "train", "trajectory_id": "train-task", "website": "example",
               "step_index": 0, "gold_operation": "CLICK", "gold_node_ids": ["missing"],
               "state": {"task": "search", "previous_actions": [], "page_text": "Search"},
               "candidates": [{"node_id": "one", "element": {"tag": "button", "text": "Search"},
                               "operations": ["CLICK"]}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = []
            for gold in [["missing"], ["one"]]:
                row["gold_node_ids"] = gold
                with gzip.open(root / "train_frames.jsonl.gz", "wt") as stream:
                    stream.write(json.dumps(row) + "\n")
                with patch.object(pilot, "GROUND", root), patch.object(pilot, "OUT", root):
                    pilot.inspect_mind2web([{"trajectory_id": "train-task", "split": "train"}])
                results.append(json.loads((root / "mind2web-inspection.jsonl").read_text()))
            self.assertEqual(results[0]["observation"], results[1]["observation"])
            self.assertFalse(results[0]["alignment"]["target_in_top64"])
            self.assertTrue(results[1]["alignment"]["target_in_top64"])

    def test_protected_websites_include_subdomains(self):
        assignments = [{"website": "amazon", "split": "pilot_holdout"},
                       {"website": "new.mta.info", "split": "pilot_holdout"},
                       {"website": "booking", "split": "train"}]
        for host in ["www.amazon.com", "www.amazon.de", "new.mta.info", "bustime.mta.info"]:
            self.assertTrue(pilot.host_protected(host, assignments))
        for host in ["www.booking.com", "amazon.example.org.evil.test", "notamazon.com"]:
            # 首个包含 amazon 的主机被保守拦截；不将此函数视为通用公共后缀解析器。
            if host == "amazon.example.org.evil.test":
                self.assertTrue(pilot.host_protected(host, assignments))
            else:
                self.assertFalse(pilot.host_protected(host, assignments))


if __name__ == "__main__":
    unittest.main()
