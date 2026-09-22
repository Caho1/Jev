"""检查候选完整性、换序标签和数据隔离，避免错误监督进入正式训练。"""

import json
import random
import unittest
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from banking77_data import BankingEncoder, group_duplicates, partition, INSTRUCTION
from benchmark_training import DEFAULT_WEIGHTS
from laya_common import build_sequence
from train_banking77 import verify_data, metrics, temperature_fit

ROOT = Path(__file__).resolve().parents[1]


class Banking77Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.labels, cls.data = verify_data(ROOT / "data/banking77-v1")
        cls.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_WEIGHTS / "tokenizer", local_files_only=True)
        cls.encoder = BankingEncoder(cls.tokenizer, cls.labels)

    def test_all_candidates_and_permutation_targets(self):
        state = self.encoder.tokenize_state("I was charged twice.")
        order = list(range(77))
        random.Random(31).shuffle(order)
        for label in range(77):
            item = self.encoder.encode(state,label,order)
            self.assertEqual(order[item["label"]], label)
            self.assertEqual(len(item["markers"]),77)
            for pos, original_index in zip(item["markers"],order):
                expected = self.encoder.options[original_index]
                self.assertEqual(item["ids"][pos:pos+len(expected)],expected)

    def test_matches_untruncated_upstream_format(self):
        state = "Please help with this payment."
        item = self.encoder.encode(self.encoder.tokenize_state(state),0)
        q = {"t":"choice","ins":INSTRUCTION,"crit":dict.fromkeys(self.labels)}
        ids,markers = build_sequence(self.tokenizer,state,q,max_len=8192,head_max_len=2048)
        self.assertEqual(item["ids"],ids)
        self.assertEqual(item["markers"],markers)
        small = BankingEncoder(self.tokenizer,self.labels,max_length=64)
        with self.assertRaises(ValueError): small.encode(state,0)

    def test_official_test_retained_and_groups_disjoint(self):
        self.assertEqual(len(self.data["test"]),3080)
        self.assertEqual(len({r["id"] for rows in self.data.values() for r in rows}),sum(map(len,self.data.values())))
        groups = [{r["group_id"] for r in rows} for rows in self.data.values()]
        for i,g in enumerate(groups):
            for h in groups[i+1:]: self.assertFalse(g & h)

    def test_duplicate_exclusion_and_conflict(self):
        rows = [{"id":str(i),"state":f"independent message {i}","label":0,"source_split":"train"} for i in range(30)]
        rows += [{"id":"a","state":"same thing!","label":0,"source_split":"train"},
                 {"id":"b","state":"Same thing","label":0,"source_split":"test"},
                 {"id":"c","state":"ambiguous request","label":0,"source_split":"train"},
                 {"id":"d","state":"ambiguous request","label":1,"source_split":"train"}]
        group_duplicates(rows)
        splits,excluded = partition(rows)
        self.assertEqual({r["id"] for r in excluded},{"a","c","d"})
        self.assertEqual([r["id"] for r in splits["test"]],["b"])

    def test_calibration_does_not_change_class_predictions(self):
        rng = np.random.default_rng(12)
        logits = rng.normal(size=(100,77))
        labels = logits.argmax(axis=1)
        temperature = temperature_fit(logits,labels)
        self.assertEqual(metrics(logits,labels)["accuracy"],1.)
        self.assertEqual(metrics(logits,labels,temperature)["accuracy"],1.)
        self.assertLess(metrics(logits,labels,temperature)["nll"],metrics(logits,labels)["nll"])


if __name__=="__main__":
    unittest.main()
