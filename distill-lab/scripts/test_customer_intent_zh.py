"""验证中文客服数据的时点、单/多诉求边界和电商去除未来回复。"""

import json
import tempfile
import unittest
from pathlib import Path

from prepare_customer_intent_zh import LAB, ecommerce_review, encoder_for, frames, request_keys


class CustomerIntentDataTests(unittest.TestCase):
    def test_prepared_length_matches_real_tokenizer_encoding(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(LAB / "checkpoints/banking77-lora-v1/base/tokenizer", local_files_only=True)
        labels = ["酒店/查询/名称", "酒店/查询/地址"]
        encoder = encoder_for(tokenizer, labels, {labels[0]: "推荐酒店", labels[1]: "查询地址"})
        for text in ("酒店地址在哪？", "此前对话：推荐甲酒店\n当前用户：有无烟房吗？"):
            tokens = encoder.tokenize_state(text)
            self.assertEqual(encoder.prefix_length + len(tokens), len(encoder.encode(tokens, 0)["ids"]))

    def test_request_is_not_inform_constraint(self):
        acts = [["Inform", "酒店", "价格", "100"], ["Request", "酒店", "名称", ""],
                ["General", "greet", "none", "none"]]
        self.assertEqual(request_keys(acts), ["酒店/查询/名称"])

    def test_multiple_requests_are_not_first_label_only(self):
        acts = [["Request", "酒店", "电话", ""], ["Request", "酒店", "地址", ""]]
        self.assertEqual(len(request_keys(acts)), 2)

    def test_facility_details_collapse_without_erasing_other_requests(self):
        acts = [["Request", "酒店", "酒店设施-热水", ""], ["Request", "酒店", "酒店设施-wifi", ""],
                ["Select", "餐馆", "源领域", "酒店"]]
        self.assertEqual(request_keys(acts), ["酒店/查询/酒店设施", "餐馆/筛选/关联对象"])

    def test_only_past_text_and_current_user_enter_state(self):
        dialogue = {"goal": "DO_NOT_LEAK_GOAL", "messages": [
            {"role": "usr", "content": "找酒店", "dialog_act": [["Request", "酒店", "名称", ""]], "user_state": "HIDDEN"},
            {"role": "sys", "content": "推荐甲酒店", "dialog_act": []},
            {"role": "usr", "content": "地址呢", "dialog_act": [["Request", "酒店", "地址", ""]]},
            {"role": "sys", "content": "FUTURE_ANSWER", "dialog_act": []}]}
        rows = list(frames("one", dialogue, "train", "train"))
        self.assertNotIn("推荐甲酒店", rows[0]["state"])
        self.assertIn("推荐甲酒店", rows[1]["state"])
        for row in rows:
            for secret in ("HIDDEN", "DO_NOT_LEAK_GOAL", "FUTURE_ANSWER", "dialog_act"):
                self.assertNotIn(secret, row["state"])
            self.assertEqual(row["turn_index"], row["history_turns"])

    def test_ecommerce_removes_future_answer_and_stays_unapproved(self):
        record = {"id": "x", "output": "商品材质", "image": ["x.jpg"],
                  "instruction": "根据多轮对话分类\n<用户与客服的对话 START>\n用户: <image>\n客服: 什么问题\n用户: 什么材质\n客服: FUTURE\n<用户与客服的对话 END>"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "www2025-train.json").write_text(json.dumps([record]))
            result = ecommerce_review(root, root)
            row = json.loads((root / "ecommerce_review.jsonl").read_text())
            self.assertNotIn("FUTURE", row["state"])
            self.assertFalse(row["approved_for_training"])
            self.assertEqual(result["records_with_future_replies_removed"], 1)


if __name__ == "__main__":
    unittest.main()
