"""验证多正例损失和有监督候选采样的划分边界。"""
import unittest
import torch
from train_browser_choice_pilot import positive_mass_loss, supervised_candidates


class ChoicePilotTests(unittest.TestCase):
    def test_multi_positive_mass_and_gradients(self):
        logits = torch.tensor([0., 0., 0.], requires_grad=True)
        loss = positive_mass_loss(logits, [0, 2])
        self.assertAlmostEqual(float(loss.detach()), -__import__('math').log(2/3), places=6)
        loss.backward()
        self.assertLess(float(logits.grad[0]), 0)
        self.assertLess(float(logits.grad[2]), 0)
        self.assertGreater(float(logits.grad[1]), 0)

    def test_sampling_rejects_development(self):
        with self.assertRaisesRegex(ValueError, "只能用于训练集"):
            supervised_candidates({"split": "development"}, 64, 1)

    def test_sampling_preserves_gold_only_in_train(self):
        candidates = [{"node_id": str(i), "element": {"text": "find" if i < 3 else "unrelated"}}
                      for i in range(5)]
        frame = {"id": "train-example", "split": "train", "gold_node_ids": ["4"],
                 "state": {"task": "find"}, "candidates": candidates}
        selected = supervised_candidates(frame, 3, 42)
        self.assertEqual(len(selected), 3)
        self.assertIn("4", {r['node_id'] for r in selected})
        self.assertEqual(frame['candidates'], candidates)
        self.assertEqual(selected, supervised_candidates(frame, 3, 42))


if __name__ == '__main__':
    unittest.main()
