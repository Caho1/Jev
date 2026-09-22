"""从完整本地文件加载 Laya，在 Mac MPS 或 CPU 上执行分类。"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from banking77_data import BankingEncoder, INSTRUCTION, sha256
from benchmark_training import load_model
from train_banking77 import reload_checkpoint
from laya_common import collate_items

LAB = Path(__file__).resolve().parents[1]
MODEL = LAB / "checkpoints/banking77-lora-v1"


class LocalLaya:
    def __init__(self, variant="tuned", device="mps"):
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS 不可用；如需 CPU，请显式指定 --device cpu")
        cfg = json.loads((MODEL / "best/config.json").read_text())
        if sha256(MODEL / "base/model.safetensors") != cfg["base_model_sha256"]:
            raise ValueError("基础权重校验失败")
        args = SimpleNamespace(weights=MODEL / "base", device=device, seq_len=cfg["seq_len"],
                               mode="lora" if variant == "tuned" else "full", rank=cfg["rank"],
                               gradient_checkpointing=cfg["gradient_checkpointing"], checkpoint_head=cfg["checkpoint_head"])
        torch.set_num_threads(4)
        torch.manual_seed(42)
        started = time.perf_counter()
        self.model = reload_checkpoint(args, MODEL / "best") if variant == "tuned" else load_model(args)[0]
        self.model.eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL / "base/tokenizer", local_files_only=True)
        self.device, self.variant = device, variant
        self.synchronize()
        self.load_seconds = time.perf_counter() - started

    def synchronize(self):
        if self.device == "mps":
            torch.mps.synchronize()

    def encoder(self, criteria, instruction=INSTRUCTION):
        # 沿用训练格式，保留所有候选和正文；不使用上游会截短候选的构造器。
        encoder = BankingEncoder(self.tokenizer, list(criteria), 8192)
        encoder.head = self.tokenizer("choice question: " + instruction, add_special_tokens=False)["input_ids"]
        options = [name if not desc else f"{name}: {desc}" for name, desc in criteria.items()]
        encoder.options = [[self.tokenizer.mask_token_id] + self.tokenizer(" " + option, add_special_tokens=False)["input_ids"] for option in options]
        return encoder

    @torch.inference_mode()
    def logits(self, texts, criteria, instruction=INSTRUCTION):
        encoder = self.encoder(criteria, instruction)
        items = [[encoder.encode(encoder.tokenize_state(text))] for text in texts]
        batch = collate_items(items, self.tokenizer.pad_token_id)
        inputs = {key: batch[key].to(self.device) for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}
        started = time.perf_counter()
        logits, _ = self.model(**inputs)
        result = logits.float().cpu().numpy()
        self.synchronize()
        elapsed = time.perf_counter() - started
        if not np.isfinite(result).all():
            raise FloatingPointError("本地预测出现非有限数值")
        return result, {"forward_seconds": elapsed, "lengths": [len(item[0]["ids"]) for item in items]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--variant", choices=["baseline", "tuned"], default="tuned")
    parser.add_argument("--criteria", type=Path)
    args = parser.parse_args()
    if args.criteria:
        criteria = json.loads(args.criteria.read_text())
        temperature = 1.0
    else:
        criteria = {name: "" for name in json.loads((MODEL / "best/config.json").read_text())["labels"]}
        temperatures = json.loads((LAB / "results/banking77-lora-v1/results.json").read_text())["temperatures"]
        temperature = temperatures[args.variant]
    model = LocalLaya(args.variant, args.device)
    logits, info = model.logits([args.text], criteria)
    z = logits[0].astype(float) / temperature
    p = np.exp(z - z.max()); p /= p.sum()
    labels = list(criteria)
    print(json.dumps({"device": args.device, "variant": args.variant, "text": args.text,
                      "choice": labels[int(p.argmax())], "top3": [{"label": labels[i], "probability": float(p[i])} for i in np.argsort(-p)[:3]],
                      "temperature": temperature, "load_seconds": model.load_seconds, **info}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
