"""加载固定版本的 Laya 底座与 BANKING77 微调权重，输出 77 类意图概率。"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from laya_common import DecisionModel, collate_items


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HeadLayer(torch.nn.Module):
    """保留训练时包装层的参数名称，推理时直接执行原层。"""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


class Banking77Classifier:
    def __init__(self, model_dir=None, base_dir=None, device="cpu", local_files_only=False):
        self.root = Path(model_dir or Path(__file__).resolve().parent)
        self.config = json.loads((self.root / "release_config.json").read_text())
        self.labels = self.config["labels"]
        self.temperature = float(json.loads((self.root / "calibration.json").read_text())["temperature"])
        if len(self.labels) != 77 or len(set(self.labels)) != 77 or self.temperature <= 0:
            raise ValueError("类别或温度配置无效")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS 不可用，请指定 device='cpu'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        self.device = torch.device(device)
        # 固定上游提交，仅下载英文底座的必需文件。
        base = Path(base_dir) if base_dir else Path(snapshot_download(
            self.config["base_model"], revision=self.config["base_revision"],
            allow_patterns=list(self.config["base_files_sha256"]),
            local_files_only=local_files_only,
        ))
        for relative, expected in self.config["base_files_sha256"].items():
            if sha256(base / relative) != expected:
                raise ValueError(f"底座校验失败：{relative}")
        if sha256(self.root / "trainable.safetensors") != self.config["checkpoint_sha256"]:
            raise ValueError("微调权重校验失败")
        cfg = json.loads((base / "rl_agent_config.json").read_text())
        encoder_cfg = AutoConfig.from_pretrained(base / "encoder", local_files_only=True)
        if self.config["max_length"] > encoder_cfg.max_position_embeddings:
            raise ValueError("输入预算超过底座支持长度")
        # 原配置以 Transformers 5 保存；4.57 需要显式恢复全局与局部 RoPE。
        rope = getattr(encoder_cfg, "rope_parameters", None)
        if rope:
            if any(rope[k]["rope_type"] != "default" for k in ("full_attention", "sliding_attention")):
                raise ValueError("当前加载器仅支持原检查点的默认 RoPE")
            encoder_cfg.global_rope_theta = rope["full_attention"]["rope_theta"]
            encoder_cfg.local_rope_theta = rope["sliding_attention"]["rope_theta"]
        encoder_cfg.reference_compile = False
        encoder = AutoModel.from_config(encoder_cfg, attn_implementation="sdpa")
        model = DecisionModel(encoder, cfg["head_layers"], len(cfg["act_costs"]) + 1)
        model.load_state_dict(load_file(str(base / "model.safetensors")), strict=True)
        model.act_head.requires_grad_(False)
        model.encoder = get_peft_model(model.encoder, LoraConfig(**self.config["lora"]))
        model.head.layers = torch.nn.ModuleList(HeadLayer(layer) for layer in model.head.layers)
        state = load_file(str(self.root / "trainable.safetensors"))
        expected = {name for name, param in model.named_parameters() if param.requires_grad}
        if set(state) != expected:
            raise ValueError("微调权重不完整，必须同时包含 LoRA 和训练后的决策头")
        model.load_state_dict(state, strict=False)
        self.model = model.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.root, local_files_only=True)
        self.head = self.tokenizer("choice question: " + self.config["instruction"], add_special_tokens=False)["input_ids"]
        self.options = [[self.tokenizer.mask_token_id] + self.tokenizer(" " + label, add_special_tokens=False)["input_ids"] for label in self.labels]

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("每条输入必须是字符串")
        tok = self.tokenizer
        ids = [tok.cls_token_id] + self.head + [tok.sep_token_id]
        markers = []
        for option in self.options:
            markers.append(len(ids))
            ids.extend(option)
        state = tok(text.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        ids += [tok.sep_token_id] + state + [tok.sep_token_id]
        if len(ids) > self.config["max_length"]:
            raise ValueError(f"含问题和全部候选的输入共 {len(ids)} tokens，超过 {self.config['max_length']}；未截断")
        return {"ids": ids, "markers": markers, "qtype": 0, "label": -1}

    @torch.inference_mode()
    def logits(self, texts, batch_size=4):
        texts = [texts] if isinstance(texts, str) else list(texts)
        if batch_size < 1:
            raise ValueError("batch_size 必须大于零")
        outputs = []
        for start in range(0, len(texts), batch_size):
            items = [[self.encode(text)] for text in texts[start:start + batch_size]]
            batch = collate_items(items, self.tokenizer.pad_token_id)
            inputs = {k: batch[k].to(self.device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}
            logits, _ = self.model(**inputs)
            logits = logits.float().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("预测出现非有限数值")
            outputs.append(logits)
        return torch.cat(outputs) if outputs else torch.empty((0, len(self.labels)))

    def predict(self, texts, top_k=3, batch_size=4):
        if not 1 <= top_k <= len(self.labels):
            raise ValueError("top_k 必须在 1 到 77 之间")
        # 双精度 softmax 沿用原评测的温度校准流程。
        probs = torch.softmax(self.logits(texts, batch_size).double() / self.temperature, dim=-1)
        results = []
        for row in probs:
            values, indices = row.topk(top_k)
            results.append({
                "label": self.labels[indices[0].item()],
                "confidence": values[0].item(),
                "top_k": [{"label": self.labels[i], "probability": p} for i, p in zip(indices.tolist(), values.tolist())],
                "probabilities": dict(zip(self.labels, row.tolist())),
            })
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--base", type=Path, help="现有底座目录；省略时下载固定版本")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = Banking77Classifier(base_dir=args.base, device=args.device, local_files_only=args.local_files_only)
    result = model.predict(args.text)[0]
    result.pop("probabilities")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
