"""用缓存的 Laya 权重实测 MPS 或 CUDA 训练速度；合成输入仅用于测速。"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# 禁止不支持的 GPU 算子悄悄退回 CPU，避免把混合执行误报为 MPS 测速。
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
from laya_common import DecisionModel, build_sequence

REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
SOURCE_REVISION = "42626c348753fbb17572a813127df2278a1ec527"
DEFAULT_WEIGHTS = (
    Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya/snapshots" / REVISION
)


class CheckpointedHeadLayer(torch.nn.Module):
    """反向时重算决策头的一层，保留其权重、注意力和 dropout 行为。"""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.layer, *args, use_reentrant=False, **kwargs)
        return self.layer(*args, **kwargs)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", choices=["mps", "cuda"], default="mps")
    parser.add_argument("--mode", choices=["full", "lora"], default="lora")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, choices=[128, 256, 512, 1024, 2048, 4096, 8192], default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--memory-limit-gib", type=float, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--checkpoint-head", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.warmup, args.steps, args.rank) < 1:
        parser.error("批量、预热步数、计时步数及 LoRA 秩必须大于零")
    if args.memory_limit_gib is not None and args.memory_limit_gib <= 0:
        parser.error("内存限制必须大于零")
    return args


def build_batch(tokenizer, batch_size, seq_len, device):
    question = {
        "t": "choice",
        "ins": "Select the support team that should handle the customer's request.",
        "crit": {
            "billing": "Duplicate charges or payment problems",
            "shipping": "Delivery delays or missing packages",
            "returns": "Return or refund requests",
            "technical": "Application errors or login problems",
        },
    }
    states = [
        "I was charged twice for the same order. Please investigate the duplicate payment. ",
        "My package has not arrived and the tracking status has not changed for a week. ",
        "The shoes are the wrong size. I would like to return them for a refund. ",
        "I cannot sign in to the application. It displays an error after I enter my password. ",
    ]
    rows, positions = [], []
    for index in range(batch_size):
        # 重复文本填满预算，排除大量 padding 对吞吐量的高估。
        state = states[index % 4]
        state_tokens = len(tokenizer(state, add_special_tokens=False)["input_ids"])
        repeats = max(80, seq_len // state_tokens + 2)
        # 重复字符串的分词边界可能合并，以实际长度为准补足。
        while True:
            ids, markers = build_sequence(
                tokenizer, state * repeats, question,
                max_len=seq_len, head_max_len=min(192, seq_len - 32),
            )
            if len(ids) == seq_len:
                break
            repeats *= 2
        assert len(ids) == seq_len and len(markers) == 4
        rows.append(ids)
        positions.append(markers)
    return {
        "input_ids": torch.tensor(rows, dtype=torch.long, device=device),
        "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long, device=device),
        "marker_pos": torch.tensor(positions, dtype=torch.long, device=device),
        "marker_mask": torch.ones((batch_size, 4), dtype=torch.bool, device=device),
        "qtype": torch.zeros(batch_size, dtype=torch.long, device=device),
    }, torch.tensor([i % 4 for i in range(batch_size)], dtype=torch.long, device=device)


def load_model(args):
    cfg = json.loads((args.weights / "rl_agent_config.json").read_text())
    encoder_cfg = AutoConfig.from_pretrained(args.weights / "encoder", local_files_only=True)
    if args.seq_len > encoder_cfg.max_position_embeddings:
        raise ValueError("输入预算超过底座模型配置的上下文长度")
    # 此缓存按 Transformers 5 保存；4.57 使用两个显式 RoPE 字段，保留相同数值。
    rope = getattr(encoder_cfg, "rope_parameters", None)
    if rope:
        for layer_type in ("full_attention", "sliding_attention"):
            if rope[layer_type]["rope_type"] != "default":
                raise ValueError("测速适配仅支持该检查点的默认 RoPE")
        encoder_cfg.global_rope_theta = rope["full_attention"]["rope_theta"]
        encoder_cfg.local_rope_theta = rope["sliding_attention"]["rope_theta"]
    # 两个平台均使用普通 SDPA 路径，编译时间不混入这次训练吞吐对照。
    encoder_cfg.reference_compile = False
    encoder = AutoModel.from_config(encoder_cfg, attn_implementation="sdpa")
    model = DecisionModel(encoder, cfg["head_layers"], len(cfg["act_costs"]) + 1)
    model.load_state_dict(load_file(str(args.weights / "model.safetensors")), strict=True)
    # 业务分类测速只优化分类损失；保留上游前向中的动作头计算，但不更新此头。
    model.act_head.requires_grad_(False)
    targets = []
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model

        targets = ["Wqkv", "Wo"]
        model.encoder = get_peft_model(model.encoder, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
            target_modules=targets, bias="none",
        ))
    if args.gradient_checkpointing:
        # 仅对编码器开启重算；保留原始决策头实现。
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    if args.checkpoint_head and model.head is not None:
        model.head.layers = torch.nn.ModuleList(
            CheckpointedHeadLayer(layer) for layer in model.head.layers
        )
    return model.to(device=args.device, dtype=torch.float32).train(), targets


def memory_sample(device):
    sample = {"process_rss_bytes": psutil.Process().memory_info().rss}
    if device == "mps":
        sample.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory())
    else:
        sample.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_bytes=torch.cuda.memory_reserved())
    return sample


def synchronize(device):
    (torch.mps if device == "mps" else torch.cuda).synchronize()


def hardware_info(device):
    info = {"physical_memory_bytes": psutil.virtual_memory().total, "os": platform.platform()}
    if device == "mps":
        info["chip"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
        ).strip()
    else:
        info.update(chip=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda,
                    compute_capability=list(torch.cuda.get_device_capability(0)),
                    gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    return info


def main():
    args = arguments()
    available = torch.backends.mps.is_available() if args.device == "mps" else torch.cuda.is_available()
    if not available:
        raise RuntimeError(f"当前 PyTorch 无法使用 {args.device}，停止测速")
    if args.device == "cuda":
        # FP32 对照不使用 TF32；BF16 通过显式 autocast 开启。
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA 设备不支持 BF16")
    if args.memory_limit_gib is not None:
        # 长序列探索为当前进程设置预算，超出时报告 OOM，不取消内存保护。
        capacity = (torch.mps.recommended_max_memory() if args.device == "mps"
                    else torch.cuda.get_device_properties(0).total_memory)
        fraction = args.memory_limit_gib * 2**30 / capacity
        if fraction > 1:
            raise ValueError("测速内存限制不得超过设备容量或建议的工作集")
        (torch.mps if args.device == "mps" else torch.cuda).set_per_process_memory_fraction(float(fraction))
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    setup_start = time.perf_counter()
    print(f"加载缓存 Laya：{args.mode}, batch={args.batch_size}, length={args.seq_len}", flush=True)
    model, lora_targets = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.weights / "tokenizer", local_files_only=True)
    batch, labels = build_batch(tokenizer, args.batch_size, args.seq_len, args.device)
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    assert all(param.device.type == args.device for param in model.parameters())
    encoder_params = [param for name, param in trainable if name.startswith("encoder.")]
    head_params = [param for name, param in trainable if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 2.5e-5 if args.mode == "full" else 1e-4},
        {"params": head_params, "lr": 1e-4},
    ], weight_decay=0.01, foreach=False)
    # 分别检查编码器和决策头：确有非零梯度，且优化器真实更新了权重。
    probe_suffix = "Wqkv.weight" if args.mode == "full" else "Wqkv.lora_B.default.weight"
    encoder_probe = next((name, param) for name, param in trainable if name.endswith(probe_suffix))
    probes = [encoder_probe, ("scorer.3.weight", model.scorer[3].weight)]
    checks = {}
    steps = []
    memory = [memory_sample(args.device)]
    synchronize(args.device)
    setup_seconds = time.perf_counter() - setup_start
    print(f"加载完成：总参数 {sum(p.numel() for p in model.parameters()):,}，"
          f"可训练参数 {sum(p.numel() for _, p in trainable):,}", flush=True)

    for step in range(args.warmup + args.steps):
        before = {name: param.detach().clone() for name, param in probes} if step == 0 else {}
        synchronize(args.device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            if args.precision == "bf16" and not torch.is_autocast_enabled(args.device):
                raise RuntimeError("当前环境未实际启用 BF16 autocast")
            logits, _ = model(**batch)
            loss = F.cross_entropy(logits, labels)
        synchronize(args.device)
        forward_end = time.perf_counter()
        loss.backward()
        synchronize(args.device)
        backward_end = time.perf_counter()
        # 仅在首个预热步检查梯度，检查开销不计入正式测速。
        if step == 0:
            for name, param in probes:
                assert param.grad is not None
                grad_norm = param.grad.float().norm().item()
                finite = bool(torch.isfinite(param.grad).all().item())
                assert finite and grad_norm > 0, (name, grad_norm, finite)
                checks[name] = {"gradient_norm": grad_norm, "gradient_finite": finite}
        optimizer.step()
        synchronize(args.device)
        end = time.perf_counter()
        value = loss.item()
        if not torch.isfinite(loss).item():
            raise RuntimeError("损失出现非有限数值，停止测速")
        if step == 0:
            for name, param in probes:
                change = (param.detach() - before[name]).abs().max().item()
                assert change > 0, (name, change)
                checks[name]["max_weight_change"] = change
        memory.append(memory_sample(args.device))
        row = {
            "step": step, "warmup": step < args.warmup, "loss": value,
            "forward_seconds": forward_end - start,
            "backward_seconds": backward_end - forward_end,
            "optimizer_seconds": end - backward_end,
            "total_seconds": end - start,
        }
        steps.append(row)
        phase = "预热" if row["warmup"] else "计时"
        print(f"{phase} {step + 1}: {row['total_seconds']:.3f} 秒，loss={value:.4f}", flush=True)

    measured = [row for row in steps if not row["warmup"]]
    totals = [row["total_seconds"] for row in measured]
    mean = statistics.mean(totals)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_info(args.device),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformers", "peft", "safetensors"]},
        "model": {
            "id": "convaiinnovations/laya", "revision": args.weights.name,
            "source_revision": SOURCE_REVISION,
            "source_sha256": hashlib.sha256((ROOT / "reference/laya_common.py").read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for _, p in trainable),
            "strict_checkpoint_load": True,
        },
        "configuration": {
            "mode": args.mode, "device": args.device, "cpu_fallback": False,
            "cuda_tf32": False if args.device == "cuda" else None,
            "dtype": "bfloat16_autocast" if args.precision == "bf16" else "float32",
            "parameter_dtype": "float32",
            "memory_limit_gib": args.memory_limit_gib,
            "batch_size": args.batch_size, "sequence_length": args.seq_len, "options": 4,
            "warmup_steps": args.warmup, "measured_steps": args.steps,
            "lora_rank": args.rank if args.mode == "lora" else None,
            "lora_targets": lora_targets, "gradient_checkpointing": args.gradient_checkpointing,
            "head_gradient_checkpointing": args.checkpoint_head,
            "optimizer": "AdamW", "foreach": False,
            "encoder_learning_rate": optimizer.param_groups[0]["lr"],
            "head_learning_rate": optimizer.param_groups[1]["lr"],
            "weight_decay": 0.01, "loss": "cross_entropy", "synthetic_data": True,
        },
        "setup_seconds": setup_seconds,
        "summary": {
            "mean_step_seconds": mean, "median_step_seconds": statistics.median(totals),
            "min_step_seconds": min(totals), "max_step_seconds": max(totals),
            "std_step_seconds": statistics.stdev(totals) if len(totals) > 1 else 0.0,
            "examples_per_second": args.batch_size / mean,
            "tokens_per_second": args.batch_size * args.seq_len / mean,
            **{f"mean_{phase}_seconds": statistics.mean(row[f"{phase}_seconds"] for row in measured)
               for phase in ("forward", "backward", "optimizer")},
        },
        "max_observed_memory": {name: max(row[name] for row in memory) for name in memory[0]},
        "cuda_peak_memory": {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        } if args.device == "cuda" else None,
        "gradient_and_update_checks": checks,
        "steps": steps,
        "limitations": [
            "只用四分类合成文本测速，不能用于判断业务准确率或收敛。",
            "计时包含前向、反向、AdamW 更新及设备同步，不含分词、数据传输、验证和保存。",
            "内存为初始化及每步结束的采样最大值，不代表步内瞬时峰值；GPU 与 RSS 不应相加。",
            "预热后仅做少量迭代，不能代表长时间训练的温度、功耗和速度稳定性。",
            "每次运行从原始缓存权重重新加载，测速后的权重不保存。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"结果已保存：{args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
