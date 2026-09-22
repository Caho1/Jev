"""本地中文客服 LoRA 试验：固定开发探针，校准集和测试集不进入模型。"""

import argparse
import collections
import json
import math
import random
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from benchmark_training import load_model, hardware_info, memory_sample, synchronize, REVISION
from banking77_data import sha256, write_json
from prepare_customer_intent_zh import LAB, encoder_for, stable
from train_banking77 import batch_indices, save_checkpoint
from train_computer_grounding import Events, grounding_batch, empty_cache


def select_per_class(rows, count, seed):
    """只按固定 ID 哈希做每类抽样，不按长度或预测对错挑选。0 表示全量。"""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    return [r for label, items in sorted(groups.items())
            for r in sorted(items, key=lambda x: stable(f"{seed}:{x['id']}"))[:count or None]]


def validate_and_select(args):
    manifest = json.loads((args.data / "manifest.json").read_text())
    selected = {}
    for name in ("labels.json", "criteria.json", "train.jsonl", "development.jsonl"):
        if sha256(args.data / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"输入文件校验失败：{name}")
    labels = json.loads((args.data / "labels.json").read_text())
    criteria = json.loads((args.data / "criteria.json").read_text())
    sessions, groups = set(), set()
    for split, limit in (("train", args.train_per_class), ("development", args.dev_per_class)):
        rows = [json.loads(line) for line in (args.data / f"{split}.jsonl").read_text().splitlines()]
        if any(r["split"] != split or r["gold_choices"] != [labels[r["label"]]] for r in rows):
            raise ValueError("样本标签或来源划分不一致")
        current_sessions, current_groups = {r["dialogue_id"] for r in rows}, {r["group_id"] for r in rows}
        if sessions & current_sessions or groups & current_groups:
            raise ValueError("训练和开发存在会话或重复组交叉")
        sessions |= current_sessions
        groups |= current_groups
        selected[split] = select_per_class(rows, limit, args.seed)
    if not all(selected.values()):
        raise ValueError("训练或开发样本为空")
    return manifest, labels, criteria, selected


def metric_summary(logits, gold, labels):
    values = logits.astype(np.float64)
    values -= values.max(axis=1, keepdims=True)
    logp = values - np.log(np.exp(values).sum(axis=1, keepdims=True))
    probability = np.exp(logp)
    predicted = values.argmax(axis=1)
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(confusion, (gold, predicted), 1)
    support = confusion.sum(axis=1)
    f1 = 2 * np.diag(confusion) / np.maximum(confusion.sum(axis=0) + support, 1)
    return {"records": len(gold), "accuracy": float(np.mean(predicted == gold)),
            "macro_f1_supported_classes": float(f1[support > 0].mean()),
            "supported_classes": int((support > 0).sum()),
            "top3_accuracy": float(np.mean([y in x for y, x in zip(gold, np.argsort(-values, axis=1)[:, :3])])),
            "nll": float(-logp[np.arange(len(gold)), gold].mean()),
            "per_class": {name: {"support": int(support[i]), "recall": float(confusion[i, i]/support[i]) if support[i] else None,
                                  "f1": float(f1[i])} for i, name in enumerate(labels)},
            "confusion_matrix": confusion.tolist(), "confidence_calibrated": False}, probability


def make_items(rows, indices, encoder, rng=None):
    items = []
    for i in indices:
        order = list(range(len(encoder.labels)))
        if rng is not None:
            rng.shuffle(order)
        row = rows[i]
        items.append(encoder.encode(row["_tokens"], row["label"], order))
    return items


@torch.inference_mode()
def evaluate(model, rows, encoder, labels, args, event, phase):
    model.eval()
    all_logits = np.zeros((len(rows), len(labels)), dtype=np.float32)
    started, last_event = time.perf_counter(), time.perf_counter()
    batches = batch_indices(rows, args.eval_batch)
    for number, indices in enumerate(batches, 1):
        items = make_items(rows, indices, encoder)
        inputs = grounding_batch(items, encoder.tok, args.device, args.shape_bucket)
        logits, _ = model(**inputs)
        values = logits[:, :len(labels)].float().cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError("开发评测输出非有限")
        all_logits[indices] = values
        if time.perf_counter() - last_event >= 15 or number == len(batches):
            event(phase, completed=sum(len(x) for x in batches[:number]), total=len(rows),
                  elapsed_seconds=time.perf_counter()-started, memory=memory_sample(args.device))
            last_event = time.perf_counter()
        del inputs, logits
        if number % 16 == 0:
            empty_cache(args.device)
    result, probabilities = metric_summary(all_logits, np.array([r["label"] for r in rows]), labels)
    result.update(elapsed_seconds=time.perf_counter()-started, split="development", selection="fixed per-class ID hash",
                  official_test=False)
    write_json(args.output / f"{phase}_metrics.json", result)
    with (args.output / f"{phase}_predictions.jsonl").open("x") as stream:
        for row, values, probs in zip(rows, all_logits, probabilities):
            order = np.argsort(-probs)
            stream.write(json.dumps({"id": row["id"], "state": row["state"], "gold": row["label_name"],
                                     "predicted": labels[int(order[0])], "correct": int(order[0]) == row["label"],
                                     "logits": values.tolist(), "input_tokens": row["_length"],
                                     "top3": [{"label": labels[int(i)], "probability": float(probs[i])} for i in order[:3]]}, ensure_ascii=False) + "\n")
    return result


def run(args, event):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS 不可用，不回退到 CPU")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    fraction = args.memory_limit_gib * 2**30 / torch.mps.recommended_max_memory()
    if not 0 < fraction <= 1:
        raise ValueError("内存预算超过 MPS 建议容量")
    torch.mps.set_per_process_memory_fraction(fraction)
    manifest, labels, criteria, data = validate_and_select(args)
    tokenizer = AutoTokenizer.from_pretrained(args.weights / "tokenizer", local_files_only=True)
    encoder = encoder_for(tokenizer, labels, criteria, args.seq_len)
    for rows in data.values():
        for row in rows:
            row["_tokens"] = encoder.tokenize_state(row["state"])
            item = encoder.encode(row["_tokens"], row["label"])
            row["_length"] = len(item["ids"])
            if row["_length"] != row["input_tokens"] or len(item["markers"]) != len(labels):
                raise ValueError("编码长度或候选数与准备阶段不一致")
    base_hash = sha256(args.weights / "model.safetensors")
    if base_hash != "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c":
        raise ValueError("基础模型哈希改变")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(base_revision=REVISION, base_model_sha256=base_hash, labels=labels, criteria=criteria,
                  instruction=manifest["instruction"], hardware=hardware_info(args.device),
                  data_manifest_sha256=sha256(args.data / "manifest.json"),
                  train_records=len(data["train"]), development_records=len(data["development"]),
                  calibration_used=False, test_used=False, scope="local fixed-sample pipeline pilot" if args.train_per_class else "all prepared training records",
                  model_selection="fixed final epoch; development is diagnostic only",
                  input_lengths={s: {"min": min(r["_length"] for r in rows), "median": float(np.median([r["_length"] for r in rows])),
                                     "max": max(r["_length"] for r in rows), "truncated": 0} for s, rows in data.items()})
    source_dir = args.output / "source"
    source_dir.mkdir()
    names = [Path(__file__).name, "prepare_customer_intent_zh.py", "benchmark_training.py", "banking77_data.py",
             "train_banking77.py", "train_computer_grounding.py", "typed_laya.py", "local_laya.py"]
    config["code_sha256"] = {}
    for path in [Path(__file__).with_name(name) for name in names] + [LAB / "reference/laya_common.py"]:
        shutil.copyfile(path, source_dir / path.name)
        config["code_sha256"][path.name] = sha256(path)
    write_json(args.output / "run_config.json", config)
    write_json(args.output / "selection.json", {s: [r["id"] for r in rows] for s, rows in data.items()})
    event("loading", train_records=len(data["train"]), development_records=len(data["development"]), input_lengths=config["input_lengths"])
    model, _ = load_model(args)
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    # 初始化副本用于复核；既有 BANKING77 权重和 Computer Use 权重保持独立。
    save_checkpoint(model, args.output / "initial", {**config, "checkpoint_role": "original_laya"})
    baseline = evaluate(model, data["development"], encoder, labels, args, event, "baseline")
    event("baseline_complete", accuracy=baseline["accuracy"], macro_f1=baseline["macro_f1_supported_classes"])
    empty_cache(args.device)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in params if n.startswith("encoder.")], "lr": args.lr},
        {"params": [p for n, p in params if not n.startswith("encoder.")], "lr": args.head_lr}], weight_decay=.01, foreach=False)
    probe_name = next(n for n, _ in params if n.endswith("Wqkv.lora_B.default.weight"))
    probes = {probe_name: dict(params)[probe_name], "scorer.3.weight": dict(params)["scorer.3.weight"]}
    before = {n: p.detach().clone() for n, p in probes.items()}
    total_updates = math.ceil(math.ceil(len(data["train"])/args.batch_size)/args.accumulation) * args.epochs
    warmup = max(1, math.ceil(total_updates * .05))
    update, seen, loss_total = 0, 0, 0.0
    started, last_event = time.perf_counter(), time.perf_counter()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    event("training", completed=0, total=len(data["train"])*args.epochs, optimizer_steps_total=total_updates)
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = batch_indices(data["train"], args.batch_size, args.seed + epoch)
        for at in range(0, len(batches), args.accumulation):
            group = batches[at:at+args.accumulation]
            size = sum(map(len, group))
            optimizer.zero_grad(set_to_none=True)
            scale = (update+1)/warmup if update < warmup else max(.1, (total_updates-update)/max(1, total_updates-warmup))
            for pg, initial_lr in zip(optimizer.param_groups, (args.lr, args.head_lr)):
                pg["lr"] = initial_lr * scale
            for indices in group:
                items = make_items(data["train"], indices, encoder, rng)
                inputs = grounding_batch(items, tokenizer, args.device, args.shape_bucket)
                target = torch.tensor([item["label"] for item in items], device=args.device)
                logits, _ = model(**inputs)
                loss = F.cross_entropy(logits[:, :len(labels)], target, reduction="sum")
                if not torch.isfinite(loss):
                    raise FloatingPointError("训练损失非有限")
                (loss/size).backward()
                loss_total += float(loss.detach().cpu())
                seen += len(indices)
                del inputs, target, logits, loss
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1, error_if_nonfinite=True)
            if update == 0:
                for parameter in probes.values():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all() or not torch.any(parameter.grad != 0):
                        raise RuntimeError("LoRA 或决策头没有有效梯度")
            optimizer.step()
            synchronize(args.device)
            if update == 0:
                if any(torch.equal(before[n], p.detach()) for n, p in probes.items()):
                    raise RuntimeError("LoRA 或决策头没有真实更新")
                del before
                event("gradient_verified", lora_and_head_updated=True)
            update += 1
            empty_cache(args.device)
            if update == 1 or time.perf_counter()-last_event >= 15 or update == total_updates:
                elapsed = time.perf_counter()-started
                event("training", epoch=epoch, epochs=args.epochs, completed=seen, total=len(data["train"])*args.epochs,
                      optimizer_step=update, optimizer_steps_total=total_updates, mean_loss=loss_total/seen,
                      gradient_norm=float(norm), elapsed_seconds=elapsed, examples_per_second=seen/elapsed,
                      eta_training_seconds=(len(data["train"])*args.epochs-seen)*elapsed/seen,
                      memory=memory_sample(args.device))
                last_event = time.perf_counter()
        save_checkpoint(model, args.output / f"epoch-{epoch}", {**config, "epoch": epoch, "optimizer_step": update})
    training_seconds = time.perf_counter()-started
    del optimizer
    empty_cache(args.device)
    # 重新读取保存的权重，验证可训练参数完整，并以落盘检查点评估。
    checkpoint = args.output / f"epoch-{args.epochs}"
    saved_config = json.loads((checkpoint / "config.json").read_text())
    if sha256(checkpoint / "trainable.safetensors") != saved_config["checkpoint_sha256"]:
        raise ValueError("保存权重校验失败")
    state = load_file(str(checkpoint / "trainable.safetensors"))
    if set(state) != {name for name, _ in params}:
        raise ValueError("保存权重缺少可训练参数")
    model.load_state_dict(state, strict=False)
    del state
    tuned = evaluate(model, data["development"], encoder, labels, args, event, "tuned")
    result = {"status": "complete", "scope": config["scope"], "train_records": len(data["train"]),
              "training_seconds": training_seconds, "optimizer_steps": update, "baseline": baseline, "tuned": tuned,
              "accuracy_difference": tuned["accuracy"]-baseline["accuracy"], "checkpoint": str(checkpoint),
              "gradient_verified": True, "checkpoint_reloaded": True, "test_used": False, "calibration_used": False,
              "limitations": ["development diagnostic, not test accuracy", "single service request only, multi-request frames reserved",
                              "uncalibrated probabilities", "native Chinese crowd-collected travel services, not production e-commerce logs",
                              "full 8k budget, no claim of 8k long-context quality"]}
    write_json(args.output / "results.json", result)
    event("complete", train_records=len(data["train"]), development_records=len(data["development"]),
          baseline_accuracy=baseline["accuracy"], tuned_accuracy=tuned["accuracy"], training_seconds=training_seconds,
          checkpoint=str(checkpoint), result=str(args.output / "results.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=LAB / "data/customer-intent-zh-v2")
    parser.add_argument("--output", type=Path, default=LAB / "results/customer-intent-zh-pilot-v2")
    parser.add_argument("--weights", type=Path, default=LAB / "checkpoints/banking77-lora-v1/base")
    parser.add_argument("--train-per-class", type=int, default=8)
    parser.add_argument("--dev-per-class", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--memory-limit-gib", type=float, default=24)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch, args.accumulation, args.rank) < 1 or min(args.train_per_class, args.dev_per_class) < 0:
        parser.error("轮次、批量、累积和秩须为正，样本上限不能为负")
    args.device, args.mode, args.seq_len, args.shape_bucket = "mps", "lora", 8192, 64
    args.gradient_checkpointing = args.checkpoint_head = True
    args.output.mkdir(parents=True, exist_ok=False)
    event = Events(args.output)
    try:
        run(args, event)
    except BaseException as error:
        event("failed", error_type=type(error).__name__, error=str(error))
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise
