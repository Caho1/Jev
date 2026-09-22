"""在 MPS 或 CUDA 微调 Laya 控件评分；训练与完整候选开发检查单独保存。"""

import argparse
import collections
import gc
import gzip
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

from benchmark_training import load_model, hardware_info, memory_sample, synchronize, REVISION, SOURCE_REVISION
from train_banking77 import batch_indices, save_checkpoint
from typed_laya import encode_question, model_batch
from prepare_computer_use import digest, stable, write_json
from laya_common import collate_items

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

LAB = Path(__file__).resolve().parents[1]


class Events:
    def __init__(self, output):
        self.output = output

    def __call__(self, phase, **values):
        row = {"phase": phase, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(), **values}
        temporary = self.output/"status.json.tmp"
        write_json(temporary, row)
        temporary.replace(self.output/"status.json")
        with (self.output/"events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def verify_inputs(data):
    manifest = json.loads((data/"manifest.json").read_text())
    # 本轮只打开训练和开发文件；校准和留出文件不参与本次运行。
    for name in ("train_pairs.jsonl", "train_frames.jsonl.gz", "development_frames.jsonl.gz", "split_assignments.json"):
        if digest(data/name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"数据校验失败：{name}")
    rows = [json.loads(line) for line in (data/"train_pairs.jsonl").read_text().splitlines()]
    train_frames = {r["id"]: r for r in read_gzip(data/"train_frames.jsonl.gz")}
    development = read_gzip(data/"development_frames.jsonl.gz")
    assignments = json.loads((data/"split_assignments.json").read_text())
    seen_sites, seen_tasks = set(), set()
    for split in ("train", "development", "calibration", "pilot_holdout"):
        subset = [a for a in assignments if a["split"] == split]
        sites = {a["website"] for a in subset}
        tasks = {a["trajectory_id"] for a in subset}
        if sites & seen_sites or tasks & seen_tasks:
            raise ValueError("网站或轨迹跨划分重叠")
        seen_sites |= sites
        seen_tasks |= tasks
    if not rows or not development:
        raise ValueError("训练或开发数据为空")
    question = rows[0]["question"]
    for row in rows:
        frame = train_frames[row["frame_id"]]
        node_id = row["id"].rsplit(":", 1)[-1]
        source_candidates = {c["node_id"]: c for c in frame["candidates"]}
        expected = "yes" if node_id in frame["gold_node_ids"] else "no"
        if row["split"] != "train" or frame["split"] != "train" or row["gold_choices"] != [expected]:
            raise ValueError("训练标签或划分不匹配")
        if row["state"] != {**frame["state"], "candidate": source_candidates[node_id]["element"]}:
            raise ValueError("训练输入与操作前观察不一致")
        if row["question"] != question or row["question"]["type"] != "noul":
            raise ValueError("控件评分题型不一致")
    for frame in list(train_frames.values()) + development:
        if len(frame["state"]["previous_actions"]) != frame["step_index"]:
            raise ValueError("历史包含缺失步骤或未来动作")
    return manifest, rows, development, question


def select_development(frames, per_site, seed):
    sites = collections.defaultdict(list)
    for frame in frames:
        sites[frame["website"]].append(frame)
    selected = []
    for site, subset in sorted(sites.items()):
        # 只用 ID 选页，不按标签、候选排名或模型表现选择容易的题。
        selected.extend(sorted(subset, key=lambda r: stable(f"{seed}:{r['id']}"))[:per_site])
    return selected


def encode_training(rows, tokenizer):
    lengths = []
    for row in rows:
        row["_encodings"] = []
        for order in ([0, 1], [1, 0]):
            item, info = encode_question(tokenizer, row["state"], row["question"], 8192, order=order)
            item["label"] = info["labels"].index(row["gold_choices"][0])
            row["_encodings"].append(item)
        row["_length"] = len(row["_encodings"][0]["ids"])
        if row["_length"] != row["input_tokens"]:
            raise ValueError("训练编码与数据准备时不一致")
        lengths.append(row["_length"])
    return {"min": min(lengths), "median": float(np.median(lengths)), "max": max(lengths), "truncated": 0}


def grounding_batch(items, tokenizer, device, shape_bucket):
    # 在 CPU 上补零掩码，再传到 MPS；将注意力形状限制在固定桶，减少不同长度造成的缓存增长。
    batch = collate_items([[item] for item in items], tokenizer.pad_token_id)
    length = batch["input_ids"].shape[1]
    padding = (-length) % shape_bucket
    if padding:
        batch["input_ids"] = F.pad(batch["input_ids"], (0, padding), value=tokenizer.pad_token_id)
        batch["attention_mask"] = F.pad(batch["attention_mask"], (0, padding), value=0)
    return {key: batch[key].to(device) for key in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}


def precision_context(args):
    return torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.precision == "bf16")


def empty_cache(device):
    (torch.cuda if device == "cuda" else torch.mps).empty_cache()


def preflight(model, tokenizer, rows, args, event):
    # 在任何正式参数更新前，先验证真实最长样本的前向/反向，以及补齐掩码的数值一致性。
    longest = sorted(rows, key=lambda row: -row["_length"])[:args.batch_size]
    items = [row["_encodings"][0] for row in longest]
    sample = rows[0]["_encodings"][0]
    model.eval()
    with torch.inference_mode():
        # 掩码等价性使用 FP32 检查，独立于后续 BF16 训练的舍入误差。
        original, _ = model(**model_batch([sample], tokenizer, args.device))
        padded, _ = model(**grounding_batch([sample], tokenizer, args.device, args.shape_bucket))
        difference = float((original-padded).abs().max().cpu())
        if not torch.allclose(original, padded, atol=1e-3, rtol=1e-4):
            raise ValueError(f"补齐掩码后模型输出不一致：最大差值 {difference}")
    del original, padded
    model.train()
    samples = []
    for attempt in range(2):
        batch = grounding_batch(items, tokenizer, args.device, args.shape_bucket)
        labels = torch.tensor([item["label"] for item in items], device=args.device)
        started = time.perf_counter()
        with precision_context(args):
            logits, _ = model(**batch)
            loss = F.cross_entropy(logits[:, :2], labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("最长样本预检损失非有限")
        after_forward = memory_sample(args.device)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
        synchronize(args.device)
        samples.append({"attempt": attempt+1, "seconds": time.perf_counter()-started,
                        "loss": float(loss.detach().cpu()), "gradient_norm": float(norm),
                        "after_forward": after_forward, "after_backward": memory_sample(args.device)})
        model.zero_grad(set_to_none=True)
        del batch, labels, logits, loss
        empty_cache(args.device)
    result = {"status": "passed", "longest_real_tokens": longest[0]["_length"],
              "batch_size": len(items), "shape_bucket": args.shape_bucket,
              "device": args.device, "training_precision": args.precision, "padding_check_precision": "fp32",
              "padding_max_absolute_logit_difference": difference,
              "samples": samples, "optimizer_updates": 0,
              "memory_note": "前向后及反向后采样，非步内峰值"}
    write_json(args.output/"preflight.json", result)
    event("preflight_passed", longest_real_tokens=longest[0]["_length"], batch_size=len(items),
          padding_max_absolute_logit_difference=difference, memory=memory_sample(args.device))
    torch.manual_seed(args.seed)
    return result


def rank_metrics(scores, node_ids, gold_node_ids):
    if len(scores) != len(node_ids) or len(set(node_ids)) != len(node_ids):
        raise ValueError("候选和评分不完整")
    if not all(math.isfinite(x) for x in scores):
        raise FloatingPointError("非有限候选分数")
    # 相同分数按稳定节点编号排序，不使用标签破除平局。
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], node_ids[i]))
    positions = [j+1 for j, i in enumerate(order) if node_ids[i] in set(gold_node_ids)]
    first = min(positions) if positions else None
    return {"target_rank": first, "reciprocal_rank": 1/first if first else 0.0,
            "gold_in_pool": bool(positions), **{f"recall_at_{k}": bool(first and first <= k) for k in (1, 8, 24, 64)}}


@torch.inference_mode()
def evaluate(model, tokenizer, frames, question, args, phase, event):
    model.eval()
    output_rows = []
    target = args.output/f"{phase}_predictions.jsonl"
    total_candidates = sum(len(f["candidates"]) for f in frames)
    done_candidates = 0
    run_started = time.perf_counter()
    with target.open("x") as stream:
        for frame_index, frame in enumerate(frames):
            started = time.perf_counter()
            encoded = []
            for candidate in frame["candidates"]:
                item, _ = encode_question(tokenizer, {**frame["state"], "candidate": candidate["element"]}, question, args.seq_len)
                encoded.append(item)
            node_ids = [c["node_id"] for c in frame["candidates"]]
            scores = np.zeros(len(encoded), dtype=np.float64)
            order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]["ids"]))
            last_report = time.perf_counter()
            for at in range(0, len(order), args.eval_batch):
                indices = order[at:at+args.eval_batch]
                batch = grounding_batch([encoded[i] for i in indices], tokenizer, args.device, args.shape_bucket)
                with precision_context(args):
                    logits, _ = model(**batch)
                values = (logits[:, 1].float()-logits[:, 0].float()).cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError("开发评测出现非有限输出")
                scores[indices] = values
                done_candidates += len(indices)
                if time.perf_counter()-last_report >= 20:
                    event(phase, pages_done=frame_index, pages_total=len(frames), candidates_done=done_candidates,
                          candidates_total=total_candidates, elapsed_seconds=time.perf_counter()-run_started)
                    last_report = time.perf_counter()
            synchronize(args.device)
            summary = rank_metrics(scores.tolist(), node_ids, frame["gold_node_ids"])
            row = {"frame_id": frame["id"], "website": frame["website"], "trajectory_id": frame["trajectory_id"],
                   "split": "development", "candidate_count": len(node_ids), "node_ids": node_ids,
                   "scores": scores.tolist(), "gold_node_ids": frame["gold_node_ids"], **summary,
                   "seconds": time.perf_counter()-started, "truncated": 0}
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
            stream.flush()
            output_rows.append(row)
            event(phase, pages_done=frame_index+1, pages_total=len(frames), candidates_done=done_candidates,
                  candidates_total=total_candidates, latest_rank=summary["target_rank"], elapsed_seconds=time.perf_counter()-run_started)
    result = {"pages": len(output_rows), "candidates": total_candidates, "candidate_subset": False,
              "selected_development_pages": True, "gold_in_pool": sum(r["gold_in_pool"] for r in output_rows),
              **{f"recall_at_{k}": sum(r[f"recall_at_{k}"] for r in output_rows)/len(output_rows) for k in (1, 8, 24, 64)},
              "mean_reciprocal_rank": float(np.mean([r["reciprocal_rank"] for r in output_rows])),
              "page_p50_seconds": float(np.median([r["seconds"] for r in output_rows])),
              "page_p95_seconds": float(np.quantile([r["seconds"] for r in output_rows], .95)),
              "elapsed_seconds": time.perf_counter()-run_started, "file": target.name,
              "sha256": digest(target), "rows": [{k: v for k, v in r.items() if k not in {"scores", "node_ids"}} for r in output_rows]}
    write_json(args.output/f"{phase}_metrics.json", result)
    return result


def run(args, event):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，未回退到 CPU")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU 不支持 BF16")
        capacity = torch.cuda.get_device_properties(0).total_memory
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        backend = torch.cuda
    else:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS 不可用，未回退到 CPU")
        if args.precision != "fp32":
            raise ValueError("本轮 MPS 训练保持 FP32")
        capacity = torch.mps.recommended_max_memory()
        backend = torch.mps
    fraction = args.memory_limit_gib*2**30/capacity
    if not 0 < fraction <= 1:
        raise ValueError("内存预算超过设备允许容量")
    backend.set_per_process_memory_fraction(fraction)
    event("validating_data")
    manifest, rows, development, question = verify_inputs(args.data)
    frames = select_development(development, args.eval_pages_per_site, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.weights/"tokenizer", local_files_only=True)
    lengths = encode_training(rows, tokenizer)
    base_hash = digest(args.weights/"model.safetensors")
    if base_hash != "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c":
        raise ValueError("基础模型校验值改变")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(base_revision=REVISION, source_revision=SOURCE_REVISION, base_model_sha256=base_hash,
                  hardware=hardware_info(args.device), train_pairs=len(rows), input_lengths=lengths,
                  data_manifest_sha256=digest(args.data/"manifest.json"), question=question,
                  development_frame_ids=[f["id"] for f in frames], development_candidates=sum(len(f["candidates"]) for f in frames),
                  development_selection="fixed SHA256 order per website, before any model evaluation",
                  calibration_used=False, holdout_used=False, model_selection="fixed epoch count candidate, no hyperparameter search",
                  labels=["no", "yes"], confidence_calibrated=False,
                  code_sha256={p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name("typed_laya.py"),
                      Path(__file__).with_name("benchmark_training.py"), Path(__file__).with_name("train_banking77.py"), LAB/"reference/laya_common.py"]})
    write_json(args.output/"run_config.json", config)
    # 每次运行保存实际代码副本，后续修复脚本不会破坏历史可追溯性。
    source_dir = args.output/"source"
    source_dir.mkdir()
    for name, expected in config["code_sha256"].items():
        path = LAB/("reference" if name == "laya_common.py" else "scripts")/name
        if digest(path) != expected:
            raise ValueError("归档期间代码发生变化")
        (source_dir/name).write_bytes(path.read_bytes())
    write_json(args.output/"development_selection.json", [{"id": f["id"], "website": f["website"],
                                                           "candidates": len(f["candidates"])} for f in frames])
    event("loading_model", train_pairs=len(rows), input_lengths=lengths)
    model, _ = load_model(args)
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    encoder_params = [p for name, p in params if name.startswith("encoder.")]
    head_params = [p for name, p in params if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}],
                                  weight_decay=.01, foreach=False)
    event("preflight", max_tokens=lengths["max"], batch_size=args.batch_size,
          gradient_checkpointing=args.gradient_checkpointing, checkpoint_head=args.checkpoint_head)
    preflight(model, tokenizer, rows, args, event)
    if args.preflight_only:
        event("complete", preflight_only=True, optimizer_updates=0, result=str(args.output/"preflight.json"))
        return
    # 保存原始可训练状态用于之后同一开发样本的基线比较；原始基础权重保持只读。
    save_checkpoint(model, args.output/"initial", {**config, "checkpoint_role": "before_computer_use_training"})
    synchronize(args.device)
    total_micro = math.ceil(len(rows)/args.batch_size)
    total_updates = math.ceil(total_micro/args.accumulation)*args.epochs
    warmup = max(1, math.ceil(total_updates*.05))
    optimizer.zero_grad(set_to_none=True)
    trained_rows, update = 0, 0
    rng = random.Random(args.seed)
    started = time.perf_counter()
    losses = []
    probe_name = next(name for name, _ in params if name.endswith("Wqkv.lora_B.default.weight"))
    probe = dict(params)[probe_name]
    head_probe = next(p for name, p in params if name.endswith("scorer.3.weight"))
    before = probe.detach().clone()
    gradient_verified = False
    event("training", epoch=1, epochs=args.epochs, completed_pairs=0, total_pairs=len(rows)*args.epochs,
          optimizer_steps_total=total_updates, trainable_parameters=sum(p.numel() for _, p in params))
    for epoch in range(1, args.epochs+1):
        model.train()
        batches = batch_indices(rows, args.batch_size, args.seed+epoch)
        epoch_loss_sum, epoch_count = 0.0, 0
        for group_start in range(0, len(batches), args.accumulation):
            group = batches[group_start:group_start+args.accumulation]
            group_size = sum(len(batch) for batch in group)
            lr_scale = (update+1)/warmup if update < warmup else max(.1, (total_updates-update)/(total_updates-warmup))
            for param_group, base_lr in zip(optimizer.param_groups, (args.lr, args.head_lr)):
                param_group["lr"] = base_lr*lr_scale
            group_loss = 0.0
            for indices in group:
                items = [rows[i]["_encodings"][rng.randrange(2)] for i in indices]
                batch = grounding_batch(items, tokenizer, args.device, args.shape_bucket)
                labels = torch.tensor([it["label"] for it in items], device=args.device)
                with precision_context(args):
                    logits, _ = model(**batch)
                    loss_sum = F.cross_entropy(logits[:, :2], labels, reduction="sum")
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError("训练损失非有限")
                (loss_sum/group_size).backward()
                value = float(loss_sum.detach().cpu())
                group_loss += value
                epoch_loss_sum += value
                epoch_count += len(indices)
                trained_rows += len(indices)
            grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0, error_if_nonfinite=True)
            if not gradient_verified:
                for parameter in (probe, head_probe):
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all() or not torch.any(parameter.grad != 0):
                        raise RuntimeError("编码器 LoRA 或决策头没有有效梯度")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            synchronize(args.device)
            if update % 25 == 0:
                empty_cache(args.device)
            if not gradient_verified:
                if torch.equal(before, probe.detach()):
                    raise RuntimeError("优化器未更新 LoRA 权重")
                gradient_verified = True
                del before
                event("gradient_verified", optimizer_step=update, completed_pairs=trained_rows,
                      encoder_lora_and_head_finite_nonzero_gradients=True, encoder_lora_weights_changed=True)
            losses.append(group_loss/group_size)
            if update == 1 or update % 5 == 0 or trained_rows == len(rows)*args.epochs:
                elapsed = time.perf_counter()-started
                event("training", epoch=epoch, epochs=args.epochs, optimizer_step=update,
                      optimizer_steps_total=total_updates, completed_pairs=trained_rows, total_pairs=len(rows)*args.epochs,
                      progress=trained_rows/(len(rows)*args.epochs), recent_loss=float(np.mean(losses[-10:])),
                      epoch_mean_loss=epoch_loss_sum/epoch_count, grad_norm=float(grad_norm), elapsed_seconds=elapsed,
                      pairs_per_second=trained_rows/elapsed, eta_training_seconds=(len(rows)*args.epochs-trained_rows)*elapsed/trained_rows,
                      lr=optimizer.param_groups[0]["lr"], memory=memory_sample(args.device))
            if update % args.checkpoint_every == 0:
                save_checkpoint(model, args.output/f"step-{update:04d}", {**config, "optimizer_step": update, "completed_pairs": trained_rows})
        save_checkpoint(model, args.output/f"epoch-{epoch}", {**config, "epoch": epoch, "optimizer_step": update,
                        "train_epoch_mean_loss": epoch_loss_sum/epoch_count})
    training_seconds = time.perf_counter()-started
    del optimizer
    gc.collect()
    empty_cache(args.device)
    event("training_complete", train_pairs_seen=trained_rows, optimizer_steps=update, training_seconds=training_seconds,
          checkpoint=str(args.output/f"epoch-{args.epochs}"), next_phase="evaluate_tuned")
    tuned = evaluate(model, tokenizer, frames, question, args, "evaluate_tuned", event)
    # 与同一次加载起点比较，避免新建随机 LoRA 或意外加载客服微调头。
    initial = args.output/"initial"
    initial_config = json.loads((initial/"config.json").read_text())
    if digest(initial/"trainable.safetensors") != initial_config["checkpoint_sha256"]:
        raise ValueError("原始状态校验失败")
    state = load_file(str(initial/"trainable.safetensors"))
    if set(state) != {name for name, _ in params}:
        raise ValueError("原始可训练参数不完整")
    model.load_state_dict(state, strict=False)
    del state
    baseline = evaluate(model, tokenizer, frames, question, args, "evaluate_baseline", event)
    result = {"status": "complete", "training_seconds": training_seconds, "train_pairs_seen": trained_rows,
              "optimizer_steps": update, "gradient_verified": gradient_verified, "baseline": baseline, "tuned": tuned,
              "checkpoint": str(args.output/f"epoch-{args.epochs}"),
              "limitations": ["four selected development pages by default, not all 78 pages", "no calibration or holdout evaluation",
                              "offline element ranking, not operation selection or end-to-end task completion", "probabilities are not calibrated"],
              "recall_at_24_difference": tuned["recall_at_24"]-baseline["recall_at_24"]}
    write_json(args.output/"results.json", result)
    event("complete", result=str(args.output/"results.json"), checkpoint=result["checkpoint"],
          baseline_recall_at_24=baseline["recall_at_24"], tuned_recall_at_24=tuned["recall_at_24"],
          training_seconds=training_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=LAB/"data/computer-use-grounding-v1")
    parser.add_argument("--output", type=Path, default=LAB/"results/computer-use-grounding-lora-v1")
    parser.add_argument("--weights", type=Path, default=LAB/"checkpoints/banking77-lora-v1/base")
    parser.add_argument("--device", choices=["mps", "cuda"], default="mps")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--memory-limit-gib", type=float, default=28)
    parser.add_argument("--eval-pages-per-site", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shape-bucket", type=int, choices=[1, 32, 64, 128], default=64)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch, args.accumulation, args.eval_pages_per_site, args.checkpoint_every) < 1:
        parser.error("轮次和批次配置必须大于零")
    args.mode, args.seq_len = "lora", 8192
    args.output.mkdir(parents=True, exist_ok=False)
    event = Events(args.output)
    try:
        run(args, event)
    except BaseException as exc:
        event("failed", error_type=type(exc).__name__, error=str(exc))
        (args.output/"failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
