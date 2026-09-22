"""本机浏览器 Choice 试训：固定训练小样本，一轮更新，完整开发集前后对照。"""

import argparse
import copy
import gc
import gzip
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

from benchmark_training import load_model, memory_sample, hardware_info, synchronize
from computer_replay import freeze, model_input, normalize_answer, read_records, serialized, summarize, write_records
from prepare_computer_replay import requests_for
from prepare_computer_use import digest, retrieve, stable, write_json
from train_banking77 import save_checkpoint
from train_computer_grounding import Events, grounding_batch
from typed_laya import encode_question

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

LAB = Path(__file__).resolve().parents[1]
GROUND = LAB / "data/computer-use-grounding-v1"
REPLAY = LAB / "data/computer-use-replay-v1"
BASE_SHA = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"


def supervised_candidates(frame, limit, seed):
    """仅训练集允许保留标注正例；开发输入完全复用此前冻结的无答案检索结果。"""
    if frame["split"] != "train":
        raise ValueError("有监督候选采样只能用于训练集")
    gold = set(frame["gold_node_ids"])
    positives = [c for c in frame["candidates"] if c["node_id"] in gold]
    if not positives or len(positives) >= limit:
        raise ValueError("正例缺失或无法留下负例")
    ranked = retrieve(frame["candidates"], frame["state"]["task"], len(frame["candidates"]))
    selected = positives + [c for c in ranked if c["node_id"] not in gold][:limit-len(positives)]
    random.Random(stable(f"{seed}:{frame['id']}")).shuffle(selected)
    return selected


def positive_mass_loss(logits, gold_indices):
    """多个人工正确目标共享概率质量，避免任意挑一个正例排斥其余标注。"""
    if not gold_indices or len(set(gold_indices)) != len(gold_indices):
        raise ValueError("正例索引为空或重复")
    if min(gold_indices) < 0 or max(gold_indices) >= logits.numel():
        raise ValueError("正例索引越界")
    return torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[gold_indices], dim=0)


def prepare(output, tokenizer, seed):
    manifest = json.loads((GROUND / "manifest.json").read_text())
    for name in ("train_frames.jsonl.gz", "split_assignments.json"):
        if digest(GROUND / name) != manifest["files"][name]["sha256"]:
            raise ValueError("原训练文件或划分校验失败")
    replay_manifest = json.loads((REPLAY / "manifest.json").read_text())
    if digest(REPLAY / "development.jsonl") != replay_manifest["files"]["development.jsonl"]["sha256"]:
        raise ValueError("冻结开发输入校验失败")
    assignments = {r["trajectory_id"]: r for r in json.loads((GROUND / "split_assignments.json").read_text())}
    inspection = LAB / "data/browser-source-pilot-v1/mind2web-inspection.jsonl"
    source_ids = {r["id"] for r in (json.loads(line) for line in inspection.read_text().splitlines())}
    with gzip.open(GROUND / "train_frames.jsonl.gz", "rt") as stream:
        frames = [json.loads(line) for line in stream]
    selected = sorted((f for f in frames if f["id"] in source_ids), key=lambda f: f["id"])
    if len(selected) != 64:
        raise ValueError("预先固定的小样本不是 64 步")
    train_rows, encoded = [], []
    for frame in selected:
        if frame["split"] != "train" or assignments[frame["trajectory_id"]]["split"] != "train":
            raise ValueError("非训练轨迹混入")
        if len(frame["state"]["previous_actions"]) != frame["step_index"]:
            raise ValueError("历史步骤不一致")
        state = {"goal": frame["state"]["task"], "website": frame["website"],
                 "previous_actions": frame["state"]["previous_actions"][-8:],
                 "page_context": frame["state"]["page_text"]}
        candidates = supervised_candidates(frame, 64, seed)
        request, refs = requests_for(state, candidates)
        gold = {"operation": [frame["gold_operation"]],
                "item": [key for key, node in refs.items() if node in frame["gold_node_ids"]]}
        row = freeze(frame["id"], request, {"split": "train", "website": frame["website"],
                     "trajectory_id": frame["trajectory_id"], "supervised_candidate_sampling": True},
                     {"gold_choices": gold, "target_recalled": True,
                      "label_source": "Mind2Web human annotation", "candidate_references": refs})
        train_rows.append(row)
        for name, q in request["questions"].items():
            # 对操作及目标选项均采用固定随机排列，不把正确项放在固定位置。
            order = list(range(len(q["criteria"])))
            random.Random(stable(f"{seed}:{frame['id']}:{name}")).shuffle(order)
            item, info = encode_question(tokenizer, state, q, 8192, order=order)
            encoded.append({"id": frame["id"] + ":" + name, "item": item,
                            "gold": [info["labels"].index(key) for key in gold[name]],
                            "question": name, "input_tokens": info["input_tokens"]})
    development = read_records(REPLAY / "development.jsonl")
    for row in development:
        metadata = row["metadata"]
        a = assignments[metadata["trajectory_id"]]
        if metadata["split"] != "development" or any(metadata[k] != a[k] for k in ("split", "website", "group_id")):
            raise ValueError("开发划分不一致")
    if {r["metadata"]["trajectory_id"] for r in train_rows} & {r["metadata"]["trajectory_id"] for r in development}:
        raise ValueError("训练开发轨迹交叉")
    if {r["metadata"]["website"] for r in train_rows} & {r["metadata"]["website"] for r in development}:
        raise ValueError("训练开发网站交叉")
    write_records(output / "train.jsonl", train_rows)
    write_records(output / "development.jsonl", development)
    write_json(output / "training_encoding.json", [{k: v for k, v in r.items() if k != "item"} for r in encoded])
    return encoded, development


@torch.inference_mode()
def evaluate(model, tokenizer, rows, args, name, event):
    model.eval()
    predictions = []
    with (args.output / f"{name}_predictions.jsonl").open("x", buffering=1) as stream:
        for row in rows:
            request = model_input(row)
            synchronize(args.device)
            start = time.perf_counter()
            answers, lengths = {}, {}
            for key, question in request["questions"].items():
                item, info = encode_question(tokenizer, request["state"], question, args.seq_len)
                batch = grounding_batch([item], tokenizer, args.device, 256)
                logits, _ = model(**batch)
                values = logits[0, :len(info["labels"])].float().cpu().double()
                if not torch.isfinite(values).all():
                    raise FloatingPointError("开发集输出非有限")
                probabilities = torch.softmax(values, dim=0).tolist()
                answer = {"type": "choice", "choice": info["labels"][int(values.argmax())],
                          "probabilities": dict(zip(info["labels"], probabilities))}
                answers[key] = normalize_answer(answer, question)
                lengths[key] = info["input_tokens"]
                del batch, logits
            synchronize(args.device)
            result = {"id": row["id"], "request_sha256": row["request_sha256"], "model": name,
                      "ok": True, "answers": answers, "input_tokens": lengths,
                      "elapsed_ms": (time.perf_counter()-start)*1000}
            stream.write(serialized(result) + "\n")
            predictions.append(result)
            if len(predictions) % 8 == 0 or len(predictions) == len(rows):
                event("evaluate_" + name, completed=len(predictions), total=len(rows), memory=memory_sample(args.device))
            # 不让不同长序列形状的 MPS 缓存随开发页数量持续积累。
            torch.mps.empty_cache()
    summary = summarize(rows, predictions)
    write_json(args.output / f"{name}_metrics.json", summary)
    return summary, predictions


def run(args, event):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS 不可用，未回退 CPU")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.mps.set_per_process_memory_fraction(min(.8, 24*2**30/torch.mps.recommended_max_memory()))
    args.weights = LAB / "checkpoints/banking77-lora-v1/base"
    if digest(args.weights / "model.safetensors") != BASE_SHA:
        raise ValueError("原始 Laya 权重校验失败")
    tokenizer = AutoTokenizer.from_pretrained(args.weights / "tokenizer", local_files_only=True)
    event("preparing_data")
    encoded, development = prepare(args.output, tokenizer, args.seed)
    lengths = [r["input_tokens"] for r in encoded]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(base_model_sha256=BASE_SHA, hardware=hardware_info(args.device), train_frames=64,
                  train_questions=len(encoded), development_frames=len(development),
                  input_tokens={"min": min(lengths), "median": sorted(lengths)[len(lengths)//2], "max": max(lengths)},
                  supervised_candidate_sampling="train only: all available positives plus BM25 negatives; shuffled options",
                  evaluation_candidate_sampling="unchanged frozen BM25 top64; no positive insertion",
                  protocol="computer-use-replay-v1 operation + item; SELECT control only, not option value",
                  ultrafast_protocol_complete=False, calibration_used=False, holdout_used=False,
                  teacher_calls=0, ui_actions=0, epochs=1, batch_size=1, accumulation=4,
                  train_input_sha256=digest(args.output / "train.jsonl"),
                  dev_input_sha256=digest(args.output / "development.jsonl"),
                  split_sha256=digest(GROUND / "split_assignments.json"),
                  pilot_selection="pre-existing 64-row inspection sample, stratified by operation; no dev result selection")
    source = args.output / "source"
    source.mkdir()
    sources = [Path(__file__), *[LAB / "scripts" / name for name in (
        "prepare_computer_use.py", "prepare_computer_replay.py", "computer_replay.py", "typed_laya.py",
        "train_computer_grounding.py", "train_banking77.py", "benchmark_training.py")], LAB / "reference/laya_common.py"]
    config["code_sha256"] = {p.name: digest(p) for p in sources}
    for p in sources:
        (source / p.name).write_bytes(p.read_bytes())
    write_json(args.output / "run_config.json", config)
    event("loading_model", train_questions=len(encoded), input_tokens=config["input_tokens"])
    model, _ = load_model(args)
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    longest = max(encoded, key=lambda r: r["input_tokens"])
    event("preflight", input_tokens=longest["input_tokens"])
    batch = grounding_batch([longest["item"]], tokenizer, args.device, 256)
    start = time.perf_counter()
    logits, _ = model(**batch)
    loss = positive_mass_loss(logits[0, :len(longest["item"]["markers"])] , longest["gold"])
    if not torch.isfinite(loss):
        raise FloatingPointError("最长样本损失非有限")
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1, error_if_nonfinite=True)
    synchronize(args.device)
    preflight = {"status": "passed", "tokens": longest["input_tokens"], "loss": float(loss.detach().cpu()),
                 "forward_backward_seconds": time.perf_counter()-start, "memory": memory_sample(args.device),
                 "optimizer_updates": 0, "memory_scope": "sample after backward, not instantaneous peak"}
    write_json(args.output / "preflight.json", preflight)
    model.zero_grad(set_to_none=True)
    del logits, loss, batch
    torch.mps.empty_cache()
    save_checkpoint(model, args.output / "initial", config)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in parameters if n.startswith("encoder.")], "lr": 5e-5},
        {"params": [p for n, p in parameters if not n.startswith("encoder.")], "lr": 2e-5}],
        weight_decay=.01, foreach=False)
    probe_names = [next(n for n, _ in parameters if n.endswith("Wqkv.lora_B.default.weight")),
                   next(n for n, _ in parameters if n.endswith("scorer.3.weight"))]
    probes = {n: dict(parameters)[n] for n in probe_names}
    before = {n: p.detach().clone() for n, p in probes.items()}
    shuffled = list(encoded)
    random.Random(args.seed).shuffle(shuffled)
    total_updates = math.ceil(len(shuffled)/4)
    losses, elapsed_steps, memory = [], [], []
    start = time.perf_counter()
    event("training", optimizer_step=0, optimizer_steps_total=total_updates, completed_questions=0, total_questions=len(encoded))
    for offset in range(0, len(shuffled), 4):
        group = shuffled[offset:offset+4]
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        group_loss = 0.
        for row in group:
            batch = grounding_batch([row["item"]], tokenizer, args.device, 256)
            logits, _ = model(**batch)
            loss = positive_mass_loss(logits[0, :len(row["item"]["markers"])], row["gold"])
            if not torch.isfinite(loss):
                raise FloatingPointError("训练损失非有限")
            (loss / len(group)).backward()
            group_loss += float(loss.detach().cpu()) / len(group)
            del batch, logits, loss
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1., error_if_nonfinite=True)
        if offset == 0:
            for p in probes.values():
                if p.grad is None or not torch.isfinite(p.grad).all() or not torch.any(p.grad != 0):
                    raise RuntimeError("LoRA 或评分头梯度无效")
        optimizer.step()
        synchronize(args.device)
        if offset == 0:
            if any(torch.equal(before[n], p.detach()) for n, p in probes.items()):
                raise RuntimeError("优化器没有实际更新参数")
            write_json(args.output / "gradient_checks.json", {"finite_nonzero_gradients": True,
                       "lora_and_scorer_changed": True, "probes": probe_names})
            del before
        done = offset + len(group)
        losses.append(group_loss)
        elapsed_steps.append(time.perf_counter()-step_start)
        memory.append(memory_sample(args.device))
        elapsed = time.perf_counter()-start
        event("training", optimizer_step=offset//4+1, optimizer_steps_total=total_updates,
              completed_questions=done, total_questions=len(encoded), loss=group_loss,
              mean_loss=sum(losses)/len(losses), gradient_norm=float(norm), elapsed_seconds=elapsed,
              eta_training_seconds=(len(encoded)-done)*elapsed/done, memory=memory[-1])
        torch.mps.empty_cache()
    training_seconds = time.perf_counter()-start
    save_checkpoint(model, args.output / "epoch-1", {**config, "optimizer_step": total_updates,
                                                     "train_epoch_mean_loss": sum(losses)/len(losses)})
    write_json(args.output / "training_metrics.json", {"seconds": training_seconds, "questions": len(encoded),
               "optimizer_steps": total_updates, "losses": losses, "optimizer_step_seconds": elapsed_steps,
               "memory_samples": memory, "memory_scope": "after optimizer steps, not instantaneous peak"})
    del optimizer
    gc.collect()
    torch.mps.empty_cache()
    event("training_complete", training_seconds=training_seconds, next_phase="evaluate_tuned")
    tuned, tuned_rows = evaluate(model, tokenizer, development, args, "tuned", event)
    initial = args.output / "initial"
    initial_config = json.loads((initial / "config.json").read_text())
    if digest(initial / "trainable.safetensors") != initial_config["checkpoint_sha256"]:
        raise ValueError("初始检查点校验失败")
    state = load_file(str(initial / "trainable.safetensors"))
    if set(state) != {n for n, _ in parameters}:
        raise ValueError("初始可训练参数集合不完整")
    model.load_state_dict(state, strict=False)
    del state
    baseline, baseline_rows = evaluate(model, tokenizer, development, args, "baseline", event)
    write_json(args.output / "results.json", {"status": "complete", "training_seconds": training_seconds,
               "optimizer_steps": total_updates, "baseline": baseline, "tuned": tuned,
               "paired_inputs_identical": True, "closed_loop_success_measured": False,
               "holdout_used": False, "note": "single small pilot; development results are not final test accuracy"})
    with (args.output / "cases.jsonl").open("x") as stream:
        for row, b, t in zip(development, baseline_rows, tuned_rows):
            if row["id"] != b["id"] or row["id"] != t["id"]:
                raise ValueError("逐题对齐失败")
            stream.write(serialized({**row, "predictions": {"baseline": b, "tuned": t}}) + "\n")
    event("complete", training_seconds=training_seconds, results=str(args.output / "results.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.device, args.mode, args.rank, args.seq_len = "mps", "lora", 16, 8192
    args.gradient_checkpointing = args.checkpoint_head = True
    event = Events(args.output)
    try:
        run(args, event)
    except Exception as exc:
        event("failed", error_type=type(exc).__name__, error=str(exc)[:700])
        raise


if __name__ == "__main__":
    main()
