"""用独立开发/校准/测试划分微调 Laya；所有注释与运行说明使用中文。"""

import argparse
import gc
import json
import math
import random
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from banking77_data import BankingEncoder, sha256, write_json
from benchmark_training import load_model, hardware_info, SOURCE_REVISION, REVISION
from laya_common import collate_items


def status(output, phase, **values):
    value = {"phase": phase, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **values}
    # 原子替换当前状态，并保留事件序列供看板绘制曲线。
    temporary = output / "status.json.tmp"
    write_json(temporary, value)
    temporary.replace(output / "status.json")
    with (output / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def verify_data(path):
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(path / name) != expected["sha256"]:
            raise ValueError(f"数据文件校验失败：{name}")
    labels = json.loads((path / "labels.json").read_text())
    data = {}
    seen = set()
    for split in ("train", "development", "calibration", "test"):
        rows = [json.loads(line) for line in (path / f"{split}.jsonl").read_text().splitlines()]
        groups = {r["group_id"] for r in rows}
        if seen & groups or {r["label"] for r in rows} != set(range(77)):
            raise ValueError("分组交叉或某个划分缺少类别")
        if any(labels[r["label"]] != r["label_name"] or r["split"] != split for r in rows):
            raise ValueError("标签映射或划分字段不一致")
        seen |= groups
        data[split] = rows
    return manifest, labels, data


def tokenize_data(data, encoder):
    summary = {}
    for split, rows in data.items():
        lengths = []
        for row in rows:
            row["_tokens"] = encoder.tokenize_state(row["state"])
            item = encoder.encode(row["_tokens"], row["label"])
            if len(item["markers"]) != 77:
                raise ValueError("候选数不是 77")
            row["_length"] = len(item["ids"])
            lengths.append(row["_length"])
        summary[split] = {"records": len(rows), "min": min(lengths), "median": float(np.median(lengths)),
                          "p95": float(np.percentile(lengths, 95)), "max": max(lengths), "truncated": 0,
                          "buckets": {"0_512": sum(n <= 512 for n in lengths),
                                      "513_2048": sum(512 < n <= 2048 for n in lengths),
                                      "2049_4096": sum(2048 < n <= 4096 for n in lengths),
                                      "4097_8192": sum(4096 < n <= 8192 for n in lengths)}}
    return summary


def batch_indices(rows, batch_size, seed=None):
    indices = list(range(len(rows)))
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(indices)
        # 在随机大组内按长度排序，减少 padding；各批次仍随机呈现。
        pool = batch_size * 30
        indices = [j for start in range(0, len(indices), pool)
                   for j in sorted(indices[start:start+pool], key=lambda i: rows[i]["_length"])]
    batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
    if seed is not None:
        rng.shuffle(batches)
    return batches


def collate(rows, indices, encoder, device, rng=None):
    items = []
    for i in indices:
        row = rows[i]
        order = list(range(77))
        if rng is not None:
            rng.shuffle(order)
        items.append([encoder.encode(row["_tokens"], row["label"], order)])
    batch = collate_items(items, encoder.tok.pad_token_id)
    model_input = {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}
    return model_input, batch["label"].to(device)


def precision_context(args):
    return torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.precision == "bf16")


@torch.inference_mode()
def predict(model, rows, encoder, args):
    model.eval()
    parts = []
    for indices in batch_indices(rows, args.eval_batch):
        batch, _ = collate(rows, indices, encoder, args.device)
        with precision_context(args):
            logits, _ = model(**batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("评测出现非有限 logits")
        parts.append(logits.float().cpu().numpy())
    return np.concatenate(parts)


def metrics(logits, labels, temperature=1.0):
    z = logits.astype(np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    probs = np.exp(log_probs)
    predicted = probs.argmax(axis=1)
    correct = predicted == labels
    confidence = probs.max(axis=1)
    confusion = np.zeros((77, 77), dtype=int)
    np.add.at(confusion, (labels, predicted), 1)
    tp = np.diag(confusion)
    f1 = 2*tp / np.maximum(1, confusion.sum(axis=0) + confusion.sum(axis=1))
    recall = tp / np.maximum(1, confusion.sum(axis=1))
    bins = []
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            accuracy, conf = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += mask.mean() * abs(accuracy-conf)
            bins.append({"lower": float(lo), "upper": float(hi), "count": int(mask.sum()), "accuracy": accuracy, "confidence": conf})
    selective = {}
    for threshold in (0.5, 0.8, 0.9, 0.95, 0.99):
        mask = confidence >= threshold
        selective[str(threshold)] = {"coverage": float(mask.mean()), "count": int(mask.sum()),
                                     "accuracy": float(correct[mask].mean()) if mask.any() else None}
    return {"records": len(labels), "accuracy": float(correct.mean()), "macro_f1": float(f1.mean()),
            "top3_accuracy": float(np.mean([y in p for y,p in zip(labels, probs.argsort(axis=1)[:, -3:])])),
            "nll": float(-log_probs[np.arange(len(labels)), labels].mean()),
            "brier_sum_over_classes": float((np.square(probs).sum(axis=1) - 2*probs[np.arange(len(labels)), labels] + 1).mean()),
            "ece_15_bins_max_probability": float(ece), "temperature": float(temperature),
            "selective": selective, "reliability_bins": bins, "per_class_f1": f1.tolist(),
            "per_class_recall": recall.tolist(), "confusion_matrix": confusion.tolist()}


def temperature_fit(logits, labels):
    # 只最小化独立校准集的 NLL；网格及范围预先固定，测试集不参与。
    z = logits.astype(np.float64)
    def nll(t):
        scaled = z / t
        scaled -= scaled.max(axis=1, keepdims=True)
        return float((np.log(np.exp(scaled).sum(axis=1)) - scaled[np.arange(len(labels)), labels]).mean())
    grid = np.geomspace(0.05, 10.0, 201)
    index = min(range(len(grid)), key=lambda i: nll(grid[i]))
    refined = np.geomspace(grid[max(0,index-1)], grid[min(len(grid)-1,index+1)], 101)
    best = min(refined, key=nll)
    return float(best)


def save_checkpoint(model, path, config):
    path.mkdir(parents=True, exist_ok=True)
    trainable = {name: p.detach().cpu().contiguous() for name,p in model.named_parameters() if p.requires_grad}
    save_file(trainable, str(path / "trainable.safetensors"))
    write_json(path / "config.json", {**config, "checkpoint_sha256": sha256(path / "trainable.safetensors"),
                                      "saved_parameters": sum(p.numel() for p in trainable.values())})


def reload_checkpoint(args, path):
    config = json.loads((path / "config.json").read_text())
    if config["checkpoint_sha256"] != sha256(path / "trainable.safetensors"):
        raise ValueError("检查点校验失败")
    model, _ = load_model(args)
    state = load_file(str(path / "trainable.safetensors"))
    expected = {name for name,p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("保存的可训练参数集合不完整")
    model.load_state_dict(state, strict=False)
    return model


def labels_for(rows):
    return np.array([r["label"] for r in rows])


def save_predictions(path, rows, logits):
    np.savez_compressed(path, ids=np.array([r["id"] for r in rows]), labels=labels_for(rows), logits=logits)


def paired_interval(rows, base, tuned):
    # 按重复组重采样，避免把相似文本误当成独立证据。
    labels = labels_for(rows)
    delta = (tuned.argmax(axis=1) == labels).astype(float) - (base.argmax(axis=1) == labels)
    groups = {}
    for i,row in enumerate(rows):
        groups.setdefault(row["group_id"], []).append(i)
    sums = np.array([delta[idx].sum() for idx in groups.values()])
    counts = np.array([len(idx) for idx in groups.values()])
    rng = np.random.default_rng(20260921)
    estimates = []
    for _ in range(2000):
        indices = rng.integers(0,len(sums),len(sums))
        estimates.append(sums[indices].sum()/counts[indices].sum())
    return {"accuracy_difference": float(delta.mean()), "group_bootstrap_95_percent": np.quantile(estimates,[.025,.975]).tolist(), "resamples": 2000}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "run_config.json").exists():
        raise FileExistsError("运行目录已存在；不覆盖模型或重新读取测试集")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif args.precision != "fp32":
        raise ValueError("本实现的 BF16 训练只在 CUDA 使用")
    manifest, labels, data = verify_data(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.weights / "tokenizer", local_files_only=True)
    encoder = BankingEncoder(tokenizer, labels, args.seq_len)
    length_report = tokenize_data(data, encoder)
    write_json(args.output / "input_lengths.json", length_report)
    config = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(base_revision=REVISION, base_model_sha256=sha256(args.weights / "model.safetensors"),
                  source_revision=SOURCE_REVISION, labels=labels, data_manifest=manifest,
                  code_sha256={p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("banking77_data.py"), Path(__file__).with_name("benchmark_training.py")]},
                  hardware=hardware_info(args.device))
    write_json(args.output / "run_config.json", config)
    started = time.monotonic()
    status(args.output, "loading", lengths=length_report)
    model, _ = load_model(args)
    encoder_parameters = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head_parameters = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": encoder_parameters, "lr": args.lr},
                                  {"params": head_parameters, "lr": args.head_lr}], weight_decay=.01, foreach=False)
    if args.smoke:
        rows = data["train"][:args.batch_size]
        model.train()
        batch, target = collate(rows,list(range(len(rows))),encoder,args.device,random.Random(args.seed))
        with precision_context(args):
            logits, _ = model(**batch)
            loss = F.cross_entropy(logits,target)
        loss.backward()
        probes = {n:p for n,p in model.named_parameters() if n.endswith("Wqkv.lora_B.default.weight") or n == "scorer.3.weight"}
        before = {n:p.detach().clone() for n,p in probes.items()}
        if not probes or any(p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max()==0 for p in probes.values()):
            raise AssertionError("梯度检查未通过")
        optimizer.step()
        if any(torch.equal(before[n],p) for n,p in probes.items()):
            raise AssertionError("优化器没有改变检查参数")
        ref = predict(model,data["development"][:args.eval_batch],encoder,args)
        save_checkpoint(model,args.output / "best",config)
        del optimizer, model, probes, before, batch, logits, loss
        gc.collect()
        if args.device == "cuda": torch.cuda.empty_cache()
        restored = reload_checkpoint(args,args.output / "best")
        new = predict(restored,data["development"][:args.eval_batch],encoder,args)
        np.testing.assert_allclose(new,ref,rtol=0,atol=1e-5)
        status(args.output,"smoke_passed",checkpoint_max_abs_difference=float(np.abs(new-ref).max()))
        return
    status(args.output,"baseline_development")
    base_dev = predict(model,data["development"],encoder,args)
    baseline_metrics = metrics(base_dev,labels_for(data["development"]))
    write_json(args.output / "baseline-development.json",baseline_metrics)
    save_predictions(args.output / "baseline-development.npz",data["development"],base_dev)
    best_nll, best_epoch, best_dev = baseline_metrics["nll"], 0, base_dev
    save_checkpoint(model,args.output / "best",{**config,"selected_epoch":0,"development_nll":best_nll})
    count_batches = math.ceil(len(data["train"])/args.batch_size)
    updates_per_epoch = math.ceil(count_batches/args.accumulation)
    total_updates = updates_per_epoch*args.epochs
    warmup = max(1,int(total_updates*.05))
    def scale(step):
        return (step+1)/warmup if step < warmup else max(0.,(total_updates-step)/(total_updates-warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,scale)
    update = 0
    history = []
    for epoch in range(1,args.epochs+1):
        batches = batch_indices(data["train"],args.batch_size,args.seed+epoch)
        rng = random.Random(args.seed+1000*epoch)
        model.train()
        epoch_start = time.monotonic()
        loss_sum, sample_count = 0., 0
        for window_start in range(0,len(batches),args.accumulation):
            window = batches[window_start:window_start+args.accumulation]
            window_size = sum(map(len,window))
            optimizer.zero_grad(set_to_none=True)
            for indices in window:
                batch, target = collate(data["train"],indices,encoder,args.device,rng)
                with precision_context(args):
                    logits, _ = model(**batch)
                    loss = F.cross_entropy(logits,target,reduction="sum")
                if not torch.isfinite(loss): raise FloatingPointError("训练损失非有限")
                (loss/window_size).backward()
                loss_sum += float(loss.detach())
                sample_count += len(indices)
            norm = torch.nn.utils.clip_grad_norm_(encoder_parameters+head_parameters,1.0,error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            update += 1
            if update % 20 == 0 or window_start+args.accumulation >= len(batches):
                status(args.output,"training",epoch=epoch,epochs=args.epochs,update=update,total_updates=total_updates,
                       training_nll=loss_sum/sample_count,examples_per_second=sample_count/(time.monotonic()-epoch_start),
                       gradient_norm=float(norm))
        status(args.output,"development",epoch=epoch)
        dev = predict(model,data["development"],encoder,args)
        measured = metrics(dev,labels_for(data["development"]))
        write_json(args.output / f"epoch-{epoch}-development.json",measured)
        history.append({"epoch":epoch,"training_nll":loss_sum/sample_count,"development_accuracy":measured["accuracy"],
                        "development_macro_f1":measured["macro_f1"],"development_nll":measured["nll"]})
        if measured["nll"] < best_nll:
            best_nll,best_epoch,best_dev = measured["nll"],epoch,dev
            save_checkpoint(model,args.output / "best",{**config,"selected_epoch":epoch,"development_nll":best_nll})
            save_predictions(args.output / "best-development.npz",data["development"],dev)
        write_json(args.output / "history.json",history)
        status(args.output,"epoch_complete",**history[-1],best_epoch=best_epoch)
    del optimizer, scheduler, model, encoder_parameters, head_parameters, batch, logits, loss
    gc.collect()
    if args.device == "cuda": torch.cuda.empty_cache()
    status(args.output,"checkpoint_reload",selected_epoch=best_epoch)
    model = reload_checkpoint(args,args.output / "best")
    reloaded = predict(model,data["development"][:args.eval_batch],encoder,args)
    np.testing.assert_allclose(reloaded,best_dev[:args.eval_batch],rtol=0,atol=1e-5)
    status(args.output,"calibration",selected_epoch=best_epoch)
    tuned_cal = predict(model,data["calibration"],encoder,args)
    tuned_temperature = temperature_fit(tuned_cal,labels_for(data["calibration"]))
    # 模型及温度选定之后才运行测试；后续没有根据测试结果调整参数的代码路径。
    write_json(args.output / "test-evaluation-lock.json",{"selected_epoch":best_epoch,"temperature":tuned_temperature,
               "checkpoint_sha256":sha256(args.output / "best/trainable.safetensors"),"test_sha256":manifest["files"]["test.jsonl"]["sha256"]})
    status(args.output,"final_test",model="selected",selected_epoch=best_epoch)
    tuned_test = predict(model,data["test"],encoder,args)
    save_predictions(args.output / "tuned-calibration.npz",data["calibration"],tuned_cal)
    save_predictions(args.output / "tuned-test.npz",data["test"],tuned_test)
    del model
    gc.collect()
    if args.device == "cuda": torch.cuda.empty_cache()
    base_args = SimpleNamespace(**vars(args))
    base_args.mode = "full"
    model,_ = load_model(base_args)
    status(args.output,"final_test",model="baseline")
    base_cal = predict(model,data["calibration"],encoder,args)
    base_temperature = temperature_fit(base_cal,labels_for(data["calibration"]))
    base_test = predict(model,data["test"],encoder,args)
    save_predictions(args.output / "baseline-calibration.npz",data["calibration"],base_cal)
    save_predictions(args.output / "baseline-test.npz",data["test"],base_test)
    results = {"selected_epoch":best_epoch,"history":history,"baseline_development":baseline_metrics,
               "test":{},"accuracy_comparison":paired_interval(data["test"],base_test,tuned_test),
               "temperatures":{"baseline":base_temperature,"tuned":tuned_temperature},
               "wall_seconds":time.monotonic()-started,"checkpoint_reload_max_abs_difference":float(np.abs(reloaded-best_dev[:args.eval_batch]).max()),
               "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if args.device=="cuda" else None}
    for name, scores, temp in [("baseline",base_test,base_temperature),("tuned",tuned_test,tuned_temperature)]:
        results["test"][name] = {"raw":metrics(scores,labels_for(data["test"])),"calibrated":metrics(scores,labels_for(data["test"]),temp)}
    write_json(args.output / "results.json",results)
    write_json(args.output / "best/calibration.json",{"temperature":tuned_temperature,"source":"calibration split","confidence_definition":"maximum class softmax probability after temperature scaling; not a correctness guarantee"})
    status(args.output,"complete",selected_epoch=best_epoch,baseline_accuracy=results["test"]["baseline"]["raw"]["accuracy"],
           tuned_accuracy=results["test"]["tuned"]["raw"]["accuracy"],wall_seconds=results["wall_seconds"])


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=["cuda","mps"],default="cuda")
    p.add_argument("--precision",choices=["bf16","fp32"],default="bf16")
    p.add_argument("--seq-len",type=int,default=8192)
    p.add_argument("--mode",choices=["lora"],default="lora")
    p.add_argument("--rank",type=int,default=16)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--eval-batch",type=int,default=16)
    p.add_argument("--accumulation",type=int,default=4)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--head-lr",type=float,default=5e-5)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--memory-limit-gib",type=float,default=28.)
    p.add_argument("--gradient-checkpointing",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--checkpoint-head",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if min(args.batch_size,args.eval_batch,args.accumulation,args.epochs,args.seq_len)<1:
        p.error("训练批量、轮数和长度必须为正")
    return args


if __name__=="__main__":
    args=arguments()
    try:
        run(args)
    except Exception as exc:
        if args.output.exists(): status(args.output,"failed",error=str(exc))
        traceback.print_exc()
        raise SystemExit(1)
