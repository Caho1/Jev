"""在本机冻结权重上评测 MInDS-14 中文与原数据英文译文。"""

import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from banking77_data import INSTRUCTION, normalize, sha256, write_json
from local_laya import LocalLaya, MODEL

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "data/minds14-zh-v1"
OUTPUT = LAB / "results/minds14-zh-mps-v1"


def status(phase, **extra):
    state = {"phase": phase, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}
    temporary = OUTPUT / "status.tmp"
    write_json(temporary, state)
    temporary.replace(OUTPUT / "status.json")
    print(json.dumps(state, ensure_ascii=False), flush=True)


def measure(logits, gold, count):
    prediction = logits.argmax(1)
    matrix = np.zeros((count, count), dtype=int)
    np.add.at(matrix, (gold, prediction), 1)
    tp = matrix.diagonal()
    f1 = 2 * tp / np.maximum(1, matrix.sum(0) + matrix.sum(1))
    return {"records": len(gold), "correct": int((prediction == gold).sum()),
            "accuracy": float((prediction == gold).mean()), "macro_f1": float(f1.mean()),
            "top3_accuracy": float(np.mean([y in options for y, options in zip(gold, np.argsort(-logits, axis=1)[:, :3])])),
            "confusion": matrix.tolist(), "support": matrix.sum(1).tolist(),
            "per_class_recall": (tp / np.maximum(1, matrix.sum(1))).tolist()}


def language_gap(rows, zh, en):
    gold = np.array([r["label"] for r in rows])
    delta = (zh.argmax(1) == gold).astype(float) - (en.argmax(1) == gold).astype(float)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(normalize(row["text_zh"]), []).append(i)
    totals = np.array([delta[indices].sum() for indices in groups.values()])
    counts = np.array([len(indices) for indices in groups.values()])
    rng = np.random.default_rng(20260922)
    estimates = []
    for _ in range(2000):
        sampled = rng.integers(0, len(totals), len(totals))
        estimates.append(totals[sampled].sum() / counts[sampled].sum())
    return {"zh_minus_en": float(delta.mean()), "group_bootstrap_95_percent": np.quantile(estimates, [.025, .975]).tolist(), "groups": len(groups)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "protocol.json").exists():
        raise FileExistsError("该评测已开始；保留原始记录，不覆盖或按结果修改协议")
    manifest = json.loads((DATA / "manifest.json").read_text())
    for relative, info in manifest["files"].items():
        assert sha256(DATA / relative) == info["sha256"]
    rows = [json.loads(line) for line in (DATA / "evaluation.jsonl").read_text().splitlines()]
    criteria = json.loads((DATA / "criteria.json").read_text())
    labels = list(criteria)
    gold = np.array([r["label"] for r in rows])
    assert len(rows) == 502 and all(labels[r["label"]] == r["label_name"] for r in rows)
    cfg = json.loads((MODEL / "best/config.json").read_text())
    protocol = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "dataset": manifest, "instruction": INSTRUCTION, "criteria": criteria,
                "device": "mps", "precision": "fp32", "batch_size": 4, "context_limit": 8192,
                "temperature": 1.0, "calibration": "不对新数据调参或拟合温度；预测概率未校准。",
                "inputs": ["text_zh", "text_en"], "variants": ["baseline", "tuned"],
                "purpose": "external evaluation only", "gold_in_model_inputs": False,
                "base_sha256": cfg["base_model_sha256"], "checkpoint_sha256": cfg["checkpoint_sha256"],
                "code_sha256": {name: sha256(LAB / "scripts" / name) for name in ["local_laya.py", "evaluate_minds14_local.py", "prepare_minds14_zh.py"]},
                "comparability": "两模型、两种语言共享14候选及英文定义；这不是BANKING77的77类测试。英文译文来自数据源，并非在线翻译调用。"}
    write_json(OUTPUT / "protocol.json", protocol)
    start = time.perf_counter()
    results = {"complete": False, "models": {}, "controls": {}, "hardware": {"platform": platform.platform(), "torch": torch.__version__, "device": "mps"}}
    scores = {}
    raw_records = []
    completed = 0
    # 从已发布的英文测试记录选固定前16条，仅验证本地加载与远程已存预测是否一致。
    anchors = [json.loads(line) for line in (LAB / "data/banking77-v1/test.jsonl").read_text().splitlines()][:16]
    anchor_criteria = {name: "" for name in cfg["labels"]}
    for variant in protocol["variants"]:
        status("loading", model=variant, completed=completed, total=len(rows)*4)
        model = LocalLaya(variant, "mps")
        control = []
        for offset in range(0, len(anchors), 4):
            logits, _ = model.logits([r["state"] for r in anchors[offset:offset+4]], anchor_criteria)
            control.append(logits)
        control = np.concatenate(control)
        with np.load(LAB / f"results/banking77-lora-v1/{variant}-test.npz") as saved:
            reference = saved["logits"][:16]
        results["controls"][variant] = {"records": 16, "top1_agreement": int((control.argmax(1) == reference.argmax(1)).sum()),
                                          "max_logit_difference": float(np.max(np.abs(control-reference))),
                                          "note": "MPS FP32 对照此前 CUDA BF16，允许精度引起数值差异，不改变权重。"}
        if results["controls"][variant]["top1_agreement"] < 14:
            raise RuntimeError("本地与已保存英文预测的类别一致率过低，先检查加载")
        results["models"][variant] = {"load_seconds": model.load_seconds}
        for language in ("zh", "en"):
            parts, timings, lengths = [], [], []
            stage_start = time.perf_counter()
            for offset in range(0, len(rows), 4):
                batch = rows[offset:offset+4]
                logits, info = model.logits([row[f"text_{language}"] for row in batch], criteria)
                parts.append(logits)
                timings.append(info["forward_seconds"])
                lengths.extend(info["lengths"])
                completed += len(batch)
                if offset % 40 == 0 or offset+4 >= len(rows):
                    status("running", model=variant, language=language, completed=completed, total=len(rows)*4,
                           stage_completed=min(offset+4,len(rows)), stage_total=len(rows))
            elapsed = time.perf_counter() - stage_start
            logits = np.concatenate(parts)
            scores[(variant, language)] = logits
            np.savez_compressed(OUTPUT / f"{variant}-{language}.npz", ids=np.array([r["id"] for r in rows]), labels=gold, logits=logits)
            measured = measure(logits, gold, len(labels))
            measured.update(elapsed_seconds=elapsed, examples_per_second=len(rows)/elapsed,
                            forward_batch_p50_seconds=float(np.median(timings)), forward_batch_p95_seconds=float(np.quantile(timings,.95)),
                            token_min=min(lengths), token_median=float(np.median(lengths)), token_max=max(lengths), truncated=0)
            results["models"][variant][language] = measured
            z = logits.astype(float) - logits.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            for i, row in enumerate(rows):
                best = int(p[i].argmax())
                raw_records.append({"id": row["id"], "model": variant, "language": language,
                                    "prediction": labels[best], "correct": best == row["label"], "confidence": float(p[i,best]),
                                    "top3": [{"label": labels[int(k)], "probability": float(p[i,k])} for k in np.argsort(-p[i])[:3]]})
            write_json(OUTPUT / "results.partial.json", results)
        results["models"][variant]["language_gap"] = language_gap(rows, scores[(variant,"zh")], scores[(variant,"en")])
        del model
        gc.collect()
        torch.mps.empty_cache()
    results.update(complete=True, wall_seconds=time.perf_counter()-start, completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (OUTPUT / "predictions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in raw_records))
    write_json(OUTPUT / "results.json", results)
    status("complete", completed=completed, total=len(rows)*4, elapsed_seconds=results["wall_seconds"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if OUTPUT.exists():
            status("failed", error=f"{type(error).__name__}: {error}")
        raise
