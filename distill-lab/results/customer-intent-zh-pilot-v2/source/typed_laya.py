"""本地 Laya 的通用三题型编码；正文与选项完整保留，超限明确报错。"""

import json
import time

import numpy as np
import torch

from local_laya import LocalLaya
from laya_common import QTYPES, build_sequence, collate_items, render_options, serialize_state


def question_labels(question):
    kind, criteria = question["type"], question["criteria"]
    if kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice 至少需要两个候选")
        return list(criteria)
    if kind == "noul":
        if not isinstance(criteria, dict):
            raise ValueError("noul 的条件必须是对象")
        return ["no", "yes"]
    if kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score 至少需要两个有序等级")
        return [str(i) for i in range(len(criteria))]
    raise ValueError(f"未知题型：{kind}")


def encode_question(tokenizer, state, question, max_length=8192, legacy=False, order=None):
    labels = question_labels(question)
    q = {"t": question["type"], "ins": question["instructions"], "crit": question["criteria"]}
    order = list(range(len(labels))) if order is None else list(order)
    if sorted(order) != list(range(len(labels))):
        raise ValueError("候选排列必须完整且无重复")
    clean = lambda value: str(value).replace(tokenizer.mask_token, " ")
    token = lambda value: tokenizer(clean(value), add_special_tokens=False)["input_ids"]
    ids = [tokenizer.cls_token_id] + token(f"{q['t']} question: {q['ins']}") + [tokenizer.sep_token_id]
    options = render_options(q)
    markers = []
    for index in order:
        markers.append(len(ids))
        ids.extend([tokenizer.mask_token_id] + token(" " + options[index]))
    ids += [tokenizer.sep_token_id] + token(serialize_state(state)) + [tokenizer.sep_token_id]
    full_length = len(ids)
    if legacy:
        ids, markers = build_sequence(tokenizer, state, q, max_len=max_length, option_order=order)
    elif full_length > max_length:
        raise ValueError(f"完整输入 {full_length} tokens 超过预算 {max_length}，未截断")
    if len(markers) != len(labels):
        raise ValueError("编码遗失了候选")
    return {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "label": -1}, {
        "labels": [labels[i] for i in order], "input_tokens": len(ids), "full_input_tokens": full_length,
        "shortened": len(ids) < full_length, "qtype": q["t"],
    }


def model_batch(items, tokenizer, device):
    batch = collate_items([[item] for item in items], tokenizer.pad_token_id)
    return {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}


class TypedLaya(LocalLaya):
    @torch.inference_mode()
    def decide(self, state, question, max_length=8192, legacy=False, temperature=1.0):
        # 只接收任务输入；答案、解析、来源与 ID 均不传入模型。
        self.synchronize()
        started = time.perf_counter()
        item, info = encode_question(self.tokenizer, state, question, max_length, legacy)
        batch = model_batch([item], self.tokenizer, self.device)
        logits, _ = self.model(**batch)
        values = logits[0, :len(info["labels"])].float().cpu().numpy()
        self.synchronize()
        elapsed = time.perf_counter() - started
        if not np.isfinite(values).all():
            raise FloatingPointError("模型 logits 非有限")
        z = values.astype(float) / temperature
        p = np.exp(z-z.max()); p /= p.sum()
        return {**info, "prediction": info["labels"][int(p.argmax())], "logits": values.tolist(),
                "probabilities": dict(zip(info["labels"], p.tolist())), "confidence": float(p.max()),
                "latency_seconds": elapsed, "temperature": temperature}


def measure(records):
    if not records:
        return {"records": 0}
    valid = [r for r in records if r.get("ok")]
    ece = 0.0
    for index in range(10):
        rows = [r for r in valid if min(int(r["confidence"]*10),9) == index]
        if rows:
            ece += len(rows)/len(valid)*abs(np.mean([r["confidence"] for r in rows])-np.mean([r["correct"] for r in rows]))
    probability = [r for r in valid if r.get("gold_probs")]
    tvds = [0.5*sum(abs(r["probabilities"][k]-r["gold_probs"].get(k,0)) for k in r["probabilities"]) for r in probability]
    briers = [sum((p-float(k==r["expected"]))**2 for k,p in r["probabilities"].items()) for r in valid]
    return {"records":len(records), "correct":sum(r.get("correct",False) for r in records),
            "accuracy":sum(r.get("correct",False) for r in records)/len(records), "valid":len(valid),
            "failures":len(records)-len(valid), "ece_10_bins":float(ece) if valid else None,
            "nll":float(np.mean([-np.log(max(r["probabilities"].get(r["expected"],0),1e-12)) for r in valid])) if valid else None,
            "brier":float(np.mean(briers)) if valid else None, "probability_records":len(probability),
            "mean_tvd":float(np.mean(tvds)) if tvds else None,
            "p50_seconds":float(np.median([r["latency_seconds"] for r in valid])) if valid else None,
            "p95_seconds":float(np.quantile([r["latency_seconds"] for r in valid],.95)) if valid else None,
            "shortened":sum(r.get("shortened",False) for r in records),
            "max_input_tokens":max((r["input_tokens"] for r in valid), default=0)}
