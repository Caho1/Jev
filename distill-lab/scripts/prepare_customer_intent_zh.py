"""固定中文客服语料来源，按会话隔离划分，保留多诉求与电商待审数据。"""

import argparse
import collections
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from banking77_data import BankingEncoder, group_duplicates, normalize, sha256, write_json

LAB = Path(__file__).resolve().parents[1]
SOURCE = LAB / "research/modelscope-customer-intent-2026-09-22"
REVISION = "df82c9fdff91b9b130f2d6b89110d3870ba6260e"
EXPECTED = {
    "train": "b81f6a0845011bb4593d21dd58e2d76af4dd5d52e300ecc68cb8dee17ccc3cf8",
    "val": "b2b717990f6f0a1c98daaf5dc465e61d42f7de69c0d31e88e674275b1f34c94c",
    "test": "e57703894995211bba8638f130dc7df81e8992b71f3c05c10022ad23d6ca5310",
}
INSTRUCTION = "根据对话历史，判断当前用户提出的业务请求。只判断当前请求，不把用户提供的条件或此前已经处理的问题当成新的请求。选择最准确的一项。"
DOMAINS = {"酒店", "餐馆", "景点", "地铁", "出租"}


def stable(value):
    return hashlib.sha256(value.encode()).hexdigest()


def dump_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def request_keys(acts):
    """设施细项合并到设施咨询；条件 Inform 保留原标注，不伪装成用户请求。"""
    labels = set()
    for act, domain, slot, value in acts:
        if domain not in DOMAINS:
            continue
        if act == "Request":
            slot = "酒店设施" if slot.startswith("酒店设施-") else slot
            labels.add(f"{domain}/查询/{slot}")
        elif act == "Select":
            labels.add(f"{domain}/筛选/关联对象")
    return sorted(labels)


def criteria_for(key):
    domain, action, slot = key.split("/")
    if action == "筛选":
        return f"基于先前提到的其他领域对象，筛选与其关联的{domain}候选。"
    if slot == "名称":
        return f"寻找、推荐或确认满足条件的{domain}对象。"
    return f"询问{domain}的{slot}，而不是仅提供该属性作为筛选条件。"


def render_state(history, current):
    before = "\n".join(("用户：" if m["role"] == "usr" else "客服：") + m["content"] for m in history)
    return "此前对话：\n" + (before or "（无）") + "\n\n当前用户：\n" + current


def frames(dialogue_id, dialogue, source_split, split):
    history = []
    for turn, message in enumerate(dialogue["messages"]):
        if message["role"] not in {"usr", "sys"}:
            raise ValueError("未知说话角色")
        if message["role"] == "usr":
            gold = request_keys(message["dialog_act"])
            yield {
                "id": f"crosswoz:{dialogue_id}:{turn}", "dialogue_id": str(dialogue_id),
                "turn_index": turn, "source_split": source_split, "split": split,
                "state": render_state(history, message["content"]),
                "gold_choices": gold, "original_dialog_acts": message["dialog_act"],
                "history_turns": len(history), "current_user": message["content"],
            }
        # 严格只复制已发生的文本；目标、状态、标注与未来回复都不进入输入。
        history.append({"role": message["role"], "content": message["content"]})


def encoder_for(tokenizer, labels, criteria, max_length=8192):
    encoder = BankingEncoder(tokenizer, labels, max_length)
    encoder.head = tokenizer("choice question: " + INSTRUCTION, add_special_tokens=False)["input_ids"]
    encoder.options = [[tokenizer.mask_token_id] + tokenizer(" " + label + ": " + criteria[label], add_special_tokens=False)["input_ids"] for label in labels]
    # CLS 以及题干、候选、正文之后的三个 SEP，一共四个特殊 token。
    encoder.prefix_length = 4 + len(encoder.head) + sum(map(len, encoder.options))
    return encoder


def ecommerce_review(source, destination):
    records = json.loads((source / "www2025-train.json").read_text())
    result = []
    for record in records:
        instruction = record["instruction"]
        if "多轮对话" not in instruction:
            continue
        match = re.search(r"<用户与客服的对话 START>\s*(.*?)\s*<用户与客服的对话 END>", instruction, re.S)
        if not match:
            raise ValueError("无法提取电商对话")
        turns = [{"role": "usr" if m.group(1) == "用户" else "sys", "content": m.group(2).strip()}
                 for m in re.finditer(r"(?:^|\n)(用户|客服)[:：]\s*(.*?)(?=\n(?:用户|客服)[:：]|\Z)", match.group(1), re.S)]
        user_indices = [i for i, turn in enumerate(turns) if turn["role"] == "usr"]
        if not user_indices:
            raise ValueError("电商记录没有用户轮次")
        last = user_indices[-1]
        result.append({"id": record["id"], "source": "smau0441/www2025-train", "original_label": record["output"],
                       "state": render_state(turns[:last], turns[last]["content"]),
                       "current_user": turns[last]["content"], "image_references": record["image"],
                       "removed_future_agent_turns": len(turns)-last-1,
                       "review_status": "pending_text_sufficiency_and_source_terms",
                       "approved_for_training": False})
    dump_rows(destination / "ecommerce_review.jsonl", result)
    return {"original_records": len(records), "intent_review_records": len(result),
            "labels": dict(collections.Counter(r["original_label"] for r in result)),
            "records_with_future_replies_removed": sum(r["removed_future_agent_turns"] > 0 for r in result),
            "approved_training_records": 0}


def prepare(args):
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    raw, sessions, sources = [], [], {}
    seen_ids = set()
    for source_split in ("train", "val", "test"):
        path = args.source / f"{source_split}-crosswoz.json.zip"
        if sha256(path) != EXPECTED[source_split]:
            raise ValueError("CrossWOZ 原始文件哈希改变")
        dialogues = json.loads(zipfile.ZipFile(path).read(f"{source_split}.json"))
        sources[source_split] = {"revision": REVISION, "sha256": sha256(path), "dialogues": len(dialogues),
                                "url": f"https://raw.githubusercontent.com/thu-coai/CrossWOZ/{REVISION}/data/crosswoz/{source_split}.json.zip"}
        # 按会话 ID 固定校准组，保留原 val/test 的来源边界。
        calibration = set(sorted(dialogues, key=lambda key: stable(f"{args.seed}:{key}"))[:round(len(dialogues)*.1)]) if source_split == "train" else set()
        for dialogue_id, dialogue in sorted(dialogues.items()):
            if dialogue_id in seen_ids:
                raise ValueError("原始会话 ID 跨划分重复")
            seen_ids.add(dialogue_id)
            split = ("calibration" if dialogue_id in calibration else "train") if source_split == "train" else ("development" if source_split == "val" else "test")
            sessions.append({"dialogue_id": dialogue_id, "source_split": source_split, "split": split})
            raw.extend(frames(dialogue_id, dialogue, source_split, split))
    labels = sorted({label for row in raw if row["split"] == "train" for label in row["gold_choices"]})
    criteria = {label: criteria_for(label) for label in labels}
    eligible = [r for r in raw if r["gold_choices"]]
    print(json.dumps({"phase": "duplicate_audit", "frames": len(eligible), "labels": len(labels)}), flush=True)
    duplicate_report = group_duplicates(eligible, args.seed)
    priority = {"train": 0, "calibration": 1, "development": 2, "test": 3}
    owner = {}
    for row in eligible:
        previous = owner.get(row["group_id"], "train")
        owner[row["group_id"]] = max((previous, row["split"]), key=priority.get)
    tokenizer = AutoTokenizer.from_pretrained(args.weights / "tokenizer", local_files_only=True)
    encoder = encoder_for(tokenizer, labels, criteria)
    # 批量分词只计算长度；禁止按长度选择更容易的基线样本。
    lengths = []
    for start in range(0, len(eligible), 256):
        tokens = tokenizer([r["state"].replace(tokenizer.mask_token, " ") for r in eligible[start:start+256]], add_special_tokens=False)["input_ids"]
        lengths.extend(encoder.prefix_length + len(t) for t in tokens)
    output = {s: [] for s in priority}
    multi, excluded = [], []
    for row, length in zip(eligible, lengths):
        row["input_tokens"] = length
        if row["split"] != owner[row["group_id"]]:
            excluded.append({"id": row["id"], "split": row["split"], "reason": "duplicate_group_in_higher_priority_split", "group_id": row["group_id"]})
        elif set(row["gold_choices"]) - set(labels):
            excluded.append({"id": row["id"], "split": row["split"], "reason": "unseen_request_schema"})
        elif length > 8192:
            excluded.append({"id": row["id"], "split": row["split"], "reason": "full_input_exceeds_8192", "input_tokens": length})
        elif len(row["gold_choices"]) > 1:
            multi.append(row)
        else:
            name = row["gold_choices"][0]
            output[row["split"]].append({**row, "label": labels.index(name), "label_name": name})
    for split, rows in output.items():
        dump_rows(args.output / f"{split}.jsonl", rows)
    dump_rows(args.output / "multiple_requests.jsonl", multi)
    dump_rows(args.output / "excluded.jsonl", excluded)
    dump_rows(args.output / "non_request_frames.jsonl", [r for r in raw if not r["gold_choices"]])
    write_json(args.output / "labels.json", labels)
    write_json(args.output / "criteria.json", criteria)
    write_json(args.output / "session_assignments.json", sessions)
    write_json(args.output / "duplicate_audit.json", duplicate_report)
    ecom = ecommerce_review(args.source, args.output)
    summary = {}
    seen_groups, seen_dialogues = set(), set()
    for split, rows in output.items():
        groups, dialogues = {r["group_id"] for r in rows}, {r["dialogue_id"] for r in rows}
        if groups & seen_groups or dialogues & seen_dialogues:
            raise AssertionError("划分仍有会话或重复组交叉")
        seen_groups |= groups
        seen_dialogues |= dialogues
        sizes = [r["input_tokens"] for r in rows]
        summary[split] = {"records": len(rows), "dialogues": len(dialogues), "labels": dict(collections.Counter(r["label_name"] for r in rows)),
                          "input_tokens": {"min": min(sizes), "median": float(np.median(sizes)), "p95": float(np.quantile(sizes, .95)), "max": max(sizes)}}
    manifest = {"schema_version": 1, "seed": args.seed, "source": "CrossWOZ", "sources": sources,
                "task": "current_user_single_service_request", "instruction": INSTRUCTION,
                "label_count": len(labels), "max_context_tokens": 8192, "state_contains_annotations": False,
                "splits": summary, "multiple_request_frames": len(multi),
                "non_request_frames": sum(not r["gold_choices"] for r in raw),
                "excluded": dict(collections.Counter(r["reason"] for r in excluded)),
                "split_policy": "official val and test boundaries; hash-selected 10% original training sessions for calibration; duplicate group priority test > development > calibration > train",
                "duplicate_method": duplicate_report["near_duplicate_method"], "ecommerce": ecom,
                "code_sha256": {Path(__file__).name: sha256(__file__), "banking77_data.py": sha256(Path(__file__).with_name("banking77_data.py"))},
                "files": {f.name: {"sha256": sha256(f), "bytes": f.stat().st_size} for f in sorted(args.output.iterdir()) if f.is_file()}}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"phase": "complete", "label_count": len(labels), "splits": {s: v["records"] for s, v in summary.items()},
                      "multi": len(multi), "excluded": manifest["excluded"], "ecommerce": ecom}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=LAB / "data/customer-intent-zh-v2")
    parser.add_argument("--weights", type=Path, default=LAB / "checkpoints/banking77-lora-v1/base")
    parser.add_argument("--seed", type=int, default=20260922)
    prepare(parser.parse_args())
