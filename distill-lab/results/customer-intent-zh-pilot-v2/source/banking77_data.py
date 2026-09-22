"""BANKING77 的来源固定、重复分组、隔离划分和无候选截断编码。"""

import argparse
import csv
import hashlib
import io
import json
import random
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SOURCE_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_ROOT = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{SOURCE_REVISION}"
INSTRUCTION = "Choose the banking customer service intent that best matches the customer message."


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def normalize(text):
    return re.sub(r"\W+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def group_duplicates(records, seed=42):
    """精确归一化匹配，加字符五元组 MinHash 候选和 Jaccard 复核。"""
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)

    exact, texts, pairs = {}, [], []
    for i, record in enumerate(records):
        text = normalize(record["state"])
        if not text:
            raise ValueError("存在空文本")
        texts.append(text)
        if text in exact:
            join(i, exact[text])
        else:
            exact[text] = i
    rng = np.random.default_rng(seed)
    prime = np.uint64(4294967311)
    a = rng.integers(1, 2**31, 32, dtype=np.uint64)
    b = rng.integers(0, 2**31, 32, dtype=np.uint64)
    bands, grams_cache = defaultdict(list), {}
    # LSH 只生成候选；是否合组由真实 Jaccard >= 0.90 决定。
    for text, i in exact.items():
        if len(text) < 30:
            continue
        grams = {text[k:k + 5] for k in range(len(text) - 4)}
        grams_cache[i] = grams
        hashes = np.array([int.from_bytes(hashlib.blake2s(g.encode(), digest_size=4).digest(), "little")
                           for g in grams], dtype=np.uint64)
        signature = ((hashes[:, None] * a + b) % prime).min(axis=0)
        candidates = set()
        keys = [(j, tuple(int(x) for x in signature[j*4:j*4+4])) for j in range(8)]
        for key in keys:
            candidates.update(bands[key])
        for other in sorted(candidates):
            g2 = grams_cache[other]
            if min(len(grams), len(g2)) / max(len(grams), len(g2)) < 0.9:
                continue
            score = len(grams & g2) / len(grams | g2)
            if score >= 0.9:
                join(i, other)
                pairs.append({"left": records[other]["id"], "right": records[i]["id"], "jaccard": score})
        for key in keys:
            bands[key].append(i)
    components = defaultdict(list)
    for i in range(len(records)):
        components[root(i)].append(i)
    for indices in components.values():
        key = "\n".join(sorted({texts[i] for i in indices}))
        group_id = hashlib.sha256(key.encode()).hexdigest()
        for i in indices:
            records[i]["group_id"] = group_id
    return {"normalized_unique_texts": len(exact), "duplicate_components": sum(len(v) > 1 for v in components.values()),
            "near_duplicate_pairs": pairs, "near_duplicate_method": "32-permutation MinHash, 8 bands x 4, char-5-gram Jaccard >= 0.90, normalized length >= 30; candidate search is approximate"}


def partition(records, seed=42):
    """官方测试集保持原样；与测试有重合的训练组全部排除。"""
    test = [dict(r, split="test") for r in records if r["source_split"] == "test"]
    test_groups = {r["group_id"] for r in test}
    groups = defaultdict(list)
    for r in records:
        if r["source_split"] == "train":
            groups[r["group_id"]].append(r)
    by_label, excluded = defaultdict(list), []
    for group_id, rows in sorted(groups.items()):
        reason = ("overlap_official_test" if group_id in test_groups else
                  "conflicting_train_labels" if len({r["label"] for r in rows}) != 1 else None)
        if reason:
            excluded.extend(dict(r, exclusion_reason=reason) for r in rows)
        else:
            by_label[rows[0]["label"]].append(rows)
    result = {"train": [], "development": [], "calibration": [], "test": test}
    for label, class_groups in sorted(by_label.items()):
        random.Random(seed + label).shuffle(class_groups)
        n = max(1, round(len(class_groups) * 0.1))
        if len(class_groups) <= 2 * n:
            raise ValueError("某类别没有足够独立样本可划分")
        for k, rows in enumerate(class_groups):
            split = "development" if k < n else "calibration" if k < 2*n else "train"
            result[split].extend(dict(r, split=split) for r in rows)
    for split, rows in result.items():
        rows.sort(key=lambda r: r["id"])
    seen = set()
    for rows in result.values():
        groups_here = {r["group_id"] for r in rows}
        if seen & groups_here:
            raise AssertionError("数据划分有分组交叉")
        seen |= groups_here
    return result, excluded


class BankingEncoder:
    """保留完整候选，按实际长度组批；超限报错，不静默删除证据。"""

    def __init__(self, tokenizer, labels, max_length=8192):
        self.tok, self.labels, self.max_length = tokenizer, labels, max_length
        self.head = tokenizer("choice question: " + INSTRUCTION, add_special_tokens=False)["input_ids"]
        self.options = [[tokenizer.mask_token_id] + tokenizer(" " + name, add_special_tokens=False)["input_ids"] for name in labels]
        if len({tuple(x) for x in self.options}) != len(labels):
            raise ValueError("类别 token 序列不唯一")
        self.prefix_length = 3 + len(self.head) + sum(map(len, self.options))

    def tokenize_state(self, state):
        return self.tok(state.replace(self.tok.mask_token, " "), add_special_tokens=False)["input_ids"]

    def encode(self, state_tokens, label=None, order=None):
        order = list(range(len(self.labels))) if order is None else list(order)
        if sorted(order) != list(range(len(self.labels))):
            raise ValueError("候选顺序必须是全部类别的一次排列")
        ids = [self.tok.cls_token_id] + self.head + [self.tok.sep_token_id]
        markers = []
        for original_index in order:
            markers.append(len(ids))
            ids.extend(self.options[original_index])
        ids += [self.tok.sep_token_id] + list(state_tokens) + [self.tok.sep_token_id]
        if len(ids) > self.max_length:
            raise ValueError(f"完整样本 {len(ids)} tokens 超过 {self.max_length}；需要明确的长文策略")
        return {"ids": ids, "markers": markers, "qtype": 0,
                "label": -1 if label is None else order.index(label)}


def prepare(output, seed):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("已存在固定数据清单，请使用新目录，避免覆盖划分")
    raw_dir = output / "raw"
    raw_dir.mkdir(exist_ok=True)
    sources = {}
    for name, relative in [("categories.json", "banking_data/categories.json"),
                           ("train.csv", "banking_data/train.csv"), ("test.csv", "banking_data/test.csv"),
                           ("upstream-README.md", "README.md")]:
        url = SOURCE_ROOT + "/" + relative
        with urllib.request.urlopen(url, timeout=90) as response:
            body = response.read()
        dest = raw_dir / name
        dest.write_bytes(body)
        sources[name] = {"url": url, "sha256": sha256(dest), "bytes": len(body)}
    labels = json.loads((raw_dir / "categories.json").read_text())
    if len(labels) != 77 or len(set(labels)) != 77:
        raise ValueError("官方类别清单不符合预期")
    write_json(output / "labels.json", labels)
    records = []
    for split, expected_count in [("train", 10003), ("test", 3080)]:
        rows = list(csv.DictReader(io.StringIO((raw_dir / f"{split}.csv").read_text())))
        if len(rows) != expected_count:
            raise ValueError(f"源数据 {split} 数量与固定版本不符")
        for i, row in enumerate(rows):
            records.append({"id": f"banking77:{split}:{i:05d}", "source_split": split,
                            "source_row": i + 2, "state": row["text"], "label_name": row["category"],
                            "label": labels.index(row["category"]), "language": "en", "label_origin": "dataset_gold"})
    audit = group_duplicates(records, seed)
    splits, excluded = partition(records, seed)
    files = {}
    for name, rows in {**splits, "excluded": excluded}.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        files[path.name] = {"records": len(rows), "sha256": sha256(path),
                            "groups": len({r["group_id"] for r in rows}),
                            "class_counts": dict(sorted(Counter(r["label_name"] for r in rows).items()))}
    files["labels.json"] = {"sha256": sha256(output / "labels.json")}
    write_json(output / "deduplication.json", audit)
    manifest = {"dataset": "BANKING77", "source_revision": SOURCE_REVISION, "seed": seed,
                "license": "CC BY 4.0", "attribution": "Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders, 2020; PolyAI",
                "source_files": sources, "files": files,
                "protocol": {"official_test_unchanged": True, "training_group_ratios": [0.8, 0.1, 0.1],
                             "selection": "development NLL", "temperature": "calibration only after model selection",
                             "final_test": "evaluate baseline and selected checkpoint after selection; no test-driven tuning",
                             "candidate_order": "random permutation per training sample; canonical evaluation",
                             "max_length": 8192, "all_77_options_kept": True},
                "audit": {"normalized_unique_texts": audit["normalized_unique_texts"],
                          "duplicate_components": audit["duplicate_components"], "near_pairs": len(audit["near_duplicate_pairs"]),
                          "exclusion_reasons": dict(Counter(r["exclusion_reason"] for r in excluded)),
                          "cross_split_group_overlap": 0}, "preparer_sha256": sha256(__file__)}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"counts": {k: len(v) for k,v in splits.items()}, "audit": manifest["audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/banking77-v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.output, args.seed)
