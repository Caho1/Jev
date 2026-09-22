"""固定公开中文银行语音转写及英文译文，仅用于本地迁移评测。"""

import json
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from banking77_data import normalize, sha256, write_json

LAB = Path(__file__).resolve().parents[1]
OUTPUT = LAB / "data/minds14-zh-v1"
REVISION = "40ce77cb32a384e4d50a568e1ec39ac804019d33"
URL = f"https://huggingface.co/datasets/PolyAI/minds14/resolve/{REVISION}/zh-CN/train-00000-of-00001.parquet"
# 根据公开意图名称编写的固定英文定义；查看模型结果之前冻结。
CRITERIA = {
    "abroad": "Using a bank card abroad or preparing to use it in another country.",
    "address": "Updating the postal or residential address registered with the bank.",
    "app_error": "A technical error or malfunction in the banking mobile application.",
    "atm_limit": "The limit on cash withdrawals from an ATM, or changing that limit.",
    "balance": "Checking the current balance or available money in a bank account.",
    "business_loan": "Applying for or asking about a loan for a business.",
    "card_issues": "A bank card is not working, is declined, or has another usage problem.",
    "cash_deposit": "Depositing physical cash into a bank account.",
    "direct_debit": "Setting up, managing, or cancelling a direct debit payment.",
    "freeze": "Freezing or blocking a bank card or account, for example after loss or theft.",
    "high_value_payment": "Making a large payment or transfer and its amount limits.",
    "joint_account": "Opening or managing a bank account shared with another person.",
    "latest_transactions": "Viewing recent transactions or account transaction history.",
    "pay_bill": "Paying a bill through a bank account or banking service.",
}
CHINESE_NAMES = dict(zip(CRITERIA, ["境外用卡", "修改地址", "银行 App 故障", "ATM 取款限额", "查询余额", "企业贷款", "卡片使用问题", "现金存款", "直接扣款", "冻结卡片或账户", "大额支付", "联名账户", "最近交易", "支付账单"]))


def main():
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("评测数据已经冻结，不覆盖")
    raw = OUTPUT / "raw/zh-CN.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        with urllib.request.urlopen(URL, timeout=180) as response:
            raw.write_bytes(response.read())
    table = pq.read_table(raw, columns=["path", "transcription", "english_transcription", "intent_class", "lang_id"])
    metadata = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    labels = metadata["intent_class"]["names"]
    assert labels == list(CRITERIA)
    records = []
    for i, row in enumerate(table.to_pylist()):
        assert row["lang_id"] == metadata["lang_id"]["names"].index("zh-CN")
        assert row["transcription"].strip() and row["english_transcription"].strip()
        records.append({"id": f"minds14:zh-CN:{i:05d}", "source_path": row["path"],
                        "text_zh": row["transcription"], "text_en": row["english_transcription"],
                        "label": row["intent_class"], "label_name": labels[row["intent_class"]],
                        "source_split": "train", "local_role": "evaluation_only"})
    assert len(records) == 502 and len({r["id"] for r in records}) == 502
    test = OUTPUT / "evaluation.jsonl"
    test.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    write_json(OUTPUT / "criteria.json", CRITERIA)
    write_json(OUTPUT / "labels_zh.json", CHINESE_NAMES)
    previous = set()
    for split in ("train", "development", "calibration"):
        previous.update(normalize(json.loads(line)["state"]) for line in (LAB / f"data/banking77-v1/{split}.jsonl").read_text().splitlines())
    overlaps = [r["id"] for r in records if normalize(r["text_zh"]) in previous or normalize(r["text_en"]) in previous]
    manifest = {
        "dataset": "PolyAI/minds14", "revision": REVISION, "subset": "zh-CN", "source_split": "train",
        "local_role": "evaluation_only", "records": len(records), "classes": 14, "license": "CC BY 4.0",
        "source_url": URL, "dataset_url": "https://huggingface.co/datasets/PolyAI/minds14",
        "paper_url": "https://arxiv.org/abs/2104.08524",
        "collection": "母语众包参与者按银行场景录音；本轮使用发布的 ASR 转写，不是实际生产客户日志。英文列为数据源提供的机器翻译。",
        "scope": "官方只提供 train split。本项目将全部 502 条留作外部评测，未用于训练、选模或校准；不是官方另设的测试集。",
        "files": {str(p.relative_to(OUTPUT)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (raw, test, OUTPUT / "criteria.json", OUTPUT / "labels_zh.json")},
        "class_counts": dict(Counter(r["label_name"] for r in records)),
        "unique_normalized_zh": len({normalize(r["text_zh"]) for r in records}),
        "banking77_train_dev_cal_exact_overlaps": overlaps,
        "deduplication_note": "保留发布的全部 502 条；仅检查归一化精确重合，不声称排除语义近重复或上游预训练暴露。",
    }
    write_json(OUTPUT / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("records", "class_counts", "unique_normalized_zh", "banking77_train_dev_cal_exact_overlaps")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
