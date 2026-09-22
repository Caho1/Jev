"""用固定的同一批中文开发题比较官方多语言底座，不读取测试集。"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from benchmark_training import load_model, hardware_info
from banking77_data import sha256, write_json
from prepare_customer_intent_zh import LAB, encoder_for
from train_customer_intent_zh import evaluate
from train_computer_grounding import Events
from transformers import AutoTokenizer
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pilot", action="store_true")
    options = parser.parse_args()
    run = LAB / "results/customer-intent-zh-pilot-v2"
    if options.wait_for_pilot:
        print("等待现有训练结束，避免争用 MPS。", flush=True)
        while True:
            phase = json.loads((run / "status.json").read_text())["phase"]
            if phase == "failed":
                raise RuntimeError("先处理现有训练失败")
            if phase == "complete":
                break
            time.sleep(10)
    output = run / "multilingual"
    output.mkdir(exist_ok=False)
    data = LAB / "data/customer-intent-zh-v2"
    weights = LAB / "checkpoints/laya-multilingual-base"
    upstream = json.loads((weights / "upstream-manifest.json").read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    for name in ("development.jsonl", "labels.json", "criteria.json"):
        assert sha256(data / name) == manifest["files"][name]["sha256"]
    for item in upstream["files"]:
        if "lfs" in item:
            assert sha256(weights / item["rfilename"]) == item["lfs"]["sha256"]
    ids = json.loads((run / "selection.json").read_text())["development"]
    lookup = {r["id"]: r for r in map(json.loads, (data / "development.jsonl").read_text().splitlines())}
    rows = [lookup[key] for key in ids]
    labels = json.loads((data / "labels.json").read_text())
    criteria = json.loads((data / "criteria.json").read_text())
    # 上游 v5 把额外 token 存为列表；本地 v4 的同名参数要求字典。
    # 用 additional_special_tokens 保留相同特殊 token，词表文件不作修改。
    token_config = json.loads((weights / "tokenizer/tokenizer_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(weights / "tokenizer", local_files_only=True,
        extra_special_tokens={}, additional_special_tokens=token_config["extra_special_tokens"])
    assert len(tokenizer) == 256000
    encoder = encoder_for(tokenizer, labels, criteria, 8192)
    for row in rows:
        row["_tokens"] = encoder.tokenize_state(row["state"])
        row["_length"] = encoder.prefix_length + len(row["_tokens"])
        item = encoder.encode(row["_tokens"], row["label"])
        assert len(item["ids"]) == row["_length"] <= 8192
        assert len(item["markers"]) == len(labels)
    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.mps.set_per_process_memory_fraction(24 * 2**30 / torch.mps.recommended_max_memory())
    args = SimpleNamespace(weights=weights, output=output, seq_len=8192, device="mps", mode="full",
                           gradient_checkpointing=False, checkpoint_head=False, eval_batch=1, shape_bucket=64)
    config = {"repo": upstream["repo"], "revision": upstream["revision"], "hardware": hardware_info("mps"),
              "trained_on_current_data": False, "test_used": False, "calibration_used": False,
              "model_sha256": sha256(weights / "model.safetensors"), "selection_sha256": sha256(run / "selection.json"),
              "script_sha256": sha256(Path(__file__)), "input_tokens": {"min": min(r["_length"] for r in rows),
              "max": max(r["_length"] for r in rows)}, "truncated": 0,
              "caveat": "同样的中文文本、题干与 35 候选；各底座使用自身 tokenizer。官方默认训练长度为 1024，这次完整输入可能更长。"}
    write_json(output / "run_config.json", config)
    event = Events(output)
    event("loading", records=len(rows))
    model, _ = load_model(args)
    metrics = evaluate(model, rows, encoder, labels, args, event, "baseline")
    event("complete", accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1_supported_classes"])
    print(json.dumps({k: metrics[k] for k in ("records", "accuracy", "macro_f1_supported_classes", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
