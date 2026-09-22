"""在相同冻结输入上比较 Jev、原始 Laya 和 computer use 检查点；不执行 UI 动作。"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from computer_replay import model_input, normalize_answer, read_records, serialized, summarize, training_exports
from prepare_computer_use import LAB, digest, write_json


class LocalBackend:
    def __init__(self, variant, device, run, weights):
        import torch
        from benchmark_training import load_model
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        torch.set_num_threads(4)
        torch.manual_seed(42)
        if device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("本机 MPS 不可用")
            torch.mps.set_per_process_memory_fraction(min(0.7, 20 * 2**30 / torch.mps.recommended_max_memory()))
        config = json.loads((run / "epoch-1/config.json").read_text())
        if digest(weights / "model.safetensors") != config["base_model_sha256"]:
            raise ValueError("底座权重校验失败")
        args = SimpleNamespace(weights=weights, device=device, seq_len=8192, mode="lora", rank=config["rank"],
                               gradient_checkpointing=False, checkpoint_head=True)
        model, _ = load_model(args)
        checkpoint = run / ("initial" if variant == "laya_base" else "epoch-1")
        checkpoint_config = json.loads((checkpoint / "config.json").read_text())
        checksum = digest(checkpoint / "trainable.safetensors")
        if checksum != checkpoint_config["checkpoint_sha256"]:
            raise ValueError("检查点校验失败")
        state = load_file(str(checkpoint / "trainable.safetensors"))
        if set(state) != {k for k, v in model.named_parameters() if v.requires_grad}:
            raise ValueError("LoRA 与决策头参数不完整")
        model.load_state_dict(state, strict=False)
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(weights / "tokenizer", local_files_only=True)
        self.device = device
        self.identity = {"variant": variant, "checkpoint_sha256": checksum,
                         "base_sha256": config["base_model_sha256"], "device": device, "dtype": "float32",
                         "temperature": 1.0, "calibrated": False,
                         "execution": "one forward per question; no shared-state KV cache"}

    def evaluate(self, request):
        import torch
        from typed_laya import encode_question, model_batch

        answers, lengths = {}, {}
        with torch.inference_mode():
            for name, question in request["questions"].items():
                item, details = encode_question(self.tokenizer, request["state"], question, 8192)
                batch = model_batch([item], self.tokenizer, self.device)
                logits, _ = self.model(**batch)
                values = logits[0, :len(details["labels"])].float().cpu().double()
                if not torch.isfinite(values).all():
                    raise FloatingPointError("本地模型出现非有限数值")
                probabilities = torch.softmax(values, dim=-1).tolist()
                options = details["labels"]
                answers[name] = {"type": "choice", "choice": options[max(range(len(options)), key=probabilities.__getitem__)],
                                 "probabilities": dict(zip(options, probabilities))}
                lengths[name] = details["input_tokens"]
        return {"answers": answers}, {"input_tokens": lengths}

    def close(self):
        import torch
        del self.model
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()


class JevBackend:
    def __init__(self):
        sys.path.insert(0, str(LAB.parent / "snake"))
        from jev_client import JevClient
        self.client = JevClient(model="jev-1.13.0", timeout=30)
        self.identity = {"variant": "jev", "model": "jev-1.13.0", "execution": "one HTTP request for all questions"}

    def evaluate(self, request):
        return self.client.evaluate(request["state"], request["questions"])

    def close(self):
        self.client.close()


def run_model(name, rows, output, args):
    directory = output / name
    directory.mkdir(exist_ok=False)
    identity = {"model": name, "input_sha256": digest(args.input),
                "requests": [{"id": r["id"], "sha256": r["request_sha256"]} for r in rows],
                "optimizer_updates": 0, "ui_actions": 0, "retries": 0,
                "purpose": "evaluation" if all(r["metadata"].get("split") != "train" for r in rows) else "training_teacher_collection",
                "code_sha256": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("computer_replay.py"))}}
    backend = None
    records = []
    write_json(directory / "protocol.json", identity)
    try:
        backend = JevBackend() if name == "jev" else LocalBackend(name, args.device, args.run, args.weights)
        identity["backend"] = backend.identity
        write_json(directory / "protocol.json", identity)
        with (directory / "responses.jsonl").open("x", buffering=1) as stream:
            for row in rows:
                request = model_input(row)
                # 写前日志标记已经尝试的请求；中断后不会自动重复可能计费的调用。
                with (directory / "attempts.jsonl").open("a", buffering=1) as journal:
                    journal.write(serialized({"id": row["id"], "request_sha256": row["request_sha256"]}) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                started = time.perf_counter()
                record = {"id": row["id"], "request_sha256": row["request_sha256"], "model": name}
                try:
                    raw, metadata = backend.evaluate(request)
                    answers = {key: normalize_answer(raw["answers"][key], q) for key, q in request["questions"].items()}
                    record.update(ok=True, answers=answers, metadata=metadata)
                except Exception as exc:
                    # 不保存网络响应正文、密钥、主机环境或未脱敏的异常字符串。
                    record.update(ok=False, error_type=type(exc).__name__)
                record["elapsed_ms"] = (time.perf_counter() - started) * 1000
                stream.write(serialized(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                records.append(record)
                print(json.dumps({"model": name, "completed": len(records), "total": len(rows), "ok": record["ok"]}), flush=True)
                if not record["ok"]:
                    break
    except Exception as exc:
        write_json(directory / "failure.json", {"error_type": type(exc).__name__})
        print(json.dumps({"model": name, "error_type": type(exc).__name__}), flush=True)
    finally:
        if backend:
            backend.close()
    summary = summarize(rows, records)
    summary["complete"] = len(records) == len(rows) and all(r["ok"] for r in records)
    write_json(directory / "results.json", summary)
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["laya_base", "laya_computer", "jev"], default=["laya_base", "laya_computer", "jev"])
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--run", type=Path, default=LAB / "results/computer-use-grounding-cuda-v1")
    parser.add_argument("--weights", type=Path, default=LAB / "checkpoints/banking77-lora-v1/base")
    args = parser.parse_args()
    rows = read_records(args.input)
    if not rows or len(set(args.models)) != len(args.models):
        parser.error("输入为空或模型重复")
    splits = {r["metadata"].get("split") for r in rows}
    if len(splits) != 1 or not splits <= {"train", "development"}:
        parser.error("当前阶段只允许单独的训练或开发划分，不运行校准与留出数据")
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, all_predictions = {}, {}
    for name in args.models:
        records, summaries[name] = run_model(name, rows, args.output, args)
        all_predictions[name] = records
    write_json(args.output / "results.json", {"source": str(args.input), "split": list(splits)[0],
               "models": summaries, "paired_inputs_identical": True, "closed_loop_success_measured": False,
               "note": "diagnostic smoke subset, not representative benchmark; unattempted/errors count incorrect"})
    mappings = {name: {r["id"]: r for r in records} for name, records in all_predictions.items()}
    with (args.output / "cases.jsonl").open("x") as stream:
        for row in rows:
            stream.write(serialized({**row, "predictions": {name: values.get(row["id"]) for name, values in mappings.items()}}) + "\n")
    if splits == {"train"}:
        exports = training_exports(rows, all_predictions.get("jev", []))
        with (args.output / "training_examples.jsonl").open("x") as stream:
            for row in exports:
                stream.write(serialized(row) + "\n")


if __name__ == "__main__":
    main()
