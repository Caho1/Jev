"""固定已训练检查点，在全部开发页面上比较控件排序；不训练、不读取留出集。"""

import argparse
import json
import traceback
from pathlib import Path

from train_computer_grounding import (
    Events, LAB, digest, empty_cache, evaluate, load_model, read_gzip, write_json,
)
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer


def run(cli, event):
    config = json.loads((cli.run / "run_config.json").read_text())
    if json.loads((cli.run / "status.json").read_text())["phase"] != "complete":
        raise ValueError("只评测完整结束的固定训练轮次")
    # 与训练归档逐项核对实际推理代码，防止评测期间悄悄换实现。
    for name, expected in config["code_sha256"].items():
        path = LAB / ("reference" if name == "laya_common.py" else "scripts") / name
        if digest(path) != expected:
            raise ValueError(f"训练与评测代码不一致：{name}")
    args = argparse.Namespace(**config)
    args.output = cli.output
    args.weights = cli.weights or Path(config["weights"])
    args.data = cli.data or Path(config["data"])
    if cli.device:
        args.device = cli.device
        args.precision = "fp32" if cli.device == "mps" else config["precision"]
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frames_path = args.data / "development_frames.jsonl.gz"
    if digest(manifest_path) != config["data_manifest_sha256"]:
        raise ValueError("数据版本改变")
    if digest(frames_path) != manifest["files"][frames_path.name]["sha256"]:
        raise ValueError("开发页面校验失败")
    if digest(args.weights / "model.safetensors") != config["base_model_sha256"]:
        raise ValueError("基础权重改变")
    frames = read_gzip(frames_path)
    if not frames or any(f["split"] != "development" for f in frames):
        raise ValueError("存在非开发集页面或数据为空")
    if len({f["id"] for f in frames}) != len(frames):
        raise ValueError("开发页面重复")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    backend = torch.cuda if args.device == "cuda" else torch.mps
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 不可用")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        capacity = torch.cuda.get_device_properties(0).total_memory
    else:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS 不可用")
        capacity = torch.mps.recommended_max_memory()
    fraction = args.memory_limit_gib * 2**30 / capacity
    if not 0 < fraction <= 1:
        raise ValueError("显存预算无效")
    backend.set_per_process_memory_fraction(fraction)
    protocol = {
        "training_run": str(cli.run), "pages": len(frames),
        "candidates": sum(len(f["candidates"]) for f in frames),
        "development_file_sha256": digest(frames_path),
        "all_development_pages": True, "calibration_used": False,
        "holdout_used": False, "optimizer_updates": 0,
        "fixed_checkpoint_epoch": args.epochs, "device": args.device,
        "precision": args.precision, "eval_batch": args.eval_batch,
        "code_sha256": {**config["code_sha256"], Path(__file__).name: digest(Path(__file__))},
    }
    write_json(cli.output / "protocol.json", protocol)
    (cli.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    event("loading_model", pages=protocol["pages"], candidates=protocol["candidates"])
    model, _ = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.weights / "tokenizer", local_files_only=True)
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    results = {}
    for name, checkpoint in (("baseline", "initial"), ("tuned", f"epoch-{args.epochs}")):
        path = cli.run / checkpoint
        checkpoint_config = json.loads((path / "config.json").read_text())
        checksum = digest(path / "trainable.safetensors")
        if checksum != checkpoint_config["checkpoint_sha256"]:
            raise ValueError("检查点权重校验失败")
        state = load_file(str(path / "trainable.safetensors"))
        if set(state) != names:
            raise ValueError("可训练参数集合不完整")
        model.load_state_dict(state, strict=False)
        del state
        empty_cache(args.device)
        metrics = evaluate(model, tokenizer, frames, config["question"], args, f"evaluate_{name}", event)
        metrics.update(selected_development_pages=False, all_development_pages=True,
                       checkpoint_sha256=checksum)
        write_json(cli.output / f"evaluate_{name}_metrics.json", metrics)
        results[name] = metrics
    results.update(status="complete", protocol=protocol,
                   limitations=["仅为开发集离线控件排序，不代表闭环任务完成率",
                                "模型权重固定，未打开校准和留出数据", "输出分数未经概率校准"])
    write_json(cli.output / "results.json", results)
    event("complete", pages=len(frames), baseline_recall_at_24=results["baseline"]["recall_at_24"],
          tuned_recall_at_24=results["tuned"]["recall_at_24"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", choices=["mps", "cuda"])
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    event = Events(cli.output)
    try:
        run(cli, event)
    except BaseException as exc:
        event("failed", error_type=type(exc).__name__, error=str(exc))
        (cli.output / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
