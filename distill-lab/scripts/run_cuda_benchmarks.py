"""顺序运行 RTX 5090 长上下文测速，逐项保存结果、日志与失败记录。"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cuda-5090"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("benchmark_training.py")
    script_hash = hashlib.sha256(script.read_bytes()).hexdigest()
    manifest = []

    def run(name, length, *, mode="lora", batch=1, precision="bf16", gc=False,
            head_gc=False, warmup=3, steps=10):
        result_path = out / f"{name}.json"
        log_path = out / f"{name}.log"
        command = [
            sys.executable, str(script), "--device", "cuda", "--weights", str(args.weights),
            "--mode", mode, "--batch-size", str(batch), "--seq-len", str(length),
            "--precision", precision, "--memory-limit-gib", "28", "--warmup", str(warmup),
            "--steps", str(steps), "--output", str(result_path),
        ]
        if gc:
            command.append("--gradient-checkpointing")
        if head_gc:
            command.append("--checkpoint-head")
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["model"].get("benchmark_sha256") != script_hash:
                raise RuntimeError(f"已有结果使用不同测速代码，请选择新输出目录：{result_path}")
            entry = {"name": name, "status": "existing", "result": result_path.name}
            print(f"复用已完成结果：{name}", flush=True)
        else:
            print(f"开始：{name}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            if code:
                text = log_path.read_text()
                status = "oom" if "out of memory" in text.lower() else "failed"
                entry = {"name": name, "status": status, "exit_code": code, "log": log_path.name}
                result = None
            else:
                result = json.loads(result_path.read_text())
                entry = {"name": name, "status": "ok", "result": result_path.name, "log": log_path.name}
        manifest.append(entry)
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if entry["status"] == "failed":
            raise RuntimeError(f"非显存错误，停止后续测试：{log_path}")
        return result

    if not run("smoke-lora-b1-l512-bf16", 512, warmup=2, steps=3):
        raise RuntimeError("最小配置未通过，停止后续测试")

    # 与本机对照的两组 FP32 配置：输入长度、批量与编码器重算设置一致。
    run("lora-b1-l2048-fp32-gc", 2048, precision="fp32", gc=True)
    run("lora-b1-l4096-fp32-gc", 4096, precision="fp32", gc=True, warmup=2, steps=5)

    longest = None
    for length in (2048, 4096, 8192):
        result = run(f"lora-b1-l{length}-bf16", length)
        if result is None:
            result = run(f"lora-b1-l{length}-bf16-gc-head", length, gc=True, head_gc=True)
        if length == 8192:
            longest = result

    # 8k 已能运行时也测一次重算，以判断能否用节省的显存增加微批量。
    if longest is not None:
        checkpointed = run("lora-b1-l8192-bf16-gc-head", 8192, gc=True, head_gc=True)
        if checkpointed is not None:
            longest = checkpointed

    # 只在当前峰值留有余量时增加 8k 微批量，最多测试到 batch 4。
    if longest is not None:
        current = longest
        for batch in (2, 4):
            peak = current["cuda_peak_memory"]["allocated_bytes"]
            if peak * 2 > 26 * 2**30:
                break
            cfg = current["configuration"]
            suffix = "-gc-head" if cfg["head_gradient_checkpointing"] else ""
            result = run(f"lora-b{batch}-l8192-bf16{suffix}", 8192, batch=batch,
                         gc=cfg["gradient_checkpointing"],
                         head_gc=cfg["head_gradient_checkpointing"])
            if result is None:
                break
            current = result
        run("full-b1-l8192-bf16-gc-head", 8192, mode="full", gc=True, head_gc=True)

    print(f"测速结束，结果目录：{out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
