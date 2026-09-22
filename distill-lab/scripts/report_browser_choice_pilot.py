"""从落盘预测重算试训指标，验证前后输入及检查点，再生成中文报告。"""

import argparse
import json
from pathlib import Path
from statistics import median

from computer_replay import read_records, summarize
from prepare_computer_use import digest, write_json


def counts(rows, predictions):
    by_id = {p["id"]: p for p in predictions}
    operation = target = joint = recalled = none = 0
    for row in rows:
        p = by_id[row["id"]]
        gold = row["reference"]["gold_choices"]
        op = p["answers"]["operation"]["choice"] in gold["operation"]
        item = p["answers"]["item"]["choice"] in gold["item"]
        present = row["reference"]["target_recalled"]
        operation += op
        target += item and present
        joint += op and item and present
        recalled += present
        none += p["answers"]["item"]["choice"] == "NONE"
    return {"records": len(rows), "operation_correct": operation, "actual_target_correct": target,
            "grounded_joint_correct": joint, "target_recalled": recalled, "predicted_none": none}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    results = json.loads((run / "results.json").read_text())
    config = json.loads((run / "run_config.json").read_text())
    training = json.loads((run / "training_metrics.json").read_text())
    dev = read_records(run / "development.jsonl")
    train = read_records(run / "train.jsonl")
    assert digest(run / "development.jsonl") == config["dev_input_sha256"]
    assert digest(run / "train.jsonl") == config["train_input_sha256"]
    assert all(r["metadata"]["split"] == "train" for r in train)
    assert all(r["metadata"]["split"] == "development" for r in dev)
    for field in ("website", "trajectory_id"):
        assert not {r["metadata"][field] for r in train} & {r["metadata"][field] for r in dev}
    models = {}
    for variant in ("baseline", "tuned"):
        predictions = [json.loads(line) for line in (run / f"{variant}_predictions.jsonl").read_text().splitlines()]
        assert len(predictions) == len(dev) and all(r["ok"] for r in predictions)
        assert summarize(dev, predictions) == results[variant]
        models[variant] = counts(dev, predictions)
    for variant in ("initial", "epoch-1"):
        checkpoint = run / variant
        metadata = json.loads((checkpoint / "config.json").read_text())
        assert digest(checkpoint / "trainable.safetensors") == metadata["checkpoint_sha256"]
    for name, expected in config["code_sha256"].items():
        assert digest(run / "source" / name) == expected
    write_json(run / "audit.json", {"status": "passed", "paired_inputs_identical": True,
               "source_and_checkpoint_hashes_verified": True, "website_and_trajectory_disjoint": True,
               "metrics_recomputed_from_predictions": True, "counts": models})
    b, t = models["baseline"], models["tuned"]
    n = len(dev)
    def display(value):
        return f"{value}/{n}（{100*value/n:.2f}%）"
    max_driver = max(x["mps_driver_bytes"] for x in training["memory_samples"]) / 2**30
    report = f"""# 浏览器 Choice 本机试训

已完成一轮小规模 LoRA + 决策头训练，32 次优化器更新。记录来自本机实际运行。
训练与开发来自 Mind2Web 原有网站隔离划分，没有使用校准或留出集；未调用 Jev 或执行网页。

## 训练设置与速度

- 硬件：{config['hardware']['chip']}；MPS FP32，禁用 CPU 算子回退。
- 起点：原始 Laya；没有加载客服或上一轮 computer-use adapter。
- 数据：64 个训练步骤拆成 128 道 Choice 题；操作类型按既有检查样本分层，不代表线上频率。
- 方法：同页操作 / 目标选择，LoRA rank 16、alpha 32；微批次 1、累积 4，一轮。
- 学习率：编码器 LoRA 5e-5、决策头 2e-5；AdamW，weight decay 0.01。
- 上下文上限：8,192 tokens；实际训练长度 {config['input_tokens']['min']}–{config['input_tokens']['max']}，未截断。
- 纯训练阶段：{training['seconds']:.1f} 秒（{training['seconds']/60:.2f} 分钟），含每步日志与缓存清理，不含准备、预检、保存和评测。
- 平均每题 {training['seconds']/training['questions']:.2f} 秒；每次优化器更新耗时中位数 {median(training['optimizer_step_seconds']):.2f} 秒。
- 更新结束时采样的 MPS 驱动内存最大值：{max_driver:.2f} GiB；不是步内峰值，也不能与 RSS 相加。
- LoRA 和评分头均通过有限非零梯度及参数实际改变检查。

## 相同开发输入上的效果

完整开发集 {n} 步；下表是离线下一步判定，不是网页任务成功率。

| 指标 | 原始 Laya | 试训后 |
|---|---:|---:|
| 操作正确 | {display(b['operation_correct'])} | {display(t['operation_correct'])} |
| 真实目标正确，不把 NONE 算成功 | {display(b['actual_target_correct'])} | {display(t['actual_target_correct'])} |
| 操作和真实目标同时正确 | {display(b['grounded_joint_correct'])} | {display(t['grounded_joint_correct'])} |
| 输入候选包含正确目标 | {display(b['target_recalled'])} | {display(t['target_recalled'])} |

每步两道题依次前向，调用延迟中位数（靠上的中位样本）原始模型 {results['baseline']['median_request_ms']:.0f} ms，
试训后 {results['tuned']['median_request_ms']:.0f} ms；不包含浏览器执行，不用作 Jev 端到端速度对比。

训练候选允许有监督保留正确目标，并随机排列选项；开发候选严格复用原来冻结的 BM25 top-64，
没有插入答案。候选命中仅 {b['target_recalled']}/{n}，是当前步骤联合准确率的上限。
训练与开发候选分布不同，不能用训练损失或训练候选召回代替开发效果。

这次协议仍为 `operation + item`：SELECT 只判断下拉控件，未预测选项值，
未覆盖滚动、等待、DONE、BLOCKED，也没有完成 jev-ultrafast 的分操作目标接口。
此次没有中文策略数据或 WebChain 数据进入训练。单次小样本结果不足以说明生产可用或超过 Jev。

## 产物

- `epoch-1/trainable.safetensors`：本轮 LoRA 和决策头；加载时还需原始底座，不能当独立完整模型。
- `initial/`：同次运行的训练前状态，已重新加载用于基线对比。
- `cases.jsonl`：每个开发题目的输入、标注及两个模型的完整回答。
- `results.json`、`training_metrics.json`、`events.jsonl`：效果、训练速度/损失和进度记录。
- `audit.json`：逐题重算、网站/轨迹隔离、归档源码和检查点校验结果。
- `source/`：本次实际运行代码快照；`run_config.json`：数据与模型校验值。

原始数据来源：[Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web)。
"""
    (run / "README.md").write_text(report)
    print(json.dumps({"counts": models, "training_seconds": training["seconds"],
                      "max_sampled_driver_gib": max_driver, "audit": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
