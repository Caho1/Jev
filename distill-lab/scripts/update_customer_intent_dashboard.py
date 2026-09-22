"""把中文客服数据与当前本地试验快照接入现有看板。"""

import argparse
import json
import time
from pathlib import Path

from banking77_dashboard import APP, LAB, LOCAL, atomic_json, build, now


def read(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


def update(data, run):
    snapshot = read(APP / "src/data.json")
    manifest = read(data / "manifest.json")
    config = read(run / "run_config.json", {})
    state = read(run / "status.json", {"phase": "waiting", "time": now()})
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line.strip()]
    training = [r for r in events if r["phase"] == "training" and "optimizer_step" in r]
    latest = training[-1] if training else {}
    source_files = [str(path.relative_to(LAB)) for path in (data / "manifest.json", data / "labels.json", data / "criteria.json", run / "run_config.json", run / "selection.json", run / "status.json", run / "events.jsonl") if path.exists()]
    def query(rows, definition, extra_files=()):
        return {"rows": rows, "source": {"label": "CrossWOZ 中文客服 · 本地 MPS 试验", "executedAt": state["time"],
            "files": source_files + [str(p.relative_to(LAB)) for p in extra_files if p.exists()],
            "links": [{"label": "CrossWOZ 原始数据", "url": "https://github.com/thu-coai/CrossWOZ"}],
            "metricDefinitions": [{"label": "口径", "definition": definition}],
            "caveats": ["中文人工任务采集，不是生产电商日志。", "单诉求分类；多诉求数据另存。", "开发探针按类别和 ID 固定抽样，不能当成正式测试或线上分布。", "未调用 Jev，未校准概率，未评测独立测试集。"],
            "evidenceFlow": [{"title": "固定来源", "detail": "CrossWOZ@df82c9fdff91b9b130f2d6b89110d3870ba6260e；原始文件与输出均记录 SHA-256。"},
                             {"title": "准备", "detail": "prepare_customer_intent_zh.py 按会话分组，排除跨划分重复组，只把当前及过去文本输入模型。"},
                             {"title": "训练与评测", "detail": "train_customer_intent_zh.py 从原始 Laya 开始，MPS FP32，LoRA 加决策头；固定样本、固定轮次，保存前后预测。"}]},
                "methods": [{"language": "text", "code": "scripts/update_customer_intent_dashboard.py：读取本次文件；按 ID 连接预测；保留缺失结果为 null。"}]}
    overview = [{"phase": state["phase"], "eventTime": state["time"], "syncTime": now(), "error": state.get("error"),
                 "classes": manifest["label_count"], "preparedTrain": manifest["splits"]["train"]["records"],
                 "pilotTrain": config.get("train_records"), "devProbe": config.get("development_records"),
                 "update": latest.get("optimizer_step", 0), "updates": latest.get("optimizer_steps_total"),
                 "completed": latest.get("completed", 0), "total": config.get("train_records", 0)*config.get("epochs", 1),
                 "progressRate": latest.get("completed", 0)/max(1, config.get("train_records", 0)*config.get("epochs", 1)),
                 "examplesPerSecond": latest.get("examples_per_second"), "remainingSeconds": latest.get("eta_training_seconds"),
                 "multiple": manifest["multiple_request_frames"], "ecommerceReview": manifest["ecommerce"]["intent_review_records"],
                 "ecommerceApproved": manifest["ecommerce"]["approved_training_records"],
                 "inputMedian": manifest["splits"]["train"]["input_tokens"]["median"],
                 "inputMax": manifest["splits"]["train"]["input_tokens"]["max"]}]
    split_names = {"train": "训练", "development": "开发", "calibration": "校准", "test": "封存测试"}
    splits = [{"split": split_names[name], "records": info["records"], "dialogues": info["dialogues"],
               "medianTokens": info["input_tokens"]["median"], "maxTokens": info["input_tokens"]["max"]} for name, info in manifest["splits"].items()]
    scores, predictions = [], {}
    variants = {"baseline": ("原始 Laya", run, "baseline"), "tuned": ("中文试验 LoRA", run, "tuned"),
                "multilingual": ("官方多语言 Laya", run / "multilingual", "baseline")}
    metric_files, prediction_files = [], []
    for variant, (name, folder, prefix) in variants.items():
        metric_files.append(folder / f"{prefix}_metrics.json")
        prediction_files.append(folder / f"{prefix}_predictions.jsonl")
        result = read(metric_files[-1])
        if result:
            scores.append({"model": name, "records": result["records"], "accuracyRate": result["accuracy"],
                           "macroF1Rate": result["macro_f1_supported_classes"], "nll": result["nll"]})
        prediction_path = prediction_files[-1]
        predictions[variant] = {r["id"]: r for r in (json.loads(line) for line in prediction_path.read_text().splitlines())} if prediction_path.exists() else {}
    ids = read(run / "selection.json", {}).get("development", [])
    rows = {r["id"]: r for r in (json.loads(line) for line in (data / "development.jsonl").read_text().splitlines())}
    cases = []
    for number, sample_id in enumerate(ids, 1):
        row = rows[sample_id]
        case = {"number": number, "id": sample_id, "text": row["current_user"], "state": row["state"], "gold": row["label_name"], "inputTokens": row["input_tokens"]}
        for variant in variants:
            prediction = predictions[variant].get(sample_id, {})
            case.update({variant+"Answer": prediction.get("predicted"), variant+"Correct": prediction.get("correct"), variant+"Tokens": prediction.get("input_tokens")})
        cases.append(case)
    curves = [{"step": r["optimizer_step"], "meanLoss": r["mean_loss"], "examplesPerSecond": r["examples_per_second"]} for r in training]
    snapshot["queries"].update({
        "czi_run": query(overview, "本地固定抽样试验的实际状态；进度为已处理训练样本/计划样本，不含评测。"),
        "czi_splits": query(splits, "完成过滤后的单一业务请求样本；不同来源划分及会话保持隔离。"),
        "czi_training": query(curves, "本轮累计训练交叉熵；横轴为优化器更新次数，不是样本数。"),
        "czi_scores": query(scores, "同一批固定开发探针上的正确率和各支持类别的等权 F1；不是测试准确率。多语言 Laya 未用本批数据微调，使用自身 tokenizer。", tuple(metric_files) + (run / "multilingual/run_config.json",)),
        "czi_cases": query(cases, "开发样本的当前用户、完整历史、金标和落盘预测按样本 ID 连接；待评测为 null。", tuple(prediction_files)),
    })
    snapshot.update(generatedAt=now(), buildStatus="complete")
    atomic_json(APP / "src/data.json", snapshot)
    build()
    atomic_json(LOCAL / "dashboard-version.json", {"generatedAt": snapshot["generatedAt"]})
    print(json.dumps({"phase": state["phase"], "updatedAt": snapshot["generatedAt"], "step": latest.get("optimizer_step")}), flush=True)
    return state["phase"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=LAB / "data/customer-intent-zh-v2")
    parser.add_argument("--run", type=Path, default=LAB / "results/customer-intent-zh-pilot-v2")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic()+7200
    while True:
        phase = update(args.data, args.run)
        if not args.watch or phase in {"complete", "failed"} or time.monotonic() >= deadline:
            break
        time.sleep(30)
