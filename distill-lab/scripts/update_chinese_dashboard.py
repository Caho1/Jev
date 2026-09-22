"""将本地中文评测和逐题证据加入现有看板，不请求远程机器或模型 API。"""

import argparse
import json
import re

from banking77_dashboard import APP, LAB, LOCAL, atomic_json, build, now

DATA = LAB / "data/minds14-zh-v1"
RUN = LAB / "results/minds14-zh-mps-v1"


def update(build_status):
    snapshot = json.loads((APP / "src/data.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    state = json.loads((RUN / "status.json").read_text())
    result = json.loads((RUN / "results.json").read_text()) if (RUN / "results.json").exists() else None
    audit = json.loads((RUN / "data-quality-audit.json").read_text()) if (RUN / "data-quality-audit.json").exists() else None
    original = [json.loads(line) for line in (DATA / "evaluation.jsonl").read_text().splitlines()]
    labels_zh = json.loads((DATA / "labels_zh.json").read_text())
    predictions = {}
    if result:
        for line in (RUN / "predictions.jsonl").read_text().splitlines():
            row = json.loads(line)
            predictions[(row["id"], row["model"], row["language"])] = row
    case_rows = []
    for i, row in enumerate(original):
        for language in ("zh", "en"):
            case = {"number": i+1, "id": row["id"], "language": language,
                    "text": row[f"text_{language}"], "textZh": row["text_zh"], "textEn": row["text_en"],
                    "expected": row["label_name"], "expectedZh": labels_zh[row["label_name"]],
                    "containsHan": bool(re.search("[\u3400-\u9fff]", row["text_zh"]))}
            for model in ("baseline", "tuned"):
                saved = predictions.get((row["id"], model, language))
                case.update({f"{model}Answer": saved["prediction"] if saved else None,
                             f"{model}AnswerZh": labels_zh[saved["prediction"]] if saved else None,
                             f"{model}Correct": saved["correct"] if saved else None,
                             f"{model}Confidence": saved["confidence"] if saved else None})
            case_rows.append(case)
    overview = [{"records": len(original), "classes": 14, "uniqueChinese": manifest["unique_normalized_zh"],
                 "phase": state["phase"], "completed": state.get("completed",0), "total": state.get("total",2008),
                 "wallSeconds": result["wall_seconds"] if result else None, "complete": bool(result),
                 "error": state.get("error"), "updatedAt": state["updated_at"]}]
    summary, performance, classes = [], [], []
    audit_rows = []
    if result:
        for key, name in [("baseline","原始 Laya"),("tuned","微调 Laya")]:
            model = result["models"][key]
            if audit:
                audit_rows.append({"model": name, "records": audit["contains_han"],
                                   "excluded": audit["total"]-audit["contains_han"],
                                   "zhAccuracyRate": audit["models"][key]["zh"]["accuracy"],
                                   "enAccuracyRate": audit["models"][key]["en"]["accuracy"]})
            gap = model["language_gap"]
            summary.append({"model": name, "zhAccuracyRate": model["zh"]["accuracy"], "enAccuracyRate": model["en"]["accuracy"],
                            "zhCorrect": model["zh"]["correct"], "enCorrect": model["en"]["correct"], "samples": len(original),
                            "zhMacroF1Rate": model["zh"]["macro_f1"], "enMacroF1Rate": model["en"]["macro_f1"],
                            "delta": gap["zh_minus_en"], "lower": gap["group_bootstrap_95_percent"][0], "upper": gap["group_bootstrap_95_percent"][1]})
            for language in ("zh", "en"):
                m = model[language]
                performance.append({"model": name, "language": "中文原文" if language=="zh" else "数据源英文译文",
                                    "seconds": m["elapsed_seconds"], "examplesPerSecond": m["examples_per_second"],
                                    "medianTokens": m["token_median"], "maxTokens": m["token_max"],
                                    "batchP50Ms": 1000*m["forward_batch_p50_seconds"], "batchP95Ms": 1000*m["forward_batch_p95_seconds"]})
        for i, (label, name) in enumerate(labels_zh.items()):
            classes.append({"intent": label, "intentZh": name, "samples": result["models"]["tuned"]["zh"]["support"][i],
                            "baselineZhRate": result["models"]["baseline"]["zh"]["per_class_recall"][i],
                            "tunedZhRate": result["models"]["tuned"]["zh"]["per_class_recall"][i],
                            "tunedEnRate": result["models"]["tuned"]["en"]["per_class_recall"][i]})
    def query(rows, definitions):
        return {"rows": rows, "source": {"label": "MInDS-14 · 本机 MPS 配对评测", "executedAt": state["updated_at"],
            "files": ["data/minds14-zh-v1/evaluation.jsonl", "data/minds14-zh-v1/manifest.json", "data/minds14-zh-v1/criteria.json",
                      "results/minds14-zh-mps-v1/protocol.json", "results/minds14-zh-mps-v1/results.json", "results/minds14-zh-mps-v1/predictions.jsonl", "results/minds14-zh-mps-v1/data-quality-audit.json"],
            "links": [{"label":"公开数据集", "url":manifest["dataset_url"]},{"label":"原始论文", "url":manifest["paper_url"]}],
            "metricDefinitions": definitions,
            "caveats": [manifest["collection"], manifest["scope"], "14 类新候选集合，不能直接与 BANKING77 的 77 类准确率比较。",
                        "保留全部 502 条，归一化中文去重后 480 条；底座及上游是否见过公开数据未知。", "英文为数据源机器译文，其错误可能影响配对结果。"],
            "evidenceFlow": [{"title":"固定数据","detail":f"PolyAI/minds14@{manifest['revision']}，zh-CN 子集。"},
                             {"title":"本地推理","detail":"evaluate_minds14_local.py：原始及微调 Laya 在 MPS FP32、batch 4 上推理；保持相同英文指令与 14 条英文候选定义，不训练或重新校准。"},
                             {"title":"逐题连接","detail":"按样本 ID、模型、输入语言连接冻结文本与保存的预测。"}]},
            "methods":[{"language":"text","code":"scripts/prepare_minds14_zh.py 导入；scripts/evaluate_minds14_local.py 评测；scripts/update_chinese_dashboard.py 整理快照。"}]}
    snapshot["queries"].update({
        "zh_run": query(overview, [{"label":"样本与推理次数","definition":"502 个来源样本 × 中文/英文两种输入 × 原始/微调两模型 = 2,008 次预测；不是 2,008 个独立问题。"}]),
        "zh_summary": query(summary, [{"label":"准确率与 Macro F1","definition":"每个语言条件的正确数 / 全部 502 条；Macro F1 为 14 类等权平均。"}, {"label":"中英文差值","definition":"同一样本中文准确率减英文译文准确率。95% 区间按归一化中文重复组配对 bootstrap 2,000 次。"}]),
        "zh_performance": query(performance, [{"label":"本地吞吐","definition":"502 条 / 该语言评测用时，包含分词、组批、MPS 推理与回传，不含加载模型；无在线翻译或网络等待。"}, {"label":"批次耗时","definition":"batch 4 的前向及结果回传耗时分位数；末批为 2 条，不等于单请求延迟。"}]),
        "zh_cases": query(case_rows, [{"label":"逐题结果","definition":"同一原始样本的中文与英文译文分别预测，标准类别来自公开数据。概率为温度 1 的 softmax，未针对本数据校准。"}]),
        "zh_classes": query(classes, [{"label":"类别召回","definition":"该类别预测正确数 / 该类别原始样本数，中文、英文及两模型使用相同样本。"}]),
        "zh_audit": query(audit_rows, [{"label":"含汉字的子集","definition":"事后数据质量补充检查：原502条中有10条原文没有汉字且为英文。对其余492条及对应英文译文单独计分，未重跑或调整模型；主结果仍保留502条。"}]),
    })
    snapshot.update(generatedAt=now(), buildStatus=build_status)
    atomic_json(APP / "src/data.json", snapshot)
    build()
    atomic_json(LOCAL / "dashboard-version.json", {"generatedAt": snapshot["generatedAt"]})
    print(json.dumps(overview[0], ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-status", choices=["updating","complete"], default="updating")
    update(parser.parse_args().build_status)
