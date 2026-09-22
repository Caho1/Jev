"""按测试 ID 对齐已保存的三模型预测，生成可核查的逐题看板数据。"""

import json
from pathlib import Path

import numpy as np

from banking77_data import sha256

LAB = Path(__file__).resolve().parents[1]


def case_rows():
    data = LAB / "data/banking77-v1"
    laya = LAB / "results/banking77-lora-v1"
    jev = LAB / "results/banking77-jev-v1"
    protocol = json.loads((jev / "protocol.json").read_text())["protocol"]
    assert sha256(data / "test.jsonl") == protocol["test_sha256"]
    assert sha256(data / "labels.json") == protocol["labels_sha256"]
    tests = [json.loads(line) for line in (data / "test.jsonl").read_text().splitlines()]
    labels = json.loads((data / "labels.json").read_text())
    responses = [json.loads(line) for line in (jev / "responses.jsonl").read_text().splitlines()]
    by_id = {row["id"]: row for row in responses}
    assert len(by_id) == len(responses) == len(tests) == 3080
    assert set(by_id) == {row["id"] for row in tests}
    temperatures = json.loads((laya / "results.json").read_text())["temperatures"]
    scores = {}
    for model in ("baseline", "tuned"):
        path = laya / f"{model}-test.npz"
        assert sha256(path) == protocol["laya_predictions_sha256"][model]
        with np.load(path, allow_pickle=False) as saved:
            assert saved["ids"].tolist() == [row["id"] for row in tests]
            assert saved["labels"].tolist() == [row["label"] for row in tests]
            logits = saved["logits"].astype(np.float64) / temperatures[model]
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        scores[model] = probabilities / probabilities.sum(axis=1, keepdims=True)

    def top_three(probabilities):
        order = sorted(range(len(labels)), key=lambda i: -probabilities[i])[:3]
        return [{"label": labels[i], "probability": float(probabilities[i])} for i in order]

    rows = []
    for index, test in enumerate(tests):
        answer = by_id[test["id"]]
        row = {"number": index + 1, "id": test["id"], "text": test["state"],
               "expected": test["label_name"], "sourceSplit": test["source_split"],
               "jevValid": answer["ok"], "jevError": answer.get("error", "")}
        for model, probabilities in scores.items():
            p = probabilities[index]
            prediction = int(p.argmax())
            row.update({f"{model}Answer": labels[prediction],
                        f"{model}Correct": prediction == test["label"],
                        f"{model}Confidence": float(p[prediction]),
                        f"{model}Top3": top_three(p)})
        if answer["ok"]:
            raw = answer["answers"]["intent"]
            assert raw["choice"] == answer["choice"] == labels[answer["prediction"]]
            p = [raw["probabilities"][label] for label in labels]
            row.update(jevAnswer=raw["choice"], jevCorrect=raw["choice"] == test["label_name"],
                       jevConfidence=raw["probabilities"][raw["choice"]], jevTop3=top_three(p))
        else:
            # 校验失败记录没有保存原始回答，保留缺失状态并沿用计错规则。
            row.update(jevAnswer=None, jevCorrect=False, jevConfidence=None, jevTop3=[])
        row["disagreement"] = len({row["baselineAnswer"], row["tunedAnswer"], row["jevAnswer"]}) > 1
        # 使用可导出的 JSON 文本，避免来源表将嵌套对象显示为 [object Object]。
        for model in ("baseline", "tuned", "jev"):
            row[f"{model}Top3"] = json.dumps(row[f"{model}Top3"], ensure_ascii=False, separators=(",", ":"))
        rows.append(row)

    summary = json.loads((jev / "results.json").read_text())["models"]
    for model in ("baseline", "tuned", "jev"):
        assert sum(row[f"{model}Correct"] for row in rows) == summary[model]["correct"]
    return rows
