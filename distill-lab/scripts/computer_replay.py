"""冻结 computer use 决策输入；答案、教师输出和任务划分只存在于外层。"""

import copy
import hashlib
import json
import math
from pathlib import Path

FORMAT = "computer-use-replay-v1"
UPSTREAM_REVISION = "cc7b5066ae1a07b5e3182e8f87a9b5b6dfdcffc1"


def serialized(value):
    # 保留候选插入顺序；顺序也是模型输入的一部分。
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(serialized(value).encode()).hexdigest()


def validate_request(request):
    if set(request) != {"state", "questions"} or not isinstance(request["state"], (str, dict, list)):
        raise ValueError("输入仅包含 state 和 questions")
    questions = request["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ValueError("题目为空")
    for question in questions.values():
        if set(question) != {"type", "instructions", "criteria"} or question["type"] != "choice":
            raise ValueError("本版回放器仅接收完整的 Choice 题目")
        options = question["criteria"]
        if not isinstance(question["instructions"], str) or not isinstance(options, dict) or not 2 <= len(options) <= 255:
            raise ValueError("Choice 须包含 2–255 个候选")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in options.items()):
            raise ValueError("候选键与描述必须为字符串")
    serialized(request)


def freeze(record_id, request, metadata, reference=None, teacher=None):
    validate_request(request)
    row = {"format": FORMAT, "id": record_id, "metadata": copy.deepcopy(metadata),
           "request": copy.deepcopy(request), "request_sha256": fingerprint(request),
           "reference": copy.deepcopy(reference or {})}
    if teacher is not None:
        row["teacher"] = copy.deepcopy(teacher)
    validate_record(row)
    return row


def validate_record(row):
    if row["format"] != FORMAT or not isinstance(row["id"], str) or not row["id"]:
        raise ValueError("回放记录格式或 ID 无效")
    validate_request(row["request"])
    if fingerprint(row["request"]) != row["request_sha256"]:
        raise ValueError("输入已被修改，不能复用旧预测")
    for name, labels in row["reference"].get("gold_choices", {}).items():
        if name not in row["request"]["questions"] or not labels:
            raise ValueError("参考答案题号无效")
        if not set(labels) <= set(row["request"]["questions"][name]["criteria"]):
            raise ValueError("参考答案不属于候选")


def model_input(row):
    validate_record(row)
    # 不展开外层字典，避免把 reference、teacher、split 等字段送入模型。
    return copy.deepcopy(row["request"])


def read_records(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        validate_record(row)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("回放 ID 重复")
    return rows


def write_records(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            validate_record(row)
            stream.write(serialized(row) + "\n")


def normalize_answer(answer, question):
    options = question["criteria"]
    values = answer.get("probabilities")
    choice = answer.get("choice")
    if answer.get("type") != "choice" or choice not in options or not isinstance(values, dict) or set(values) != set(options):
        raise ValueError("返回候选集合或题型不匹配")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values.values()):
        raise ValueError("返回概率无效")
    total = sum(values.values())
    if abs(total - 1) > 0.025 or total <= 0:
        raise ValueError("概率总和超出舍入容差")
    if values[choice] < max(values.values()) - 0.001:
        raise ValueError("choice 与最高概率不一致")
    probs = {key: values[key] / total for key in options}
    return {"choice": choice, "probabilities": probs, "max_probability": max(probs.values()),
            "raw_probability_sum": total, "vendor_confidence": answer.get("confidence")}


def summarize(rows, predictions):
    by_id = {p["id"]: p for p in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("预测 ID 重复")
    questions = {}
    joint = grounded = labeled = recalled = 0
    for row in rows:
        pred = by_id.get(row["id"], {})
        if pred and pred["request_sha256"] != row["request_sha256"]:
            raise ValueError("预测与请求校验值不一致")
        gold = row["reference"].get("gold_choices", {})
        all_correct = bool(gold)
        for name, expected in gold.items():
            result = questions.setdefault(name, {"records": 0, "correct": 0, "valid": 0})
            result["records"] += 1
            answer = pred.get("answers", {}).get(name, {}) if pred.get("ok") else {}
            valid = answer.get("choice") in row["request"]["questions"][name]["criteria"]
            correct = valid and answer["choice"] in expected
            result["valid"] += int(valid)
            result["correct"] += int(correct)
            all_correct = all_correct and correct
        if gold:
            labeled += 1
            joint += int(all_correct)
            target_present = row["reference"].get("target_recalled")
            if target_present is not None:
                recalled += int(target_present)
                grounded += int(all_correct and target_present)
    for value in questions.values():
        value["accuracy"] = value["correct"] / value["records"]
    latencies = sorted(p["elapsed_ms"] for p in predictions if p.get("ok"))
    return {"records": len(rows), "attempted": len(predictions), "valid": sum(p.get("ok", False) for p in predictions),
            "labeled_records": labeled, "questions": questions,
            "joint_accuracy": joint / labeled if labeled else None,
            "grounded_joint_accuracy": grounded / labeled if labeled and all("target_recalled" in r["reference"] for r in rows) else None,
            "candidate_recall": recalled / labeled if labeled and all("target_recalled" in r["reference"] for r in rows) else None,
            "median_request_ms": latencies[len(latencies) // 2] if latencies else None,
            "metric_scope": "offline next-step decisions; not closed-loop task success"}


def training_exports(rows, predictions):
    """人工答案保留为主监督；只有同意人工答案的训练集教师预测可作辅助监督。"""
    pred_by_id = {r["id"]: r for r in predictions}
    if len(pred_by_id) != len(predictions):
        raise ValueError("重复预测")
    exported = []
    for row in rows:
        if row["metadata"].get("split") != "train":
            raise ValueError("导出器拒绝混入非训练划分")
        validate_record(row)
        if row["reference"].get("label_source") != "Mind2Web human annotation":
            raise ValueError("本版导出器仅接受有来源记录的 Mind2Web 人工标签")
        pred = pred_by_id.get(row["id"])
        if pred and pred["request_sha256"] != row["request_sha256"]:
            raise ValueError("教师响应不属于当前请求")
        for name, gold in row["reference"].get("gold_choices", {}).items():
            answer = pred.get("answers", {}).get(name, {}) if pred and pred.get("ok") else {}
            teacher = answer.get("probabilities") if answer.get("choice") in gold else None
            exported.append({"id": row["id"] + ":" + name, "split": "train",
                             "trajectory_id": row["metadata"]["trajectory_id"],
                             "state": row["request"]["state"], "question": row["request"]["questions"][name],
                             "gold_choices": gold, "teacher_probabilities": teacher,
                             "teacher_model": pred.get("model") if teacher else None,
                             "source_request_sha256": row["request_sha256"],
                             "target_recalled": row["reference"].get("target_recalled")})
    return exported
