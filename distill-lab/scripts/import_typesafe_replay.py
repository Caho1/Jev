"""导入 typesafe-computer-use 的保存日志，恢复固定观察；不重新截图或读取当前 AX。"""

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess

from computer_replay import UPSTREAM_REVISION, freeze, write_records
from prepare_computer_use import LAB, digest, stable, write_json


def json_section(text, title):
    match = re.search(re.escape(title) + r"\n={10,}\n", text)
    if not match:
        raise ValueError(f"日志缺少字段：{title}")
    return json.JSONDecoder().raw_decode(text[match.end():].lstrip())[0]


def source_instructions(source):
    """只解析固定源文件的字符串常量，不导入或执行上游程序。"""
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decide")
    result = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        values = []
        if isinstance(target, ast.Name) and target.id == "questions" and isinstance(node.value, ast.Dict):
            values = [(ast.literal_eval(k), v) for k, v in zip(node.value.keys, node.value.values)]
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "questions":
            values = [(ast.literal_eval(target.slice), node.value)]
        for key, call in values:
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "Choice":
                result[key] = ast.literal_eval(next(k.value for k in call.keywords if k.arg == "instructions"))
    if set(result) != {"kind", "site", "item", "offscreen"}:
        raise ValueError("上游题目结构变更，需要重新审查导入器")
    return result


def import_step(payload, answers, instructions, record_id, metadata):
    state = json_section(payload, "STATE  (sent as `state`)")
    questions = {}
    for name in ("kind", "site", "item", "offscreen"):
        title = f"QUESTION {name}  (Choice criteria)"
        if title not in payload:
            continue
        criteria = json_section(payload, title)
        if criteria:
            questions[name] = {"type": "choice", "instructions": instructions[name], "criteria": criteria}
    # 原日志在写 payload 和真正调用时分别计算 now；不得声称旧教师响应与重建请求逐字一致。
    metadata = {**metadata, "source": "typesafe-computer-use saved run", "upstream_revision": UPSTREAM_REVISION,
                "instructions_reconstructed": True, "legacy_teacher_request_verified": False,
                "replay_reads_current_screen": False}
    record = freeze(record_id, {"state": state, "questions": questions}, metadata,
                    teacher={"origin": "legacy run, diagnostic only", "answers": answers,
                             "eligible_for_distillation": False})
    # 观察层保留原始定位证据，但执行器句柄在离线文件中不可复用。
    record["observations"] = {k: answers.get(k) for k in ("items", "offscreen_controls", "field", "app", "url")}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-id", required=True, help="同一场景/模板的所有运行必须使用同一组")
    parser.add_argument("--split", choices=["train", "development"], required=True)
    parser.add_argument("--upstream", type=Path, default=LAB / "reference/typesafe-computer-use")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("上游提交与已审查版本不符")
    source_path = args.upstream / "typesafe_computer_use/decide.py"
    source = source_path.read_text()
    expected = subprocess.check_output(["git", "-C", str(args.upstream), "show", f"{revision}:typesafe_computer_use/decide.py"], text=True)
    if source != expected:
        raise ValueError("上游决策源文件存在修改")
    instructions = source_instructions(source)
    run_config = json.loads((args.run / "run.json").read_text())
    payloads = sorted(args.run.glob("step-*-payload.txt"))
    if not payloads:
        raise ValueError("运行目录没有保存步骤")
    trajectory = "typesafe:" + stable(args.run.name + serialized_goal(run_config))[:20]
    rows = []
    for path in payloads:
        stem = path.name.removesuffix("-payload.txt")
        answers = json.loads((args.run / f"{stem}-answers.json").read_text())
        record = import_step(path.read_text(), answers, instructions, f"{trajectory}:{stem}", {
            "trajectory_id": trajectory, "group_id": args.group_id, "split": args.split,
            "step": stem, "source_payload_sha256": digest(path)})
        image = args.run / f"{stem}-raw.png"
        if image.exists():
            record["observations"]["screenshot"] = {"filename": image.name, "sha256": digest(image)}
        rows.append(record)
    args.output.mkdir(parents=True, exist_ok=False)
    write_records(args.output / "replay.jsonl", rows)
    write_json(args.output / "manifest.json", {"records": len(rows), "sha256": digest(args.output / "replay.jsonl"),
        "human_labels_available": False, "upstream_revision": revision,
        "note": "完整保存日志中的观察；旧教师回答不视为人工答案，重建请求需重新对照。"})
    print(json.dumps({"records": len(rows), "labeled": 0}, ensure_ascii=False))


def serialized_goal(config):
    return str(config.get("goal", ""))


if __name__ == "__main__":
    main()
