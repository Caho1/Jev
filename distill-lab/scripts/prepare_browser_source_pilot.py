"""核对浏览器决策数据的小样本；只生成审计材料，不导出训练集、不调用模型。"""

import argparse
import collections
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
from bs4 import BeautifulSoup

from prepare_computer_use import retrieve

LAB = Path(__file__).resolve().parents[1]
OUT = LAB / "data/browser-source-pilot-v1"
RAW = OUT / "raw"
PRIOR = LAB / "research/browser-dataset-search-2026-09-22"
GROUND = LAB / "data/computer-use-grounding-v1"
SPLIT_SHA = "22771eba4af39e6683882bf40dc7f33ec37b8a1d653f2bbcac8f6a01b076dbb3"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_lines(name, rows):
    (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def inspect_mind2web(assignments):
    """只读现有 train 文件，不重分组；抽样不依据检索是否命中答案。"""
    split_by_id = {r["trajectory_id"]: r["split"] for r in assignments}
    with gzip.open(GROUND / "train_frames.jsonl.gz", "rt") as stream:
        frames = [json.loads(line) for line in stream]
    counts, samples = collections.Counter(), []
    limits = {"CLICK": 30, "TYPE": 20, "SELECT": 14}
    chosen = collections.Counter()
    # ID 排序使样本选择与文件行顺序无关；操作分层仅用于数据检查。
    for row in sorted(frames, key=lambda r: r["id"]):
        if row["split"] != "train" or split_by_id.get(row["trajectory_id"]) != "train":
            raise ValueError("发现非训练轨迹，停止处理")
        selected = retrieve(row["candidates"], row["state"]["task"], 64)
        gold = set(row["gold_node_ids"])
        matches = [c for c in selected if c["node_id"] in gold]
        compatible = [c for c in matches if row["gold_operation"] in c["operations"]]
        counts["frames"] += 1
        counts["gold_in_full_pool"] += any(c["node_id"] in gold for c in row["candidates"])
        counts["gold_in_top64"] += bool(matches)
        counts["gold_operation_compatible_top64"] += bool(compatible)
        counts["operation_" + row["gold_operation"]] += 1
        if chosen[row["gold_operation"]] >= limits[row["gold_operation"]]:
            continue
        chosen[row["gold_operation"]] += 1
        samples.append({
            "id": row["id"], "split": "train", "trajectory_id": row["trajectory_id"],
            "source": "osunlp/Mind2Web", "source_revision": "17ece8eb89862368edc0cc806acee6fca5163474",
            "website": row["website"], "step_index": row["step_index"],
            "observation": {"state": row["state"], "candidates": selected},
            "reference": {"operation": row["gold_operation"], "gold_node_ids": row["gold_node_ids"]},
            "alignment": {"target_in_top64": bool(matches), "operation_compatible": bool(compatible)},
            "limitations": ["候选来自静态标注池，未证明符合实时视口可见性", "现有中间格式未保留输入参数及完整下拉选项"],
            "purpose": "inspection_only_not_training_export",
        })
    write_lines("mind2web-inspection.jsonl", samples)
    return {**counts, "inspection_rows": len(samples), "inspection_operation_counts": dict(chosen),
            "train_trajectories": len({r["trajectory_id"] for r in frames}),
            "retrieval_uses_gold": False, "training_exported": False}


def host_protected(host, assignments):
    """保守排除既有非训练网站；所有新来源仍保持未分配状态。"""
    host = host.lower().strip(".")
    for row in assignments:
        if row["split"] == "train":
            continue
        site = row["website"]
        if site == "new.mta.info":
            site = "mta.info"
        if host == site or host.endswith("." + site) or site in host.split("."):
            return True
    return False


def fetch_dom(url, path):
    """取公开静态结构，不执行页面；只从元数据提供的 HTTP(S) 链接取样。"""
    import requests
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("结构地址不是 HTTP(S)")
    temp = path.with_suffix(".partial")
    try:
        with requests.get(url, stream=True, timeout=(15, 40)) as response:
            response.raise_for_status()
            size = 0
            with temp.open("wb") as stream:
                for block in response.iter_content(65536):
                    size += len(block)
                    if size > 6_000_000:
                        raise ValueError("单份结构超过 6 MB 抽样上限")
                    stream.write(block)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def inspect_webchain(assignments, fetch):
    with (RAW / "webchain-test-trace-ids.tsv").open() as stream:
        test_ids = {r["trace_uid"] for r in csv.DictReader(stream, delimiter="\t")}
    traces_path = PRIOR / "webagentlab__webchain-data__seed_sft__parts__part_00__metadata__traces.parquet"
    # 只用 UID 栏计算交集；过滤完成后才读取其余内容。
    all_ids = set(pq.read_table(traces_path, columns=["uid"])["uid"].to_pylist())
    traces = pq.read_table(traces_path, filters=[("uid", "not in", list(test_ids))]).to_pylist()
    protected = {r["uid"] for r in traces if host_protected(r["primary_host"] or "", assignments)}
    allowed = {r["uid"] for r in traces} - protected
    columns = ["trace_uid", "source_step_id", "source_step_index", "action_type", "original_action_type",
               "seed_function", "html_dom_url", "ax_tree_url", "selector", "host", "href"]
    rows = pq.read_table(RAW / "webchain-actions.parquet", columns=columns,
                         filters=[("trace_uid", "in", sorted(allowed))]).to_pylist()
    # 同一个源步骤可被不同滑动窗口重复列出，不能按窗口行计训练量。
    unique = {}
    for row in rows:
        key = (row["trace_uid"], row["source_step_id"])
        if key in unique and unique[key] != row:
            raise ValueError("同一源步骤的元数据冲突")
        unique[key] = row
    actions = sorted(unique.values(), key=lambda r: (r["trace_uid"], r["source_step_index"]))
    selected, per_type, per_trace = [], collections.Counter(), collections.Counter()
    for row in actions:
        op = row["original_action_type"]
        if op not in {"click", "type", "select"} or not row["html_dom_url"] or not row["selector"]:
            continue
        if per_type[op] >= 4 or per_trace[row["trace_uid"]] >= 2:
            continue
        if host_protected(row["host"] or "", assignments):
            continue
        per_type[op] += 1
        per_trace[row["trace_uid"]] += 1
        selected.append(row)
    probes = []
    for row in selected:
        path = RAW / ("webchain-dom-" + hashlib.sha256(row["html_dom_url"].encode()).hexdigest()[:20] + ".html")
        result = {"trace_uid": row["trace_uid"], "step_id": row["source_step_id"],
                  "source_step_index": row["source_step_index"], "original_operation": row["original_action_type"],
                  "seed_function": row["seed_function"], "host": row["host"],
                  "dom_url": row["html_dom_url"], "selector": row["selector"],
                  "split": "unassigned", "training_eligible": False,
                  "observation_timing": "unverified", "quarantine_reason": "DOM 操作前后时序尚未核实"}
        if fetch and not path.exists():
            try:
                fetch_dom(row["html_dom_url"], path)
            except Exception as exc:
                result["fetch_error"] = type(exc).__name__
        if path.exists():
            soup = BeautifulSoup(path.read_bytes(), "html.parser")
            try:
                nodes = soup.select(row["selector"])
                result.update(selector_match_count=len(nodes), unique_target=len(nodes) == 1,
                              matched_tags=sorted({node.name for node in nodes}))
                # 唯一 CSS 匹配只证明静态结构对齐，不足以证明观察时序和可见性。
                if len(nodes) == 1 and nodes[0].name in {"select", "option"}:
                    select = nodes[0] if nodes[0].name == "select" else nodes[0].find_parent("select")
                    result["native_select_parent_found"] = select is not None
                    if select is not None:
                        result["native_option_count"] = len(select.find_all("option"))
                    if nodes[0].name == "option":
                        result["target_option_has_selected_attribute"] = nodes[0].has_attr("selected")
            except Exception as exc:
                result["selector_error"] = type(exc).__name__
            result.update(dom_file=str(path.relative_to(OUT)), dom_bytes=path.stat().st_size, dom_sha256=sha(path))
        probes.append(result)
    write_lines("webchain-alignment.jsonl", probes)
    write_json("webchain-split-exclusions.json", {
        "official_test_uid_count": len(test_ids), "excluded_test_uids_in_part00": sorted(all_ids & test_ids),
        "excluded_existing_nontrain_site_uids": sorted(protected), "remaining_split": "unassigned",
        "note": "还需全站点/跨来源任务去重后，才能冻结新数据划分。未把剩余行自动归为 train。"})
    return {"part00_traces": len(all_ids), "official_test_traces_excluded": len(all_ids & test_ids),
            "existing_nontrain_site_traces_excluded": len(protected), "remaining_traces_unassigned": len(allowed),
            "metadata_rows_after_filter": len(rows), "unique_source_steps": len(actions),
            "original_action_counts": dict(collections.Counter(r["original_action_type"] for r in actions)),
            "changed_seed_action_counts": dict(collections.Counter(
                r["original_action_type"] + "→" + r["seed_function"] for r in actions
                if r["seed_function"] and r["seed_function"] != r["original_action_type"])),
            "dom_probes": len(probes), "dom_files_loaded": sum("dom_file" in r for r in probes),
            "unique_selector_matches": sum(r.get("unique_target", False) for r in probes),
            "verified_pre_action_samples": 0, "training_exported": False}


def inspect_webworld():
    rows = [json.loads(line) for line in (RAW / "webworld-prefix-sample.jsonl").read_text().splitlines()]
    audit = []
    for index, row in enumerate(rows):
        turns = row.get("conversations", [])
        prompt = turns[0].get("value", "") if turns else ""
        # 网页内容里的 Task/Goal 字样不算独立任务字段；只检查外层指令和顶层字段。
        prefix = prompt.split("Initial Page State:", 1)[0]
        goal_field = any(row.get(k) for k in ("goal", "task", "instruction", "user_query"))
        outer_goal_header = bool(re.search(r"(?:^|\n)(?:User (?:Task|Goal)|Goal|Task|用户目标|用户任务)\s*:", prefix, re.I))
        world_format = prompt.startswith("You are a web world model.") and "First Action:" in prompt
        characters = sum(len(re.findall(r"[\u3400-\u9fff]", t.get("value", ""))) for t in turns)
        audit.append({"source_row": index, "world_model_format": world_format,
                      "explicit_goal_field_or_outer_header": bool(goal_field or outer_goal_header),
                      "cjk_characters": characters, "native_chinese_provenance_verified": False,
                      "training_eligible": False,
                      "quarantine_reason": "需要独立用户目标及目标对应的动作监督；含汉字本身不证明原生中文来源"})
    write_lines("webworld-audit.jsonl", audit)
    return {"rows": len(rows), "world_model_format_rows": sum(r["world_model_format"] for r in audit),
            "explicit_goal_field_or_outer_header_rows": sum(r["explicit_goal_field_or_outer_header"] for r in audit),
            "rows_with_at_least_10_cjk_characters": sum(r["cjk_characters"] >= 10 for r in audit),
            "representative_sample": False, "verified_native_chinese_policy_rows": 0, "training_exported": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-dom", action="store_true", help="下载至多 12 个静态 DOM 样本")
    args = parser.parse_args()
    downloads = json.loads((RAW / "download-manifest.json").read_text())
    downloads.append(json.loads((RAW / "webworld-sample-manifest.json").read_text()))
    for item in downloads:
        if sha(RAW / item["file"]) != item["sha256"]:
            raise ValueError("下载样本校验失败：" + item["file"])
    if sha(GROUND / "split_assignments.json") != SPLIT_SHA:
        raise ValueError("原划分校验值变化，需要先核查来源")
    assignments = json.loads((GROUND / "split_assignments.json").read_text())
    summary = {"purpose": "browser_source_alignment_pilot", "date": "2026-09-22",
               "mind2web": inspect_mind2web(assignments),
               "webchain": inspect_webchain(assignments, args.fetch_dom),
               "webworld": inspect_webworld(),
               "split_assignments_sha256": SPLIT_SHA, "training_started": False, "teacher_calls": 0}
    write_json("summary.json", summary)
    inputs = [Path(__file__).resolve(), LAB / "scripts/prepare_computer_use.py",
              GROUND / "train_frames.jsonl.gz", GROUND / "split_assignments.json",
              PRIOR / "webagentlab__webchain-data__seed_sft__parts__part_00__metadata__traces.parquet"]
    write_json("manifest.json", {"external_inputs": [{"path": str(p), "sha256": sha(p)} for p in inputs],
               "files": [{"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha(p)}
                         for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "manifest.json"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
