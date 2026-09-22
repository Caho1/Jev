"""读取本次远程训练的聚合日志，生成可追溯看板快照；不读取凭据或客户原文。"""

import argparse
import datetime as dt
import functools
import hashlib
import http.server
import json
import math
import shlex
import subprocess
import threading
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
APP = LAB / "dashboard"
LOCAL = LAB / "results/banking77-lora-v1"
REMOTE = "/root/autodl-tmp/laya-context-bench-20260921/runs/banking77-lora-v1"
NODE = "/Users/bystanders/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
BUILDER = "/Users/bystanders/.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.9/scripts/data-app.mjs"
FILES = ["status.json", "events.jsonl", "run_config.json", "input_lengths.json", "history.json",
         "baseline-development.json", "results.json", "test-evaluation-lock.json",
         "epoch-1-development.json", "epoch-2-development.json", "epoch-3-development.json"]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def collect():
    # 仅允许明确列出的聚合产物。训练中的非原子指标文件若未写完，下一次再读。
    code = f'''import json,pathlib,subprocess
p=pathlib.Path({REMOTE!r})
out={{}}
for name in {FILES!r}:
 try:
  text=(p/name).read_text()
  out[name]=[json.loads(x) for x in text.splitlines()] if name.endswith('.jsonl') else json.loads(text)
 except (FileNotFoundError,json.JSONDecodeError): pass
try:
 values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0].split(',')
 out['gpu'] = dict(zip(['utilization','memory_mib','total_mib','temperature_c'],map(float,values)))
except Exception: pass
print(json.dumps(out))'''
    cmd = ["ssh", "-S", str(LAB / ".runtime/ssh/5090.sock"), "-p", "15253",
           "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "root@connect.bjb2.seetacloud.com",
           "/root/autodl-tmp/laya-context-bench-20260921/.venv/bin/python -c " + shlex.quote(code)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=18, check=True)
    bundle = json.loads(result.stdout)
    bundle["_collected_at"] = now()
    if not bundle.get("status.json"):
        raise RuntimeError("未读取到训练状态")
    for name, value in bundle.items():
        if name.endswith(".json"):
            atomic_json(LOCAL / name, value)
    atomic_json(LOCAL / "dashboard-bundle.json", bundle)
    return bundle


def snapshot(bundle, connection="connected"):
    existing = json.loads((APP / "src/data.json").read_text())
    manifest = json.loads((LAB / "data/banking77-v1/manifest.json").read_text())
    stamp = now()
    events = bundle.get("events.jsonl", [])
    state = bundle["status.json"]
    captured_at = bundle.get("_collected_at", state["time"])
    cfg = bundle.get("run_config.json", {})
    train_events = [e for e in events if e["phase"] == "training"]
    latest = train_events[-1] if train_events else {}
    history = bundle.get("history.json", [])
    baseline = bundle.get("baseline-development.json")
    results = bundle.get("results.json")
    terminal = state["phase"] in ("complete", "failed")
    batch = cfg.get("batch_size", 8)
    accumulation = cfg.get("accumulation", 4)
    epochs = cfg.get("epochs", 3)
    ntrain = manifest["files"]["train.jsonl"]["records"]
    total_updates = math.ceil(math.ceil(ntrain / batch) / accumulation) * epochs
    update = latest.get("update", 0)
    throughput = latest.get("examples_per_second")
    remaining = max(0, (total_updates-update)*batch*accumulation/throughput) if throughput else None
    elapsed = (dt.datetime.fromisoformat(state["time"])-dt.datetime.fromisoformat(events[0]["time"])).total_seconds() if events else None
    current = {"phase":state["phase"],"eventTime":state["time"],"syncTime":captured_at,
               "connection":connection,"epoch":state.get("epoch",latest.get("epoch",0)),"epochs":epochs,
               "update":update,"totalUpdates":total_updates,"progressRate":update/total_updates,
               "examplesPerSecond":throughput,"remainingTrainingSeconds":remaining if not terminal else None,
               "elapsedSeconds":results.get("wall_seconds") if results else elapsed,
               "selectedEpoch":results.get("selected_epoch") if results else state.get("best_epoch"),
               "contextLimit":cfg.get("seq_len",8192),"batchSize":batch,"accumulation":accumulation,
               "run":"banking77-lora-v1","error":state.get("error"), **bundle.get("gpu",{})}
    query = {}
    def add(key, rows, label, files, definitions=(), caveats=()):
        query[key] = {"rows":rows,"source":{
            "label":label,"files":files,"executedAt":captured_at,
            "description":"本次 RTX 5090 实测日志；时间使用 UTC+8 显示。",
            "metricDefinitions":list(definitions),"caveats":list(caveats),
            "evidenceFlow":[{"title":"读取训练产物","detail":"读取 runs/banking77-lora-v1 中明确列出的 JSON/JSONL 聚合文件。"},
                            {"title":"生成快照","detail":"运行 scripts/banking77_dashboard.py；完整转换代码保存在本地，未测量的指标保持为空。"}]},
            "methods":[{"language":"text","code":"从标明的 JSON 文件提取同名指标。开发集按 epoch 整理；测试值仅在 results.json 完成后读取。百分比由比例乘 100 展示。"}]}
    add("run",[current],"训练进度与 GPU 采样",["status.json","events.jsonl","run_config.json"],
        [{"label":"进度","definition":"已完成优化器更新次数 / 计划更新次数；完成更新后仍需校准与测试。"},
         {"label":"剩余训练时间","definition":"剩余更新 × 批量 × 梯度累积 / 当前 epoch 累计样本吞吐，近似估计，不含评测。"},
         {"label":"吞吐","definition":"当前 epoch 已处理样本数 / 本轮已用训练秒数。"}])
    add("training",[{"step":e["update"],"epoch":e["epoch"],"time":e["time"],"训练 NLL":e["training_nll"],"样本/秒":e["examples_per_second"]} for e in train_events],
        "训练损失与吞吐",["events.jsonl"],
        [{"label":"训练 NLL","definition":"当前 epoch 从开始到该更新的每样本平均交叉熵，越低越好；每轮重置累计。"}])
    dev = ([{"epoch":0,"stage":"原始 Laya","accuracyRate":baseline["accuracy"],"macroF1Rate":baseline["macro_f1"],"开发 NLL":baseline["nll"],"samples":998}] if baseline else [])
    dev += [{"epoch":e["epoch"],"stage":f"第 {e['epoch']} 轮","accuracyRate":e["development_accuracy"],"macroF1Rate":e["development_macro_f1"],"开发 NLL":e["development_nll"],"samples":998} for e in history]
    add("development",dev,"开发集评测 · 998 条",["baseline-development.json","history.json"],
        [{"label":"准确率","definition":"正确预测样本数 / 998。"}, {"label":"Macro F1","definition":"77 类各自 F1 的等权平均，避免大类主导。"},
         {"label":"模型选择","definition":"比较原始模型及每轮模型的开发集 NLL，选择最小者；不使用测试集选择。"}])
    finals=[]
    selective=[]
    perclass=[]
    if results:
        for name,label in [("baseline","原始 Laya"),("tuned","微调 Laya")]:
            for variant,cal in [("raw","未校准"),("calibrated","温度校准")]:
                m=results["test"][name][variant]
                finals.append({"model":label,"calibration":cal,"accuracyRate":m["accuracy"],"macroF1Rate":m["macro_f1"],"nll":m["nll"],"ece":m["ece_15_bins_max_probability"],"samples":3080})
            for threshold, metric in results["test"][name]["calibrated"]["selective"].items():
                selective.append({"model":label,"threshold":float(threshold),"coverageRate":metric["coverage"],"accuracyRate":metric["accuracy"],"count":metric["count"]})
        base=results["test"]["baseline"]["raw"]
        tuned=results["test"]["tuned"]["raw"]
        perclass=[{"intent":label,"samples":sum(tuned["confusion_matrix"][i]),"baselineRecallRate":base["per_class_recall"][i],"tunedRecallRate":tuned["per_class_recall"][i],"f1Rate":tuned["per_class_f1"][i]} for i,label in enumerate(cfg["labels"])]
        current["accuracyDelta"] = results["accuracy_comparison"]
    add("test",finals,"官方测试集 · 3,080 条",["results.json","test-evaluation-lock.json"],
        [{"label":"准确率","definition":"正确预测样本数 / 3,080；使用原始和微调模型相同的完整候选编码。","componentIds":["test-accuracy","test-probabilities"]},
         {"label":"Macro F1","definition":"77 个类别的 F1 等权平均，每类 F1 = 2TP / (2TP + FP + FN)。","componentIds":["test-accuracy","test-probabilities"]},
         {"label":"NLL","definition":"真实类别概率的负自然对数在 3,080 条样本上的平均值，越低越好。","componentIds":["test-probabilities"]},
         {"label":"ECE","definition":"15 个等宽置信度区间，按区间样本占比加权的置信度与正确率绝对差，越低越好。","componentIds":["test-probabilities"]},
         {"label":"温度校准","definition":"仅在独立校准集 997 条上最小化 NLL 选择温度；正温度不改变分类结果。"}],
        ["本轮仅测英文银行短句的 77 类闭集分类；尚未验证中文、长上下文、未知意图或线上分布。", "没有在相同数据上测量通用 LLM，不构成优于 LLM 的证据。"])
    add("selective",selective,"置信度分流 · 最终测试集",["results.json"],
        [{"label":"覆盖率","definition":"温度校准后最高类别概率 ≥ 阈值的样本数 / 3,080。"},
         {"label":"保留样本准确率","definition":"满足置信度阈值的样本中正确预测的比例；没有样本时为空。"}],
        ["阈值预先固定，只作描述性评估。上线需在真实业务开发集上选择阈值，并另测未知意图。"])
    add("classes",perclass,"每类召回与 F1 · 最终测试集",["results.json","run_config.json"],
        [{"label":"召回率","definition":"真实属于该类且预测正确的样本数 / 真实属于该类的样本数。"}])
    splits=[]
    lengths=bundle.get("input_lengths.json",{})
    names={"train":"训练集","development":"开发集","calibration":"校准集","test":"测试集"}
    purposes={"train":"更新模型参数","development":"按 NLL 选择模型","calibration":"拟合概率温度","test":"最终独立评测"}
    for key in names:
        entry=manifest["files"][key+".jsonl"]
        splits.append({"split":names[key],"count":entry["records"],"groups":entry["groups"],"purpose":purposes[key],
                       "medianTokens":lengths.get(key,{}).get("median"),"maxTokens":lengths.get(key,{}).get("max"),"sha256":entry["sha256"]})
    add("splits",splits,"固定数据划分",["data/banking77-v1/manifest.json","input_lengths.json"],
        [{"label":"划分","definition":"官方训练来源按类别及重复组分配。排除与官方测试重叠的 40 条及冲突标签组 4 条；官方测试保留 3,080 条。"},
         {"label":"输入长度","definition":"包含指令、全部 77 个候选标签和原文；8,192 为上限，按实际长度组批。"}],
        ["近重复为 MinHash 候选 + 字符 5-gram Jaccard≥0.90；不保证检测所有语义重复。", "数据许可 CC BY 4.0；PolyAI / Casanueva et al. 2020。"])
    return {**existing,"generatedAt":stamp,"status":"observed","queries":{**existing.get("queries",{}),**query},"filters":[]}


def build():
    subprocess.run([NODE,BUILDER,"build","--project-dir",str(APP),"--separate-data"],check=True,stdout=subprocess.DEVNULL)


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control","no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/api/version":
            value=json.loads((APP/"src/data.json").read_text())
            # 构建完成后才发布版本，避免刷新到一半写入的静态产物。
            version=(LOCAL/"dashboard-version.json")
            data=version.read_bytes() if version.exists() else b'{}'
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(data)
        else:
            super().do_GET()

    def log_message(self,*args):
        pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch",action="store_true")
    parser.add_argument("--no-build",action="store_true")
    parser.add_argument("--serve",action="store_true")
    parser.add_argument("--serve-only",action="store_true")
    parser.add_argument("--port",type=int,default=4173)
    parser.add_argument("--interval",type=int,default=20)
    args=parser.parse_args()
    server=None
    if args.serve or args.serve_only:
        server=http.server.ThreadingHTTPServer(("127.0.0.1",args.port),functools.partial(Handler,directory=str(APP/"dist")))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        print(f"http://127.0.0.1:{args.port}/",flush=True)
    while not args.serve_only:
        try:
            bundle=collect()
            value=snapshot(bundle)
            atomic_json(APP/"src/data.json",value)
            if not args.no_build:
                build()
                atomic_json(LOCAL/"dashboard-version.json",{"generatedAt":value["generatedAt"]})
            print(json.dumps({"syncedAt":value["generatedAt"],"phase":bundle["status.json"]["phase"]}),flush=True)
            if bundle["status.json"]["phase"] in ("complete","failed") or not args.watch:
                break
        except Exception as exc:
            print(f"同步失败，保留上一份快照：{type(exc).__name__}",flush=True)
            if not args.watch: raise
        time.sleep(max(10,args.interval))
    if server:
        # 训练结束即停止 SSH 轮询；本地看板继续可访问。
        threading.Event().wait()


if __name__=="__main__":
    main()
