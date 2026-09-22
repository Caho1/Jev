"""冻结 BANKING77 测试集上的 Jev 准确性对照；API 结果只用于评测。"""

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

from banking77_data import INSTRUCTION, sha256
from banking77_dashboard import atomic_json

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "snake"))
from jev_client import JevClient, Halt

MODEL = "jev-1.13.0"
OUTPUT = LAB / "results/banking77-jev-v1"
DATA = LAB / "data/banking77-v1"
LAYA = LAB / "results/banking77-lora-v1"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def request_for(row, labels):
    # 与 Laya 使用同一个指令和原始类别名；不发送 ID、标签或其他评测元数据。
    return row["state"], {"intent": {"type":"choice", "instructions":INSTRUCTION,
                                   "criteria":{name:name for name in labels}}}


def decode(data, labels):
    answer = data.get("answers", {}).get("intent", {})
    values = answer.get("probabilities")
    choice = answer.get("choice")
    if answer.get("type") != "choice" or choice not in labels or not isinstance(values, dict) or set(values) != set(labels):
        raise ValueError("返回类型或候选集合不匹配")
    probs = np.array([values[k] for k in labels], dtype=np.float64)
    if any(type(values[k]) not in (int,float) for k in labels) or not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any():
        raise ValueError("返回概率无效")
    total = float(probs.sum())
    if abs(total-1)>0.025:
        raise ValueError("概率总和超出预先约定的舍入容差")
    prediction = labels.index(choice)
    if probs[prediction] < probs.max()-0.001:
        raise ValueError("choice 与概率最大值不一致")
    # API 概率有舍入，分类以 API 的 choice 为准，不用并列概率的字典位置替换它。
    return prediction, probs/total, total


def classification(prediction, expected):
    prediction=np.asarray(prediction,dtype=int)
    expected=np.asarray(expected,dtype=int)
    matrix=np.zeros((77,77),dtype=int)
    valid=prediction>=0
    np.add.at(matrix,(expected[valid],prediction[valid]),1)
    support=np.bincount(expected,minlength=77)
    tp=matrix.diagonal()
    f1=np.divide(2*tp,support+matrix.sum(0),out=np.zeros(77),where=(support+matrix.sum(0))>0)
    recall=np.divide(tp,support,out=np.zeros(77),where=support>0)
    return {"records":len(expected),"valid":int(valid.sum()),"failures":int((~valid).sum()),
            "correct":int((prediction==expected).sum()),"accuracy":float((prediction==expected).mean()),
            "macro_f1":float(f1.mean()),"per_class_f1":f1.tolist(),"per_class_recall":recall.tolist(),
            "confusion_matrix":matrix.tolist(),"support":support.tolist()}


def paired(rows, a, b):
    # 正值表示 b 比 a 准确；按已有重复组重采样。
    gold=np.array([r["label"] for r in rows])
    delta=(b==gold).astype(float)-(a==gold).astype(float)
    groups={}
    for i,row in enumerate(rows): groups.setdefault(row["group_id"],[]).append(i)
    sums=np.array([delta[idx].sum() for idx in groups.values()])
    counts=np.array([len(idx) for idx in groups.values()])
    rng=np.random.default_rng(20260921)
    estimates=[]
    for _ in range(2000):
        idx=rng.integers(0,len(sums),len(sums))
        estimates.append(sums[idx].sum()/counts[idx].sum())
    return {"accuracy_difference_b_minus_a":float(delta.mean()),
            "group_bootstrap_95_percent":np.quantile(estimates,[.025,.975]).tolist(),
            "b_only_correct":int(((b==gold)&(a!=gold)).sum()),
            "a_only_correct":int(((a==gold)&(b!=gold)).sum()),
            "both_correct":int(((a==gold)&(b==gold)).sum()),
            "both_wrong":int(((a!=gold)&(b!=gold)).sum()),"resamples":2000}


class Runner:
    def __init__(self, labels, rate):
        self.labels=labels
        self.rate=rate
        self.local=threading.local()
        self.clients=[]
        self.lock=threading.Lock()
        self.next_time=0.
        self.stop=threading.Event()

    def call(self,row):
        if self.stop.is_set(): return None
        with self.lock:
            start=max(time.monotonic(),self.next_time)
            self.next_time=start+1/self.rate
        time.sleep(max(0,start-time.monotonic()))
        if self.stop.is_set(): return None
        if not hasattr(self.local,"client"):
            self.local.client=JevClient(model=MODEL,timeout=30)
            with self.lock:self.clients.append(self.local.client)
        state,questions=request_for(row,self.labels)
        record={"id":row["id"],"started_at":now(),"request_sha256":hashlib.sha256(json.dumps({"model":MODEL,"state":state,"questions":questions},ensure_ascii=False,separators=(",",":")).encode()).hexdigest()}
        started=time.monotonic()
        try:
            data,meta=self.local.client.evaluate(state,questions)
            prediction,probs,total=decode(data,self.labels)
            record.update(ok=True,prediction=prediction,choice=self.labels[prediction],probabilities=probs.tolist(),
                          probability_sum_before_normalization=total,answers=data["answers"],metadata=meta,
                          returned_model=data.get("model"))
        except (Halt,ValueError) as exc:
            record.update(ok=False,prediction=-1,error=str(exc))
            # 不自动重试已计费或不确定的调用。中止派发，保留已完成项以便明确恢复。
            self.stop.set()
        except Exception as exc:
            record.update(ok=False,prediction=-1,error=type(exc).__name__)
            self.stop.set()
        record.update(elapsed_ms=(time.monotonic()-started)*1000,completed_at=now())
        return record

    def close(self):
        for client in self.clients:client.close()


def save_report(rows, records, protocol, directory):
    mapping={r["id"]:r for r in records}
    selected=[row for row in rows if row["id"] in mapping]
    gold=np.array([r["label"] for r in selected])
    jev=np.array([mapping[r["id"]]["prediction"] for r in selected])
    report={"model":MODEL,"completed_at":now(),"complete":len(selected)==len(rows),"records":len(selected),"total":len(rows),
            "protocol":protocol,"models":{},"paired":{},"failures":[{"id":r["id"],"error":r["error"]} for r in records if not r["ok"]]}
    if not selected:
        return report
    predictions={"jev":jev}
    for name in ("baseline","tuned"):
        raw=np.load(LAYA/f"{name}-test.npz")
        by_id={key:int(label) for key,label in zip(raw["ids"].tolist(),raw["logits"].argmax(1))}
        predictions[name]=np.array([by_id[row["id"]] for row in selected])
    for name,prediction in predictions.items():report["models"][name]=classification(prediction,gold)
    report["paired"]={name:paired(selected,predictions[name],jev) for name in ("baseline","tuned")}
    valid=jev>=0
    report["valid_subset"]={name:classification(pred[valid],gold[valid]) for name,pred in predictions.items()} if valid.any() else {}
    latency=[r["elapsed_ms"] for r in records if r["ok"]]
    report["jev_request_latency_ms"]={"p50":float(np.quantile(latency,.5)),"p95":float(np.quantile(latency,.95))} if latency else None
    report["usage"]={}
    for record in records:
        for key,value in record.get("metadata",{}).get("usage",{}).items():
            if type(value) in (int,float): report["usage"][key]=report["usage"].get(key,0)+value
    atomic_json(directory/"results.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--requests-per-second",type=float,default=8.)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if not os.environ.get("TYPESAFE_API_KEY"): raise SystemExit("环境缺少 TYPESAFE_API_KEY")
    manifest=json.loads((DATA/"manifest.json").read_text())
    for name in ["test.jsonl","labels.json"]:
        if sha256(DATA/name)!=manifest["files"][name]["sha256"]:raise ValueError("固定数据校验失败")
    rows=[json.loads(line) for line in (DATA/"test.jsonl").read_text().splitlines()]
    labels=json.loads((DATA/"labels.json").read_text())
    if len(rows)!=3080 or len(labels)!=77: raise ValueError("数据数量异常")
    for name in ("baseline","tuned"):
        data=np.load(LAYA/f"{name}-test.npz")
        if data["ids"].tolist()!=[r["id"] for r in rows] or data["labels"].tolist()!=[r["label"] for r in rows]:
            raise ValueError("Laya 预测与冻结测试集不一致")
    protocol={"model":MODEL,"endpoint":"https://api.typesafe.ai/v1/systemone","test_sha256":sha256(DATA/"test.jsonl"),
              "labels_sha256":sha256(DATA/"labels.json"),"instructions":INSTRUCTION,"criteria":"canonical label name -> identical label name",
              "state":"original customer message string only","gold_labels_sent":False,"purpose":"evaluation_only_not_training",
              "laya_predictions_sha256":{name:sha256(LAYA/f"{name}-test.npz") for name in ["baseline","tuned"]},
              "selected_laya_checkpoint":json.loads((LAYA/"test-evaluation-lock.json").read_text()),
              "seed":20260921,"workers":args.workers,"requests_per_second":args.requests_per_second,
              "retries":"none; preserve successes; explicit resume only continues unattempted cases",
              "accuracy":"API choice; errors count incorrect; also report success-only paired subset",
              "schema":"77 finite probabilities in [0,1], sum tolerance .025; choice within .001 of maximum; normalize for storage only",
              "calibration":"no extra Jev calibration calls; comparison focuses on classification accuracy and Macro F1"}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    manifest_path=OUTPUT/"protocol.json"
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous["protocol"]!=protocol:raise ValueError("协议已冻结，不能更改")
        if not args.resume:raise ValueError("已有评测记录；如需继续未请求样本，使用 --resume")
    else:
        atomic_json(manifest_path,{"created_at":now(),"protocol":protocol,"code_sha256":sha256(__file__)})
    raw_path=OUTPUT/"responses.jsonl"
    records=[json.loads(line) for line in raw_path.read_text().splitlines()] if raw_path.exists() else []
    completed={r["id"] for r in records}
    if len(completed)!=len(records):raise ValueError("重复评测 ID")
    order=list(rows);random.Random(protocol["seed"]).shuffle(order)
    pending=[row for row in order if row["id"] not in completed]
    if args.limit is not None:pending=pending[:args.limit]
    runner=Runner(labels,args.requests_per_second)
    started=time.monotonic()
    def progress(phase):
        current={"phase":phase,"updated_at":now(),"completed":len(records),"total":len(rows),"success":sum(r["ok"] for r in records),"errors":sum(not r["ok"] for r in records),"wall_seconds_this_session":time.monotonic()-started}
        atomic_json(OUTPUT/"status.json",current)
        print(json.dumps(current,ensure_ascii=False),flush=True)
    progress("running")
    try:
        with raw_path.open("a",buffering=1) as stream, cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            iterator=iter(pending)
            active={pool.submit(runner.call,row) for row in [next(iterator,None) for _ in range(args.workers)] if row is not None}
            while active:
                done,active=cf.wait(active,return_when=cf.FIRST_COMPLETED)
                for future in done:
                    record=future.result()
                    if record:
                        stream.write(json.dumps(record,ensure_ascii=False)+"\n");stream.flush();os.fsync(stream.fileno())
                        records.append(record)
                        if len(records)%80==0:progress("running")
                    if not runner.stop.is_set():
                        next_row=next(iterator,None)
                        if next_row is not None:active.add(pool.submit(runner.call,next_row))
        report=save_report(rows,records,protocol,OUTPUT)
        phase="complete" if report["complete"] else "stopped_on_error" if runner.stop.is_set() else "partial"
        progress(phase)
        print(json.dumps({"records":report["records"],"models":{k:{m:v[m] for m in ["accuracy","macro_f1","failures"]} for k,v in report["models"].items()}},ensure_ascii=False),flush=True)
    finally:runner.close()


if __name__=="__main__":main()
