"""本机串行评测 JevBench 公开 231 题；不声称完整榜单分数。"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from banking77_data import sha256, write_json
from benchmark_training import hardware_info
from local_laya import MODEL
from typed_laya import TypedLaya, measure, question_labels

LAB = Path(__file__).resolve().parents[1]
SOURCE = LAB / "research/jevbench-source-2026-09-22"


def public_tasks():
    manifest = json.loads((SOURCE / "source-manifest.json").read_text())
    files = {r["path"]:r for r in manifest["files"]}
    rows = []
    for name, tier in [("easy","easy"),("original","standard"),("hard","hard")]:
        relative = f"datasets/public/{name}.jsonl"
        assert sha256(SOURCE / relative) == files[relative]["sha256"]
        for line in (SOURCE / relative).read_text().splitlines():
            row = json.loads(line)
            assert set(question_labels(row["question"])) == set(map(str,row["labels"]))
            rows.append({**row, "tier":tier})
    assert len(rows)==231 and len({r["id"] for r in rows})==231
    return rows, manifest


def summarize(records):
    return {"all":measure(records), **{key:{value:measure([r for r in records if r[key]==value]) for value in sorted({r[key] for r in records})} for key in ["tier","family","type"]}}


def status(output, phase, **extra):
    row={"phase":phase,"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),**extra}
    temporary=output/'status.tmp';write_json(temporary,row);temporary.replace(output/'status.json')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'protocol.json').exists():raise FileExistsError("不覆盖已有评测")
    tasks,manifest=public_tasks()
    configs=[('baseline_512','baseline',512,True),('baseline_8k','baseline',8192,False),('banking_8k','tuned',8192,False)]
    protocol={"created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"scope":"231 public decisions; development diagnostic, not full official ranking",
              "source":manifest,"configs":configs,"temperature":1.0,"precision":"fp32","device":"mps","batch_size":1,
              "hardware":hardware_info('mps'),"gold_in_model_inputs":False,"warmup":"three separate authored smoke questions, one per primitive; no public item selected as warmup",
              "comparability":"512 reference uses locally pinned upstream encoder; source/package revision and hardware differ from official row, not an exact reproduction. 8k preserves instructions and all option descriptions.",
              "model_sha256":{str(p.relative_to(MODEL)):sha256(p) for p in [MODEL/'base/model.safetensors',MODEL/'best/trainable.safetensors']},
              "code_sha256":{p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('typed_laya.py'),Path(__file__).with_name('local_laya.py'),LAB/'reference/laya_common.py']}}
    write_json(args.output/'protocol.json',protocol)
    started=time.perf_counter();results={"complete":False,"configs":{}};completed=0
    with (args.output/'predictions.jsonl').open('w') as stream:
        for key,variant,length,legacy in configs:
            status(args.output,'loading',config=key,completed=completed,total=693)
            model=TypedLaya(variant,'mps')
            warmups=[{"type":"choice","instructions":"Which color is stated?","criteria":{"red":"red","blue":"blue"}},
                     {"type":"noul","instructions":"Is the stated color red?","criteria":{"false":"The color is not red.","true":"The color is red."}},
                     {"type":"score","instructions":"Rate the explicitly stated severity.","criteria":["minor","major","critical"]}]
            for q in warmups:model.decide('The color is red. The severity is minor.',q,length,legacy)
            rows=[]
            for i,task in enumerate(tasks):
                record={"id":task['id'],"config":key,"tier":task['tier'],"family":task['family'],"type":task['question']['type'],"group":task['group'],"expected":str(task['expected']),"gold_probs":task.get('provenance',{}).get('gold_probs')}
                try:
                    prediction=model.decide(task['state'],task['question'],length,legacy)
                    record.update(prediction,ok=True,correct=prediction['prediction']==record['expected'])
                except (ValueError,RuntimeError,FloatingPointError) as error:
                    record.update(ok=False,correct=False,error=f'{type(error).__name__}: {error}')
                    torch.mps.empty_cache()
                rows.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();completed+=1
                if (i+1)%20==0 or i+1==len(tasks):
                    results['configs'][key]=summarize(rows)
                    write_json(args.output/'results.partial.json',results)
                    status(args.output,'running',config=key,completed=completed,total=693,correct=sum(r['correct'] for r in rows),config_completed=len(rows),elapsed_seconds=time.perf_counter()-started)
            results['configs'][key]=summarize(rows)
            results['configs'][key]['load_seconds']=model.load_seconds
            del model;gc.collect();torch.mps.empty_cache()
    results.update(complete=True,wall_seconds=time.perf_counter()-started,completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    write_json(args.output/'results.json',results);status(args.output,'complete',completed=completed,total=693,elapsed_seconds=results['wall_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=LAB/'results/jevbench-public-mps-v1');args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if args.output.exists():status(args.output,'failed',error=f'{type(error).__name__}: {error}')
        raise
