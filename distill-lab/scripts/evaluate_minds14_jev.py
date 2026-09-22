"""在冻结中文外部评测上获取 Jev 对照；预测仅用于评测，不进入训练。"""
import concurrent.futures as cf
import hashlib
import json
import re
import sys
import time
from pathlib import Path
import numpy as np
from banking77_data import INSTRUCTION,normalize,sha256,write_json
from banking77_dashboard import atomic_json
from evaluate_banking77_jev import Runner,decode,now
from evaluate_minds14_local import measure

LAB=Path(__file__).resolve().parents[1]
DATA=LAB/'data/minds14-zh-v1'
OUT=LAB/'results/minds14-jev-v1'


class ChineseRunner(Runner):
    def call(self,row):
        if self.stop.is_set():return None
        with self.lock:
            start=max(time.monotonic(),self.next_time);self.next_time=start+1/self.rate
        time.sleep(max(0,start-time.monotonic()))
        if self.stop.is_set():return None
        from evaluate_banking77_jev import JevClient,MODEL
        if not hasattr(self.local,'client'):
            self.local.client=JevClient(model=MODEL,timeout=30)
            with self.lock:self.clients.append(self.local.client)
        criteria=json.loads((DATA/'criteria.json').read_text())
        questions={'intent':{'type':'choice','instructions':INSTRUCTION,'criteria':criteria}}
        record={'id':row['id'],'started_at':now(),'request_sha256':hashlib.sha256(json.dumps({'model':MODEL,'state':row['text_zh'],'questions':questions},ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
        start=time.monotonic()
        try:
            response,meta=self.local.client.evaluate(row['text_zh'],questions)
            prediction,probs,total=decode(response,self.labels)
            record.update(ok=True,prediction=prediction,choice=self.labels[prediction],probabilities=probs.tolist(),probability_sum_before_normalization=total,answers=response['answers'],metadata=meta,returned_model=response.get('model'))
        except Exception as error:
            # 客户端错误已经脱敏；其余异常只保存类型，不泄漏网络响应和凭据。
            from evaluate_banking77_jev import Halt
            record.update(ok=False,prediction=-1,error=str(error) if isinstance(error,(Halt,ValueError)) else type(error).__name__)
            self.stop.set()
        record.update(elapsed_ms=(time.monotonic()-start)*1000,completed_at=now())
        return record


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('不覆盖已发出的 API 评测')
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert sha256(DATA/'evaluation.jsonl')==manifest['files']['evaluation.jsonl']['sha256']
    rows=[json.loads(x) for x in (DATA/'evaluation.jsonl').read_text().splitlines()]
    labels=list(json.loads((DATA/'criteria.json').read_text()))
    write_json(OUT/'protocol.json',{'created_at':now(),'model':'jev-1.13.0','rows':502,'purpose':'evaluation only; never training or checkpoint selection','instruction':INSTRUCTION,'criteria':json.loads((DATA/'criteria.json').read_text()),'dataset_sha256':sha256(DATA/'evaluation.jsonl'),'source_revision':manifest['revision'],'gold_in_request':False,'language':'zh-CN source transcription','rate_per_second':4,'workers':4,'code_sha256':sha256(Path(__file__))})
    runner=ChineseRunner(labels,4);completed=[];started=time.perf_counter()
    try:
        with (OUT/'responses.jsonl').open('w') as stream,cf.ThreadPoolExecutor(max_workers=4) as executor:
            # 单个失败停止派发；没有不确定请求的隐式重试。
            pending={executor.submit(runner.call,r) for r in rows[:4]};index=4
            while pending:
                ready,pending=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
                for future in ready:
                    record=future.result()
                    if record:
                        completed.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                    if index<len(rows) and not runner.stop.is_set():pending.add(executor.submit(runner.call,rows[index]));index+=1
                if len(completed)%20<4:
                    state={'phase':'running','completed':len(completed),'total':len(rows),'updated_at':now()}
                    atomic_json(OUT/'status.json',state);print(json.dumps(state),flush=True)
    finally:runner.close()
    by_id={r['id']:r for r in completed};summary={}
    for scope in ['all','han']:
        subset=[r for r in rows if scope=='all' or re.search('[\u3400-\u9fff]',r['text_zh'])]
        expected=np.array([r['label'] for r in subset]);pred=np.array([by_id.get(r['id'],{}).get('prediction',-1) for r in subset])
        correct=pred==expected;matrix=np.zeros((14,14),dtype=int)
        for y,p in zip(expected,pred):
            if p>=0:matrix[y,p]+=1
        support=np.bincount(expected,minlength=14);tp=matrix.diagonal()
        summary[scope]={'records':len(subset),'completed':sum(r['id'] in by_id for r in subset),'valid':int((pred>=0).sum()),'correct':int(correct.sum()),'accuracy':float(correct.mean()),'macro_f1':float((2*tp/np.maximum(1,support+matrix.sum(0))).mean()),'per_class_recall':(tp/np.maximum(1,support)).tolist(),'confusion':matrix.tolist()}
    result={'complete':len(completed)==len(rows) and all(r['ok'] for r in completed),'scopes':summary,'wall_seconds':time.perf_counter()-started,'failures':[{'id':r['id'],'error':r['error']} for r in completed if not r['ok']],'model':'jev-1.13.0','completed_at':now()}
    write_json(OUT/'results.json',result)
    atomic_json(OUT/'status.json',{'phase':'complete' if result['complete'] else 'failed','completed':len(completed),'total':len(rows),'updated_at':now()})
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
