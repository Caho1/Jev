"""Paired classification + synchronized burst benchmark; no credential persistence."""
import argparse,concurrent.futures as cf,hashlib,json,math,os,random,statistics,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'snake'))
from jev_client import JevClient,Halt


def percentile(xs,q):return sorted(xs)[max(0,math.ceil(len(xs)*q)-1)] if xs else None

def grade(task,answers):
 out={}
 for key,q in task['questions'].items():
  try:
   a=answers[key];probs=a['probabilities'];labels=list(q['criteria'])
   assert a['type']=='choice' and set(probs)==set(labels)
   assert all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in probs.values())
   assert abs(sum(probs.values())-1)<.025 and a['choice'] in labels
   pred=a['choice'];assert probs[pred]>=max(probs.values())-.001
   y=task['expected'][key]
   out[key]=dict(valid=True,prediction=pred,expected=y,correct=pred==y,pmax=max(probs.values()),brier=sum((p-(k==y))**2 for k,p in probs.items()))
  except (KeyError,TypeError,ValueError,AssertionError):out[key]=dict(valid=False,correct=False)
 return out

def audit(agent,t):
 from laya_mlx.common import build_prefix,render_options,serialize_state
 tok=agent.tok;nt=lambda s:len(tok(s,add_special_tokens=False)['input_ids'])
 st=nt(serialize_state(t['state']).replace(tok.mask_token,' '));out={}
 for key,d in t['questions'].items():
  q=agent._to_internal(d);prefix,markers=build_prefix(tok,q,agent.cfg['head_max_len'])
  head=nt(f"{q['t']} question: {q['ins']}".replace(tok.mask_token,' '))
  lens=[1+min(48,nt(' '+v.replace(tok.mask_token,' '))) for v in render_options(q)]
  option_crop=any(nt(' '+v.replace(tok.mask_token,' '))>48 for v in render_options(q))
  budget=agent.cfg['head_max_len']-sum(lens)
  if budget<16:
   per=max(4,(agent.cfg['head_max_len']-16)//len(lens))
   option_crop |= any(n>per for n in lens)
   lens=[min(n,per) for n in lens];budget=agent.cfg['head_max_len']-sum(lens)
  room=max(0,agent.cfg['max_len']-len(prefix)-1)
  out[key]=dict(state_tokens=st,state_budget=room,state_truncated=st>room,rubric_truncated=option_crop or head>max(8,budget))
 return out

def batched_predict(agent,tasks):
 """Cross-request batching using public prepare/forward; choice-only corpus."""
 import numpy as np
 from laya_mlx.agent import collate_items
 from laya_mlx.common import confidence_from_probs,temp_bucket
 entries=[];results=[{'answers':{}} for _ in tasks]
 for ti,t in enumerate(tasks):
  items,qs=agent.prepare(t['state'],t['questions'])
  entries.extend((ti,key,item,q) for key,item,q in zip(t['questions'],items,qs))
 for start in range(0,len(entries),agent.batch_size):
  chunk=entries[start:start+agent.batch_size]
  batch=collate_items([x[2] for x in chunk],agent.tok.pad_token_id,max_length=agent.cfg['max_len'])
  logits,act=agent.forward(batch);logits=np.asarray(logits)
  if not np.isfinite(logits).all():raise ValueError('Nonfinite logits')
  for row,(ti,key,item,q) in enumerate(chunk):
   assert q['t']=='choice'
   k=len(item['markers']);temp=agent.temperature_by_options.get(temp_bucket(item['qtype'],k),agent.temperature[item['qtype']])
   z=logits[row,:k]/max(1e-3,float(temp));p=np.exp(z-z.max());p/=p.sum();labels=list(q['crit'])
   results[ti]['answers'][key]=dict(type='choice',choice=labels[int(p.argmax())],probabilities={k:round(float(v),4) for k,v in zip(labels,p)},confidence=round(confidence_from_probs(p,len(labels)),4))
 return results

class Runner:
 def __init__(self,name):
  self.name=name;self.agent=None;self.clients=[];self.local=threading.local();self.stop=threading.Event()
  if name!='jev':
   from laya_mlx import Agent
   import mlx.core as mx
   start=time.perf_counter()
   model='convaiinnovations/laya'+('-typed-decisions' if name=='typed' else '')
   self.agent=Agent(model,device='gpu',dtype='float16',batch_size=16)
   self.meta=dict(model=model,revision=self.agent.model_dir.name,context=self.agent.cfg['max_len'],head_budget=self.agent.cfg['head_max_len'],load_ms=(time.perf_counter()-start)*1000,mlx_version=mx.__version__,runtime='MLX FP16, no compile, no result cache',batch_question_limit=16)
  else:self.meta=dict(model='jev-1.13.0',runtime='HTTPS connection per worker, no retries')
 def call(self,t):
  if self.agent:return self.agent.predict(t['state'],t['questions'])
  if self.stop.is_set():raise RuntimeError('Skipped after API failure')
  if not hasattr(self.local,'client'):
   self.local.client=JevClient(timeout=20);self.clients.append(self.local.client)
  try:
   data,metrics=self.local.client.evaluate(t['state'],t['questions'])
   return data
  except Halt:
   self.stop.set();raise
 def close(self):
  for c in self.clients:c.close()

def result(t,d,ms,error=None,**extra):
 gs=grade(t,d.get('answers',{}))
 return dict(id=t['id'],suite=t['suite'],ok=error is None,ms=ms,answers=d.get('answers',{}),grades=gs,error=error,usage=d.get('usage',{}),**extra)

def save(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)

def aggregate(rows):
 gs=[g for r in rows for g in r['grades'].values()];times=[r['ms'] for r in rows if r['ok']]
 return dict(requests=len(rows),success=sum(r['ok'] for r in rows),questions=len(gs),correct=sum(g['correct'] for g in gs),accuracy=sum(g['correct'] for g in gs)/len(gs) if gs else None,exact_match=sum(r['ok'] and all(g['correct'] for g in r['grades'].values()) for r in rows)/len(rows) if rows else None,p50_ms=percentile(times,.5),p95_ms=percentile(times,.95),schema_valid=sum(g['valid'] for g in gs),brier=statistics.mean(g['brier'] for g in gs if 'brier'in g) if any('brier'in g for g in gs) else None,deadline_success={str(deadline):sum(r['ok'] and r['ms']<=deadline and all(g['correct'] for g in r['grades'].values()) for r in rows)/len(rows) if rows else None for deadline in [50,100,500,1000]})

def run_accuracy(runner,tasks,out):
 path=out/f'{runner.name}-accuracy.json'
 if path.exists():print('Existing accuracy reused:',runner.name,flush=True);return json.loads(path.read_text())
 for t in tasks[:3]:runner.call(t)
 rows=[];started=time.time()
 raw=out/f'{runner.name}-accuracy.jsonl'
 if raw.exists():raise RuntimeError('Partial run exists; inspect before retrying')
 for i,t in enumerate(tasks):
  a=audit(runner.agent,t) if runner.agent else None
  start=time.perf_counter();error=None;data={}
  try:data=runner.call(t)
  except Exception as e:error=str(e)
  row=result(t,data,(time.perf_counter()-start)*1000,error,audit=a)
  rows.append(row)
  with raw.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  if (i+1)%32==0:print(runner.name,'accuracy',i+1,'/',len(tasks),flush=True)
  if runner.stop.is_set():break
 report=dict(metadata=runner.meta,started=started,completed=time.time(),rows=rows,summary={s:aggregate([r for r in rows if r['suite']==s]) for s in sorted({t['suite'] for t in tasks})})
 save(path,report);return report

def run_load(runner,tasks,out,resume=False):
 path=out/f'{runner.name}-load.json'
 if path.exists():print('Existing load reused:',runner.name,flush=True);return
 if runner.stop.is_set():return
 groups={
  'short':[t for t in tasks if t['suite'] in ['intent_en','intent_zh','sentiment']],
  'mixed':[t for t in tasks if t['suite'] in ['policy','workflow','arithmetic','multifield']]}
 rawpath=out/f'{runner.name}-load.jsonl'
 if rawpath.exists() and not resume:raise RuntimeError('Partial load exists: use --resume-load to preserve completed rounds')
 records=[json.loads(line) for line in rawpath.read_text().splitlines()] if rawpath.exists() else []
 completed={(r['workload'],r['concurrency'],r['repeat'],r['mode']) for r in records}
 parity=[]
 if runner.agent:
  probe=[next(t for t in tasks if t['suite']==s) for s in ['intent_en','intent_zh','policy','multifield']]
  plain=[runner.call(t) for t in probe];batched=batched_predict(runner.agent,probe)
  for t,a,b in zip(probe,plain,batched):
   delta=max(abs(a['answers'][k]['probabilities'][v]-b['answers'][k]['probabilities'][v]) for k,q in t['questions'].items() for v in q['criteria'])
   same=all(a['answers'][k]['choice']==b['answers'][k]['choice'] for k in t['questions'])
   parity.append(dict(id=t['id'],max_probability_delta=delta,same_choice=same))
  if not all(x['same_choice'] and x['max_probability_delta']<.01 for x in parity):raise RuntimeError('Cross-request batch parity failed')
 modes=['serial_queue','microbatch'] if runner.agent else ['parallel_api']
 for workload,eligible in groups.items():
  for concurrency in [1,4,8,16]:
   for repeat in range(2):
    rng=random.Random(901+repeat);jobs=rng.sample(eligible,48)
    for mode in modes:
     if (workload,concurrency,repeat,mode) in completed:continue
     pool=cf.ThreadPoolExecutor(max_workers=concurrency) if not runner.agent else None
     try:
      if pool:
       barrier=threading.Barrier(concurrency)
       def warm():barrier.wait(timeout=30);return runner.call(jobs[0])
       list(pool.map(lambda _:warm(),range(concurrency)))
      elif mode=='microbatch':batched_predict(runner.agent,jobs[:concurrency])
      else:runner.call(jobs[0])
      rows=[];wall_start=time.perf_counter()
      for start in range(0,len(jobs),concurrency):
       wave=jobs[start:start+concurrency];release=time.perf_counter()
       if mode=='microbatch':
        try:
         data=batched_predict(runner.agent,wave);elapsed=(time.perf_counter()-release)*1000
         rows.extend(result(t,d,elapsed,wave=start//concurrency) for t,d in zip(wave,data))
        except Exception as e:
         rows.extend(result(t,{},(time.perf_counter()-release)*1000,str(e)) for t in wave)
       elif mode=='serial_queue':
        for t in wave:
         begin=time.perf_counter()
         try:d=runner.call(t);error=None
         except Exception as e:d={};error=str(e)
         rows.append(result(t,d,(time.perf_counter()-release)*1000,error,service_ms=(time.perf_counter()-begin)*1000,wave=start//concurrency))
       else:
        def job(t):
         begin=time.perf_counter();data={};error=None
         try:data=runner.call(t)
         except Exception as e:error=str(e)
         return result(t,data,(time.perf_counter()-release)*1000,error,service_ms=(time.perf_counter()-begin)*1000,wave=start//concurrency)
        rows.extend(pool.map(job,wave))
       if runner.stop.is_set():break
      wall=time.perf_counter()-wall_start;s=aggregate(rows)
      s.update(wall_s=wall,requests_per_s=sum(r['ok'] for r in rows)/wall,correct_requests_per_s=sum(r['ok'] and all(g['correct'] for g in r['grades'].values()) for r in rows)/wall,questions_per_s=sum(len(r['grades']) for r in rows if r['ok'])/wall)
      rec=dict(workload=workload,concurrency=concurrency,repeat=repeat,mode=mode,summary=s,rows=rows)
      records.append(rec)
      with (out/f'{runner.name}-load.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
      print(runner.name,workload,mode,'C',concurrency,'round',repeat,round(s['requests_per_s'],1),'req/s',flush=True)
     except Exception as e:
      failure=dict(timestamp=time.time(),model=runner.name,workload=workload,concurrency=concurrency,repeat=repeat,mode=mode,error=str(e),stage='warmup_or_setup',measured_round_recorded=False)
      with (out/'setup-failures.jsonl').open('a') as f:f.write(json.dumps(failure)+'\n')
      raise
     finally:
      if pool:
       pool.shutdown()
       # Each subsequent executor owns fresh thread-local connections.
       runner.close();runner.clients=[]
     if runner.stop.is_set():break
    if runner.stop.is_set():break
   if runner.stop.is_set():break
  if runner.stop.is_set():break
 save(path,dict(metadata=runner.meta,parity=parity,records=records))

def main():
 p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',choices=['english','typed','jev'],default=['english','typed','jev']);p.add_argument('--load',action='store_true');p.add_argument('--resume-load',action='store_true');a=p.parse_args()
 tasks=json.loads((ROOT/'corpus.json').read_text());out=ROOT/'results';out.mkdir(exist_ok=True)
 if 'jev' in a.models and not os.environ.get('TYPESAFE_API_KEY'):p.error('TYPESAFE_API_KEY is missing; source the local environment first')
 manifest=dict(corpus_sha256=hashlib.sha256((ROOT/'corpus.json').read_bytes()).hexdigest(),seed=260921,cases=len(tasks),questions=sum(len(t['questions']) for t in tasks),python=sys.version,method='Frozen synthetic cases; labels never sent. Accuracy single pass after 3 warmups. Load: synchronized bursts at C=1/4/8/16, 48 requests per round, 2 rounds per workload. All clients in a burst share release time; next burst waits for all. Local serial queue and explicit cross-request microbatch tested separately. GPU calls never run concurrently from Python threads. API one warm TLS connection per worker. No result caching, no silent retries. Successful latency quantiles plus errors and quality-adjusted throughput. Repeated trials and translated/position variants are not independent population samples.')
 existing=out/'manifest.json'
 if existing.exists() and json.loads(existing.read_text())['corpus_sha256']!=manifest['corpus_sha256']:raise RuntimeError('Corpus changed; use a new results folder')
 save(existing,manifest)
 for model in a.models:
  r=Runner(model)
  try:
   run_accuracy(r,tasks,out)
   if a.load:run_load(r,tasks,out,resume=a.resume_load)
  finally:r.close()
  del r
  if model!='jev':
   import gc,mlx.core as mx
   gc.collect();mx.clear_cache()
 print('Completed requested runs',flush=True)
if __name__=='__main__':main()
