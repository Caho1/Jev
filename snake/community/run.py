import sys,os,json,time,math,getpass,hashlib,random,statistics,resource,subprocess
from pathlib import Path
from collections import defaultdict,Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_client import JevClient
from laya_mlx import Agent
from laya_mlx.common import build_prefix,serialize_state,render_options
import mlx.core as mx
ROOT=Path(__file__).parent

def audit(agent,t):
 result={}
 state_tokens=len(agent.tok(serialize_state(t['state']).replace(agent.tok.mask_token,' '),add_special_tokens=False)['input_ids'])
 for key,definition in t['questions'].items():
  q=agent._to_internal(definition);prefix,_=build_prefix(agent.tok,q,agent.cfg['head_max_len']);room=max(0,agent.cfg['max_len']-len(prefix)-1)
  nt=lambda s:len(agent.tok(s,add_special_tokens=False)['input_ids'])
  opt_lengths=[1+nt(' '+s.replace(agent.tok.mask_token,' ')) for s in render_options(q)]
  original_head=nt(q['t']+' question: '+str(q['ins']).replace(agent.tok.mask_token,' '))
  rubric_truncated=any(n>49 for n in opt_lengths) or original_head+sum(opt_lengths)+3>len(prefix)
  result[key]={'state_tokens':state_tokens,'state_budget':room,'state_truncated':state_tokens>room,'rubric_truncated':rubric_truncated}
 return result

def grade(t,answers):
 out={}
 for k,q in t['questions'].items():
  try:
   a=answers[k];kind=q['type'];expected=t['expected'][k]
   if a.get('type')!=kind:raise ValueError('answer type mismatch')
   if kind=='noul':
    p=a['noul'];assert type(p) in (int,float) and math.isfinite(p) and 0<=p<=1
    probs={'no':1-p,'yes':p};pred=p>.5 if t['scoring']=='reviews' else ('yes' if p>.5 else 'no')
   else:
    probs=a['probabilities'];labels=list(q['criteria']) if kind=='choice' else [str(i) for i in range(len(q['criteria']))]
    assert set(probs)==set(labels) and all(type(p) in (int,float) and math.isfinite(p) and 0<=p<=1 for p in probs.values()) and abs(sum(probs.values())-1)<=.02
    z=sum(probs.values());probs={k:p/z for k,p in probs.items()}
    pred=max(labels,key=probs.get)
    if kind=='score' and t['scoring']=='reviews':pred=min(5,math.floor(a['score']+1.5))
   correct=expected[0]<=pred<=expected[1] if isinstance(expected,list) else pred==expected
   out[k]={'valid':True,'prediction':pred,'expected':expected,'correct':correct}
   if not isinstance(expected,list):
    label=('yes' if expected else 'no') if isinstance(expected,bool) else expected
    out[k]['brier']=sum((p-(key==label))**2 for key,p in probs.items())
  except Exception as e:out[k]={'valid':False,'correct':False,'error':type(e).__name__}
 return out

def percentile(xs,q):return sorted(xs)[max(0,math.ceil(len(xs)*q)-1)] if xs else None

def summarize(rows,meta,total):
 suites={}
 for suite in sorted({r['suite'] for r in rows}):
  subset=[r for r in rows if r['suite']==suite];models={}
  for model in ['laya','jev']:
   entries=[r[model] for r in subset];lat=[e['ms'] for e in entries if e.get('ok')];grades=[g for e in entries for g in e['grades'].values()]
   full=[g for r in subset for k,g in r[model]['grades'].items() if not any(r['audit'][k][f] for f in ['state_truncated','rubric_truncated'])]
   fields={}
   for k in {k for r in subset for k in r[model]['grades']}:
    gs=[r[model]['grades'][k] for r in subset if k in r[model]['grades']];fields[k]={'correct':sum(g['correct'] for g in gs),'n':len(gs)}
   models[model]={'requests':len(entries),'success':sum(e.get('ok',False) for e in entries),'questions':len(grades),'correct':sum(g['correct'] for g in grades),'accuracy':sum(g['correct'] for g in grades)/len(grades),'exact_match':sum(all(g['correct'] for g in e['grades'].values()) for e in entries)/len(entries),'p50_ms':percentile(lat,.5),'p95_ms':percentile(lat,.95),'full_context_n':len(full),'full_context_correct':sum(g['correct'] for g in full),'schema_valid':sum(g['valid'] for g in grades),'brier':statistics.mean([g['brier'] for g in grades if 'brier'in g]) if any('brier'in g for g in grades) else None,'fields':fields}
  suites[suite]={'models':models,'state_truncated_questions':sum(a['state_truncated'] for r in subset for a in r['audit'].values()),'rubric_truncated_questions':sum(a['rubric_truncated'] for r in subset for a in r['audit'].values())}
 report={'metadata':meta,'completed':len(rows),'total':total,'suites':suites,'updated':time.strftime('%Y-%m-%d %H:%M:%S'),'errors':[{'id':r['id'],'model':m,'error':r[m].get('error')} for r in rows for m in ['laya','jev'] if not r[m].get('ok')]}
 tmp=ROOT/'summary.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(ROOT/'summary.json')
 return report

def main():
 key=os.environ.get('TYPESAFE_API_KEY') or getpass.getpass('TypeSafe API key: ')
 tasks=json.loads((ROOT/'corpus.json').read_text());random.Random(2026).shuffle(tasks)
 agent=Agent('convaiinnovations/laya',device='gpu',dtype='float16',batch_size=16)
 client=JevClient(key,timeout=20)
 meta={'hardware':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'laya':'convaiinnovations/laya','revision':agent.model_dir.name,'runtime':'laya-mlx 0.1.0 / MLX 0.32.2 / FP16 / batch_size 16','jev':'jev-1.13.0','seed':2026,'corpus_sha256':hashlib.sha256((ROOT/'corpus.json').read_bytes()).hexdigest(),'sources':json.loads((ROOT/'sources.json').read_text()),'method':'Identical original state/questions per request. Single sequential paired pass, alternating model order. No retries. Three warmups excluded. Remote latency includes network. Invalid/error answers count wrong. Truncation audited; full-context subset reported separately.'}
 warm={'state':'A customer asks for a refund.','questions':{'d':{'type':'choice','instructions':'Classify the request.','criteria':{'billing':'refund or payment','other':'other'}}}}
 for _ in range(3):agent.predict(**warm);client.evaluate(**warm)
 rows=[];raw=ROOT/'raw.jsonl'
 if raw.exists():raise RuntimeError('Existing raw.jsonl: archive it explicitly before a new run')
 for i,t in enumerate(tasks):
  row={'id':t['id'],'suite':t['suite'],'family':t['family'],'group':t['group'],'audit':audit(agent,t)}
  for model in (['laya','jev'] if i%2==0 else ['jev','laya']):
   start=time.perf_counter()
   try:
    if model=='laya':data=agent.predict(t['state'],t['questions']);ms=(time.perf_counter()-start)*1000
    else:data,metrics=client.evaluate(t['state'],t['questions']);ms=metrics['api_ms']
    row[model]={'ok':True,'ms':ms,'answers':data['answers'],'usage':data.get('usage',{}),'grades':grade(t,data['answers'])}
   except Exception as e:
    row[model]={'ok':False,'ms':(time.perf_counter()-start)*1000,'error':str(e),'grades':{k:{'valid':False,'correct':False} for k in t['questions']}}
    if model=='jev':client.close();client=JevClient(key,timeout=20)
  rows.append(row)
  with raw.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  if i%10==0 or i==len(tasks)-1:
   meta['mlx_peak_bytes']=mx.get_peak_memory();meta['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
   summarize(rows,meta,len(tasks));print(f'{i+1}/{len(tasks)} completed',flush=True)
  if len(rows)>=3 and all(not r['jev']['ok'] for r in rows[-3:]):break
 client.close();report=summarize(rows,meta,len(tasks));print(json.dumps(report['suites'],indent=2),flush=True)
if __name__=='__main__':main()
