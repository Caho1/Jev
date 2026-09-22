import json,time,sys,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from laya_mlx import Agent
from run import audit
from live_tests import grade
ROOT=Path(__file__).parent
print('Loading typed-decisions checkpoint',flush=True)
a=Agent('convaiinnovations/laya-typed-decisions',device='gpu',dtype='float16',batch_size=16)
tasks=json.loads((ROOT/'corpus.json').read_text())
for _ in range(3):a.predict(tasks[0]['state'],tasks[0]['questions'])
rows=[]
for i,t in enumerate(tasks):
 start=time.perf_counter()
 try:
  d=a.predict(t['state'],t['questions']);ms=(time.perf_counter()-start)*1000
  r={'id':t['id'],'suite':t['suite'],'audit':audit(a,t),'ms':ms,'grades':grade(t,d['answers']),'answers':d['answers'],'ok':True}
 except Exception as e:r={'id':t['id'],'suite':t['suite'],'ok':False,'error':str(e)}
 rows.append(r)
 if i%50==0:print(i+1,'/',len(tasks),flush=True)
summary={}
for suite in sorted({r['suite'] for r in rows}):
 rs=[r for r in rows if r['suite']==suite];gs=[g for r in rs for g in r.get('grades',{}).values()];times=sorted(r['ms'] for r in rs if r['ok'])
 summary[suite]={'n':len(rs),'correct':sum(g['correct'] for g in gs),'questions':len(gs),'accuracy':sum(g['correct'] for g in gs)/len(gs),'p50_ms':times[(len(times)-1)//2],'state_truncated':sum(v['state_truncated'] for r in rs for v in r.get('audit',{}).values()),'errors':sum(not r['ok'] for r in rs)}
result={'model':'convaiinnovations/laya-typed-decisions','revision':a.model_dir.name,'context':a.cfg['max_len'],'head_budget':a.cfg['head_max_len'],'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),'comparison_note':'Fresh typed-decisions run on unchanged corpus; English and Jev baselines are previous saved measurements, not simultaneous reruns.','summary':summary,'rows':rows}
(ROOT/'typed-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(summary,indent=2),flush=True)
