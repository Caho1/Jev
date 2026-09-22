import json, math, time, uuid
from pathlib import Path
ROOT=Path(__file__).parent
TASKS={t["id"]:t for t in json.loads((ROOT/"community/corpus.json").read_text())}
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

def task_for(body):
 if not isinstance(body,dict) or body.get('id') not in TASKS: raise ValueError('Unknown test id')
 return TASKS[body['id']]
def test_result(task,data,ms):
 return {'id':task['id'],'request_id':str(uuid.uuid4()),'timestamp':time.time(),'ms':ms,'answers':data['answers'],'grades':grade(task,data['answers']),'usage':data.get('usage',{}),'ok':True}
