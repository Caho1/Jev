"""Local MLX worker for the paired Snake comparison."""
import json, time, threading, subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
from live_tests import task_for, test_result
POOL=ThreadPoolExecutor(max_workers=1)
AGENT=None
TYPED=None
STATUS={'ready':False,'error':None,'model':'convaiinnovations/laya','runtime':'MLX FP16','hardware':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()}
def prompt(moves):
 safe=[m for m in moves if m['safe']]
 preferred=max(safe,key=lambda m:m.get('advance',0))['direction'] if safe else None
 criteria={m['direction']:('Blocked. Collision.' if not m['legal'] else 'Unsafe. Traps the snake.' if not m['safe'] else 'Safe. Eat food now. Best.' if m.get('eat') else 'Safe. Best route to food.' if m['direction']==preferred else 'Safe. Slower route.') for m in moves}
 return 'Safe route: '+('yes.' if safe else 'no.'),{'move':{'type':'choice','instructions':'Choose the best safe move toward food.','criteria':criteria}}
def load():
 global AGENT, TYPED
 try:
  from laya_mlx import Agent
  start=time.perf_counter();AGENT=Agent('convaiinnovations/laya',device='gpu',dtype='float16',batch_size=16)
  STATUS['load_ms']=(time.perf_counter()-start)*1000
  state,q=prompt([{'direction':d,'safe':d=='RIGHT','legal':d=='RIGHT','advance':1} for d in ['UP','DOWN','LEFT','RIGHT']])
  for _ in range(3):AGENT.predict(state,q)
  TYPED=Agent('convaiinnovations/laya-typed-decisions',device='gpu',dtype='float16',batch_size=16)
  TYPED.predict(state,q)
  STATUS.update(ready=True,revision=AGENT.model_dir.name,test_default='typed-decisions',test_models=['typed-decisions','english'])
  print('Laya ready',json.dumps(STATUS),flush=True)
 except Exception as e:STATUS['error']=type(e).__name__+': '+str(e);print(STATUS['error'],flush=True)
def predict(body):
 if '_test_id' in body:
  task=task_for({'id':body['_test_id']})
  variant=body.get('variant','typed-decisions')
  if variant not in ['english','typed-decisions']:raise ValueError('Unknown Laya variant')
  agent=TYPED if variant=='typed-decisions' else AGENT
  start=time.perf_counter();result=agent.predict(task['state'],task['questions']);ms=(time.perf_counter()-start)*1000
  output=test_result(task,result,ms)
  from community.run import audit
  output.update(model=agent.model_id,revision=agent.model_dir.name,context=agent.cfg['max_len'],audit=audit(agent,task))
  return output
 state,q=prompt(body['candidates']);start=time.perf_counter();result=AGENT.predict(state,q);ms=(time.perf_counter()-start)*1000
 answer=result['answers']['move']
 return {'choice':answer['choice'],'probabilities':answer['probabilities'],'ms':ms,'model':'convaiinnovations/laya','usage':result.get('usage',{})}
class Handler(BaseHTTPRequestHandler):
 def cors(self):
  origin=self.headers.get('Origin')
  if origin in ['http://127.0.0.1:8877','http://localhost:8877']:self.send_header('Access-Control-Allow-Origin',origin)
  self.send_header('Access-Control-Allow-Headers','Content-Type');self.send_header('Access-Control-Allow-Methods','GET, POST, OPTIONS')
 def reply(self,status,data):
  b=json.dumps(data).encode();self.send_response(status);self.cors();self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_OPTIONS(self):self.send_response(204);self.cors();self.end_headers()
 def do_GET(self):self.reply(200,STATUS)
 def do_POST(self):
  if self.headers.get('Origin') not in [None,'http://127.0.0.1:8877','http://localhost:8877']:return self.reply(403,{'error':'Origin rejected'})
  if not STATUS['ready']:return self.reply(503,{'error':STATUS['error'] or 'Laya is loading'})
  try:
   n=int(self.headers.get('Content-Length','0'))
   if not 0<n<40000:raise ValueError('Invalid request size')
   body=json.loads(self.rfile.read(n))
   if self.path=='/api/test':body={'_test_id':task_for(body)['id'],'variant':body.get('variant','typed-decisions')}
   self.reply(200,POOL.submit(predict,body).result())
  except Exception as e:self.reply(502,{'error':type(e).__name__+': '+str(e)})
 def log_message(self,*args):pass
if __name__=='__main__':
 POOL.submit(load)
 print('Laya worker: http://127.0.0.1:8878',flush=True)
 ThreadingHTTPServer(('127.0.0.1',8878),Handler).serve_forever()
