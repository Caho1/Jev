"""Local snake UI and Jev bridge. Credentials stay in the server environment."""
import json, os, time, math, threading
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from jev_client import JevClient, decode_choice, Halt
from live_tests import task_for, test_result
ROOT=Path(__file__).parent
CLIENT=None
LOCK=threading.Lock()
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
 def do_GET(self):
  if self.path=='/api/status':return self.reply(200,{'available':bool(os.environ.get('TYPESAFE_API_KEY')),'model':'jev-1.13.0'})
  super().do_GET()
 def reply(self,status,data):
  raw=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def do_POST(self):
  if self.path=='/api/test':return self.run_test()
  if self.path!='/api/decision':return self.reply(404,{'error':'Not found'})
  if self.headers.get('Origin') not in (None,'http://127.0.0.1:8877','http://localhost:8877'):return self.reply(403,{'error':'Origin rejected'})
  if not os.environ.get('TYPESAFE_API_KEY'):return self.reply(503,{'error':'未配置 TYPESAFE_API_KEY，请使用本地模式。'})
  try:
   n=int(self.headers.get('Content-Length','0'))
   if not 0<n<40000:raise ValueError('Invalid state size')
   state=json.loads(self.rfile.read(n));start=time.perf_counter()
   moves=state['candidates']
   safe=[m for m in moves if m['safe']]
   preferred=max(safe,key=lambda m:m.get('advance',0))['direction'] if safe else None
   criteria={m['direction']:('Blocked. Collision.' if not m['legal'] else 'Unsafe. Traps the snake.' if not m['safe'] else 'Safe. Eat food now. Best.' if m.get('eat') else 'Safe. Best route to food.' if m['direction']==preferred else 'Safe. Slower route.') for m in moves}
   state='Safe route: '+('yes.' if safe else 'no.')
   global CLIENT
   with LOCK:
    if CLIENT is None: CLIENT=JevClient()
    data,metrics=CLIENT.evaluate(state,{'move':{'type':'choice','instructions':'Choose the best safe move toward food.','criteria':criteria}})
   answer=data['answers']['move'];decode_choice(answer,criteria,0,0)
   self.reply(200,{'choice':answer['choice'],'probabilities':answer['probabilities'],'ms':metrics['api_ms'],'model':metrics['model']})
  except Halt as e:
   if CLIENT: CLIENT.close()
   CLIENT=None
   self.reply(502,{'error':str(e)})
  except Exception as e:self.reply(502,{'error':'Jev 请求失败：'+type(e).__name__+'。请检查网络和服务端配置后重试。'})
 def run_test(self):
  global CLIENT
  if self.headers.get('Origin') not in (None,'http://127.0.0.1:8877','http://localhost:8877'):return self.reply(403,{'error':'Origin rejected'})
  try:
   n=int(self.headers.get('Content-Length','0'))
   if not 0<n<1024:raise ValueError('Invalid request size')
   task=task_for(json.loads(self.rfile.read(n)))
   with LOCK:
    try:
     if CLIENT is None:CLIENT=JevClient(timeout=20)
     data,metrics=CLIENT.evaluate(task['state'],task['questions'])
    except Exception:
     if CLIENT:CLIENT.close()
     CLIENT=None
     raise
   self.reply(200,test_result(task,data,metrics['api_ms']))
  except Exception as e:self.reply(502,{'error':str(e)})
if __name__=='__main__':
 import sys, getpass
 if '--ask-key' in sys.argv: os.environ['TYPESAFE_API_KEY']=getpass.getpass('TypeSafe API key: ')
 print('Snake: http://127.0.0.1:8877',flush=True)
 ThreadingHTTPServer(('127.0.0.1',8877),Handler).serve_forever()
