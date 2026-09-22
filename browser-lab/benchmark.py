import sys,json,getpass,subprocess,time,io,argparse
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0,str(Path.home()/'.agents/skills/jev-mac-use/scripts'))
from jev import JevClient,Halt
from mac_use import Bridge,run

def press(label,role='AXButton'):return {'op':'press','selector':{'label':label,'role':role}}
def fill(label,text):return {'op':'set_value','selector':{'label':label,'role':'AXTextField'},'text':text}
def task(goal,actions,result,role='AXHeading'):
 return {'bundle_id':'com.google.Chrome','goal':goal,'actions':actions,'success':[{'selector':{'label':result,'role':role}}]}
cases={
'form':task('Complete the profile with Display name Lin, City Shanghai, Pro plan, and check I reviewed the test details. Finish profile. Start form only if no profile form is visible.',[press('Start form'),fill('Display name','Lin'),fill('City','Shanghai'),press('Continue'),press('Pro plan','AXRadioButton'),press('I reviewed the test details','AXCheckBox'),press('Finish profile')],'Profile complete: Lin / Shanghai / Pro / reviewed'),
'search':task('Search catalog for penguin, then choose Penguin notebook, not the poster. Wait while Loading results is visible. Start search only if the search form is absent.',[press('Start search'),fill('Search catalog','penguin'),press('Search products'),press('Choose Penguin notebook'),press('Choose Penguin poster')],'Selected: Penguin notebook'),
'modal':task('Edit workspace, set Workspace name to Jev Lab, enable notifications, and Save settings. Start modal only if Workspace settings is absent.',[press('Start modal'),press('Edit workspace'),fill('Workspace name','Jev Lab'),press('Enable notifications','AXCheckBox'),press('Save settings'),press('Cancel edit')],'Workspace saved: Jev Lab / notifications on','AXStaticText')}
# Chrome exposes static text via value rather than label.
cases['modal']['success']=[{'selector':{'label':'','role':'AXStaticText','value':'Workspace saved: Jev Lab / notifications on'}}]
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--case',choices=list(cases));p.add_argument('--repeats',type=int,default=1);a=p.parse_args()
key=getpass.getpass('TypeSafe API key (hidden): ')
subprocess.run(['open','-a','Google Chrome'],check=True)
b=Bridge('com.google.Chrome');c=None;report={'runs':[]}
try:
 b.request({'cmd':'snapshot'});c=JevClient(key);_,report['warmup']=c.evaluate('Connection warmup.',{'ready':{'type':'noul','instructions':'Does state contain warmup?'}})
 for repeat in range(a.repeats):
  for name,t in cases.items():
   if a.case and name!=a.case:continue
   # Reset using an observed start button, separate from the measured task.
   snap=b.request({'cmd':'snapshot'});label='Start '+name
   nodes=[n for n in snap['nodes'] if n['label']==label and n['role']=='AXButton']
   if len(nodes)!=1:raise Halt('Reset unavailable')
   b.request({'cmd':'act','snapshot':snap['snapshot'],'target':nodes[0]['id'],'op':'press'})
   b.request({'cmd':'settle','timeout_ms':600})
   pre=b.request({'cmd':'snapshot'});Path(a.out+'.'+name+'.before.json').write_text(json.dumps(pre,indent=2))
   trace=io.StringIO();row={'case':name,'repeat':repeat};start=time.perf_counter()
   try:
    with redirect_stdout(io.StringIO()):result=run(t,c,b,'flat',16,30,True,trace)
    row.update(result)
   except Halt as e:row.update(status='stopped',error=str(e))
   row['task_ms']=round((time.perf_counter()-start)*1000,3);row['trace']=[json.loads(x) for x in trace.getvalue().splitlines()]
   report['runs'].append(row)
   try:Path(a.out+'.'+name+'.after.json').write_text(json.dumps(b.request({'cmd':'snapshot'}),indent=2))
   except Halt:pass
   Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
   if row['status']!='verified':break
  if report['runs'][-1]['status']!='verified':break
finally:
 b.close()
 if c:c.close()
