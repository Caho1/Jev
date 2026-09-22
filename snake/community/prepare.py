from pathlib import Path
import ast,json,hashlib,subprocess,random,csv,io,urllib.request
ROOT=Path(__file__).parent
sources={}
for name,path,url in [('jevbench','/tmp/jevbench-reference','https://github.com/fstandhartinger/jevbench'),('reviews','/tmp/jev-vs-luna-reference','https://github.com/mameli/jev-vs-luna'),('phishing','/tmp/jev-phishing-bench-reference','https://github.com/anisselbd/jev-phishing-bench')]:
 sources[name]={'url':url,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()}
def assignment(path,name):
 tree=ast.parse(Path(path).read_text())
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):return ast.literal_eval(node.value)
 raise ValueError(name)
tasks=[]
for file,suite in [('easy','JevBench / easy'),('original','JevBench / standard'),('hard','JevBench / hard')]:
 p=Path('/tmp/jevbench-reference/datasets/public')/(file+'.jsonl')
 sources['jevbench'][file+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 for line in p.read_text().splitlines():
  t=json.loads(line);tasks.append({'id':t['id'],'suite':suite,'family':t['family'],'group':t['group'],'state':t['state'],'questions':{'decision':t['question']},'expected':{'decision':t['expected']},'labels':{'decision':t['labels']},'scoring':'jevbench'})
questions=assignment('/tmp/jev-vs-luna-reference/classification.py','QUESTIONS')
reviews=json.loads(Path('/tmp/jev-vs-luna-reference/data/reviews_100.json').read_text())
for r in reviews:tasks.append({'id':r['id'],'suite':'Reviews / 5 fields','family':'reviews','group':r['case_id'],'state':{'text':r['text']},'questions':questions,'expected':r['expected'],'scoring':'reviews'})
url='https://huggingface.co/datasets/AreLit/PhishNChips/resolve/main/core_emails.csv'
raw=urllib.request.urlopen(url,timeout=90).read()
digest=hashlib.sha256(raw).hexdigest()
assert digest=='cebb407ff8630491a97400e37464b8db8dfc4299164fca51fcb4ac7eec8204ef'
sources['phishing'].update(dataset=url,sha256=digest,sampling='50 legitimate + 50 phishing, seeded 2026, sorted IDs before random sample; verdict-only question, not full 9-question reproduction')
rows=list(csv.DictReader(io.StringIO(raw.decode())))
rng=random.Random(2026);selected=[]
for label in ['0','1']:selected+=rng.sample(sorted([r for r in rows if r['phish_label']==label],key=lambda r:r['id']),50)
rng.shuffle(selected)
verdict=assignment('/tmp/jev-phishing-bench-reference/run_jev.py','QUESTIONS')['verdict']
for r in selected:
 email=json.loads(r['email_content']);state={k:email[k] for k in ['sender','from','subject','body','link_display_text','link_url']}
 tasks.append({'id':'phish-'+r['id'],'suite':'Phishing / subset 100','family':'phishing','group':r['id'],'state':state,'questions':{'verdict':verdict},'expected':{'verdict':'phishing' if r['phish_label']=='1' else 'legitimate'},'labels':{'verdict':['phishing','legitimate']},'scoring':'jevbench'})
(ROOT/'corpus.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2))
(ROOT/'sources.json').write_text(json.dumps(sources,indent=2))
from collections import Counter
print(dict(Counter(t['suite'] for t in tasks)), 'requests/model',len(tasks),'decisions/model',sum(len(t['questions']) for t in tasks))
