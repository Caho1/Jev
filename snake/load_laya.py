import json,time
from laya_mlx import Agent
start=time.perf_counter()
agent=Agent('convaiinnovations/laya',device='gpu',dtype='float16',batch_size=1)
print('MODEL_LOADED',round(time.perf_counter()-start,2),flush=True)
for i in range(3):
 t=time.perf_counter();r=agent.predict('Safe route: yes.',{'move':{'type':'choice','instructions':'Choose the best safe move toward food.','criteria':{'UP':'Blocked. Collision.','DOWN':'Safe. Slower route.','LEFT':'Blocked. Collision.','RIGHT':'Safe. Best route to food.'}}});print(json.dumps({'ms':(time.perf_counter()-t)*1000,'answer':r['answers']['move']}),flush=True)
print('CHECKPOINT',agent.model_dir,flush=True)
