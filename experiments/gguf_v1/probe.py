import json,time,urllib.request,importlib.util,sys
from pathlib import Path
from transformers import AutoTokenizer
from relational_decisions import Candidate
from relational_decisions.h2o import H2OLightningBackend
from relational_decisions.prompts import WORLD_PREFIX,build_tasks

def post(port,route,payload):
 t=time.perf_counter(); req=urllib.request.Request(f'http://127.0.0.1:{port}/{route}',json.dumps(payload).encode(),{'Content-Type':'application/json'})
 r=json.load(urllib.request.urlopen(req,timeout=180));return r,time.perf_counter()-t
world='A report says Oak is ready. A separate report says Elm is not ready.'
c=(Candidate('oak',('p',),('n',),'Oak is ready','Oak is not ready'),Candidate('elm',('p2',),('n2',),'Elm is ready','Elm is not ready'))
tasks=build_tasks(c,'binary-reports-v2');qs={t.name:{'type':'choice','instructions':t.instruction,'criteria':t.labels} for t in tasks}
result={'d1':[],'h2o':[]}
for rep in range(2):
 r,s=post(8766,'v1/systemone',{'state':WORLD_PREFIX+world,'questions':qs});result['d1'].append({'mode':'batch','seconds':s,'response':r});print('d1 batch',s,r,flush=True)
for name,q in qs.items():
 r,s=post(8766,'v1/systemone',{'state':WORLD_PREFIX+world,'questions':{name:q}});result['d1'].append({'mode':'single','seconds':s,'response':r});print('d1 single',s,r,flush=True)
p=Path('/home/agent/.cache/huggingface/hub/models--h2oai--h2o-lightning-4b/snapshots/672dc01ed37a516357cd3c7da777c96699d3f16c')
spec=importlib.util.spec_from_file_location('h2o_shim',p/'h2o_lightning_shim.py');shim=importlib.util.module_from_spec(spec);spec.loader.exec_module(shim)
b=H2OLightningBackend.__new__(H2OLightningBackend);b.tokenizer=AutoTokenizer.from_pretrained(p);b.contract=shim.Contract(json.loads((p/'serve_config.json').read_text()),env={})
ts,rs=b.requests(world,c)
prefix=[]
for xs in zip(*(r[1] for r in rs)):
 if len(set(xs))!=1:break
 prefix.append(xs[0])
for mode in ('cold','warm','warm'):
 if mode=='warm':
  r,s=post(8765,'completion',{'prompt':prefix,'n_predict':0,'cache_prompt':False,'temperature':-1,'samplers':[]});print('warmup',s,r.get('timings'),flush=True)
 for task,(_,tokens,ids) in zip(ts,rs):
  r,s=post(8765,'completion',{'prompt':tokens,'n_predict':1,'temperature':-1,'n_probs':128,'post_sampling_probs':False,'cache_prompt':mode=='warm','seed':0,'samplers':[]})
  scores={e['id']:e['logprob'] for e in r['completion_probabilities'][0]['top_logprobs']}
  rec={'mode':mode,'name':task.name,'seconds':s,'probabilities':shim.probabilities([scores[i] for i in ids],.8),'timings':r['timings'],'tokens_evaluated':r['tokens_evaluated'],'truncated':r['truncated']}
  print('h2o',rec,flush=True);result['h2o'].append(rec)
Path('/workspace/gguf-smoke-v2.json').write_text(json.dumps(result,indent=2)+'\n')
