"""Infrastructure check only; no scientific outcomes or experiment checkpoints."""
import argparse,copy,json,sys,time,importlib.metadata,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);a=p.parse_args()
arm=Path(a.arm).resolve();sys.path.insert(0,str(arm))
from src.data import load_config,write_json
from src.model import load_model,seed_everything
from src.train import train_stage
from src.evaluate import evaluate
import torch
start=time.time();cfg=load_config(arm/'followup_config.json')
assert torch.cuda.is_available(),'CUDA unavailable; do not silently run on CPU'
seed_everything(909013)
model,tok=load_model(cfg)
assert next(model.parameters()).device.type=='cuda'
rows=[{'example_id':str(i),'original_text':'Infrastructure test only. '+('A news report describes an event. '*80),'gold_label':i%4} for i in range(16)]
smoke=copy.deepcopy(cfg);smoke['training']['epochs']=1
t=time.time();exposure=train_stage(model,tok,cfg['tasks'][0],rows,smoke,909013)
assert exposure['optimizer_steps']==1
assert all(torch.isfinite(v).all().item() for v in model.parameters() if v.requires_grad)
checks=[]
for task in cfg['tasks']:
 r=evaluate(model,tok,task,rows[:1],cfg)[0]
 assert len(r['candidate_scores'])==len(task['labels'])
 assert all(torch.isfinite(torch.tensor(r[k])).all().item() for k in ['candidate_scores','probabilities'])
 checks.append({'task':task['key'],'candidate_count':len(task['labels'])})
report={'status':'passed','scope':'synthetic one optimizer step and candidate scoring; no scientific result','model':cfg['model'],'gpu':torch.cuda.get_device_name(0),'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','peft','datasets','numpy','pandas','scipy','scikit-learn','accelerate']},'elapsed_seconds':time.time()-start,'step_and_scoring_seconds':time.time()-t,'checks':checks,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'scientific_results_created':False}
write_json(arm/'gpu_smoke.json',report);print(json.dumps(report,indent=2),flush=True)
