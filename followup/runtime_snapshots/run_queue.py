"""Sequential GPU queue for authorized follow-up arms. Stops on any failure."""
import sys,json,os,time,subprocess,hashlib,traceback
from datetime import datetime,timezone
from pathlib import Path
from filelock import FileLock
root=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)
state={'status':'starting','pid':os.getpid(),'python':sys.executable,'started_utc':now(),'arms':['more_seeds','reverse_order','second_model'],'K5_status':'waiting_for_actual_human_review_then_new_bank','human_200_status':'waiting_for_author','completed_commands':[]}
def run(label,args,arm):
 state.update(status='running',current_command=label,current_arm=arm,updated_utc=now())
 with (root/(label+'.stdout.log')).open('a',encoding='utf-8') as out,(root/(label+'.stderr.log')).open('a',encoding='utf-8') as err:
  child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=root/arm,stdout=out,stderr=err,env={**os.environ,'PYTHONUNBUFFERED':'1'})
  state['child_pid']=child.pid;write(root/'queue_status.json',state)
  while child.poll() is None:
   time.sleep(10);state['updated_utc']=now();write(root/'queue_status.json',state)
  if child.returncode:raise RuntimeError(f'{label} failed, exit code {child.returncode}; inspect preserved logs; no automatic retry')
 state['completed_commands'].append({'label':label,'completed_utc':now()});write(root/'queue_status.json',state)
def main():
 with FileLock(str(root/'.gpu_queue.lock'),timeout=0):
  baseline=json.loads((root/'execution_identity.json').read_text(encoding='utf-8'))
  assert sys.executable.lower()==baseline['python'].lower()
  for name,h in baseline['code_hashes'].items():assert sha(root/name)==h,'Execution code changed: '+name
  # The read-only saved-record hash ledger verifies the original evidence at both ends.
  for name,h in baseline['original_evidence_hashes'].items():assert sha(Path(name))==h,'Original evidence changed'
  write(root/'queue_status.json',state)
  for arm in state['arms']:
   smoke=root/arm/'gpu_smoke.json'
   if not smoke.exists():run(arm+'_smoke',[root/'gpu_smoke.py','--arm',root/arm],arm)
   assert json.loads(smoke.read_text())['status']=='passed'
   run(arm+'_training',[root/arm/'run.py','--config',root/arm/'followup_config.json'],arm)
   cfg=json.loads((root/arm/'followup_config.json').read_text())
   for seed in cfg['seeds']:
    runpath=root/arm/'results/main'/f'seed_{seed}'
    record=json.loads((runpath/'run.json').read_text(encoding='utf-8'))
    assert record['status']=='complete' and record['completed']==list('ABC')
    for stage in 'ABC':
     with (runpath/f'after_{stage}.jsonl').open(encoding='utf-8') as f:rows=[json.loads(t) for t in f if t.strip()]
     assert len(rows)==7600==len({r['example_id'] for r in rows})
    assert all(record['exposure'][s]['examples_seen']==10000 and record['exposure'][s]['optimizer_steps']==625 for s in 'ABC')
   run(arm+'_posthoc',[root/arm/'followup_analysis.py'],arm)
  for name,h in baseline['original_evidence_hashes'].items():assert sha(Path(name))==h
  state.update(status='three_K3_arms_complete_K5_and_human_review_pending',completed_utc=now(),updated_utc=now());write(root/'queue_status.json',state)
try:main()
except Exception as e:
 state.update(status='failed',error=str(e),traceback=traceback.format_exc(),updated_utc=now());write(root/'queue_failure.json',state);write(root/'queue_status.json',state);raise
