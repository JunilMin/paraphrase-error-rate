"""Infrastructure-only queue revision. Experiment code/configuration remain frozen."""
import sys,json,os,time,subprocess,hashlib,traceback,csv
from datetime import datetime,timezone
from pathlib import Path
from filelock import FileLock
from control_io_v2 import atomic_json
ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def jl(p):return [json.loads(t) for t in p.read_text(encoding='utf-8-sig').splitlines() if t.strip()]
state={'control_version':'v2_20261118','status':'starting','pid':os.getpid(),'python':sys.executable,'started_utc':now(),'arms':['more_seeds','reverse_order','second_model'],'K5_status':'awaiting_new_bank_and_actual_quality_review','human_200_status':'author_context_review_completed','completed_commands':[],'reused_complete_arms':[]}
def event(obj):
 with (ROOT/'control_events_v2.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=now(),**obj),ensure_ascii=False)+'\n');f.flush()
def status(heartbeat=False):
 state['updated_utc']=now()
 try:atomic_json(ROOT/'queue_status.json',state)
 except PermissionError as e:
  event({'kind':'status_write_blocked','error':str(e),'heartbeat_only':heartbeat})
  if not heartbeat:raise
def run(label,args,arm):
 state.update(status='running',current_command=label,current_arm=arm);status()
 with (ROOT/(label+'.stdout.log')).open('a',encoding='utf-8') as out,(ROOT/(label+'.stderr.log')).open('a',encoding='utf-8') as err:
  child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT/arm,stdout=out,stderr=err,env={**os.environ,'PYTHONUNBUFFERED':'1'})
  state['child_pid']=child.pid
  # A brief heartbeat publication failure must not orphan or cancel valid training.
  status(heartbeat=True)
  while child.poll() is None:time.sleep(10);status(heartbeat=True)
  if child.returncode:raise RuntimeError(f'{label} failed with exit code {child.returncode}; no automatic training retry')
 state['completed_commands'].append({'label':label,'completed_utc':now()});status()
def validate_arm(arm):
 cfg=read(ROOT/arm/'followup_config.json');report=[]
 for seed in cfg['seeds']:
  run=ROOT/arm/'results/main'/f'seed_{seed}'
  if not (run/'run.json').exists() or read(run/'run.json')['status']!='complete':return None
  rec=read(run/'run.json');expected=dict(cfg,mode='main')
  assert rec['config']==expected and rec['seed']==seed and rec['completed']==list('ABC')
  idsets=[]
  for stage in 'ABC':
   rows=jl(run/f'after_{stage}.jsonl');assert len(rows)==7600==len({r['example_id'] for r in rows})
   idsets.append({r['example_id']:(r['original_text'],r['gold_label'],r['text_group']) for r in rows})
   assert rec['exposure'][stage]['examples_seen']==10000 and rec['exposure'][stage]['optimizer_steps']==625
  assert idsets[0]==idsets[1]==idsets[2]
  pool={r['example_id'] for r in jl(run/'after_A.jsonl') if r['correct']};inc=jl(run/'inconsistency.jsonl')
  assert len(inc)==len(pool)==len({r['example_id'] for r in inc}) and {r['example_id'] for r in inc}==pool
  assert all(r['n_paraphrases']==3 for r in inc)
  report.append({'seed':seed,'status':'complete','test_sources':7600,'A_correct':len(pool),'results_csv_sha256':sha(run/'results.csv')})
 return report
def posthoc_ready(arm,report):
 folder=ROOT/arm/'results/followup_paper_analysis';meta=folder/'analysis_metadata.json'
 if not meta.exists():return False
 m=read(meta)
 if {int(r['seed']):r['sha256'] for r in m['sources']}!={r['seed']:r['results_csv_sha256'] for r in report}:return False
 required=['three_feature_scores.csv','three_feature_oof.csv','three_feature_paired_differences.csv','three_feature_summary.csv','per_distribution.csv','per_binary_rates.csv','confidence_conditioned.csv','reverse_transitions.csv','feature_redundancy.csv']
 return all((folder/'tables'/n).exists() for n in required) and (folder/'figures/per_distribution.png').exists()
def main():
 with FileLock(str(ROOT/'.gpu_queue.lock'),timeout=0):
  identity=read(ROOT/'execution_identity.json');control=read(ROOT/'control_revision_20261118.json')
  assert sys.executable.lower()==identity['python'].lower()
  for name,h in identity['code_hashes'].items():assert sha(ROOT/name)==h,'Frozen experiment input/code changed'
  for name,h in control['new_control_hashes'].items():assert sha(ROOT/name)==h,'Control revision changed'
  for name,h in identity['original_evidence_hashes'].items():assert sha(Path(name))==h
  state['status']='running';status()
  reused=[]
  for arm in state['arms']:
   state['current_arm']=arm;report=validate_arm(arm)
   if report is None:
    smoke=ROOT/arm/'gpu_smoke.json'
    if not smoke.exists():run(arm+'_smoke',[ROOT/'gpu_smoke.py','--arm',ROOT/arm],arm)
    assert read(smoke)['status']=='passed'
    run(arm+'_training',[ROOT/arm/'run.py','--config',ROOT/arm/'followup_config.json'],arm)
    report=validate_arm(arm);assert report is not None
   else:
    reused.append({'arm':arm,'runs':report});event({'kind':'completed_training_reused','arm':arm,'seeds':[r['seed'] for r in report]})
   if not posthoc_ready(arm,report):run(arm+'_posthoc',[ROOT/arm/'followup_analysis.py'],arm)
   else:event({'kind':'completed_posthoc_reused','arm':arm})
   state['reused_complete_arms']=[r['arm'] for r in reused];status()
  atomic_json(ROOT/'v2_reuse_audit.json',reused)
  for name,h in identity['original_evidence_hashes'].items():assert sha(Path(name))==h
  state.update(status='three_K3_arms_complete_K5_and_human_review_pending',completed_utc=now());status()
if __name__=='__main__':
 try:main()
 except Exception as e:
  state.update(status='failed',error=str(e),traceback=traceback.format_exc());event({'kind':'manager_failed','error':str(e)})
  atomic_json(ROOT/'queue_failure.json',state);status();raise
