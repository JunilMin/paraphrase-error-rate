"""One-task read-only collection of completed follow-up cohorts and review gates."""
import json,csv,math,statistics,hashlib,time,os,sys,shutil,datetime
from pathlib import Path
from filelock import FileLock
ROOT=Path(__file__).resolve().parent
ORIGINAL=Path(r'C:\Users\user\Dropbox\research paper\Research_Project_v03\Paraphrase_Error_Rate_20261021\Research_Project')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def csvrows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def jl(p):return [json.loads(t) for t in p.read_text(encoding='utf-8-sig').splitlines() if t.strip()]
def write(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def writecsv(p,rows):
 assert rows
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def finite(v):
 try:v=float(v)
 except (TypeError,ValueError):return None
 return v if math.isfinite(v) else None
def collect():
 out=ROOT/'completed_K3_reports';out.mkdir(exist_ok=True)
 original_identity=read(ROOT/'execution_identity.json')
 assert all(sha(Path(n))==h for n,h in original_identity['original_evidence_hashes'].items())
 sources={r['example_id']:(r['original_text'],r['gold_label'],r['text_group']) for r in jl(ORIGINAL/'results/main/seed_13/after_A.jsonl')}
 effects=[];inputs={};coverage=[]
 for cohort,base,seeds,tab in [('original',ORIGINAL,[13,42,77],ORIGINAL/'results/paper_analysis/tables')]+[(arm,ROOT/arm,read(ROOT/arm/'followup_config.json')['seeds'],ROOT/arm/'results/followup_paper_analysis/tables') for arm in ['more_seeds','reverse_order','second_model']]:
  for seed in seeds:
   run=base/'results/main'/f'seed_{seed}';record=read(run/'run.json');assert record['status']=='complete'
   for stage in 'ABC':
    path=run/f'after_{stage}.jsonl';rows=jl(path)
    assert len(rows)==7600==len({r['example_id'] for r in rows})
    assert {r['example_id']:(r['original_text'],r['gold_label'],r['text_group']) for r in rows}==sources
    inputs[str(path)]=sha(path)
   eligible={r['example_id'] for r in jl(run/'after_A.jsonl') if r['correct']}
   inc=jl(run/'inconsistency.jsonl');assert len(inc)==len(eligible)==len({r['example_id'] for r in inc})
   assert {r['example_id'] for r in inc}==eligible
   assert all(r['n_paraphrases']==3 and any(math.isclose(r['inconsistency'],v/3,abs_tol=1e-12) for v in range(4)) for r in inc)
   assert all(record['exposure'][s]['examples_seen']==10000 and record['exposure'][s]['optimizer_steps']==625 for s in 'ABC')
   coverage.append({'cohort':cohort,'seed':seed,'test_sources':7600,'A_correct':len(eligible),'complete_stages':'ABC','K':3,'model':record['config']['model']['name'],'task_order':' -> '.join(t['key'] for t in record['config']['tasks'])})
  scores_path=tab/'three_feature_scores.csv';ci_path=tab/'three_feature_paired_differences.csv'
  scores=csvrows(scores_path);cis=csvrows(ci_path);inputs[str(scores_path)]=sha(scores_path);inputs[str(ci_path)]=sha(ci_path)
  assert {int(r['seed']) for r in scores}==set(seeds)
  for seed in seeds:
   for endpoint in 'BC':
    pair={r['predictor']:r for r in scores if int(r['seed'])==seed and r['endpoint']==endpoint}
    assert set(pair)=={'baseline_combined','baseline_plus_inconsistency'}
    cfg=read(base/'results/main'/f'seed_{seed}'/'run.json')['config']
    for metric in ['auroc','average_precision']:
     lower=finite(pair['baseline_combined'][metric]);upper=finite(pair['baseline_plus_inconsistency'][metric]);delta=upper-lower if lower is not None and upper is not None else None
     matched=[r for r in cis if int(r['seed'])==seed and r['endpoint']==endpoint and r['metric']==metric]
     assert len(matched)==1
     ci=matched[0];saved=finite(ci['difference'])
     assert (delta is None and saved is None) or (delta is not None and saved is not None and math.isclose(delta,saved,abs_tol=1e-10))
     effects.append({'cohort':cohort,'seed':seed,'endpoint':endpoint,'actual_later_task':cfg['tasks'][1 if endpoint=='B' else 2]['key'],'metric':metric,'baseline':lower,'baseline_plus_PER':upper,'difference':delta,'ci_low':finite(ci['ci_low']),'ci_high':finite(ci['ci_high']),'valid_bootstrap_repeats':ci['valid_repeats'],'requested_bootstrap_repeats':ci['requested_repeats'],'primary':endpoint=='B' and metric=='auroc'})
 summary=[]
 for cohort in ['original','more_seeds','reverse_order','second_model']:
  for endpoint in 'BC':
   for metric in ['auroc','average_precision']:
    subset=[r for r in effects if r['cohort']==cohort and r['endpoint']==endpoint and r['metric']==metric];values=[r['difference'] for r in subset if r['difference'] is not None]
    summary.append({'cohort':cohort,'endpoint':endpoint,'actual_later_task':subset[0]['actual_later_task'],'metric':metric,'n_runs':len(subset),'n_defined':len(values),'n_undefined':len(subset)-len(values),'mean_difference':statistics.mean(values) if values else None,'sample_SD':statistics.stdev(values) if len(values)>1 else None,'primary':endpoint=='B' and metric=='auroc'})
 writecsv(out/'seed_effects.csv',effects);writecsv(out/'cohort_summary.csv',summary);writecsv(out/'coverage_audit.csv',coverage)
 write(out/'report_provenance.json',{'status':'K3_cohorts_collected','source_hashes':inputs,'coverage_rows':len(coverage),'report_script_sha256':sha(Path(__file__)),'report_version':'PER-followup-report-v1','primary':'B delta OOF AUROC adding PER to confidence/entropy/margin baseline','interval_scope':'Fixed OOF predictions, paired text-group percentile bootstrap; no cross-arm pooled significance test or multiplicity correction','human_review':'200 author context judgments are separate; original independent agreement unchanged','K5_status':'Not included in K3 report; new inputs require actual human quality review','language_models_retrained_by_this_report':False})
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4.5))
 for x,cohort in enumerate(['original','more_seeds','reverse_order','second_model']):
  values=[r['difference'] for r in effects if r['cohort']==cohort and r['primary'] and r['difference'] is not None]
  ax.scatter([x]*len(values),values,label=cohort)
  if values:ax.plot([x-.12,x+.12],[statistics.mean(values)]*2,color='black')
 ax.axhline(0,color='gray',linestyle='--');ax.set_xticks(range(4),['Original','Additional seeds','Reversed order','Second model']);ax.set_ylabel('PER added to 3-feature baseline: AUROC difference');ax.set_title('First later-task endpoint, seed-wise descriptive effects');fig.tight_layout();fig.savefig(out/'primary_seed_effects.png',dpi=200);plt.close(fig)
 text=['# Completed K3 follow-up cohorts','', 'Every configured run is included. Original and follow-up cohorts are reported separately. B and C denote positions; the actual task is provided in each table. Differences are descriptive. No pooled cross-arm significance claim is made.','', '| Cohort | B delta AUROC mean | Sample SD | Defined / planned runs |','|---|---:|---:|---:|']
 for r in summary:
  if r['primary']:text.append(f"| {r['cohort']} | {r['mean_difference']} | {r['sample_SD']} | {r['n_defined']}/{r['n_runs']} |")
 text+=['','The original 200 unresolved pairs received a separate author context-preservation review (163 accepted, 37 rejected; all retained category). This is not strict semantic equivalence or original-reviewer consensus. K5 results are absent until new input preparation, quality review and separate evaluation are complete.']
 (out/'K3_results_summary.md').write_text('\n'.join(text),encoding='utf-8')
 delivery=Path(r'C:\Users\user\Dropbox\research paper\PER_followup_20261113\completed_K3_reports')
 delivery.mkdir(exist_ok=True)
 for p in out.iterdir():
  if p.is_file():shutil.copy2(p,delivery/p.name)
 return str(out)
def main():
 with FileLock(str(ROOT/'.finish_reports.lock'),timeout=0):
  report=None
  while True:
   queue=read(ROOT/'queue_status.json');generation=read(ROOT/'k5_generation/status.json');human=read(ROOT/'human_review_status.json')
   overall={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'core_queue':queue['status'],'current_arm':queue.get('current_arm'),'human_200_completed':human['pairs_completed']==200,'generation':generation['status'],'K5_evaluation_started':False,'K3_report':report}
   if queue['status']=='three_K3_arms_complete_K5_and_human_review_pending' and report is None:report=collect();overall['K3_report']=report
   failure=ROOT/'k5_generation/failure.json'
   if queue['status']=='failed' or failure.exists():overall['status']='needs_failure_investigation';write(ROOT/'overall_status.json',overall);return
   if generation['status'] in ['generated_pending_actual_human_review','needs_generation_recovery']:
    overall['status']='needs_new_variant_human_review' if generation['status']=='generated_pending_actual_human_review' else 'needs_generation_recovery';write(ROOT/'overall_status.json',overall);return
   overall['status']='running';write(ROOT/'overall_status.json',overall);time.sleep(30)
if __name__=='__main__':
 try:main()
 except Exception as e:write(ROOT/'finish_reports_failure.json',{'error':str(e),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});raise
