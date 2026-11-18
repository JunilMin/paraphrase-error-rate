"""Prepare isolated follow-up workspaces. This command never trains or downloads weights."""
import argparse,copy,csv,hashlib,json,re,shutil,subprocess,sys,tempfile
from pathlib import Path

NEW_SEEDS=[101,202,303,404,505]
SMOL_REV='31b70e2e869a7173562077fd711b654946d38674'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readjl(p):return [json.loads(t) for t in Path(p).read_text(encoding='utf-8').splitlines() if t.strip()]

def generic_analysis(source):
    replacements={
        "        seed = state['seed']":"        count = int(protocol['paraphrase']['count'])\n        if count < 1: raise ValueError('Positive rewrite count required')\n        levels = [i/count for i in range(count+1)]\n        seed = state['seed']",
        '[0, 1/3, 2/3, 1]':'levels',
        'PER is not a three-paraphrase fraction':'PER does not match the configured rewrite count',
        "pool.n_paraphrases.eq(3)":"pool.n_paraphrases.eq(count)",
        'Expected three paraphrases':'Unexpected rewrite count',
        'range(4)':'range(count+1)',
        "['0', '1/3', '2/3', '1']":"[str(i) if i in (0,1) else str(i) for i in levels]",
    }
    for old,new in replacements.items():
        expected=2 if old in ('[0, 1/3, 2/3, 1]','range(4)') else 1
        if source.count(old)!=expected:raise ValueError('Source version changed: '+old)
        source=source.replace(old,new)
    compile(source,'followup_paper_analysis.py','exec')
    return source
def plan(project,scenario,bank=None,attestation=None):
    project=Path(project).resolve()
    cfg=copy.deepcopy(json.loads((project/'results/main/seed_13/run.json').read_text(encoding='utf-8'))['config'])
    cfg.pop('mode',None)
    cfg['seeds']=NEW_SEEDS if scenario=='more_seeds' else [13,42,77]
    if scenario=='reverse_order':cfg['tasks']=[cfg['tasks'][0],cfg['tasks'][2],cfg['tasks'][1]]
    if scenario=='second_model':
        cfg['model'].update(name='HuggingFaceTB/SmolLM2-1.7B-Instruct',revision=SMOL_REV,tokenizer_revision=SMOL_REV)
    selected=project/'results/cache/paraphrases/ag_news.jsonl'
    if scenario=='more_rewrites':
        if bank is None or attestation is None:raise ValueError('K=5 requires --bank and --human-bank-attestation; no automatic generation or approval')
        selected=Path(bank).resolve();rows=readjl(selected);original=readjl(project/'results/cache/paraphrases/ag_news.jsonl')
        old={str(r['example_id']):r['paraphrases'] for r in original}
        assert len(rows)==7600==len({str(r['example_id']) for r in rows})
        assert {str(r['example_id']) for r in rows}==set(old)
        for r in rows:
            values=r['paraphrases'];assert isinstance(values,list) and len(values)==5
            assert all(isinstance(v,str) and v.strip() for v in values)
            assert len(set(v.strip() for v in values))==5
            assert values[:3]==old[str(r['example_id'])],'Keep original three candidates in the same positions'
        att=json.loads(Path(attestation).read_text(encoding='utf-8'))
        assert att.get('bank_sha256')==sha(selected)
        assert att.get('human_review_completed') is True and att.get('reviewer') and att.get('review_date')
        assert att.get('protocol_version')=='PER-followup-20261109-v1'
        cfg['paraphrase']['count']=5
    return cfg,selected

def prepare(project,destination,scenario,bank=None,attestation=None):
    project=Path(project).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Destination must be new; never reuse or clear an existing experiment')
    if project==destination or project in destination.parents or destination in project.parents:raise ValueError('Destination must be separate from the original project')
    cfg,selected=plan(project,scenario,bank,attestation)
    files=[project/'run.py']+list((project/'src').glob('*.py'))+list(project.glob('requirements*.txt'))
    before={str(p.relative_to(project)):sha(p) for p in files}
    destination.mkdir(parents=True)
    for source in files:
        target=destination/source.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    target=destination/cfg['paraphrase']['file'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(selected,target)
    # Original manifest copied verbatim as evidence, not relabeled as this new bank's provenance.
    provenance=destination/'provenance';provenance.mkdir()
    oldmanifest=project/'results/cache/paraphrases/ag_news.manifest.json'
    if oldmanifest.exists():shutil.copy2(oldmanifest,provenance/'original_bank_manifest.json')
    if attestation:shutil.copy2(attestation,provenance/'new_bank_human_attestation.json')
    (destination/'followup_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    extra=destination/'src/followup_paper_analysis.py'
    extra.write_text(generic_analysis((project/'src/paper_analysis.py').read_text(encoding='utf-8')),encoding='utf-8')
    wrapper=destination/'followup_analysis.py'
    wrapper.write_text("from pathlib import Path\nimport json\nfrom src.followup_paper_analysis import paper_analysis\nroot=Path(__file__).resolve().parent\ncfg=json.loads((root/'followup_config.json').read_text())\nruns=[root/'results/main'/f'seed_{seed}' for seed in cfg['seeds']]\npaper_analysis(runs,root/'results/followup_paper_analysis',5000)\n",encoding='utf-8')
    manifest={'protocol_version':'PER-followup-20261109-v1','scenario':scenario,'status':'prepared_not_executed','source_project':str(project),'source_code_hashes':before,'bank_sha256':sha(selected),'model':cfg['model'],'seeds':cfg['seeds'],'task_order':[t['key'] for t in cfg['tasks']],'K':cfg['paraphrase']['count'],'provenance_note':'An imported shared-bank manifest created by the original runner describes its validation contract, not proof of regeneration or human semantic validation.'}
    manifest['additional_code_hashes']={str(p.relative_to(destination)):sha(p) for p in [extra,wrapper]}
    manifest['analysis_change']='Separate module generalizes only rewrite-count validation, levels and plot ticks. Original modules are unchanged. Runtime analysis not tested.'
    (destination/'followup_identity.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (destination/'RUN_COMMANDS.txt').write_text('Training has NOT run. Use a configured training environment in this new workspace.\n\npython run.py --config followup_config.json\npython followup_analysis.py\n\nDo not combine scenarios as independent examples; B/C refer to positional stages and task names must accompany every report.\n',encoding='utf-8')
    assert before=={n:sha(project/n) for n in before}
    return manifest

def selftest(project):
    with tempfile.TemporaryDirectory(prefix='per_followup_test_', dir=Path(__file__).resolve().parent) as tmp:
        root=Path(tmp)
        for scenario in ['more_seeds','reverse_order','second_model']:
            m=prepare(project,root/scenario,scenario)
            assert m['status']=='prepared_not_executed'
            assert not (root/scenario/'results/main').exists()
            assert all(sha(root/scenario/n)==v for n,v in m['source_code_hashes'].items())
        assert json.loads((root/'reverse_order/followup_config.json').read_text())['tasks'][1]['key']=='dbpedia_14'
        assert json.loads((root/'second_model/followup_config.json').read_text())['model']['revision']==SMOL_REV
        try:prepare(project,root/'more_seeds','more_seeds')
        except FileExistsError:pass
        else:raise AssertionError('Existing workspace was accepted')
        try:plan(project,'more_rewrites')
        except ValueError:pass
        else:raise AssertionError('Unreviewed K=5 bank was accepted')
        # Positive K=5 staging is tested with synthetic extension text, strictly inside temp.
        # This validates software behavior, never produces scientific input or human evidence.
        original=readjl(Path(project)/'results/cache/paraphrases/ag_news.jsonl')
        for r in original:r['paraphrases']+=['SYNTHETIC TEST ONLY alpha '+str(r['example_id']),'SYNTHETIC TEST ONLY beta '+str(r['example_id'])]
        b=root/'synthetic_bank.jsonl';b.write_text(''.join(json.dumps(r)+'\n' for r in original),encoding='utf-8')
        a=root/'synthetic_attestation.json';a.write_text(json.dumps({'bank_sha256':sha(b),'human_review_completed':True,'reviewer':'SYNTHETIC TEST ONLY','review_date':'TEST','protocol_version':'PER-followup-20261109-v1'}),encoding='utf-8')
        assert prepare(project,root/'k5_test','more_rewrites',b,a)['K']==5
    return {'status':'passed','scope':'offline staging, source hashes, refusal of existing workspaces and missing review evidence, synthetic K=5 staging','not_tested':['GPU training','SmolLM2 runtime compatibility','Scientific validity of a new K=5 bank']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--scenario',choices=['more_seeds','reverse_order','second_model','more_rewrites']);p.add_argument('--destination');p.add_argument('--bank');p.add_argument('--human-bank-attestation');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:result=selftest(a.project)
    else:
        if not a.scenario or not a.destination:p.error('--scenario and --destination required')
        result=prepare(a.project,a.destination,a.scenario,a.bank,a.human_bank_attestation)
    print(json.dumps(result,indent=2))

