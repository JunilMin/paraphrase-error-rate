"""Read-only verification of the released PER records; no model fitting or inference."""
import argparse,csv,json,math,hashlib,statistics
from pathlib import Path
from collections import Counter

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p): return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def jl(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def auroc_ap(y,s):
    n1=sum(y);n0=len(y)-n1
    if not n1 or not n0:return None,None
    groups={}
    for a,b in zip(y,s):
        n,p=groups.get(b,(0,0));groups[b]=(n+1,p+a)
    wins=0.;neg=0
    for score,(n,p) in sorted(groups.items()):
        wins+=p*(neg+(n-p)/2);neg+=n-p
    tp=seen=0;ap=0.
    for score,(n,p) in sorted(groups.items(),reverse=True):
        seen+=n;tp+=p;ap+=(p/n1)*(tp/seen)
    return wins/(n1*n0),ap
def close(a,b,tol=1e-10):assert abs(float(a)-float(b))<=tol,(a,b)
def audit(project,out):
    project=Path(project).resolve();out=Path(out).resolve()
    assert out!=project and project not in out.parents,'Write audit outside source project'
    out.mkdir(parents=True,exist_ok=True)
    report={'scope':'saved-record consistency audit; no training, inference, OOF refit, or bootstrap rerun','seeds':{},'checks':[]}
    tracked=[p for p in project.rglob('*') if p.is_file() and (p.suffix in {'.py','.yaml'} or any(x in p.parts for x in ['main','human_review','paraphrases','tables'])) and '__pycache__' not in p.parts]
    hashes={str(p.relative_to(project)):sha(p) for p in tracked}
    sample={int(r['seed']):r for r in readcsv(project/'results/tables/sample_summary.csv')}
    score_rows=readcsv(project/'results/tables/predictor_scores.csv')
    calc_summary=[]
    for seed in [13,42,77]:
        run=project/f'results/main/seed_{seed}'
        raw={stage:jl(run/f'after_{stage}.jsonl') for stage in 'ABC'}
        maps={s:{r['example_id']:r for r in raw[s]} for s in 'ABC'}
        assert all(len(v)==7600 and len(raw[s])==7600 for s,v in maps.items())
        assert set(maps['A'])==set(maps['B'])==set(maps['C'])
        for ident,a in maps['A'].items():
            for s in 'ABC':
                r=maps[s][ident]
                assert (r['original_text'],r['gold_label'],r['text_group'])==(a['original_text'],a['gold_label'],a['text_group'])
                assert r['correct']==(r['pred']==r['gold_label'])
                scores=r['candidate_scores'];m=max(scores);w=[math.exp(t-m) for t in scores];ps=[t/sum(w) for t in w]
                assert r['pred']==ps.index(max(ps))
                close(r['confidence'],max(ps));close(r['gold_nll'],-math.log(max(ps[r['gold_label']],1e-300)))
                for x,z in zip(ps,r['probabilities']):close(x,z)
        pool=[r for r in raw['A'] if r['correct']]
        assert len({r['text_group'] for r in pool})==len(pool)
        per=jl(run/'inconsistency.jsonl');pm={r['example_id']:r for r in per}
        assert len(pm)==len(per)==len(pool) and set(pm)=={r['example_id'] for r in pool}
        assert all(r['n_paraphrases']==3 and any(abs(r['inconsistency']-k/3)<1e-12 for k in range(4)) for r in per)
        frame=readcsv(run/'results.csv');fm={r['example_id']:r for r in frame};assert len(fm)==7600
        for ident,a in maps['A'].items():
            r=fm[ident]
            for feature in ['confidence','entropy','margin','gold_nll']:close(r[feature],a[feature])
            if a['correct']:close(r['inconsistency'],pm[ident]['inconsistency'])
            else:assert r['inconsistency']==''
            for s in 'BC':
                expected=str(not maps[s][ident]['correct']) if a['correct'] else ''
                assert r[f'forgotten_after_{s}']==expected
        counts={'evaluated':7600,'A_correct':len(pool),'PER_positive':sum(r['inconsistency']>0 for r in per)}
        assert int(sample[seed]['n_correct_after_A'])==len(pool)
        for s in 'BC':
            y=[int(not maps[s][r['example_id']]['correct']) for r in pool]
            counts[s+'_errors']=sum(y);counts[s+'_rate']=sum(y)/len(y)
            assert int(sample[seed]['n_forgotten_after_'+s])==sum(y)
            close(sample[seed]['forgetting_rate_'+s],sum(y)/len(y))
            for signal,sign in [('inconsistency',1),('confidence',-1),('entropy',1),('margin',-1),('gold_nll',1)]:
                sc=[sign*(pm[r['example_id']]['inconsistency'] if signal=='inconsistency' else r[signal]) for r in pool]
                a,b=auroc_ap(y,sc)
                row=next(x for x in score_rows if int(x['seed'])==seed and x['endpoint']==s and x['predictor']==signal)
                close(a,row['auroc']);close(b,row['average_precision'])
                calc_summary.append((s,signal,a,b))
        report['seeds'][str(seed)]=counts
    # Validate OOF target alignment, fold separation, metrics, and saved summary means.
    for folder,summary_name in [('results/tables','seed_summary.csv'),('results/paper_analysis/tables','three_feature_summary.csv')]:
        f=project/folder
        candidates=list(f.glob('*oof*predictions*.csv'))
        if not candidates:
            report['checks'].append(folder+': OOF file not present; no refit claimed');continue
        oo=readcsv(candidates[0]);groups={}
        for r in oo:groups.setdefault((r['seed'],r['endpoint'],r['predictor']),[]).append(r)
        for (seed,endpoint,predictor),rr in groups.items():
            base={r['example_id']:r for r in readcsv(project/f'results/main/seed_{seed}/results.csv') if r['correct_after_A']=='True'}
            assert len(rr)==len(base)==len({r['example_id'] for r in rr})
            assert len({r['fold'] for r in rr})==5
            for r in rr:assert int(r['target'])==int(base[r['example_id']][f'forgotten_after_{endpoint}']=='True')
            a,b=auroc_ap([int(r['target']) for r in rr],[float(r['probability']) for r in rr])
            sr=readcsv(f/summary_name)
            # Individual metrics in original file; aggregate check below covers both.
        for endpoint in 'BC':
            for predictor in ['baseline_combined','baseline_plus_inconsistency']:
                values=[auroc_ap([int(r['target']) for r in rr],[float(r['probability']) for r in rr]) for (se,ep,pr),rr in groups.items() if ep==endpoint and pr==predictor]
                assert len(values)==3
                for j,metric in enumerate(['auroc','average_precision']):
                    row=next(r for r in sr if r['endpoint']==endpoint and r['predictor']==predictor and r['metric']==metric)
                    close(row['mean'],statistics.mean(v[j] for v in values))
        report['checks'].append(folder+': OOF targets, coverage, folds, and summary metrics match saved records')
    bank=jl(project/'results/cache/paraphrases/ag_news.jsonl')
    assert len(bank)==7600==len({r['example_id'] for r in bank})
    assert all(len(r['paraphrases'])==3 for r in bank)
    review=project/'results/human_review';a=readcsv(review/'reviewer_1.csv');b=readcsv(review/'reviewer_2.csv')
    assert len(a)==len(b)==600 and len({r['review_id'].split(':')[0] for r in a})==200
    assert [r['review_id'] for r in a]==[r['review_id'] for r in b]
    matches=sum(x['meaning_preserved']==y['meaning_preserved'] for x,y in zip(a,b));assert matches==412
    ca=Counter(r['meaning_preserved'] for r in a);cb=Counter(r['meaning_preserved'] for r in b)
    pe=sum(ca[k]*cb[k] for k in set(ca)|set(cb))/600**2;kappa=(matches/600-pe)/(1-pe)
    close(kappa,json.loads((review/'analysis/summary.json').read_text())['meaning_cohens_kappa'])
    pending=readcsv(review/'assistant_adjudication_200.csv');assert len(pending)==200 and all(r['consensus_status']=='pending_human_confirmation' for r in pending)
    intervals=readcsv(project/'results/tables/paired_differences.csv');assert len(intervals)==12
    report['intervals_containing_zero']=sum(float(r['ci_low'])<=0<=float(r['ci_high']) for r in intervals)
    report['human_review']={'pairs':600,'agreement':matches/600,'kappa':kappa,'pending':200}
    assert all(sha(project/n)==v for n,v in hashes.items()),'Source changed during audit'
    report['source_files_unchanged']=True;report['tracked_files']=len(hashes);report['status']='passed'
    report['not_verified_by_this_audit']=['New LLM inference on paraphrases','Refitting OOF logistic regressions','Regenerating the 5,000 bootstrap samples','Human adjudication of pending pairs']
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    (out/'results_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--out',required=True);a=p.parse_args();audit(a.project,a.out)
