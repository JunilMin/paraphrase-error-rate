"""Create blank human-only forms; never convert AI judgments into human decisions."""
import argparse,csv,json,hashlib
from pathlib import Path
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,rows,cols):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def prepare(project,out):
    source=Path(project)/'results/human_review';out=Path(out)
    if out.exists():raise FileExistsError('Keep existing review responses; choose a new folder')
    out.mkdir(parents=True)
    unresolved=read(source/'assistant_adjudication_200.csv')
    assert len(unresolved)==200==len({r['review_id'] for r in unresolved})
    blank=[{k:r[k] for k in ['review_id','original_text','paraphrase_text']}|{'meaning_preserved':'','label_preserved':'','error_type':'','notes':'','reviewer':'','review_date':''} for r in unresolved]
    cols=list(blank[0]);write(out/'independent_reviewer_A_200.csv',blank,cols);write(out/'independent_reviewer_B_200.csv',blank,cols)
    a={r['review_id']:r for r in read(source/'reviewer_1.csv')};b={r['review_id']:r for r in read(source/'reviewer_2.csv')}
    final=[]
    for r in unresolved:
        key=r['review_id'];assert r['original_text']==a[key]['original_text']==b[key]['original_text']
        final.append({k:r[k] for k in ['review_id','original_text','paraphrase_text']}|{'original_r1_meaning':a[key]['meaning_preserved'],'original_r2_meaning':b[key]['meaning_preserved'],'final_meaning_preserved':'','final_label_preserved':'','decision_reason':'','human_adjudicator':'','review_date':''})
    write(out/'human_adjudication_200.csv',final,list(final[0]))
    (out/'input_hashes.json').write_text(json.dumps({n:sha(source/n) for n in ['reviewer_1.csv','reviewer_2.csv','assistant_adjudication_200.csv']},indent=2),encoding='utf-8')
    return {'pairs':len(final),'forms':'blank; no model scores, later outcomes, or AI resolutions exported'}
def validate(project,completed,out):
    source=Path(project)/'results/human_review';pending=read(source/'assistant_adjudication_200.csv');expected={r['review_id']:r for r in pending}
    rows=read(Path(completed));assert len(rows)==200==len({r['review_id'] for r in rows})
    assert {r['review_id'] for r in rows}==set(expected)
    for r in rows:
        e=expected[r['review_id']]
        assert r['original_text']==e['original_text'] and r['paraphrase_text']==e['paraphrase_text']
        assert r['final_meaning_preserved'] in ['yes','no','uncertain']
        assert r['final_label_preserved'] in ['yes','no','uncertain']
        assert all(r.get(k,'').strip() for k in ['decision_reason','human_adjudicator','review_date'])
    from collections import Counter
    report={'status':'completed_fields_validated','provenance':'human identity and review independence are author-reported, not authenticated by this script','pairs':200,'meaning':dict(Counter(r['final_meaning_preserved'] for r in rows)),'label':dict(Counter(r['final_label_preserved'] for r in rows)),'completed_form_sha256':sha(Path(completed)),'source_records_modified':False}
    out=Path(out)
    if out.exists():raise FileExistsError('Use a new report path')
    out.write_text(json.dumps(report,indent=2),encoding='utf-8');return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--out',required=True);p.add_argument('--validate-completed');a=p.parse_args()
    print(json.dumps(validate(a.project,a.validate_completed,a.out) if a.validate_completed else prepare(a.project,a.out),indent=2))
