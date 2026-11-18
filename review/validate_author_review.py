"""Validate actual author responses; never infer or fill a human judgment."""
import argparse,csv,json,hashlib
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('completed_csv',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
def read(path):return list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
original=read(root/'human_review/author_review_200.csv');expected={r['review_id']:r for r in original}
rows=read(a.completed_csv)
assert len(rows)==200==len({r['review_id'] for r in rows})
assert set(expected)=={r['review_id'] for r in rows}
for r in rows:
 e=expected[r['review_id']]
 assert all(r[k]==e[k] for k in ['original_text','paraphrase_text'])
 assert all(r[k] in ['yes','no','uncertain'] for k in ['meaning_preserved','label_preserved'])
 assert all(r[k].strip() for k in ['notes','reviewer','review_date'])
if a.out.exists():raise FileExistsError('Do not overwrite a review report')
report={'status':'all_200_response_fields_validated','review_provenance':'author additional human adjudication; identity and independence are author-reported; not consensus between original reviewers','pairs':200,'meaning_counts':dict(Counter(r['meaning_preserved'] for r in rows)),'label_counts':dict(Counter(r['label_preserved'] for r in rows)),'reviewers':sorted({r['reviewer'] for r in rows}),'completed_csv_sha256':hashlib.sha256(a.completed_csv.read_bytes()).hexdigest(),'original_records_modified':False}
a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
