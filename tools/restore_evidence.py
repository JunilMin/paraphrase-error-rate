"""Restore byte-verified saved evidence; never overwrite differing files."""
import argparse,gzip,hashlib,json,os,tempfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
def restore(cohort,destination,repo=REPO,bank_only=False):
 repo=Path(repo).resolve();destination=Path(destination).resolve()
 manifest=json.loads((repo/'evidence/manifest.json').read_text(encoding='utf-8'))
 entries=[r for r in manifest['entries'] if r['cohort']==cohort]
 if bank_only:entries=[r for r in entries if r['path'].startswith('cache/paraphrases/')]
 if not entries:raise ValueError('Unknown or empty cohort: '+cohort)
 restored=skipped=0
 for row in entries:
  target=(destination/row['path']).resolve();archive=(repo/row['archive']).resolve()
  if not target.is_relative_to(destination) or not archive.is_relative_to(repo/'evidence'):raise ValueError('Unsafe evidence path')
  packed=archive.read_bytes()
  if hashlib.sha256(packed).hexdigest()!=row['gzip_sha256']:raise ValueError('Archive digest mismatch: '+row['archive'])
  content=gzip.decompress(packed)
  if len(content)!=row['uncompressed_bytes'] or hashlib.sha256(content).hexdigest()!=row['sha256']:raise ValueError('Uncompressed evidence mismatch: '+row['path'])
  if target.exists():
   if hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:raise FileExistsError('Preserve existing differing file: '+str(target))
   skipped+=1;continue
  target.parent.mkdir(parents=True,exist_ok=True)
  # Exclusive creation protects any file introduced after the existence check.
  with target.open('xb') as f:f.write(content)
  restored+=1
 return {'cohort':cohort,'restored':restored,'already_identical':skipped,'destination':str(destination)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cohort',default='original',choices=['original','more_seeds','reverse_order']);p.add_argument('--destination',type=Path,default=REPO/'results');p.add_argument('--bank-only',action='store_true');a=p.parse_args()
 print(json.dumps(restore(a.cohort,a.destination,bank_only=a.bank_only),indent=2))
