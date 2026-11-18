"""Offline safety and scientific invariants; no downloaded models or GPU training."""
import unittest,sys,json,gzip,hashlib,tempfile,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('restore',ROOT/'tools/restore_evidence.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from src.data import clean_training
from src.paraphrase import numbers,valid_paraphrases
from src.evaluate import probability_features
class ReproductionTests(unittest.TestCase):
 def test_evaluation_overlap_and_conflicting_labels(self):
  rows=[dict(example_id=str(i),text_group=g,gold_label=y) for i,(g,y) in enumerate([('test',0),('conflict',0),('conflict',1),('keep',1),('keep',1)])]
  cleaned,checks=clean_training(rows,[dict(text_group='test')]);self.assertEqual([r['example_id'] for r in cleaned],['3']);self.assertEqual(checks['train_eval_overlap_removed'],1);self.assertEqual(checks['train_conflicting_rows_removed'],2);self.assertEqual(checks['train_duplicates_removed'],1)
 def test_numeric_screen_is_not_semantic_validation(self):
  self.assertEqual(numbers('Sales rose 3 percent in <b>2025</b>.'),['2025','3%'])
  self.assertEqual(valid_paraphrases('A 3% increase.',['A 4% increase.','An increase of 3%.','An increase of 3%.']),['An increase of 3%.'])
 def test_confidence_is_candidate_normalization(self):
  import math
  r=probability_features([0,math.log(3)],1);self.assertAlmostEqual(r['confidence'],.75);self.assertEqual(r['pred'],1);self.assertTrue(r['correct']);self.assertAlmostEqual(r['gold_nll'],-math.log(.75))
 def test_restore_preserves_existing_files_and_rejects_tampering(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'evidence').mkdir();content=b'original evidence\n';packed=gzip.compress(content,mtime=0);p=root/'evidence/file.gz';p.write_bytes(packed)
   entry={'cohort':'original','path':'main/file.txt','archive':'evidence/file.gz','gzip_sha256':hashlib.sha256(packed).hexdigest(),'sha256':hashlib.sha256(content).hexdigest(),'uncompressed_bytes':len(content)}
   (root/'evidence/manifest.json').write_text(json.dumps({'entries':[entry]}))
   dest=root/'restored';self.assertEqual(module.restore('original',dest,root)['restored'],1);self.assertEqual(module.restore('original',dest,root)['already_identical'],1)
   (dest/'main/file.txt').write_bytes(b'user changes')
   with self.assertRaises(FileExistsError):module.restore('original',dest,root)
   self.assertEqual((dest/'main/file.txt').read_bytes(),b'user changes')
   p.write_bytes(b'corrupt')
   with self.assertRaises(ValueError):module.restore('original',root/'another',root)
 def test_restoration_rejects_path_escape(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'evidence').mkdir();(root/'evidence/manifest.json').write_text(json.dumps({'entries':[{'cohort':'original','path':'../escape','archive':'evidence/x'}]}))
   with self.assertRaises(ValueError):module.restore('original',root/'target',root)
if __name__=='__main__':unittest.main()
