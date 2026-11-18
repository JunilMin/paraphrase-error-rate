"""Control-plane JSON writes tolerate brief Windows reader locks."""
import json,os,time,uuid
from pathlib import Path
def atomic_json(path,obj,attempts=30,delay=.1,replace=None):
 path=Path(path);replace=replace or os.replace
 temp=path.with_name('.'+path.stem+'.'+uuid.uuid4().hex+'.tmp')
 with temp.open('w',encoding='utf-8') as f:
  f.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));f.flush();os.fsync(f.fileno())
 for attempt in range(attempts):
  try:replace(temp,path);return
  except PermissionError:
   if attempt+1==attempts:raise
   time.sleep(delay)
