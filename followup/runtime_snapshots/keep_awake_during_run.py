"""Temporary process-scoped idle-sleep request; no persistent power settings changed."""
import ctypes,json,time,os,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
kernel=ctypes.windll.kernel32
kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint]
kernel.SetThreadExecutionState.restype=ctypes.c_uint
kernel.OpenProcess.restype=ctypes.c_void_p
def alive(pid):
 h=kernel.OpenProcess(0x1000,False,int(pid))
 if not h:return False
 try:
  code=ctypes.c_ulong();return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(h),ctypes.byref(code))) and code.value==259
 finally:kernel.CloseHandle(ctypes.c_void_p(h))
def log(status):
 (ROOT/'power_request_status.json').write_text(json.dumps({'status':status,'pid':os.getpid(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'permanent_power_settings_changed':False,'display_forced_on':False,'does_not_prevent_manual_shutdown_or_reboot':True},indent=2),encoding='utf-8')
try:
 assert kernel.SetThreadExecutionState(0x80000001),'Windows idle-sleep request failed'
 log('system_idle_sleep_request_active')
 while True:
  time.sleep(30)
  try:state=json.loads((ROOT/'overall_status.json').read_text(encoding='utf-8'))
  except (OSError,ValueError):continue
  if state.get('status')!='running' or not alive(state['pid']):break
finally:
 kernel.SetThreadExecutionState(0x80000000);log('request_released')
