"""Use the same saved-result analysis with robust control-plane writes only."""
import finish_reports as reports
from control_io_v2 import atomic_json
reports.write=atomic_json
if __name__=='__main__':
 try:reports.main()
 except Exception as e:
  atomic_json(reports.ROOT/'finish_reports_failure.json',{'error':str(e),'control_version':'v2_20261118','utc':reports.datetime.datetime.now(reports.datetime.timezone.utc).isoformat()});raise
