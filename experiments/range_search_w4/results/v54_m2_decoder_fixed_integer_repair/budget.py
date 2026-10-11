import atexit,json,os,time
from pathlib import Path
O=Path(__file__).resolve().parent;P=O/'ACTIVE_JOB_LEDGER.json';ACTIVE=None;CAP=10800.;VERSION='v54'
def load():return json.loads(P.read_text())
def save(x):
 t=P.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\
');t.replace(P)
def seconds():
 x=load();vals=[(s.get('version'),s.get('charged_seconds',max(0,time.time()-s['start_epoch']))) for s in x['segments']];return x['v51_charged_seconds']+sum(v for _,v in vals),sum(v for k,v in vals if k==VERSION)
def start(label):
 global ACTIVE
 x=load();assert all('charged_seconds' in s for s in x['segments']);total,local=seconds();assert total<CAP
 ACTIVE=len(x['segments']);x['segments'].append(dict(version=VERSION,label=label,pid=os.getpid(),start_epoch=time.time()));save(x);atexit.register(finish)
def finish():
 global ACTIVE
 if ACTIVE is None:return
 x=load();s=x['segments'][ACTIVE]
 if 'charged_seconds' not in s:s.update(end_epoch=time.time());s['charged_seconds']=s['end_epoch']-s['start_epoch'];save(x)
 ACTIVE=None

def elapsed():return seconds()[0]
