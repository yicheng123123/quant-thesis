import atexit, json, os, time
from pathlib import Path
O=Path(__file__).resolve().parent; P=O/'ACTIVE_JOB_LEDGER.json'; CAP=9000.0; VERSION='v56'; ACTIVE=None

def load(): return json.loads(P.read_text(encoding='utf8'))
def save(x):
    tmp=P.with_suffix('.json.tmp'); tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8'); tmp.replace(P)
def seconds():
    x=load(); base=float(x['preflight_charged_seconds']); vals=[]
    for s in x['segments']:
        if 'charged_seconds' in s: v=float(s['charged_seconds'])
        else: v=max(0.0,time.time()-float(s['start_epoch']))
        vals.append((s.get('version'),v))
    local=sum(v for k,v in vals if k==VERSION)
    return base+sum(v for _,v in vals),local
def charge_cpu(label,seconds,**details):
    x=load(); x['segments'].append(dict(version=VERSION,label=label,charged_seconds=float(seconds),cpu_only=True,GPU_calls=0,**details)); save(x)
def start(label):
    global ACTIVE
    x=load(); total,_=seconds(); assert total<CAP, f'v56 cap already exhausted: {total}'
    ACTIVE=len(x['segments']); x['segments'].append(dict(version=VERSION,label=label,pid=os.getpid(),start_epoch=time.time())); save(x); atexit.register(finish)
def finish():
    global ACTIVE
    if ACTIVE is None:return
    x=load(); s=x['segments'][ACTIVE]
    if 'charged_seconds' not in s: s.update(end_epoch=time.time(),charged_seconds=max(0.0,time.time()-float(s['start_epoch']))); save(x)
    ACTIVE=None
def elapsed(): return seconds()[0]
