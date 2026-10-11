"""Shared active-job ledger; idle application intervals are not charged."""
import atexit,json,os,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
PATH=OUT/'ACTIVE_JOB_LEDGER.json'
CAP=10800.;HISTORICAL=657.305
ACTIVE=None

def load():
    return json.loads(PATH.read_text()) if PATH.exists() else dict(cap_seconds=CAP,v51_charged_seconds=HISTORICAL,segments=[],idle_intervals_excluded=True)

def save(x):
    t=PATH.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(PATH)

def elapsed():
    d=load();return d['v51_charged_seconds']+sum(s.get('charged_seconds', max(0,time.time()-s['start_epoch'])) for s in d['segments'])

def start(label):
    global ACTIVE
    d=load()
    assert not any('charged_seconds' not in s for s in d['segments']), 'unclosed job segment needs explicit audit before resume'
    ACTIVE=len(d['segments']);d['segments'].append(dict(label=label,pid=os.getpid(),start_epoch=time.time()))
    save(d);atexit.register(finish)

def finish():
    global ACTIVE
    if ACTIVE is None:return
    d=load();s=d['segments'][ACTIVE]
    if 'charged_seconds' not in s:
        s['end_epoch']=time.time();s['charged_seconds']=s['end_epoch']-s['start_epoch'];save(d)
    ACTIVE=None
