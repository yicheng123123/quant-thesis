from pathlib import Path
import argparse, csv, ctypes as ct, json, os, platform, subprocess, sys, threading, time, traceback
import psutil

ROOT=Path(__file__).resolve().parent

class MemoryInfo(ct.Structure):
    _fields_=[('total',ct.c_ulonglong),('free',ct.c_ulonglong),('used',ct.c_ulonglong)]

class ProcessInfo(ct.Structure):
    _fields_=[('pid',ct.c_uint),('usedGpuMemory',ct.c_ulonglong),('gpuInstanceId',ct.c_uint),('computeInstanceId',ct.c_uint)]

class Nvml:
    def __init__(self):
        self.dll=ct.WinDLL('nvml.dll')
        ret=self.dll.nvmlInit_v2()
        if ret: raise RuntimeError(('nvmlInit',ret))
        self.handle=ct.c_void_p()
        ret=self.dll.nvmlDeviceGetHandleByIndex_v2(0,ct.byref(self.handle))
        if ret: raise RuntimeError(('nvmlHandle',ret))

    def memory(self):
        m=MemoryInfo(); ret=self.dll.nvmlDeviceGetMemoryInfo(self.handle,ct.byref(m))
        if ret: raise RuntimeError(('nvmlMemory',ret))
        return m.total,m.used,m.free

    def processes(self):
        count=ct.c_uint(512); rows=(ProcessInfo*512)()
        ret=self.dll.nvmlDeviceGetComputeRunningProcesses_v2(self.handle,ct.byref(count),rows)
        if ret: return {},'NOT_AVAILABLE: nvml process query status '+str(ret)
        found={}; available=False
        for r in rows[:count.value]:
            if r.usedGpuMemory==2**64-1: found[int(r.pid)]=None
            else: found[int(r.pid)]=int(r.usedGpuMemory); available=True
        return found, 'AVAILABLE' if available else 'NOT_AVAILABLE: WDDM returned unavailable process memory or no attributable process'

class Sampler:
    def __init__(self,name):
        self.name=name; self.start=time.monotonic_ns(); self.pid=None; self.stop=threading.Event()
        self.errors=[]; self.process_peaks={}; self.device_peak=None; self.aggregate_peak=0; self.count=0
        self.gpu_status=None; self.nvml=None; self.last_phase='PRE_RUN_BASELINE'
        try: self.nvml=Nvml()
        except BaseException as e: self.gpu_status='NOT_AVAILABLE: '+repr(e)
        self.gf=(ROOT/'memory'/('gpu_'+name+'.csv')).open('x',newline='')
        self.cf=(ROOT/'memory'/('rss_'+name+'.csv')).open('x',newline='')
        self.ef=(ROOT/'memory'/('errors_'+name+'.jsonl')).open('x')
        self.gw=csv.DictWriter(self.gf,fieldnames=['monotonic_ns','elapsed_seconds','phase','block','device_used_bytes','device_used_GiB','device_total_bytes','device_free_bytes','target_pid','target_gpu_bytes','target_gpu_GiB','process_gpu_status'])
        self.cw=csv.DictWriter(self.cf,fieldnames=['monotonic_ns','elapsed_seconds','phase','block','pid','creation_time','role','name','rss_bytes','rss_GiB','cpu_user_seconds','cpu_system_seconds'])
        self.gw.writeheader(); self.cw.writeheader()

    def tick(self):
        now=time.monotonic_ns(); elapsed=(now-self.start)/1e9; phase=self.last_phase; block=-1; workload_pid=self.pid
        f=ROOT/'timings'/'STAGE_EVENTS.jsonl'
        if self.pid and f.exists():
            try:
                x=json.loads(f.read_text().splitlines()[-1])
                if x.get('monotonic_ns',0)>=self.start:
                    phase=x['phase']; block=x['block']; self.last_phase=phase; workload_pid=x.get('workload_pid',self.pid)
            except (OSError,ValueError,KeyError,IndexError): pass
        used=total=free=None; process_gpu=None; process_status=self.gpu_status
        if self.nvml:
            total,used,free=self.nvml.memory()
            self.device_peak=max(self.device_peak or 0,used)
            processes,process_status=self.nvml.processes()
            if workload_pid in processes: process_gpu=processes[workload_pid]
            if self.pid and process_gpu is None:
                process_status='NOT_AVAILABLE: target process memory unavailable on this platform'
        self.gpu_status=process_status
        self.gw.writerow(dict(monotonic_ns=now,elapsed_seconds=elapsed,phase=phase,block=block,
            device_used_bytes=used,device_used_GiB=used/2**30 if used is not None else None,device_total_bytes=total,
            device_free_bytes=free,target_pid=workload_pid,target_gpu_bytes=process_gpu,
            target_gpu_GiB=process_gpu/2**30 if process_gpu is not None else None,process_gpu_status=process_status))
        procs=[]
        if self.pid:
            try:
                p=psutil.Process(self.pid); procs=[(p,'MAIN')]+[(c,'CHILD') for c in p.children(recursive=True)]
            except psutil.NoSuchProcess: pass
        aggregate=0
        for p,role in procs:
            try:
                if p.pid==workload_pid: role='MAIN'
                elif p.pid==self.pid: role='VENV_LAUNCHER'
                with p.oneshot():
                    rss=p.memory_info().rss; created=p.create_time(); name=p.name(); cpu=p.cpu_times()
                key=str(p.pid)+'_'+str(created)
                old=self.process_peaks.setdefault(key,dict(pid=p.pid,creation_time=created,role=role,name=name,peak_rss_bytes=0))
                old['peak_rss_bytes']=max(old['peak_rss_bytes'],rss); old['peak_rss_GiB']=old['peak_rss_bytes']/2**30
                old['last_cpu_user_seconds']=cpu.user; old['last_cpu_system_seconds']=cpu.system
                aggregate+=rss
                self.cw.writerow(dict(monotonic_ns=now,elapsed_seconds=elapsed,phase=phase,block=block,pid=p.pid,
                    creation_time=created,role=role,name=name,rss_bytes=rss,rss_GiB=rss/2**30,cpu_user_seconds=cpu.user,cpu_system_seconds=cpu.system))
            except psutil.NoSuchProcess: pass
            except BaseException as e: self.record_error(e)
        self.aggregate_peak=max(self.aggregate_peak,aggregate); self.count+=1
        for f in [self.gf,self.cf,self.ef]: f.flush()

    def record_error(self,e):
        x=dict(monotonic_ns=time.monotonic_ns(),error=repr(e)); self.errors.append(x)
        self.ef.write(json.dumps(x)+'\n'); self.ef.flush()

    def run(self):
        deadline=time.monotonic()
        while not self.stop.is_set():
            try: self.tick()
            except BaseException as e: self.record_error(e)
            deadline+=.2; self.stop.wait(max(0,deadline-time.monotonic()))
            if deadline<time.monotonic()-.2: deadline=time.monotonic()

    def finish(self):
        self.stop.set(); self.thread.join(timeout=10)
        try: self.tick()
        except BaseException as e: self.record_error(e)
        for f in [self.gf,self.cf,self.ef]: f.close()
        x=dict(samples=self.count,target_interval_seconds=.2,device_sampled_peak_bytes=self.device_peak,
            device_sampled_peak_GiB=self.device_peak/2**30 if self.device_peak is not None else None,
            process_gpu_status=self.gpu_status,process_rss_peaks=list(self.process_peaks.values()),
            aggregate_concurrent_target_tree_rss_peak_bytes=self.aggregate_peak,
            aggregate_concurrent_target_tree_rss_peak_GiB=self.aggregate_peak/2**30,
            errors=self.errors,sampled_peak_is_not_exact_instantaneous_peak=True,
            device_total_includes_desktop_and_other_processes=True,
            short_lived_children_between_samples_may_be_missed=True)
        (ROOT/'memory'/('summary_'+self.name+'.json')).write_text(json.dumps(x,indent=2))
        if self.nvml: self.nvml.dll.nvmlShutdown()
        return x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--resume',action='store_true'); a=ap.parse_args()
    name=time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    for p in [ROOT/'memory',ROOT/'timings',ROOT/'M3/logs',ROOT/'M3/checkpoints']:
        if not p.is_dir(): raise RuntimeError('missing output directory: '+str(p))
        with (p/('WRITE_CHECK_'+name+'.txt')).open('x') as f: f.write('write check before CUDA child start\n')
    sampler=Sampler(name); sampler.tick()
    sampler.thread=threading.Thread(target=sampler.run,daemon=True); sampler.thread.start()
    launch=time.monotonic_ns(); cpu0=time.process_time_ns(); child=None; code=None; error=None
    log=ROOT/'M3/logs'/('process_'+name+'.log')
    env=dict(os.environ); env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
    cmd=[sys.executable,'-B',str(ROOT/'M3/run_formal_m3.py')]+(['--resume'] if a.resume else [])
    try:
        with log.open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=ROOT.parents[1])
            sampler.pid=child.pid
            (ROOT/'ACTIVE_PROCESS.json').write_text(json.dumps(dict(pid=child.pid,supervisor_pid=os.getpid(),command=cmd,log=str(log),launch_monotonic_ns=launch),indent=2))
            print('RUNNING',child.pid,log,flush=True)
            code=child.wait()
    except BaseException as e:
        error=repr(e)
        if child and child.poll() is None:
            # Graceful Ctrl+C is forwarded first; final sampler flush is guaranteed.
            try: child.send_signal(__import__('signal').CTRL_BREAK_EVENT); child.wait(timeout=15)
            except BaseException: child.terminate(); child.wait()
        raise
    finally:
        end=time.monotonic_ns(); memory=sampler.finish()
        report=dict(status='COMPLETE' if code==0 else 'FAILED_OR_INTERRUPTED',exit_code=code,error=error,
            attempt=name,resume=a.resume,command=cmd,python=sys.version,executable=sys.executable,
            launch_monotonic_ns=launch,end_monotonic_ns=end,end_to_end_wall_seconds=(end-launch)/1e9,
            supervisor_cpu_process_seconds=(time.process_time_ns()-cpu0)/1e9,memory_summary=str(ROOT/'memory'/('summary_'+name+'.json')),
            main_cpu_process_seconds_source='stage meter plus RSS timeline; separate from supervisor',log=str(log))
        (ROOT/('PROCESS_ACCOUNTING_'+name+'.json')).write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
    return code or 0

if __name__=='__main__': raise SystemExit(main())
