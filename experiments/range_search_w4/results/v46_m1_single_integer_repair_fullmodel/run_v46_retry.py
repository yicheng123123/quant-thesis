import csv, hashlib, importlib.util, json, os, sys, threading, time, traceback
from pathlib import Path
import torch
from safetensors import safe_open
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
sys.path.insert(0,str(OUT));sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('run_v42',OUT/'runtime_v46.py')
rt=importlib.util.module_from_spec(spec);sys.modules['run_v42']=rt;spec.loader.exec_module(rt)
import repair_v45 as repair
# Critical launcher binding: formal run_layer resolves this exact byte-identical v45 function.
rt.batch_repair=repair.batch_repair
CAP=10800;RESERVE=360
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def hashfile(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def precision_seconds():
    p=OUT/'STAGE_TIMINGS.csv'
    return sum(float(r['seconds']) for r in csv.DictReader(p.open(encoding='utf8')) if r['stage']=='independent_FP64_increment_recheck') if p.exists() else 0.
def deadline(reserve=0):
    if rt.WALL_START is not None and rt.elapsed()>CAP-reserve:raise TimeoutError(f'v46 cumulative 180-minute cap/reserve reached at {rt.elapsed():.1f}s')
def verify_final_parameters():
    src=rt.V18/'M1/checkpoint/final/model.safetensors';dst=OUT/'R1/checkpoint/final/model.safetensors'
    allowed={f'model.layers.{li}.{name}.weight' for li in range(24) for group in rt.GROUPS.values() for name in group}
    with safe_open(str(src),framework='pt',device='cpu') as a,safe_open(str(dst),framework='pt',device='cpu') as b:
        assert set(a.keys())==set(b.keys());unchanged=0
        for k in a.keys():
            if k not in allowed:
                x=a.get_tensor(k);y=b.get_tensor(k)
                assert torch.equal(x.view(torch.uint8),y.view(torch.uint8)),f'non-quantized M1 tensor changed: {k}'
                unchanged+=1
    ca=json.loads((rt.V18/'M1/checkpoint/final/config.json').read_text());cb=json.loads((OUT/'R1/checkpoint/final/config.json').read_text())
    ignore={'_name_or_path','transformers_version'}
    assert {k:v for k,v in ca.items() if k not in ignore}=={k:v for k,v in cb.items() if k not in ignore}
    rt.dump('DEPLOYMENT_LOCK.json',{'pass':True,'configuration_equal_except_paths_and_transformers_version':True,
       'non_target_model_tensors_bitwise_equal':True,'non_target_tensor_count':unchanged,
       'final_model_sha256':hashfile(dst),'config_sha256':hashfile(OUT/'R1/checkpoint/final/config.json'),
       'checkpoint_lock_sha256':hashfile(OUT/'CHECKPOINT_LOCK.json'),'completed_unix':time.time()})
def main():
    assert read('SOURCE_AUDIT.json')['pass'] and read('BASELINE_AUDIT.json')['pass'] and read('COST_PROJECTION.json')['go']
    assert read('validation/PREFLIGHT_GPU.json')['pass'],'reuse only the already-passed v45 preflight'
    freeze=read('PRE_GPU_FREEZE.json')
    for n,h in freeze['frozen_file_sha256'].items():assert hashfile(OUT/n)==h,f'frozen source drift: {n}'
    amendment=read('EXECUTION_AMENDMENT_001.json')
    assert amendment['corrected_launcher_sha256']==hashfile(OUT/'run_v46_retry.py')
    assert amendment['formal_binding']=='runtime.batch_repair = repair_v45.batch_repair'
    assert read('GPU_START.json')['hard_cap_seconds']==CAP
    assert not (OUT/'CHECKPOINT_LOCK.json').exists()
    assert len(list((OUT/'formal/groups').glob('*.pt')))==0,'invalid V42-dispatch groups must never enter the corrected R1'
    rt.WALL_START=read('GPU_START.json')['unix'];rt.START=time.perf_counter();rt.deadline=deadline;rt.MODE='preflight'
    def hard_stop():
        rt.dump('RUN_STATUS.json',{'status':'INCOMPLETE_HARD_CAP','category':'D','elapsed_seconds':rt.elapsed(),'complete_model':False,'PPL_run':False})
        os._exit(124)
    watchdog=threading.Timer(max(1,CAP-rt.elapsed()),hard_stop);watchdog.daemon=True;watchdog.start()
    try:
        with rt.timer('retry_warmup'):
            z=torch.zeros((64,64),device='cuda');z@z
        model=rt.model_load()
        ids=rt.zero_model_check(model)
        hidden=rt.embed(model)
        projection=read('COST_PROJECTION.json')
        extra=max(0,projection['FP64_recheck_allowance_seconds']-precision_seconds())
        remaining=24*projection['measured_full_block_max_seconds']*projection['safety_multiplier']+extra+RESERVE
        rt.dump('BUDGET_STATUS.json',{'phase':'corrected_restart_from_original_M1','elapsed_seconds':rt.elapsed(),'blocks_remaining':24,
          'new_formal_groups':96,'bad_attempt_groups_reused':0,'remaining_FP64_allowance_prediction_seconds':extra,
          'finish_reserve_seconds':RESERVE,'conservative_remaining_seconds':remaining,'go':rt.elapsed()+remaining<=CAP})
        assert rt.elapsed()+remaining<=CAP,'cumulative failure cost leaves insufficient frozen conservative forecast'
        rt.MODE='formal';rt.batch_repair=repair.batch_repair
        for li in range(24):
            remain=24-li;extra=max(0,projection['FP64_recheck_allowance_seconds']-precision_seconds())
            forecast=remain*projection['measured_full_block_max_seconds']*projection['safety_multiplier']+extra+RESERVE
            rt.dump('BUDGET_STATUS.json',{'phase':'formal_before_layer','layer':li,'elapsed_seconds':rt.elapsed(),
             'remaining_blocks':remain,'safety_multiplier':projection['safety_multiplier'],
             'remaining_FP64_allowance_prediction_seconds':extra,'finish_reserve_seconds':RESERVE,
             'conservative_remaining_seconds':forecast,'go':rt.elapsed()+forecast<=CAP})
            assert rt.elapsed()+forecast<=CAP,'conservative completion reserve no longer fits'
            start=time.perf_counter();hidden=rt.run_layer(model,li,hidden);sec=time.perf_counter()-start
            with (OUT/'BLOCK_TIMINGS.csv').open('a',encoding='utf8',newline='') as f:
                fields=['layer','seconds','elapsed_seconds','new_groups'];w=csv.DictWriter(f,fieldnames=fields)
                if f.tell()==0:w.writeheader()
                w.writerow({'layer':li,'seconds':sec,'elapsed_seconds':rt.elapsed(),'new_groups':4})
            rt.dump('formal/PREFIX_STATUS.json',{'completed_layer':li,'hidden_sha256':hashlib.sha256(hidden.tobytes()).hexdigest(),'elapsed_seconds':rt.elapsed()})
            print('BLOCK_COMPLETE',li,'seconds',round(sec,3),'elapsed',round(rt.elapsed(),3),flush=True)
        assert len(rt.STATS)==168 and sum(int(s['rows']) for s in rt.STATS)==304128
        rt.dump('FORMAL_REPAIR_COMPLETE.json',{'modules':168,'rows':304128,'elapsed_seconds':rt.elapsed(),
          'accepted_updates':sum(int(s['accepted_updates']) for s in rt.STATS),
          'modified_coordinates':sum(int(s['modified_coordinates']) for s in rt.STATS),
          'candidate_evaluations':sum(int(s['candidate_evaluations']) for s in rt.STATS),
          'rejected_candidates':sum(int(s['rejected_candidates']) for s in rt.STATS)})
        assert rt.elapsed()+RESERVE<=CAP,'finish reserve exhausted before final save/evaluation'
        rt.MODE='deployment';model=rt.final_save_check(model,ids);verify_final_parameters()
        rt.MODE='evaluation';rt.evaluate(model)
        status={'status':'COMPLETE','category':'quality_pending_aggregation','complete_model':True,'PPL_run':True,'modules':168,'rows':304128,
         'elapsed_seconds':rt.elapsed(),'hard_cap_seconds':CAP,'gpu_name':torch.cuda.get_device_name(),'torch':torch.__version__,
         'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
         'FP64_recheck_count':len(list(csv.DictReader((OUT/'NUMERICAL_RECHECKS.csv').open(encoding='utf8')))) if (OUT/'NUMERICAL_RECHECKS.csv').exists() else 0,
         'failed_dispatch_attempt_elapsed_seconds':read('EXECUTION_AMENDMENT_001.json')['failed_attempt_elapsed_seconds']}
        rt.dump('RUN_STATUS.json',status)
    finally:watchdog.cancel()
if __name__=='__main__':
    try:main()
    except BaseException as ex:
        tb=traceback.format_exc()
        try:
            rt.dump('RUN_STATUS.json',{'status':'FAILED_STOP','category':'D','error':repr(ex),'traceback':tb,'elapsed_seconds':rt.elapsed() if rt.WALL_START else 0,'complete_model':False,'PPL_run':False})
            with (OUT/'FAILED_ATTEMPTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'error':repr(ex),'traceback':tb,'elapsed_seconds':rt.elapsed() if rt.WALL_START else 0})+'\n')
        except Exception:pass
        raise
