import csv, hashlib, importlib.util, json, math, shutil, sys, time, traceback
from pathlib import Path
import torch
from safetensors import safe_open
OUT=Path(__file__).resolve().parent
ROOT=Path(r'E:\\quant-thesis')
BASE=ROOT/'experiments/range_search_w4/results'
V18=BASE/'range_w4_fullmodel_v18'
V45=BASE/'v45_m2_single_integer_repair_fullmodel'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def dump(path,obj):
    p=OUT/path
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    tmp.replace(p)

sys.path.insert(0,str(OUT))
import budget
budget.start('v55 fresh frozen WT2 and C4 evaluation continuation')
try:
    spec=importlib.util.spec_from_file_location('v55_runner_eval_only',OUT/'run_v55.py')
    runner=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=runner
    spec.loader.exec_module(runner)
    rt=runner.rt
    modelpath=OUT/'R1/checkpoint/final/model.safetensors'
    cfgpath=OUT/'R1/checkpoint/final/config.json'
    lock=json.loads((OUT/'CHECKPOINT_LOCK.json').read_text(encoding='utf8'))
    assert sha(modelpath)==lock['final_model_sha256']
    assert sha(cfgpath)==lock['config_sha256']
    assert not (OUT/'R1_wikitext.json').exists()
    assert not (OUT/'R1_c4_fixed_budget.json').exists()

    from transformers import AutoConfig
    config_paths={
        'M1':V18/'M1/checkpoint/final',
        'M2':V18/'M2/checkpoint/final',
        'v55':OUT/'R1/checkpoint/final',
        'historic_R2':V45/'R/checkpoint/final'
    }
    volatile={'_name_or_path','name_or_path','transformers_version','_commit_hash'}
    raw={}; effective={}
    for name,path in config_paths.items():
        cp=path/'config.json'
        raw[name]={'path':str(cp),'sha256':sha(cp)}
        obj=AutoConfig.from_pretrained(str(path),local_files_only=True).to_dict()
        effective[name]={k:v for k,v in obj.items() if k not in volatile}
    canonical={k:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf8')).hexdigest() for k,v in effective.items()}
    dump('CONFIG_COMPARISON.json',{
        'raw_config_files':raw,
        'current_loader':'Transformers '+__import__('transformers').__version__,
        'effective_config_hashes':canonical,
        'effective_configs_equal':len(set(canonical.values()))==1,
        'volatile_fields_ignored':sorted(volatile),
        'meaning':'Raw JSON serialization differs across Transformers versions; effective configs are normalized by the same current loader.'
    })

    model_r2=V45/'R/checkpoint/final/model.safetensors'
    model_m2=V18/'M2/checkpoint/final/model.safetensors'
    tensor_rows=[]
    with safe_open(str(modelpath),framework='pt',device='cpu') as ours, safe_open(str(model_r2),framework='pt',device='cpu') as r2, safe_open(str(model_m2),framework='pt',device='cpu') as m2:
        keys=set(ours.keys()); assert keys==set(r2.keys())==set(m2.keys())
        for key in sorted(keys):
            a=ours.get_tensor(key); b=r2.get_tensor(key); c=m2.get_tensor(key)
            eqr=torch.equal(a.view(torch.uint8),b.view(torch.uint8))
            eqm=torch.equal(a.view(torch.uint8),c.view(torch.uint8))
            if not eqr or not eqm:
                row={'tensor':key,'equal_R2':eqr,'equal_M2':eqm}
                if not eqr: row['max_abs_vs_R2']=float((a.float()-b.float()).abs().max())
                if not eqm: row['max_abs_vs_M2']=float((a.float()-c.float()).abs().max())
                tensor_rows.append(row)
    with (OUT/'reports/TENSOR_DIFFERENCES_VS_R2_AND_M2.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['tensor','equal_R2','equal_M2','max_abs_vs_R2','max_abs_vs_M2'])
        w.writeheader(); w.writerows(tensor_rows)
    dump('DEPLOYMENT_COMPARISON.json',{
        'model_sha256':sha(modelpath),'M2_model_sha256':sha(model_m2),'historic_R2_model_sha256':sha(model_r2),
        'tensor_keys_compared':len(keys),'named_tensors_different_vs_R2':len(tensor_rows),
        'named_tensors_different_vs_M2':sum(not r['equal_M2'] for r in tensor_rows),
        'effective_config_hashes':canonical,'effective_config_equal_R2':canonical['v55']==canonical['historic_R2'],
        'raw_config_serialization_difference_is_not_model_difference':True
    })

    hashes=runner.in_memory_eval_protocol()
    rt.OUT=OUT
    rt.elapsed=runner.elapsed
    rt.deadline=lambda reserve=0: runner.deadline(max(360.0,float(reserve)))
    rt.MODE='evaluation'; rt.TIMINGS=[]
    assert rt.P['v18']['source_protocol']=={'v16_wikitext_manifest_sha256':hashes['wikitext'],'v16_c4_manifest_sha256':hashes['c4_fixed_budget']}
    model=rt.AutoModelForCausalLM.from_pretrained(str(OUT/'R1/checkpoint/final'),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()
    torch.cuda.reset_peak_memory_stats()
    rt.evaluate(model)
    results=[]
    for ds,key in [('wikitext','wikitext'),('c4_fixed_budget','c4_fixed_budget')]:
        src=OUT/f'R1_{ds}.json'; win=OUT/f'R1_{ds}_WINDOWS.csv'
        assert src.exists() and win.exists()
        res=json.loads(src.read_text(encoding='utf8'))
        assert res['model_sha256']==sha(modelpath)
        assert res['manifest_sha256']==hashes[key]
        dst=OUT/f'evaluation/v55_{ds}.json'; dstwin=OUT/f'evaluation/v55_{ds}_WINDOWS.csv'
        src.replace(dst);win.replace(dstwin)
        res=json.loads(dst.read_text(encoding='utf8'))
        results.append({'dataset':ds,'result_sha256':sha(dst),'windows_sha256':sha(dstwin),'ppl':res['ppl'],'nll':res['nll'],'valid_tokens':res['scored_tokens'],'input_tokens':res['input_tokens'],'windows':res['windows'],'manifest_sha256':res['manifest_sha256'],'model_sha256':res['model_sha256']})
    rt.csvwrite('STAGE_TIMINGS_EVAL_CONTINUATION.csv',rt.TIMINGS)
    dump('QUALITY_EVALUATION_SOURCE.json',{
        'pass':len(results)==2,'new_PPL_evaluations':2,'reused_history':False,'P3_read':False,
        'evaluator_source_sha256':sha(OUT/'runtime_v54.py'),'manifest_hashes':hashes,'results':results,
        'gpu_name':torch.cuda.get_device_name(),'torch':torch.__version__,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()
    })
    dump('EVALUATION_CONTINUATION.json',{'status':'PASS','model_sha256':sha(modelpath),'datasets_completed':[r['dataset'] for r in results],'results':results,'finished_unix':time.time()})
    del model
except BaseException as exc:
    with (OUT/'FAILED_ATTEMPTS.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps({'time_unix':time.time(),'error':repr(exc),'traceback':traceback.format_exc(),'segment':'evaluation-only continuation'},allow_nan=False)+'\n')
    dump('EVALUATION_CONTINUATION_FAILURE.json',{'error':repr(exc),'traceback':traceback.format_exc(),'finished_unix':time.time()})
    raise
finally:
    budget.finish()
