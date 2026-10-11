from pathlib import Path
import hashlib, json, time
from safetensors import safe_open
import torch

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parents[1]/'experiments/range_search_w4/results/range_w4_omniquant_resume_v21_4/M3/dense_export/model.safetensors'
NEW=ROOT/'M3/dense_export/model.safetensors'

def main():
    wall=time.monotonic(); cpu=time.process_time(); rows=[]
    if not NEW.is_file(): raise RuntimeError('complete dense export required')
    with safe_open(str(OLD),framework='pt',device='cpu') as a, safe_open(str(NEW),framework='pt',device='cpu') as b:
        ka=set(a.keys()); kb=set(b.keys())
        for key in sorted(ka|kb):
            if key not in ka or key not in kb:
                rows.append(dict(name=key,exact_equal=False,reason='key missing')); continue
            x=a.get_tensor(key); y=b.get_tensor(key)
            shape_equal=x.shape==y.shape; dtype_equal=x.dtype==y.dtype
            same=shape_equal and dtype_equal and torch.equal(x,y)
            rows.append(dict(name=key,shape=list(y.shape),dtype=str(y.dtype),shape_equal=shape_equal,dtype_equal=dtype_equal,
                exact_equal=same,max_abs_difference=float((x.float()-y.float()).abs().max()) if shape_equal and not same else 0,
                old_sha256=hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest(),
                new_sha256=hashlib.sha256(y.contiguous().numpy().tobytes()).hexdigest()))
    result=dict(old_checkpoint=str(OLD),new_checkpoint=str(NEW),scope='CPU named saved tensor comparison; no forward/PPL',
        all_saved_tensors_equal=all(r['exact_equal'] for r in rows),different_tensor_count=sum(not r['exact_equal'] for r in rows),
        tensors=rows,new_quality_measured=False,historical_M3_PPL_inherited=False,
        cpu_process_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
    (ROOT/'HISTORICAL_M3_TENSOR_COMPARISON.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='tensors'},indent=2))

if __name__=='__main__': main()
