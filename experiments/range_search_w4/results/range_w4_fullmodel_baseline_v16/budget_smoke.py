"""Two widths, alpha=.9 only, B=3/32 timing. Not full-model calibration or quality evaluation."""
from pathlib import Path
import importlib.util, json, time, math
import torch
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[3]
def load_module(name,path):
    import sys
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
    if (OUT/'BUDGET_SMOKE.json').exists(): raise RuntimeError('already complete; do not rerun')
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter()
    core=load_module('v16_dynamic_core',ROOT/'experiments/range_search_w4/dynamic_core.py')
    old=OUT.parent/'range_w4_quantized_prefix_v15/cache'
    pristine=torch.load(old/'pristine_target_weights.pt',map_location='cpu',weights_only=False)
    rows=[]
    for proj in ['q_proj','down_proj']:
        entry=pristine['L2/'+proj]
        w=entry['weight_fp16'][:32].float().cuda()
        hinv=torch.load(old/f'hessian/L2_{proj}_QPREFIX_INPUT.pt',map_location='cpu',weights_only=False)['hinv'].cuda()
        whash=w.clone();hhash=hinv.clone();outputs={}
        for b in [3,32]:
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
            q=core.dynamic_gptq(w[:b],hinv,alpha=.9)
            torch.cuda.synchronize();seconds=time.perf_counter()-t
            outputs[b]=q.weight_fp32.half().clone()
            rows.append(dict(projection=proj,rows=b,input_dim=w.shape[1],alpha=.9,seconds=seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                finite=bool(torch.isfinite(q.weight_fp32).all()),candidate_slots=b))
        delta=(outputs[3].float()-outputs[32][:3].float())
        rows[-1]['batch_first3_half_exact']=bool(torch.equal(outputs[3],outputs[32][:3]))
        rows[-1]['batch_first3_half_max_abs']=float(delta.abs().max())
        rows[-1]['pristine_and_hinv_unchanged']=bool(torch.equal(w,whash) and torch.equal(hinv,hhash))
        # Direct projection versus mathematically equivalent Gram loss; tiny deterministic input only.
        gen=torch.Generator(device='cuda').manual_seed(16)
        x=torch.randn((64,w.shape[1]),device='cuda',generator=gen,dtype=torch.float32)
        e=outputs[32][:3].float()-w[:3]
        direct=(x@e.T).square().mean(0)
        gram=x.T@x/len(x)
        quadratic=((e@gram)*e).sum(1)
        rows[-1]['quadratic_loss_max_abs']=float((direct-quadratic).abs().max())
        rows[-1]['quadratic_loss_allclose']=bool(torch.allclose(direct,quadratic,rtol=1e-5,atol=1e-8))
        del w,hinv,q,whash,hhash,outputs,x,e,gram,delta,direct,quadratic
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    obj=dict(scope='budget smoke only, reused historical H, first32 pristine L2 q/down rows, alpha0.9; no grid search or PPL',
        gpu=torch.cuda.get_device_name(0),total_device_bytes=torch.cuda.get_device_properties(0).total_memory,
        cases=rows,candidate_slots=sum(r['candidate_slots'] for r in rows),wall_seconds=time.perf_counter()-started,
        timing_note='single synchronized cold call per geometry; not a steady-state benchmark or speedup measurement')
    (OUT/'BUDGET_SMOKE.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(obj,indent=2))
if __name__=='__main__': main()
