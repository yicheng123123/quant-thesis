"""Trace group-entry drift and compare subset/full-row native parity only."""
from pathlib import Path
import copy, hashlib, importlib.util, json, sys
import torch
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[3]; V16=OUT.parent/'range_w4_fullmodel_baseline_v16'
REV='7ae557604adf67be50417f59c2c2f167def9a775'
class StopAtFirstBlock(RuntimeError): pass
def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def main():
    spec=importlib.util.spec_from_file_location('v17_dynamic_core',ROOT/'experiments/range_search_w4/dynamic_core.py')
    core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
    from transformers import AutoModelForCausalLM
    from gptqmodel.quantization.gptq import GPTQ
    from gptqmodel.quantization import QuantizeConfig
    manifest=json.loads((V16/'CALIBRATION_MANIFEST.json').read_text(encoding='utf-8'))
    mp=ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/REV
    model=AutoModelForCausalLM.from_pretrained(str(mp),torch_dtype=torch.float16,low_cpu_mem_usage=True,local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
    docs=[]
    def hook(module,args,kwargs):docs.append(args[0].detach().clone());raise StopAtFirstBlock
    handle=model.model.layers[0].register_forward_pre_hook(hook,with_kwargs=True)
    with torch.inference_mode():
        for s in manifest['samples'][:2]:
            try:model(input_ids=torch.tensor(s['token_ids'][:64],device='cuda')[None],use_cache=False)
            except StopAtFirstBlock:pass
    handle.remove(); target=model.model.layers[0].self_attn.k_proj; pristine=target.weight.detach().clone()
    cfg=QuantizeConfig(bits=4,group_size=128,sym=True,desc_act=False,static_groups=False,true_sequential=True,damp_percent=.1,damp_auto_increment=0,mse=0)
    native=GPTQ(copy.deepcopy(target),qcfg=cfg);native.quantizer.configure(perchannel=True,bits=4,sym=True)
    for d in docs:native.process_batch(d)
    H=native.H.clone();H.diagonal().add_(.1*H.diagonal().mean());chol=torch.linalg.cholesky(H)
    Hinv=torch.linalg.cholesky(native.block_cholesky_inverse(chol,block_size=H.shape[0]),upper=True)
    captured=[];orig_find=native.quantizer.find_params
    def capture_find(x,weight=False):
        r=orig_find(x,weight=weight)
        captured.append({'input':x.detach().clone(),'scale':native.quantizer.scale.detach().clone(),'zero':native.quantizer.zero.detach().clone()})
        return r
    native.quantizer.find_params=capture_find
    nq,ns,nz,ng,*_=native.quantize(blocksize=128)
    full=core.dynamic_gptq(pristine.float(),Hinv,alpha=1,group_size=128,block_size=128,trace=True)
    subset=core.dynamic_gptq(pristine[:32].float(),Hinv,alpha=1,group_size=128,block_size=128,trace=True)
    groups=[]
    for g in range(7):
        n=captured[g+1]['input'].detach().cpu(); c=full.group_inputs[g]
        diff=(n.float()-c.float()).abs();nz_count=int((n!=c).sum())
        groups.append({'group':g,'entry_shape':list(n.shape),'entry_max_abs':float(diff.max()),'entry_exact':bool(torch.equal(n,c)),
            'entry_diff_count':nz_count,'native_scale_max_abs':float((captured[g+1]['scale'].cpu()-full.scales[:,g:g+1].cpu()).abs().max()),
            'native_scale_exact':bool(torch.equal(captured[g+1]['scale'].cpu(),full.scales[:,g:g+1].cpu())),
            'subset_scale_max_abs':float((captured[g+1]['scale'][:32].cpu()-subset.scales[:,g:g+1].cpu()).abs().max()),
            'native_scale_first3':captured[g+1]['scale'][:3].flatten().cpu().tolist(),
            'custom_scale_first3':full.scales[:3,g].cpu().tolist()})
    full_half=full.weight_fp32.half(); subset_half=subset.weight_fp32.half()
    d=(full_half!=nq).nonzero(); first=None
    if len(d):
        r,c=map(int,d[0].tolist());first={'row':r,'col':c,'group':c//128,'native':float(nq[r,c]),'custom_full':float(full_half[r,c])}
    cols=[]
    for g in range(7):
        sl=slice(g*128,min((g+1)*128,pristine.shape[1]))
        cols.append({'group':g,'native_custom_full_fp16_exact':bool(torch.equal(nq[:,sl],full_half[:,sl])),
            'native_custom_subset_first32_fp16_exact':bool(torch.equal(nq[:32,sl],subset_half[:,sl])),
            'full_diff_count':int((nq[:,sl]!=full_half[:,sl]).sum()),'subset_diff_count':int((nq[:32,sl]!=subset_half[:,sl]).sum())})
    out={'status':'DIAGNOSTIC_ONLY','native_find_params_calls':len(captured),'groups':groups,'weight_by_group':cols,
         'full_row_weight_exact':bool(torch.equal(nq,full_half)),'subset_first32_weight_exact':bool(torch.equal(nq[:32],subset_half)),
         'first_full_row_mismatch':first,'native_hinv_sha256':sha(Hinv)}
    (OUT/'M2_PARITY_TRACE.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
