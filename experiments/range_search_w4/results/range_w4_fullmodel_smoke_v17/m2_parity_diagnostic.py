"""Locate, but do not work around, the first M2 native parity mismatch."""
from pathlib import Path
import copy, hashlib, importlib.util, json, sys
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
V16=OUT.parent/'range_w4_fullmodel_baseline_v16'
REV='7ae557604adf67be50417f59c2c2f167def9a775'
class StopAtFirstBlock(RuntimeError): pass

def sha(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def main():
    spec=importlib.util.spec_from_file_location('v17_dynamic_core',ROOT/'experiments/range_search_w4/dynamic_core.py')
    core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
    from transformers import AutoModelForCausalLM
    from gptqmodel.models.definitions.qwen2 import Qwen2GPTQ
    from gptqmodel.quantization.gptq import GPTQ
    from gptqmodel.quantization import QuantizeConfig
    manifest=json.loads((V16/'CALIBRATION_MANIFEST.json').read_text(encoding='utf-8'))
    model_path=ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/REV
    model=AutoModelForCausalLM.from_pretrained(str(model_path),torch_dtype=torch.float16,low_cpu_mem_usage=True,
        local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
    docs=[]
    def hook(module,args,kwargs):
        docs.append(args[0].detach().clone());raise StopAtFirstBlock
    h=model.model.layers[0].register_forward_pre_hook(hook,with_kwargs=True)
    with torch.inference_mode():
        for s in manifest['samples'][:2]:
            try:model(input_ids=torch.tensor(s['token_ids'][:64],device='cuda')[None],use_cache=False)
            except StopAtFirstBlock:pass
    h.remove()
    target=model.model.layers[0].self_attn.k_proj
    pristine=target.weight.detach().clone()
    qcfg=QuantizeConfig(bits=4,group_size=128,sym=True,desc_act=False,static_groups=False,true_sequential=True,
                        damp_percent=.1,damp_auto_increment=0,mse=0)
    native=GPTQ(copy.deepcopy(target),qcfg=qcfg);native.quantizer.configure(perchannel=True,bits=4,sym=True)
    for doc in docs:native.process_batch(doc)
    hess=native.H.clone();hess.diagonal().add_(.1*hess.diagonal().mean())
    chol=torch.linalg.cholesky(hess)
    hinv=torch.linalg.cholesky(native.block_cholesky_inverse(chol,block_size=hess.shape[0]),upper=True)
    nq,ns,nz,ng,*_=native.quantize(blocksize=128)
    custom=core.dynamic_gptq(pristine.float(),hinv,alpha=1.0,group_size=128,block_size=128)
    qnative=nq.float();qcustom=custom.weight_fp32
    nsc=ns.float();csc=custom.scales.float();nze=nz.float();cze=custom.zeros.float()
    report={
        'status':'DIAGNOSTIC_ONLY','rows':int(pristine.shape[0]),'columns':int(pristine.shape[1]),
        'native_nsamples':native.nsamples,'docs_shapes':[list(d.shape) for d in docs],
        'native_weight_dtype':str(nq.dtype),'native_scale_shape':list(ns.shape),'custom_scale_shape':list(custom.scales.shape),
        'weight_exact_after_native_dtype_cast':bool(torch.equal(qcustom.to(nq.dtype),nq)),
        'weight_max_abs':float((qcustom-qnative).abs().max()),
        'weight_diff_count_fp32':int((qcustom!=qnative).sum()),
        'weight_diff_count_fp16':int((qcustom.half()!=nq.half()).sum()),
        'scale_exact':bool(torch.equal(csc,nsc)),'scale_max_abs':float((csc-nsc).abs().max()),
        'scale_diff_count':int((csc!=nsc).sum()),
        'zero_exact':bool(torch.equal(cze,nze)),'zero_max_abs':float((cze-nze).abs().max()),
        'zero_diff_count':int((cze!=nze).sum()),
        'g_idx_expected':bool(torch.equal(ng.cpu(),torch.arange(pristine.shape[1],dtype=ng.dtype)//128)),
        'first_scale_native':nsc[0].detach().cpu().tolist(),'first_scale_custom':csc[0].detach().cpu().tolist(),
        'first_zero_native':nze[0].detach().cpu().tolist(),'first_zero_custom':cze[0].detach().cpu().tolist(),
        'pristine_hash':sha(pristine),'hinv_hash':sha(hinv)}
    # Locate group/row/column for the first weight mismatch without saving large tensors.
    diff=(qcustom.half()!=nq.half()).nonzero()
    if len(diff):
        r,c=map(int,diff[0].tolist());report['first_weight_mismatch']={'row':r,'column':c,
            'native':float(nq[r,c]),'custom':float(qcustom.half()[r,c]),'group':c//128}
    report['local_output_mse_native_rows0_32']=float(((torch.cat([d[0] for d in docs]).float()@(nq[:32].float()-pristine[:32].float()).T).square().mean()).item())
    report['local_output_mse_custom_rows0_32']=float(((torch.cat([d[0] for d in docs]).float()@(qcustom[:32].half().float()-pristine[:32].float()).T).square().mean()).item())
    (OUT/'M2_PARITY_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
