from pathlib import Path
import hashlib, importlib.util, json, sys, inspect, copy
import torch
from safetensors import safe_open
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];V16=OUT.parent/'range_w4_fullmodel_baseline_v16'
REV='7ae557604adf67be50417f59c2c2f167def9a775'
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def meta(x):return dict(dtype=str(x.dtype),device=str(x.device),shape=list(x.shape),stride=list(x.stride()),contiguous=x.is_contiguous(),hash=sha(x))
def compare(a,b):
    a=a.detach().cpu();b=b.detach().cpu();diff=(a.double()-b.double()).abs();idx=(a!=b).nonzero()
    return dict(exact=torch.equal(a,b),max_abs_diff=float(diff.max()) if diff.numel() else 0,mean_abs_diff=float(diff.mean()) if diff.numel() else 0,num_diff=int((a!=b).sum()),first_index=idx[0].tolist() if len(idx) else None,native=meta(a),custom=meta(b))
def cfg():return QuantizeConfig(bits=4,group_size=128,sym=True,desc_act=False,static_groups=False,true_sequential=True,damp_percent=.1,damp_auto_increment=0,mse=0)
def prepare():
    torch.manual_seed(20261005);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    path=ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/REV/'model.safetensors'
    m=json.loads((V16/'CALIBRATION_MANIFEST.json').read_text(encoding='utf-8'))
    with safe_open(str(path),framework='pt',device='cpu') as f:
        emb=f.get_tensor('model.embed_tokens.weight').half().to('cuda')
        w=f.get_tensor('model.layers.0.self_attn.k_proj.weight').half().to('cuda')
        b=f.get_tensor('model.layers.0.self_attn.k_proj.bias').half().to('cuda')
    ids=[torch.tensor(s['token_ids'][:64],dtype=torch.long,device='cuda')[None] for s in m['samples'][:2]]
    with torch.inference_mode():docs=[torch.nn.functional.embedding(i,emb) for i in ids]
    del emb
    layer=torch.nn.Linear(896,128,bias=True,device='cuda',dtype=torch.float16)
    with torch.no_grad():layer.weight.copy_(w);layer.bias.copy_(b)
    g=GPTQ(layer,qcfg=cfg());g.quantizer.configure(perchannel=True,bits=4,sym=True)
    for d in docs:g.process_batch(d)
    return layer,docs,g.H.clone()
def make_native(layer,H):
    g=GPTQ(copy.deepcopy(layer),qcfg=cfg());g.quantizer.configure(perchannel=True,bits=4,sym=True)
    g.H=H.clone();g.nsamples=2
    return g
def capture_native(g):
    fn=inspect.unwrap(GPTQ.quantize);lines=Path(fn.__code__.co_filename).read_text().splitlines();captured={}
    def tracer(frame,event,arg):
        if frame.f_code is fn.__code__ and event=='line':
            line=lines[frame.f_lineno-1].strip();loc=frame.f_locals
            if line=='diag = torch.arange(self.columns, device=self.device)':captured['damp']=loc['damp'].clone()
            elif line=='with lock:' and 'damped_H' not in captured:captured['damped_H']=loc['H'].clone()
            elif line=='H = self.block_cholesky_inverse(H, block_size=self.columns)':captured['chol']=loc['H'].clone()
            elif line=='H = torch.linalg.cholesky(H, upper=True)':captured['inverse']=loc['H'].clone()
            elif line=='break' and 'Hinv' in loc:captured['Hinv']=loc['Hinv'].clone()
        return tracer
    sys.settrace(tracer)
    try:result=g.quantize(blocksize=128)
    finally:sys.settrace(None)
    return result,captured
def main():
    layer,docs,H=prepare();native,n=capture_native(make_native(layer,H))
    manual=H.clone();mean_view=manual.diagonal().mean();mean_diag=torch.mean(torch.diag(manual));damp=.1*mean_view
    manual.diagonal().add_(damp);chol=torch.linalg.cholesky(manual)
    inv=make_native(layer,H).block_cholesky_inverse(chol,block_size=896);hinv=torch.linalg.cholesky(inv,upper=True)
    c={'damp':damp,'damped_H':manual,'chol':chol,'inverse':inv,'Hinv':hinv}
    comparisons={k:compare(n[k],c[k]) for k in n}
    report=dict(W=meta(layer.weight),H=meta(H),input_docs=[meta(d) for d in docs],
        mean_diag_contiguous=float(mean_diag),mean_diagonal_view=float(mean_view),
        means=compare(mean_diag,mean_view),stages=comparisons,
        v17_manual_hinv_hash_expected='26171c3a97741883b151b69cfd4563197e8047900f3b7dd37ef4248fdca682fe',
        v17_input_reproduced=sha(hinv)=='26171c3a97741883b151b69cfd4563197e8047900f3b7dd37ef4248fdca682fe')
    dump('UPSTREAM_PROBE.json',report)
    torch.save(dict(W=layer.weight.detach().cpu(),bias=layer.bias.detach().cpu(),docs=[d.cpu() for d in docs],H=H.cpu(),native=n,custom=c,native_output=[x.cpu() for x in native[:4]]),OUT/'repro_state.pt')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
