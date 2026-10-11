"""CPU-only provenance and exact native-grid inversion. Historical inputs read only."""
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
V18=OUT.parent/'range_w4_fullmodel_v18'
V41=OUT.parent/'v41_integer_pair_repair_pilot'
FP=ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
torch.set_num_threads(4)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def th(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def pack(c):return (c[:,0::2]|(c[:,1::2]<<4)).contiguous()
def unpack(p):return torch.stack((p&15,p>>4),dim=-1).flatten(1).contiguous()
def invert(s):
    q=s['weight'];sc=s['scales'];z=s['zeros'];g=s['g_idx'].long()
    assert q.dtype==torch.float16 and sc.dtype==torch.float32
    assert torch.equal(z,torch.full_like(z,8)) and torch.equal(g,torch.arange(q.shape[1])//128)
    assert bool(torch.isfinite(sc).all() and (sc>0).all())
    codes=torch.zeros_like(q,dtype=torch.uint8);counts=torch.zeros_like(codes)
    for lo in range(0,q.shape[0],128):
        w=q[lo:lo+128].view(torch.int16);scol=sc[lo:lo+128][:,g]
        for k in range(16):
            match=(scol*(k-8)).half().view(torch.int16)==w
            counts[lo:lo+128]+=match.to(torch.uint8)
            codes[lo:lo+128][match]=k
    assert bool((counts==1).all()),f'nonunique grid mapping: {int((counts!=1).sum())}'
    assert torch.equal(unpack(pack(codes)),codes)
    dq=(sc[:,g]*(codes.float()-8)).half()
    assert torch.equal(dq.view(torch.int16),q.view(torch.int16))
    return codes
def main():
    start=time.perf_counter();(OUT/'original_codes').mkdir(exist_ok=True)
    protocol=json.loads((V18/'PROTOCOL.json').read_text());calpath=V18/'CALIBRATION_MANIFEST.json'
    assert sha(calpath)==protocol['source_protocol']['v16_calibration_manifest_sha256']
    cal=json.loads(calpath.read_text());assert len(cal['samples'])==128
    for s in cal['samples']:
        assert len(s['token_ids'])==2048
        assert hashlib.sha256(json.dumps(s['token_ids']).encode()).hexdigest()==s['token_sha256']
    model=V18/'M1/checkpoint/final/model.safetensors'
    assert sha(model)==json.loads((V18/'FORMAL_RUNTIME.json').read_text())['m1']['final_model_sha256']
    rows=[];sources=[]
    with safe_open(str(model),framework='pt',device='cpu') as sf:
      for li in range(24):
        p=V18/f'checkpoints/M1/layer_{li:02d}.pt';payload=torch.load(p,map_location='cpu',weights_only=False)
        assert payload['metadata']['calibration_manifest_sha256']==sha(calpath)
        out={}
        for name,s in payload['modules'].items():
            h=hashlib.sha256()
            for k in ('weight','scales','zeros','g_idx','selected_alpha'):h.update(k.encode());h.update(s[k].contiguous().numpy().tobytes())
            assert h.hexdigest()==s['state_sha256']
            assert torch.equal(sf.get_tensor(f'model.layers.{li}.{name}.weight').view(torch.int16),s['weight'].view(torch.int16))
            assert torch.equal(s['selected_alpha'],torch.ones_like(s['selected_alpha']))
            c=invert(s);out[name]={k:s[k] for k in ('weight','scales','zeros','g_idx')};out[name]['codes']=c;out[name]['packed_u4']=pack(c)
            rows.append({'layer':li,'module':name,'rows':c.shape[0],'input_dim':c.shape[1],'coordinates':c.numel(),'code_sha256':th(c),'weight_sha256':th(s['weight']),'scales_sha256':th(s['scales']),'zeros_sha256':th(s['zeros']),'g_idx_sha256':th(s['g_idx']),'exact_unique_mapping':True,'pack_roundtrip':True})
        dst=OUT/f'original_codes/layer_{li:02d}.pt';assert not dst.exists();torch.save(out,dst)
        sources.append({'source':str(p),'source_sha256':sha(p),'derived':str(dst),'derived_sha256':sha(dst)})
        print('audited_layer',li,flush=True)
    assert len(rows)==168 and sum(r['rows'] for r in rows)==304128
    baseline=[]
    for m in ('M0','M1','M2'):
      for ds,key in [('wikitext','v16_wikitext_manifest_sha256'),('c4_fixed_budget','v16_c4_manifest_sha256')]:
        p=V18/f'evaluation/raw/{m}_{ds}.json';r=json.loads(p.read_text())
        assert r['status']=='PASS' and r['evaluation_manifest_sha256']==protocol['source_protocol'][key]
        assert r['max_length']==2048 and r['stride']==512 and r['evaluated_tokens']==r['input_tokens']-1
        baseline.append({'method':m,'dataset':ds,'ppl':r['perplexity'],'evaluation_seconds':r['evaluation_wall_seconds'],'result_sha256':sha(p),'path':str(p),'weight_files':r['model_weight_files']})
    with (OUT/'SOURCE_MODULES.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump('SOURCE_AUDIT.json',{'pass':True,'CPU_seconds':time.perf_counter()-start,'modules':168,'rows':304128,'model_sha256':sha(model),'calibration_sha256':sha(calpath),'v41_protocol_sha256':sha(V41/'PROTOCOL.json'),'v41_implementation_sha256':sha(V41/'run_pilot.py'),'sources':sources,'baselines':baseline,'representation':'Original M2 deployment is dense FP16. Auxiliary row-major low-nibble-first unsigned4 packing is storage-only, never substituted as a new deployment kernel.','code_provenance':'Unique bitwise enumeration of all16 values from trusted saved FP32 scale, zero and g_idx; exact effective integer code, not original GPTQ forward trajectory; no approximate weight rounding.'})
    print('SOURCE_AUDIT_PASS',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
