"""M1 native-grid source audit and unique exact integer-code recovery."""
import csv, hashlib, json, math, time
from pathlib import Path
import torch
from safetensors import safe_open
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[3]; BASE=OUT.parent
V18=BASE/'range_w4_fullmodel_v18'; V41=BASE/'v41_integer_pair_repair_pilot'
FP=ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
torch.set_num_threads(4)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def th(t): return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def dump(n,v):
 p=OUT/n; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def pack(c): return (c[:,0::2] | (c[:,1::2]<<4)).contiguous()
def unpack(p): return torch.stack((p&15,p>>4),dim=-1).flatten(1).contiguous()
def invert(s):
 q,sc,z,g=s['weight'],s['scales'],s['zeros'],s['g_idx']
 rows,d=q.shape; groups=d//128
 assert q.dtype==torch.float16 and sc.dtype==torch.float32 and z.dtype==torch.float32 and g.dtype==torch.int32
 assert d%128==0 and sc.shape==(rows,groups) and torch.equal(z,torch.full_like(z,8))
 assert torch.equal(g,torch.arange(d,dtype=torch.int32)//128)
 assert bool(torch.isfinite(q).all() and torch.isfinite(sc).all() and (sc>0).all())
 offsets=torch.arange(16,dtype=torch.float32)-8.0; codes=torch.empty((rows,d),dtype=torch.uint8); ambiguous=0
 for group in range(groups):
  lo=group*128; hi=lo+128
  grid=(sc[:,group,None]*offsets[None,:]).half().float().contiguous()
  vals=q[:,lo:hi].float().contiguous()
  idx=torch.searchsorted(grid,vals,right=False)
  in_range=idx<16; safe=idx.clamp(max=15)
  found=torch.gather(grid,1,safe)==vals
  next_idx=(safe+1).clamp(max=15)
  duplicate=found & (idx<15) & (torch.gather(grid,1,next_idx)==vals)
  ok=in_range & found & ~duplicate
  if not bool(ok.all()):
   ambiguous+=int((~ok).sum())
   raise AssertionError(f'grid inversion not unique/exact in group {group}: bad={int((~ok).sum())}')
  codes[:,lo:hi]=idx.to(torch.uint8)
 dq=(sc[:,g.long()]*(codes.float()-8)).half()
 assert torch.equal(dq.view(torch.int16),q.view(torch.int16))
 assert torch.equal(unpack(pack(codes)),codes)
 return codes,ambiguous
def main():
 t=time.perf_counter(); (OUT/'original_codes').mkdir(exist_ok=True)
 v18p=json.loads((V18/'PROTOCOL.json').read_text(encoding='utf8')); calp=V18/'CALIBRATION_MANIFEST.json'; calsha=sha(calp)
 assert calsha==v18p['source_protocol']['v16_calibration_manifest_sha256']
 cal=json.loads(calp.read_text(encoding='utf8')); assert len(cal['samples'])==128
 for x in cal['samples']:
  assert len(x['token_ids'])==2048 and hashlib.sha256(json.dumps(x['token_ids']).encode()).hexdigest()==x['token_sha256']
 runtime=json.loads((V18/'FORMAL_RUNTIME.json').read_text(encoding='utf8')); m1model=V18/'M1/checkpoint/final/model.safetensors'; m2model=V18/'M2/checkpoint/final/model.safetensors'
 assert sha(m1model)==runtime['m1']['final_model_sha256'] and sha(m2model)==runtime['m2']['final_model_sha256']
 rows=[]; srcs=[]; alpha_summary=[]; scales_diff=[]
 with safe_open(str(m1model),framework='pt',device='cpu') as sf:
  for li in range(24):
   p=V18/f'checkpoints/M1/layer_{li:02d}.pt'; payload=torch.load(p,map_location='cpu',weights_only=False)
   p2=V18/f'checkpoints/M2/layer_{li:02d}.pt'; payload2=torch.load(p2,map_location='cpu',weights_only=False)
   assert payload['metadata']['method']=='M1' and payload2['metadata']['method']=='M2'
   assert payload['metadata']['calibration_manifest_sha256']==calsha and payload2['metadata']['calibration_manifest_sha256']==calsha
   out={}
   for name,s in payload['modules'].items():
    h=hashlib.sha256()
    for k in ('weight','scales','zeros','g_idx','selected_alpha'): h.update(k.encode()); h.update(s[k].contiguous().numpy().tobytes())
    assert h.hexdigest()==s['state_sha256']
    assert torch.equal(sf.get_tensor(f'model.layers.{li}.{name}.weight').view(torch.int16),s['weight'].view(torch.int16))
    assert torch.equal(s['selected_alpha'],torch.ones_like(s['selected_alpha']))
    codes,ambiguous=invert(s); out[name]={k:s[k] for k in ('weight','scales','zeros','g_idx')}; out[name]['codes']=codes; out[name]['packed_u4']=pack(codes)
    s2=payload2['modules'][name]
    scales_diff.append({'layer':li,'module':name,'M1_M2_scales_bitwise_equal':torch.equal(s['scales'],s2['scales']),'M1_M2_zero_bitwise_equal':torch.equal(s['zeros'],s2['zeros']),'M1_M2_g_idx_bitwise_equal':torch.equal(s['g_idx'],s2['g_idx']),'M1_alpha_values':sorted(set(float(x) for x in s['selected_alpha'].flatten().tolist())),'M2_alpha_values':sorted(set(float(x) for x in s2['selected_alpha'].flatten().tolist()))})
    rows.append({'layer':li,'module':name,'rows':codes.shape[0],'input_dim':codes.shape[1],'coordinates':codes.numel(),'codes_sha256':th(codes),'weight_sha256':th(s['weight']),'scales_sha256':th(s['scales']),'zeros_sha256':th(s['zeros']),'g_idx_sha256':th(s['g_idx']),'source_state_sha256':s['state_sha256'],'exact_unique_mapping':True,'pack_roundtrip':True,'ambiguous_coordinates':ambiguous})
   dst=OUT/f'original_codes/layer_{li:02d}.pt'; tmp=dst.with_suffix('.pt.tmp'); torch.save(out,tmp); tmp.replace(dst)
   srcs.append({'layer':li,'M1_layer_state_path':str(p),'M1_layer_state_sha256':sha(p),'M1_model_path':str(m1model),'M1_model_sha256':sha(m1model),'derived_path':str(dst),'derived_sha256':sha(dst)})
   print('M1_GRID_RECOVERED',li,flush=True)
 assert len(rows)==168 and sum(r['rows'] for r in rows)==304128 and sum(r['coordinates'] for r in rows)>0
 with (OUT/'SOURCE_MODULES.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 with (OUT/'INITIALIZATION_GRID_COMPARISON.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(scales_diff[0])); w.writeheader(); w.writerows(scales_diff)
 baselines=[]; ev=V18/'evaluation/raw'
 for m,ckpt in [('M1',m1model),('M2',m2model)]:
  for ds,key in [('wikitext','v16_wikitext_manifest_sha256'),('c4_fixed_budget','v16_c4_manifest_sha256')]:
   p=ev/f'{m}_{ds}.json'; r=json.loads(p.read_text(encoding='utf8'))
   assert r['status']=='PASS' and r['method']==m and r['model_weight_files'][0]['sha256']==sha(ckpt)
   assert r['evaluation_manifest_sha256']==v18p['source_protocol'][key] and r['max_length']==2048 and r['stride']==512 and r['evaluated_tokens']==r['input_tokens']-1
   baselines.append({'arm':m,'dataset':ds,'path':str(p),'sha256':sha(p),'checkpoint_sha256':sha(ckpt),'ppl':r['perplexity'],'nll':r.get('weighted_mean_nll',r.get('exact_window_weighted_average_nll',r.get('average_nll',r.get('nll',r.get('mean_nll'))))),'valid_tokens':r['evaluated_tokens'],'windows':r['window_count'],'manifest_sha256':r['evaluation_manifest_sha256']})
 v55=OUT.parent/'v55_m2_decoder_fixed_integer_repair'
 for ds,file in [('wikitext','v55_wikitext.json'),('c4_fixed_budget','v55_c4_fixed_budget.json')]:
  p=v55/'evaluation'/file; r=json.loads(p.read_text(encoding='utf8'))
  assert r['model_sha256']=='0fa87105e36deac2ead4c871e6e0a802590418e6ccacd616929cdf91ecc8ec0c'
  baselines.append({'arm':'v55_M2_plus_repair','dataset':ds,'path':str(p),'sha256':sha(p),'checkpoint_sha256':r['model_sha256'],'ppl':r['ppl'],'nll':r['nll'],'valid_tokens':r.get('valid_tokens',r.get('scored_tokens',r.get('evaluated_tokens'))),'windows':r.get('windows',r.get('window_count')),'manifest_sha256':r['manifest_sha256']})
 dump('SOURCE_AUDIT.json',{'pass':True,'M1_model_sha256':sha(m1model),'M2_model_sha256':sha(m2model),'model_revision':v18p['model']['revision'],'calibration_manifest_sha256':calsha,'modules':168,'rows':304128,'native_M1_layer_sources':srcs,'derived_code_maps':168,'all_coordinate_codes_uniquely_recovered':True,'M1_vs_M2_grid_comparison':'INITIALIZATION_GRID_COMPARISON.csv','M1_M2_effective_config_json_sha_equal':sha(V18/'M1/checkpoint/final/config.json')==sha(V18/'M2/checkpoint/final/config.json'),'baseline_evaluations':baselines,'v55_quality_source_sha256':sha(v55/'QUALITY_EVALUATION_SOURCE.json'),'no_v46_repair_parameters_read':True,'no_P3_read':True,'seconds':time.perf_counter()-t})
 print('SOURCE_AUDIT_PASS',time.perf_counter()-t,flush=True)
if __name__=='__main__':main()
