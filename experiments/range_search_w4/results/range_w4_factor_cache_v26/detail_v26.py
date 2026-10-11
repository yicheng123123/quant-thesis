"""Finish frozen v26 stage accounting; no factor-cache implementation or new search."""
import json,statistics,time
import numpy as np
import torch
from pathlib import Path
import run_v26 as r
from gptqmodel.quantization.gptq import GPTQ
O=Path(__file__).resolve().parent

def stage(label,fn):
 torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();t=time.perf_counter();v=fn();b.record();torch.cuda.synchronize()
 return v,dict(wall_seconds=time.perf_counter()-t,gpu_stream_seconds=a.elapsed_time(b)/1000)

def detailed_score(x,w,q):
 records=[];old_array=np.array;old_to=torch.Tensor.to;old_matmul=torch.matmul
 def array(*a,**kw):
  t=time.perf_counter();v=old_array(*a,**kw)
  if len(a) and isinstance(a[0],np.memmap):records.append(('host_memmap_copy',time.perf_counter()-t,None))
  return v
 def to(tensor,*a,**kw):
  kind='H2D_transfer' if tensor.device.type=='cpu' and len(a)>0 and str(a[0]).startswith('cuda') else 'other_to'
  e=torch.cuda.Event(enable_timing=True) if kind=='H2D_transfer' else None;f=torch.cuda.Event(enable_timing=True) if e else None
  if e:e.record()
  st=time.perf_counter();v=old_to(tensor,*a,**kw);wall=time.perf_counter()-st
  if e:f.record()
  if kind=='H2D_transfer':records.append((kind,wall,(e,f)))
  return v
 def matmul(*a,**kw):
  e=torch.cuda.Event(enable_timing=True);f=torch.cuda.Event(enable_timing=True);e.record();st=time.perf_counter();v=old_matmul(*a,**kw);wall=time.perf_counter()-st;f.record();records.append(('full_row_matmul',wall,(e,f)));return v
 np.array=array;torch.Tensor.to=to;torch.matmul=matmul
 try:loss,cost=r.measure_score(x,w,q)
 finally:np.array=old_array;torch.Tensor.to=old_to;torch.matmul=old_matmul
 stats={}
 for k,c,ev in records:
  s=stats.setdefault(k,dict(calls=0,cpu_dispatch_envelope_seconds=0,gpu_stream_seconds=0));s['calls']+=1;s['cpu_dispatch_envelope_seconds']+=c
  if ev:s['gpu_stream_seconds']+=ev[0].elapsed_time(ev[1])/1000
 return loss,cost,stats

def main():
 r.verify_sources();decision=json.loads((O/'PROFILE_DECISION.json').read_text('utf-8'));assert not decision['implement_full_factor_cache']
 if (O/'PROFILE_DETAIL.json').exists():print('DETAIL_REUSED');return
 budget=r.Budget();model,w=r.load_model_and_weights(budget);x=r.capture_inputs(model,budget)
 result={}
 for name in r.WNAMES:
  budget.check('detail_'+name,55)
  pristine=w[name];mod=model.model.layers[0].mlp.get_submodule(name)
  H,ns,hcost=r.rp.build_hessian(mod,x[name],[],f'v26 detail L0 {name}');assert ns==128
  hhash=r.thash(H);whash=r.thash(pristine)
  Hwork,clone=stage('H clone',lambda:H.clone())
  dead,deadcalc=stage('dead-mask from original H',lambda:torch.diag(Hwork)==0)
  _,deadfill=stage('dead column fill',lambda:Hwork.__setitem__((dead,dead),1))
  damp,dampcalc=stage('damp calculation',lambda:.1*torch.mean(torch.diag(Hwork)))
  diag=torch.arange(H.shape[0],device=r.D)
  _,dampfill=stage('damped diagonal write',lambda:Hwork.__setitem__((diag,diag),Hwork[diag,diag]+damp))
  lower,lowercost=stage('lower cholesky',lambda:torch.linalg.cholesky(Hwork))
  middle,invcost=stage('native inverse',lambda:GPTQ.block_cholesky_inverse(None,lower,block_size=H.shape[0]))
  upper,uppercost=stage('upper cholesky',lambda:torch.linalg.cholesky(middle,upper=True))
  assert torch.isfinite(upper).all().item()
  assert r.thash(H)==hhash and r.thash(pristine)==whash
  prepared=r.om.prepare_native_state(H,nsamples=128)
  assert torch.equal(prepared.damped_hessian,Hwork) and torch.equal(prepared.cholesky,lower) and torch.equal(prepared.inverse_before_final_cholesky,middle)
  q,candcost=r.timed(lambda:r.om.optimized_dynamic_gptq(pristine,prepared,alpha=1.0,check_cholesky=True))
  native,nativecost=r.timed(lambda:r.native_dynamic_gptq(pristine,H,nsamples=128,alpha=1.0))
  parity={f:torch.equal(getattr(q,f),getattr(native,f)) for f in ['weight_fp16','scales','zeros','g_idx']}
  if not all(parity.values()):raise RuntimeError('alpha1 native parity failed '+name+' '+str(parity))
  losses,scorecost,split=detailed_score(x[name],pristine,q)
  io={}
  for path in [r.ACT/f'L0_{name}_128x2048.f16']:
   _,sec=r.timed(lambda:r.sha(path));io['activation_file_read_hash_seconds']=sec
  result[name]=dict(shape=list(pristine.shape),H_sha256=hhash,weight_sha256=whash,dead_columns=int(dead.sum()),H_build_seconds=hcost['seconds'],
    H_and_factor_stages=dict(H_clone=clone,dead_mask=deadcalc,dead_column_fill=deadfill,damping_calculation=dampcalc,damping_diagonal_write=dampfill,lower=lowercost,inverse=invcost,upper=uppercost),
    alpha1_original_candidate_seconds=candcost,alpha1_native_candidate_seconds=nativecost,alpha1_native_parity=parity,
    scoring=scorecost,scoring_components=split,score_rows=int(losses.numel()),score_finite=bool(torch.isfinite(losses).all()),io=io,
    H_unchanged=r.thash(H)==hhash,W_unchanged=r.thash(pristine)==whash,
    warning='Standalone faithful factor-operation decomposition; not additive to nested candidate times. Scoring transfer GPU events include dispatch gaps.')
  r.write('PROFILE_DETAIL.json',dict(status='IN_PROGRESS',targets=result,gpu_wall_seconds=budget.used()))
  r.event('detail_complete',projection=name,gpu_wall_seconds=budget.used())
 budget.save('DETAIL_COMPLETE');r.write('PROFILE_DETAIL.json',dict(status='COMPLETE',targets=result,gpu_wall_seconds=budget.used()))
 print(json.dumps(dict(status='COMPLETE',gpu_wall_seconds=budget.used(),alpha1_native={n:v['alpha1_native_parity'] for n,v in result.items()},stage_counts={n:v['scoring_components'] for n,v in result.items()}),indent=2))
if __name__=='__main__':main()
