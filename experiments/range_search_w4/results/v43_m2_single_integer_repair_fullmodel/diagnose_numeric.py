"""Bounded numerical diagnosis only: no repaired state is committed or selected."""
import gc,json,os,threading,time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import run_v42 as rt
OUT=Path(__file__).resolve().parent
def main():
 start=json.loads((OUT/'GPU_START.json').read_text());rt.WALL_START=start['unix'];rt.START=time.perf_counter();rt.MODE='diagnostic'
 def deadline(reserve=0):
  if rt.elapsed()>10800-reserve:raise TimeoutError('original v43 global10800s deadline')
 rt.deadline=deadline;deadline();threading.Timer(min(180,10800-rt.elapsed()),lambda:os._exit(124)).start()
 model=rt.model_load();hidden=rt.embed(model)
 # Reconstruct the exact committed R prefix; no candidate search or repair.
 for li in range(4):
  rt.LAYER=li
  for group in rt.GROUPS:
   p=OUT/f'formal/groups/layer_{li:02d}_{group}.pt'
   if p.exists():
    ck=torch.load(p,map_location='cpu',weights_only=False)
    with torch.no_grad():
     for name,s in ck['modules'].items():model.model.layers[li].get_submodule(name).weight.copy_(s['weight'].cuda())
  if li<3:hidden=rt.advance_block(model,li,hidden)
 rt.LAYER=3;rt.MODULE='mlp.up_proj';X=rt.capture(model,3,hidden,'up_gate');C=rt.build_C(X)
 failure=torch.load(OUT/'NUMERICAL_FAILURE_formal_3_mlp_up_proj.pt',map_location='cpu',weights_only=True)
 rows=failure['rows'];s=torch.load(OUT/'original_codes/layer_03.pt',map_location='cpu',weights_only=True)['mlp.up_proj']
 with safe_open(str(rt.FP/'model.safetensors'),framework='pt',device='cpu') as fp:W=fp.get_tensor('model.layers.3.mlp.up_proj.weight').half()[rows]
 code=failure['code'][rows].cuda();sc=s['scales'][rows].cuda()[:,s['g_idx'].long().cuda()];q=(sc*(code.float()-8)).half().float();E=q.double()-W.cuda().double();p=E@C;diag=C.diag()
 minus=(sc*(code.float()-9)).half().float()-q;plus=(sc*(code.float()-7)).half().float()-q;dv=torch.stack((minus,plus),-1).double()
 score=2*dv*p[:,:,None]+dv.square()*diag[None,:,None];valid=torch.stack((code>0,code<15),-1);score.masked_fill_(~valid,float('inf'));pred,k=score.flatten(1).min(1);col=k//2;off=k%2*2-1;delta=dv[torch.arange(3,device='cuda'),col,k%2];new=E.clone();new[torch.arange(3,device='cuda'),col]+=delta
 # Same exact X and grid/error values, alternative accumulation paths for diagnosis.
 with rt.timer('diagnostic_direct_FP32_FP64'):
  x32=torch.from_numpy(X.copy()).cuda().float();old_mv=[];new_mv=[]
  for j in range(3):
   old_mv.append(float((x32@E[j].float()).double().square().mean()));new_mv.append(float((x32@new[j].float()).double().square().mean()))
  old_mm=(x32@E.float().T).double().square().mean(0).cpu();new_mm=(x32@new.float().T).double().square().mean(0).cpu();del x32
  old64=torch.zeros(3,device='cuda',dtype=torch.float64);new64=old64.clone();paired64=old64.clone()
  for lo in range(0,len(X),4096):
   xx=torch.from_numpy(X[lo:lo+4096].copy()).cuda().double();yy=xx@E.T;zz=xx@new.T;old64+=yy.square().sum(0);new64+=zz.square().sum(0);paired64+=((zz-yy)*(zz+yy)).sum(0)
  old64/=len(X);new64/=len(X);paired64/=len(X)
 result=[]
 for j,row in enumerate(rows.tolist()):
  result.append({'row':row,'coordinate':int(col[j]),'offset':int(off[j]),'actual_FP16_grid_delta':float(delta[j]),'saved_predicted_delta':float(failure['pred'][j]),'fresh_predicted_delta':float(pred[j]),'fresh_minus_saved_prediction':float(pred[j])-float(failure['pred'][j]),'saved_FP32_batch_direct_delta':float(failure['actual'][j]),'saved_old_loss':float(failure['old_loss'][j]),'saved_new_loss':float(failure['new_loss'][j]),'saved_relative_rise':float(failure['actual'][j]/failure['old_loss'][j]),'FP32_single_matvec_delta':new_mv[j]-old_mv[j],'FP32_three_row_matmul_delta':float(new_mm[j]-old_mm[j]),'FP64_direct_delta':float(new64[j]-old64[j]),'FP64_paired_output_delta':float(paired64[j]),'FP64_direct_minus_prediction':float(new64[j]-old64[j]-pred[j]),'old_FP64_loss':float(old64[j]),'new_FP64_loss':float(new64[j])})
 rt.dump('NUMERICAL_DIAGNOSIS.json',{'rows':result,'input_sha256':__import__('hashlib').sha256(X.tobytes()).hexdigest(),'C_sha256':rt.th(C.cpu()),'mode':'diagnosis only; no commit, no accept-rule change, no full model or PPL','global_elapsed_seconds_including_diagnosis':rt.elapsed(),'interpretation':'Compare exact quadratic/FP64 and frozen FP32 batched output scoring. Alternate reductions are diagnostic only; they do not authorize accepting a frozen-scorer increase.'})
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 finally:os._exit(0)
