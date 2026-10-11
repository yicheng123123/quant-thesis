"""Bounded diagnosis of a material check failure. No repair/accept/commit/eval."""
import hashlib,json,os,threading,time
import numpy as np
import torch
from safetensors import safe_open
import run_v42 as rt
OUT=rt.OUT
def main():
 rt.WALL_START=json.loads((OUT/'GPU_START.json').read_text())['unix'];rt.START=time.perf_counter();rt.MODE='diagnostic'
 def deadline(reserve=0):
  if rt.elapsed()>10800-reserve:raise TimeoutError('original v44 global180min budget exhausted')
 rt.deadline=deadline;deadline();threading.Timer(min(240,10800-rt.elapsed()),lambda:os._exit(124)).start()
 meta=json.loads((OUT/'ACTIVE_GROUP_META.json').read_text());model=rt.model_load();hidden=rt.embed(model)
 for li in range(4):
  rt.LAYER=li
  for group in rt.GROUPS:
   p=OUT/f'formal/groups/layer_{li:02d}_{group}.pt'
   if p.exists():
    ck=torch.load(p,map_location='cpu',weights_only=False)
    with torch.no_grad():
     for name,s in ck['modules'].items():model.model.layers[li].get_submodule(name).weight.copy_(s['weight'].cuda())
  if li<3:hidden=rt.advance_block(model,li,hidden)
 rt.LAYER=3;rt.MODULE='mlp.down_proj';X=rt.capture(model,3,hidden,'down');xh=hashlib.sha256(X.tobytes()).hexdigest();assert xh==meta['input_sha256']
 C=rt.build_C(X);ch=rt.th(C.cpu());assert ch==meta['C_sha256']
 f=torch.load(OUT/'CRITICAL_FAILURE_formal_3_mlp_down_proj.pt',map_location='cpu',weights_only=False);st=f['state'];pr=f['proposal'];rr=pr['rows'];actual=f['direct_new_loss'][rr]-st['loss'][rr];pred=pr['pred'];tol=torch.maximum(st['loss'][rr]*2e-6,pred.abs()*1e-3);bad=((actual-pred).abs()>tol).nonzero().flatten();assert bad.numel()==1
 o=torch.load(OUT/'original_codes/layer_03.pt',map_location='cpu',weights_only=True)['mlp.down_proj']
 with safe_open(str(rt.FP/'model.safetensors'),framework='pt',device='cpu') as sf:W=sf.get_tensor('model.layers.3.mlp.down_proj.weight').half()
 SC=o['scales'][:,o['g_idx'].long()];q=(SC*(st['code'].float()-8)).half().float();qn=(SC*(pr['code'].float()-8)).half().float()
 assert torch.equal(q,st['Q']) and torch.equal(q.double()-W.double(),st['E']) and torch.equal(qn.double()-W.double(),pr['E'])
 assert bool(((st['code']>=0)&(st['code']<=15)).all() and ((pr['code']>=0)&(pr['code']<=15)).all())
 E=st['E'].cuda();newE=pr['E'].cuda();p=E@C;drift=float((p-st['p'].cuda()).abs().max())
 old32=rt.direct(X,E);new32=rt.direct(X,newE)
 assert torch.equal(old32.cpu(),st['loss']) and torch.equal(new32.cpu(),f['direct_new_loss'])
 details=[]
 with rt.timer('diagnostic_FP64_paired_output_only'):
  for idx in bad.tolist():
   row=int(rr[idx]);col=int(pr['col'][idx]);delta=pr['delta'][idx].cuda();fresh=2*delta*p[row,col]+delta.square()*C[col,col]
   old64=torch.zeros((),device='cuda',dtype=torch.float64);new64=old64.clone();paired=old64.clone()
   # Alternate precision is diagnosis only; never used to accept the failed move.
   for lo in range(0,len(X),4096):
    xx=torch.from_numpy(np.array(X[lo:lo+4096],copy=True)).cuda().double();a=xx@E[row];b=xx@newE[row]
    old64+=a.square().sum();new64+=b.square().sum();paired+=((b-a)*(b+a)).sum()
   old64/=len(X);new64/=len(X);paired/=len(X)
   details.append({'row':row,'coordinate':col,'offset':int(pr['off'][idx]),'actual_grid_delta':float(delta),'saved_prediction':float(pred[idx]),'fresh_prediction':float(fresh),'fresh_minus_saved_prediction':float(fresh)-float(pred[idx]),'saved_FP32_direct_delta':float(actual[idx]),'fresh_original_fullbatch_FP32_delta':float(new32[row]-old32[row]),'FP64_direct_delta_diagnostic_only':float(new64-old64),'FP64_paired_delta_diagnostic_only':float(paired),'FP64_paired_minus_prediction':float(paired-fresh),'original_tolerance':float(tol[idx]),'absolute_discrepancy':float((actual[idx]-pred[idx]).abs()),'error_over_original_tolerance':float((actual[idx]-pred[idx]).abs()/tol[idx]),'old_loss':float(st['loss'][row]),'direct_relative_delta':float(actual[idx]/st['loss'][row])})
 rt.dump('NUMERICAL_DIAGNOSIS.json',{'pass_diagnosis':True,'rows':details,'full_direct_output_rows':896,'input_dim':4864,'N':262144,'chunk':4096,'input_sha256':xh,'C_sha256':ch,'exact_actual_grid_E_and_proposal_verified':True,'code_grid_legal':True,'all_state_values_finite':all(bool(torch.isfinite(v).all()) for v in st.values() if torch.is_tensor(v)),'p_max_absolute_drift':drift,'original_fullbatch_FP32_old_new_loss_bitwise_reproduced':True,'no_commit_or_accept_or_PPL':True,'failed_group_checkpoint_exists':(OUT/'formal/groups/layer_03_down.pt').exists(),'interpretation':'FP32 full-batch scoring discrepancy exceeds frozen tolerance even though actual loss descends. The allowed near-zero tie/rise rejection branch does not cover this case. Alternate FP64 output is diagnostic only; no tolerance change or recovery authorized.','original_gpu_budget_global_wall_seconds_including_diagnosis':rt.elapsed()})
 print(json.dumps(details,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 finally:os._exit(0)
