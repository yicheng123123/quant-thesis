"""First GPU work is raw-X diagnosis and saved-failure branch validation only."""
import gc,hashlib,json,os,threading,time,traceback
import torch
from safetensors import safe_open
import run_v42 as rt
from precision_check import verify
OUT=rt.OUT;V44=OUT.parent/'v44_m2_single_integer_repair_fullmodel';V43=OUT.parent/'v43_m2_single_integer_repair_fullmodel'
def deadline(reserve=0):
 if rt.elapsed()>10800-reserve:raise TimeoutError('original v45 180min clock')
def main():
 assert not (OUT/'GPU_START.json').exists()
 rt.WALL_START=time.time();rt.START=time.perf_counter();rt.MODE='preflight';rt.deadline=deadline
 rt.dump('GPU_START.json',{'unix':rt.WALL_START,'pid':os.getpid(),'hard_cap_seconds':10800,'protocol_sha256':rt.sha(OUT/'PROTOCOL.json'),'precision_sha256':rt.sha(OUT/'precision_check.py'),'legacy_runtime_sha256':rt.sha(OUT/'run_v42.py')})
 threading.Timer(10800,lambda:os._exit(124)).start()
 with rt.timer('warmup'):z=torch.zeros((64,64),device='cuda');z@z
 model=rt.model_load();hidden=rt.embed(model);changed={}
 for li in range(4):
  rt.LAYER=li
  for group in rt.GROUPS:
   cp=OUT/f'formal/groups/layer_{li:02d}_{group}.pt'
   if cp.exists():
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    with torch.no_grad():
     for name,s in ck['modules'].items():model.model.layers[li].get_submodule(name).weight.copy_(s['weight'].cuda());changed[f'model.layers.{li}.{name}.weight']=s['weight']
  if li<3:hidden=rt.advance_block(model,li,hidden)
 with rt.timer('parameter_integrity'):
  keys=0
  with safe_open(str(rt.V18/'M2/checkpoint/final/model.safetensors'),framework='pt',device='cpu') as sf:
   for k in sf.keys():
    expected=changed[k] if k in changed else sf.get_tensor(k);assert torch.equal(model.get_parameter(k).detach().cpu().view(torch.int16),expected.view(torch.int16)),k;keys+=1
  rt.dump('validation/PARAMETER_INTEGRITY.json',{'pass':True,'tensor_keys':keys,'committed_keys':len(changed),'other_parameters_exact_M2':True,'failed_down_parameters_exact_original_M2':True})
 rt.LAYER=3;rt.MODULE='v44_down_full_state';X=rt.capture(model,3,hidden,'down');xh=hashlib.sha256(X.tobytes()).hexdigest();assert xh==rt.P['recovery_boundary_input_sha256']
 C=rt.build_C(X);ch=rt.th(C.cpu());assert ch==rt.P['recovery_boundary_C_sha256']
 f=torch.load(V44/'CRITICAL_FAILURE_formal_3_mlp_down_proj.pt',map_location='cpu',weights_only=False);st=f['state'];pr=f['proposal']
 o=torch.load(OUT/'original_codes/layer_03.pt',map_location='cpu',weights_only=True)['mlp.down_proj']
 with safe_open(str(rt.FP/'model.safetensors'),framework='pt',device='cpu') as sf:rawW=sf.get_tensor('model.layers.3.mlp.down_proj.weight');W=rawW.half()
 SC=o['scales'][:,o['g_idx'].long()];q=(SC*(st['code'].float()-8)).half().float();qn=(SC*(pr['code'].float()-8)).half().float()
 assert torch.equal(q,st['Q']) and torch.equal(q.double()-W.double(),st['E']) and torch.equal(qn.double()-W.double(),pr['E'])
 rr=pr['rows'];D=torch.zeros_like(st['E'][rr]);D[torch.arange(len(rr)),pr['col']]=pr['delta'];assert torch.equal(D,pr['E'][rr]-st['E'][rr])
 E=st['E'].cuda();newE=pr['E'].cuda();old=rt.direct(X,E);new=rt.direct(X,newE);assert torch.equal(old.cpu(),st['loss']) and torch.equal(new.cpu(),f['direct_new_loss'])
 with rt.timer('independent_FP64_v44_fullbatch'):
  check=verify(X,E[rr],D.cuda(),C,st['p'][rr].cuda(),pr['pred'].cuda())
 actual=(new-old)[rr].cpu();pred=pr['pred'];tol=torch.maximum(st['loss'][rr]*2e-6,pred.abs()*1e-3);bad=((actual-pred).abs()>tol).nonzero().flatten()
 result=[]
 for j in range(len(rr)):
  result.append({'row':int(rr[j]),'coordinate':int(pr['col'][j]),'offset':int(pr['off'][j]),'actual_delta_weight':float(pr['delta'][j]),'predict64':float(check['predict64'][j]),'direct64_from_raw_X':float(check['direct64'][j]),'FP32_loss_difference':float(actual[j]),'FP32_difference_minus_predict64':float(actual[j]-pred[j]),'old_cross_precision_check_pass':bool((actual[j]-pred[j]).abs()<=tol[j]),'same_precision_absolute_error':float(check['absolute_error'][j]),'same_precision_allowed_error':float(check['allowed_error'][j]),'same_precision_check_pass':bool(check['absolute_error'][j]<=check['allowed_error'][j]),'FP64_increment_descends':bool(check['direct64'][j]<0)})
 rt.csvwrite('validation/V44_FULL_FAILURE_INCREMENTS.csv',result)
 assert bad.tolist()==[211] and bool((check['direct64'][bad]<0).all())
 rt.dump('validation/V44_DIAGNOSIS.json',{'pass':True,'checked_candidates':len(rr),'original_failed_rows':rr[bad].tolist(),'N':len(X),'input_shape':list(X.shape),'input_sha256':xh,'C_sha256':ch,'e_actual_grid_and_pristine_W_exact':True,'delta_exact_proposal_minus_current_E':True,'FP32_fullbatch_reproduced_bitwise':True,'independent_FP64_expression':'a=rawX@e;d=rawX@dense_actual_delta;mean(2*a*d+d*d); no C/p dependency in reference','all_same_precision_checks_pass':True,'cache_max_error':check['cache_max_absolute_error'],'cache_max_error_over_tolerance':check['cache_max_error_over_tolerance'],'dtype':{'stored_X':str(X.dtype),'scoring_X':'float32','verification_X':'float64','current_code':str(st['code'].dtype),'grid_scales':str(o['scales'].dtype),'zero':str(o['zeros'].dtype),'g_idx':str(o['g_idx'].dtype),'pristine_checkpoint_W':str(rawW.dtype),'pristine_target_W':str(W.dtype),'q_dequant_storage':'float16','q_analysis':str(q.dtype),'e':str(st['E'].dtype),'C':str(C.dtype),'cached_p':str(st['p'].dtype),'actual_delta':str(D.dtype),'verification_outputs_a_d':'float64','FP32_outputs':'float32','loss_SSE_accumulation':'float64'},'failed_rows':[result[j] for j in bad.tolist()],'global_elapsed_seconds':rt.elapsed()})
 del X,C,f,st,pr,E,newE,D,check,W,rawW;gc.collect();torch.cuda.empty_cache()
 # Original v43 failure: same up/gate input is independent of its own weights.
 rt.MODULE='v43_up_failure';X=rt.capture(model,3,hidden,'up_gate');xh=hashlib.sha256(X.tobytes()).hexdigest();origdiag=json.loads((V43/'NUMERICAL_DIAGNOSIS.json').read_text());assert xh==origdiag['input_sha256'];C=rt.build_C(X);assert rt.th(C.cpu())==origdiag['C_sha256']
 failure=torch.load(V43/'NUMERICAL_FAILURE_formal_3_mlp_up_proj.pt',map_location='cpu',weights_only=True);codes=failure['code'];o=torch.load(OUT/'original_codes/layer_03.pt',map_location='cpu',weights_only=True)['mlp.up_proj']
 with safe_open(str(rt.FP/'model.safetensors'),framework='pt',device='cpu') as sf:W=sf.get_tensor('model.layers.3.mlp.up_proj.weight').half()
 SC=o['scales'].cuda()[:,o['g_idx'].long().cuda()];code=codes.cuda();Q=(SC*(code.float()-8)).half().float();E=Q.double()-W.cuda().double();p=E@C
 predrows=failure['rows'].cuda();col=torch.full_like(predrows,490);off=torch.tensor([1,-1,-1],device='cuda');dv=(SC[predrows,col]*(code[predrows,col].float()+off-8)).half().float()-Q[predrows,col];newE=E.clone();newE[predrows,col]+=dv.double()
 old=rt.direct(X,E);new=rt.direct(X,newE);actual=(new-old)[predrows].cpu();assert torch.equal(actual,failure['actual']) and bool((actual>=0).all())
 rt.dump('validation/V43_REJECTIONS.json',{'pass':True,'rows':failure['rows'].tolist(),'FP32_full_output_rows':4864,'original_direct_rises_reproduced':actual.tolist(),'no_FP64_override_can_accept_them':True,'input_sha256':xh,'global_elapsed_seconds':rt.elapsed()})
 rt.dump('PREFLIGHT_GATE.json',{'pass':True,'same_precision_v44_full_failure_pass':True,'v43_FP32_rises_preserved':True,'historical_stop_states_not_changed':True,'global_elapsed_seconds':rt.elapsed()});print('PREFLIGHT_PASS',rt.elapsed(),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as ex:traceback.print_exc();rt.dump('RUN_STATUS.json',{'status':'PREFLIGHT_FAILED','category':'D','error':repr(ex),'elapsed_seconds':rt.elapsed(),'complete_model':False,'PPL_run':False})
 finally:os._exit(0)
