import hashlib,json,sys,time,threading,os,traceback
from pathlib import Path
import numpy as np,torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O));import run_v52 as r;import budget
B=O.parent;V51=B/'v51_m2_fast_integer_repair_fullmodel'
assert json.loads((O/'CPU_TESTS.json').read_text())['pass']
budget.start('v51 row122 independent FP64 regression and prefix/source validation')
r.GPU_START=time.time();rt=r.rt;rt.WALL_START=r.GPU_START;rt.elapsed=budget.elapsed;rt.deadline=r.deadline;r.PHASE='validation';rt.MODE='diagnostic';rt.LAYER=-1;rt.MODULE=''
def stop():
 budget.finish();os._exit(124)
watch=threading.Timer(budget.CAP-budget.elapsed(),stop);watch.daemon=True;watch.start()
try:
 diag=json.loads((V51/'diagnostics/INITIAL_QUAD_FAILURE.json').read_text())
 assert r.sha(r.V18/'M2/checkpoint/final/model.safetensors')==r.read('PROTOCOL.json')['initialization']['model_sha256']
 with rt.timer('validation_original_M2_model_load'):
  model=AutoModelForCausalLM.from_pretrained(str(r.V18/'M2/checkpoint/final'),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()
 ids=rt.zero_model_check(model);hidden=rt.embed(model);restored=[]
 src0=torch.load(O/'original_codes/layer_00.pt',map_location='cpu',weights_only=True)
 for g in r.GROUP_ORDER:
  rt.LAYER=0;rt.MODULE=g;cp=torch.load(O/f'formal/groups/layer_00_{g}.pt',map_location='cpu',weights_only=False);r.assert_group(cp,0,g,src0)
  X=rt.capture(model,0,hidden,g);xh=r.xhash(X);assert xh==cp['input_sha256']
  source=Path(cp['inherited_from']['path']);assert r.sha(source)==cp['inherited_from']['sha256']
  for n,s in cp['modules'].items():
   with torch.no_grad():model.model.layers[0].get_submodule(n).weight.copy_(s['weight'].cuda())
  restored.append(dict(group=g,input_sha256=xh,source_sha256=r.sha(source),pass_=True))
  del X,cp
 hidden=rt.advance_block(model,0,hidden);assert hashlib.sha256(hidden.tobytes()).hexdigest()==r.read('formal/PREFIX_STATUS_layer_00.json')['hidden_sha256']
 rt.MODE='diagnostic';rt.LAYER=1;rt.MODULE='qkv';X=rt.capture(model,1,hidden,'qkv');C=rt.build_C(X)
 assert r.xhash(X)==diag['input_sha256'] and r.tensor_hash(C)==diag['C_sha256']
 r.ORIG[1]=torch.load(O/'original_codes/layer_01.pt',map_location='cpu',weights_only=True);s=r.ORIG[1]['self_attn.k_proj']
 with safe_open(str(rt.FP/'model.safetensors'),framework='pt',device='cpu') as f:W=f.get_tensor('model.layers.1.self_attn.k_proj.weight').half()
 code=s['codes'].to(C.device).long();SC=s['scales'].to(C.device)[:,s['g_idx'].to(C.device).long()];Q=(SC*(code.float()-8)).half().float();E=Q.double()-W.to(C.device).double();p=E@C;z=torch.zeros(len(code),device=C.device,dtype=torch.long)
 st=dict(code=code,Q=Q,E=E,p=p,B=E.abs()@C.abs(),updates=z.clone(),evaluations=z.clone(),status=z.clone(),rejections=z.clone(),blocked=torch.zeros((*code.shape,2),device=C.device,dtype=torch.bool));quad=(E*p).sum(1);st['loss']=quad.clone();initial=rt.direct(X,E)
 assert r.tensor_hash(E)==diag['E_sha256'] and r.tensor_hash(p)==diag['p_sha256']
 failed=(~torch.isclose(initial,quad,rtol=2e-5,atol=1e-12)).nonzero().flatten().cpu().tolist();assert failed==[122]
 before={k:r.tensor_hash(v) for k,v in st.items()};ih=r.tensor_hash(initial);qh=r.tensor_hash(quad);xh=r.xhash(X);ch=r.tensor_hash(C);wh=r.tensor_hash(W)
 r.fast.initialization_review.SOURCE_CHECK=lambda:r.initial_source_check(s['codes'],s['scales'],s['g_idx'],W,X,C,'self_attn.k_proj')
 info=r.fast.initialization_review.review(st,SC,W,X,C,initial,quad,'self_attn.k_proj')
 assert info['fallback'] and info['original_failed_rows']==1
 assert {k:r.tensor_hash(v) for k,v in st.items()}==before
 assert ih==r.tensor_hash(initial) and qh==r.tensor_hash(quad) and xh==r.xhash(X) and ch==r.tensor_hash(C) and wh==r.tensor_hash(W)
 result=dict(**{'pass':True},original_check_still_fails_on_row122=True,original_failed_rows=failed,FP64_check_passes=True,all_initial_state_bytes_unchanged=True,X_C_W_unchanged=True,original_threshold_input_unchanged=True,groups_replayed_without_repair=restored,v51_failure_X_C_E_p_hashes_reproduced=True,review=info,CPU_tests_sha256=r.sha(O/'CPU_TESTS.json'),protocol_sha256=r.sha(O/'PROTOCOL.json'),core_sha256=r.sha(O/'fast_accept_v52.py'),review_sha256=r.sha(O/'initialization_review.py'),elapsed_cumulative_seconds=budget.elapsed())
 r.dump('NUMERICAL_VALIDATION.json',result);print('V52_ROW122_REGRESSION_PASS',json.dumps(result),flush=True)
 del model;torch.cuda.empty_cache()
except BaseException as ex:
 r.dump('NUMERICAL_VALIDATION.json',dict(**{'pass':False},error=repr(ex),traceback=traceback.format_exc(),elapsed_cumulative_seconds=budget.elapsed()));raise
finally:
 watch.cancel();budget.finish()
