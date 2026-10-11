"""v49 full M1 repair. Exact fast_accept is bound explicitly before rt.run_layer."""
import time
SCRIPT_START=time.time()
import csv, gc, hashlib, importlib.util, json, math, os, shutil, sys, threading, traceback
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM
ROOT=Path(r'E:\quant-thesis');BASE=ROOT/'experiments/range_search_w4/results';OUT=BASE/'v54_m2_decoder_fixed_integer_repair';V52=BASE/'v52_m2_fast_integer_repair_fp64_initial_review';V53=BASE/'v53_decoder_interface_propagation_audit';V46=BASE/'v46_m1_single_integer_repair_fullmodel';V45=BASE/'v45_m2_single_integer_repair_fullmodel';V49=BASE/'v49_fullmodel_fast_integer_repair';V48=BASE/'v48_fast_numerical_acceptance';V18=BASE/'range_w4_fullmodel_v18'
CAP=10800;FINISH_RESERVE=360;UNCERTAINTY_RESERVE=120;GROUP_ORDER=['qkv','o','up_gate','down']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def dump(n,x):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def elapsed():return budget.elapsed()
def deadline(reserve=0):
 if elapsed()>CAP-reserve:raise TimeoutError(f'v52 180-minute cumulative GPU budget exhausted: {elapsed():.1f}s; reserve={reserve:.1f}s')

def runtime_deadline(reserve=0):
 frozen=FINISH_RESERVE if globals().get('rt') is not None and rt.MODE in {'preflight','formal','propagation'} else 0
 deadline(max(float(reserve),frozen))

sys.path.insert(0,str(ROOT));sys.path.insert(0,str(V46));sys.path.insert(0,str(V48));sys.path.insert(0,str(OUT))
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
rt=loadmod('run_v42',OUT/'runtime_v54.py');rt.OUT=OUT;rt.WALL_START=None;rt.START=None;rt.deadline=runtime_deadline
assert Path(rt.FP).resolve()==(ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775').resolve(),'pristine FP target source mismatch'
old=loadmod('repair_control',V46/'repair_v45.py')
sys.path.insert(0,str(OUT))
import budget
fast=loadmod('v52_fast_accept',OUT/'fast_accept_v52.py')
V48_RULE_SHA=sha(V48/'ACCEPTANCE_RULES.md');FAST_SHA=sha(OUT/'fast_accept_v52.py');V48_RUN_SHA=sha(V48/'run_v48.py')
SOURCE_EXPECTED=read('SOURCE_AUDIT.json') if (OUT/'SOURCE_AUDIT.json').exists() else {}
ORIG={};RUN_STATS={};STAGE_MODULE_ROWS=[];PROPAGATION=[];VERIFIED_GROUPS=set();GPU_START=0.;EVAL_KEY_MAP={}

RUN_ID='v54-'+sha(OUT/'PROTOCOL.json')[:16]+'-'+str(os.getpid())
PHASE='formal'
SOURCE_RECORDS={}
CURRENT_SOURCE_KEY=None
V53_GROUP_HASHES={x['group']:x['native_reference_input_sha256'] for x in read(V53/'BLOCK0_GROUP_SOURCE_CHECKS.json')['groups']}
V53_BOUNDARY=read(V53/'VALID_RESUME_BOUNDARY.json')['expected_boundary']
ORIGINAL_CAPTURE=rt.capture
ORIGINAL_BUILD_C=rt.build_C

def xhash(X):
 if torch.is_tensor(X):return tensor_hash(X)
 return hashlib.sha256(X.tobytes()).hexdigest()

def capture_with_source(model,li,hidden,group):
 global CURRENT_SOURCE_KEY
 X=ORIGINAL_CAPTURE(model,li,hidden,group)
 with rt.timer('v54_input_source_hash'):
  xh=xhash(X)
  if li==0:assert xh==V53_GROUP_HASHES[group],f'v53 approved block0 input mismatch: {group}'
  if li==1 and group=='qkv':assert xh==V53_BOUNDARY['layer1_qkv_X_raw_content_sha256'], 'corrected full-token layer1/qkv boundary mismatch'
  key=(RUN_ID,PHASE,int(li),str(group));CURRENT_SOURCE_KEY=key
  SOURCE_RECORDS[key]=dict(run_id=RUN_ID,phase=PHASE,block=int(li),dependency_group=str(group),input_sha256=xh,input_shape=list(X.shape),input_dtype=str(X.dtype),v53_block0_source_match=(li==0),v53_layer1_boundary_match=(li==1 and group=='qkv'))
 return X

def build_C_with_source(X):
 key=CURRENT_SOURCE_KEY;assert key is not None
 with rt.timer('v52_input_pre_Gram_source_check'):
  m=SOURCE_RECORDS[key];assert m['input_sha256']==xhash(X) and m['input_shape']==list(X.shape) and m['input_dtype']==str(X.dtype), 'input source changed before original Gram builder'
 C=ORIGINAL_BUILD_C(X)
 with rt.timer('v52_Gram_source_hash'):
  m.update(C_sha256=tensor_hash(C),C_shape=list(C.shape),C_dtype=str(C.dtype),builder='unchanged v49 FP64 complete Gram; no C reuse')
  with (OUT/'SOURCE_RECORDS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(m,allow_nan=False)+'\n')
 return C

def initial_source_check(codes,scales,gidx,W,X,C,target):
 group=next(g for g,names in rt.GROUPS.items() if target in names)
 key=(RUN_ID,PHASE,int(rt.LAYER),group)
 assert key in SOURCE_RECORDS, 'missing explicit run/block/group source'
 m=SOURCE_RECORDS[key]
 assert m['input_sha256']==xhash(X) and m['input_shape']==list(X.shape) and m['input_dtype']==str(X.dtype), 'initialization input source mismatch'
 assert m['C_sha256']==tensor_hash(C) and m['C_shape']==list(C.shape) and m['C_dtype']==str(C.dtype), 'initialization Gram source mismatch'
 orig=ORIG[int(rt.LAYER)][target]
 assert torch.equal(codes.cpu(),orig['codes']) and torch.equal(scales.cpu(),orig['scales']) and torch.equal(gidx.cpu(),orig['g_idx']), 'initial M2 grid source mismatch'
 with safe_open(str(rt.FP/'model.safetensors'),framework='pt',device='cpu') as f:
  pristine=f.get_tensor(f'model.layers.{rt.LAYER}.{target}.weight').half()
 assert torch.equal(W.cpu(),pristine), 'pristine target source mismatch'
 return dict(pass_=True,**m,pristine_weight_sha256=tensor_hash(pristine),codes_sha256=tensor_hash(codes),scales_sha256=tensor_hash(scales),g_idx_sha256=tensor_hash(gidx),**{'pass':True})

rt.capture=capture_with_source
rt.build_C=build_C_with_source
rt.elapsed=elapsed

def record_module(li,name,orig,result):
 key=(int(li),str(name))
 if key in RUN_STATS:return
 r=result;mods=int((r['codes'].cpu()!=orig['codes'].cpu()).sum());rows,d=r['codes'].shape
 stat={'mode':'formal','layer':li,'module':name,'rows':rows,'input_dim':d,'accepted_updates':int(r['updates'].sum()),'modified_coordinates':mods,'candidate_evaluations':int(r['evaluations'].sum()),'initial_search_mse_sum':float(r['initial_mse'].sum()),'final_search_mse_sum':float(r['final_mse'].sum()),'initial_raw64_mse_sum':float(r['terminal_initial_raw64'].sum()),'final_raw64_mse_sum':float(r['terminal_final_raw64'].sum()),'max_p_drift':r['max_p_drift'],'rejected_candidates':int(r['rejections'].sum()),'rows_with_rejection':int((r['rejections']>0).sum()),'FP64_rechecks':len(r['recheck_events']),'terminal_max_abs_error':r['terminal_max_abs_error'],'terminal_max_error_ratio':r['terminal_max_error_ratio'],'N':262144,'complete':True}
 RUN_STATS[key]=stat;rt.STATS=[RUN_STATS[k] for k in sorted(RUN_STATS)];rt.csvwrite('MODULE_RESULTS.csv',rt.STATS)
 rr=[];byrow={}
 for e in r['recheck_events']:byrow[int(e['row'])]=byrow.get(int(e['row']),0)+1
 for i in range(rows):rr.append({'layer':li,'module':name,'row':i,'input_dim':d,'updates':int(r['updates'][i]),'evaluations':int(r['evaluations'][i]),'eval_cap':128*d,'status':int(r['status'][i]),'rejections':int(r['rejections'][i]),'raw_FP64_rechecks':byrow.get(i,0),'initial_raw64_mse':float(r['terminal_initial_raw64'][i]),'final_raw64_mse':float(r['terminal_final_raw64'][i])})
 rt.csvwrite(f'formal/rows/layer_{li:02d}_{name.replace(".","_")}.csv',rr)

def terminal_check(X,C,W,original,result):
 """Exact full-row independent raw-X terminal path from frozen v48 runner."""
 with rt.timer('independent_full_terminal_FP64'):
  d=C.shape[0];N=len(X);codes0=original['codes'].to(C.device).long();codes1=result['codes'].to(C.device).long()
  sc=original['scales'].to(C.device)[:,original['g_idx'].long().to(C.device)]
  e0=(sc*(codes0.float()-8)).half().float().double()-W.to(C.device).double()
  e1=(sc*(codes1.float()-8)).half().float().double()-W.to(C.device).double()
  EE=torch.cat((e0,e1),0);s=torch.zeros(EE.shape[0],device=C.device,dtype=torch.float64)
  for lo in range(0,N,4096):
   rt.deadline(FINISH_RESERVE);xx=torch.from_numpy(np.array(X[lo:lo+4096],copy=True)).to(C.device,dtype=torch.float64);y=xx@EE.T;s+=y.square().sum(0)
  raw=s/N;r=len(e0);quad=(EE*(EE@C)).sum(1);op=(EE.abs()*(EE.abs()@C.abs())).sum(1)
  tol=64*fast.gamma(N+d+math.ceil(N/4096)+32)*op+64*2**-53*raw
  assert bool(torch.isfinite(raw).all() and torch.isfinite(quad).all() and torch.isfinite(tol).all()),'nonfinite terminal reconstruction'
  err=(raw-quad).abs();ratio=err/tol.clamp_min(torch.finfo(torch.float64).tiny)
  assert bool((err<=tol).all()),'v48 frozen terminal FP64/quadratic error exceeded'
  assert bool((raw[r:]<=raw[:r]).all()),'confirmed terminal objective ascent'
  assert bool(((result['final_mse'].to(C.device)-quad[r:]).abs()<=tol[r:]).all()),'accepted increment bookkeeping drift'
  return {'initial':raw[:r].cpu(),'final':raw[r:].cpu(),'max_abs_error':float(err.max()),'max_error_ratio':float(ratio.max()),'all_rows':r}

def bound_fast(codes,scales,gidx,W,X,C,target):
 """Explicit dispatch wrapper; terminal check completes before a group can commit."""
 assert rt.batch_repair is bound_fast,'fast terminal-validation wrapper binding changed'
 fast.initialization_review.SOURCE_CHECK=lambda:initial_source_check(codes,scales,gidx,W,X,C,target)
 t=time.perf_counter();r=fast.batch_repair(codes,scales,gidx,W,X,C,target)
 r['packed_u4']=rt.pack(r['codes'])
 term=terminal_check(X,C,W,{'codes':codes,'scales':scales,'g_idx':gidx},r)
 r['terminal_initial_raw64']=term['initial'];r['terminal_final_raw64']=term['final'];r['terminal_max_abs_error']=term['max_abs_error'];r['terminal_max_error_ratio']=term['max_error_ratio']
 orig={'codes':codes,'scales':scales,'g_idx':gidx,'zeros':torch.full_like(scales,8)};record_module(rt.LAYER,target,orig,r)
 allowed=128*C.shape[0]
 assert bool((r['codes']<=15).all() and (r['updates']<=32).all() and (r['evaluations']<=allowed).all())
 assert torch.equal(r['scales'],scales) and torch.equal(r['g_idx'],gidx) and torch.equal(r['zeros'],orig['zeros'])
 STAGE_MODULE_ROWS.append({'layer':rt.LAYER,'module':target,'complete_fast_repair_and_terminal_wall_seconds':time.perf_counter()-t,'terminal_seconds_included':True})
 rt.csvwrite('MATRIX_TIMINGS.csv',STAGE_MODULE_ROWS)
 return r

def assert_group(cp,li,group,src):
 assert cp.get('committed') and cp['mode']=='formal' and cp['layer']==li and cp['group']==group
 assert cp['protocol_sha256']==sha(OUT/'PROTOCOL.json')
 names=rt.GROUPS[group];assert set(cp['modules'])==set(names)
 for name in names:
  s=src[name];r=cp['modules'][name];codes=r['codes'].cpu().long();assert codes.shape==s['codes'].shape and bool(((codes>=0)&(codes<=15)).all())
  for key in ['scales','zeros','g_idx']:assert torch.equal(r[key].cpu(),s[key].cpu()),f'{li}/{name}/{key} changed'
  assert torch.equal(rt.unpack(r['packed_u4'].cpu()),codes.byte())
  q=(s['scales'][:,s['g_idx'].long()]*(codes.float()-8)).half()
  assert torch.equal(q.view(torch.int16),r['weight'].cpu().half().view(torch.int16))
  assert (r['updates']<=32).all() and (r['evaluations']<=128*codes.shape[1]).all()
  assert 'terminal_final_raw64' in r and (r['terminal_final_raw64']<=r['terminal_initial_raw64']).all()

def log_restore_and_validate(model,li,hidden,src):
 """Replay only completed groups and check each checkpoint against today's prefix."""
 directory=OUT/'formal/groups';positions=[];seen_missing=False
 paths={g:directory/f'layer_{li:02d}_{g}.pt' for g in GROUP_ORDER}
 have={g:p.exists() for g,p in paths.items()}
 for g in GROUP_ORDER:
  if not have[g]:seen_missing=True
  else:assert not seen_missing,f'checkpoint gap before existing later group at layer {li}/{g}'
 if li>0:
  prev=[directory/f'layer_{li-1:02d}_{g}.pt' for g in GROUP_ORDER]
  assert all(p.exists() for p in prev),'downstream group exists without complete preceding block'
  assert (OUT/'formal'/f'PREFIX_STATUS_layer_{li-1:02d}.json').exists()
 with rt.timer('committed_group_prefix_restore_validation'):
  for g in GROUP_ORDER:
   if not have[g]:break
   rt.MODE='formal';rt.LAYER=li;rt.MODULE=g
   cp=torch.load(paths[g],map_location='cpu',weights_only=False);assert_group(cp,li,g,src)
   X=rt.capture(model,li,hidden,g);xh=hashlib.sha256(X.tobytes()).hexdigest()
   assert xh==cp['input_sha256'],f'current repair-prefix input does not match committed group: {li}/{g}'
   for name,res in cp['modules'].items():
    m=model.model.layers[li].get_submodule(name)
    with torch.no_grad():m.weight.copy_(res['weight'].cuda())
    record_module(li,name,src[name],res)
   positions.append(dict(layer=li,group=g,input_sha256=xh,C_sha256=cp['C_sha256'],restored_without_repair=True,checkpoint_sha256=sha(paths[g])))
   VERIFIED_GROUPS.add((li,g))
   del X,cp;gc.collect()
 PROPAGATION.extend(positions)
 return positions,have

def restore_saved_stats():
 rt.STATS=[];RUN_STATS.clear()
 for p in sorted((OUT/'formal/groups').glob('layer_*.pt')):
  parts=p.stem.split('_');li=int(parts[1]);group='_'.join(parts[2:]);src=ORIG[li];cp=torch.load(p,map_location='cpu',weights_only=False);assert_group(cp,li,group,src)
  for name,r in cp['modules'].items():record_module(li,name,src[name],r)

def verify_block0():
 rows=[r for r in rt.STATS if int(r['layer'])==0]
 assert len(rows)==7 and sum(int(r['rows']) for r in rows)==12672
 files=[OUT/'formal/groups'/f'layer_00_{g}.pt' for g in GROUP_ORDER];assert all(p.exists() for p in files)
 result=[]
 for g,p in zip(GROUP_ORDER,files):
  cp=torch.load(p,map_location='cpu',weights_only=False);assert_group(cp,0,g,ORIG[0]);
  for n,r in cp['modules'].items():result.append(dict(layer=0,group=g,module=n,rows=len(r['codes']),accepted=int(r['updates'].sum()),evaluations=int(r['evaluations'].sum()),rejected=int(r['rejections'].sum()),raw64_rechecks=len(r['recheck_events']),input_sha256=cp['input_sha256'],C_sha256=cp['C_sha256'],committed=True))
 assert len(result)==7
 dump('BLOCK0_GATE.json',{'pass':True,'role':'v51 block0 restored after v53 gate; not repaired again in v54','linear_modules':7,'output_rows':12672,'groups':['qkv','o','up_gate','down'],'group_code_grid_terminal_checks_pass':True,'no_old_R1_activation_cache_substituted':True,'metrics':result,'timestamp_unix':time.time()})

def v46_measurement():
 blocks=list(csv.DictReader((V46/'BLOCK_TIMINGS.csv').open(encoding='utf8')));vals=[float(r['seconds']) for r in blocks]
 return sum(vals),max(vals),json.loads((V46/'COST_PROJECTION.json').read_text(encoding='utf8'))

def in_memory_eval_protocol():
 global EVAL_KEY_MAP
 baseline=read(V46/'BASELINE_AUDIT.json');names={'wikitext':'WIKITEXT_TEST_MANIFEST.json','c4_fixed_budget':'C4_EVAL_MANIFEST.json'};hashes={}
 for ds,name in names.items():
  p=V18/'evaluation/manifests'/name;expected=baseline['evaluation_manifests'][ds];assert sha(p)==expected
  data=json.loads(p.read_text(encoding='utf8'));assert len(data['token_ids'])==baseline['token_counts'][ds][0]+1
  hashes['wikitext' if ds=='wikitext' else 'c4_fixed_budget']=expected
 rt.P.setdefault('v18',{})['source_protocol']={'v16_wikitext_manifest_sha256':hashes['wikitext'],'v16_c4_manifest_sha256':hashes['c4_fixed_budget']}
 EVAL_KEY_MAP=hashes
 return hashes

def group_restore_audit(model,li,hidden):
 src=ORIG[li];paths,have=log_restore_and_validate(model,li,hidden,src)
 return paths,have


def verify_native_boundary(model,li,hidden):
 """Same v53 reference contract: full model forward + independent hook, never decoder extraction."""
 rows=[];t0=time.perf_counter();target=li+1
 for doc in [0,64,127]:
  deadline(FINISH_RESERVE)
  sample=rt.CAL['samples'][doc];ids=torch.tensor(sample['token_ids'],device='cuda')[None]
  active={};handles=[]
  if target<24:
   nxt=model.model.layers[target]
   def hh(module,args):
    assert 'hidden' not in active
    v=args[0];assert torch.is_tensor(v) and tuple(v.shape)==(1,2048,896) and v.dtype==torch.float16
    active['hidden']=v.detach().cpu().numpy().copy()
   def hq(module,args):
    assert 'qkv' not in active
    v=args[0];assert torch.is_tensor(v) and tuple(v.shape)==(1,2048,896) and v.dtype==torch.float16
    active['qkv']=v.detach().cpu().numpy().copy()
   handles=[nxt.register_forward_pre_hook(hh),nxt.self_attn.q_proj.register_forward_pre_hook(hq)]
  else:
   def hn(module,args):
    assert 'hidden' not in active
    v=args[0];assert torch.is_tensor(v) and tuple(v.shape)==(1,2048,896) and v.dtype==torch.float16
    active['hidden']=v.detach().cpu().numpy().copy()
   handles=[model.model.norm.register_forward_pre_hook(hn)]
  try:
   with torch.inference_mode():
    out=model(input_ids=ids,use_cache=False,output_attentions=False,logits_to_keep=1)
   del out
  finally:
   for h in handles:h.remove()
  assert 'hidden' in active and ('qkv' in active if target<24 else True),'native reference hooks not reached exactly once'
  actual=hidden[doc:doc+1]
  def cmp(a,b,label):
   assert a.shape==b.shape and a.dtype==b.dtype==np.float16,(label,a.shape,b.shape,a.dtype,b.dtype)
   bitwise=np.array_equal(a,b);delta=np.abs(a.astype(np.float32)-b.astype(np.float32))
   ok=bool(np.allclose(a,b,rtol=1e-5,atol=1e-6))
   row={'block_completed':li,'target':target if target<24 else 'final_norm','article':doc,'name':label,'shape':list(a.shape),'dtype':str(a.dtype),'bitwise_equal':bitwise,'max_abs_error':float(delta.max(initial=0)),'rtol':1e-5,'atol':1e-6,'pass':ok}
   assert ok,f'native propagation mismatch {row}'
   rows.append(row)
  cmp(actual,active['hidden'],'next_hidden')
  if target<24:
   with torch.inference_mode():
    h=torch.from_numpy(np.array(actual,copy=True)).cuda()
    q=nxt.input_layernorm(h).cpu().numpy()
   cmp(q,active['qkv'],'next_qkv_input')
  torch.cuda.synchronize()
 outpath=OUT/'PROPAGATION_NATIVE_RESULTS.csv'
 exists=outpath.exists()
 with outpath.open('a',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]))
  if not exists:w.writeheader()
  w.writerows(rows);f.flush();os.fsync(f.fileno())
 seconds=time.perf_counter()-t0
 with (OUT/'PROPAGATION_NATIVE_TIMINGS.csv').open('a',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['block_completed','seconds','articles','native_full_forward_calls','comparisons','elapsed_seconds'])
  if f.tell()==0:w.writeheader()
  w.writerow({'block_completed':li,'seconds':seconds,'articles':'0;64;127','native_full_forward_calls':3,'comparisons':len(rows),'elapsed_seconds':elapsed()});f.flush();os.fsync(f.fileno())
 rt.dump(f'formal/NATIVE_PROPAGATION_block_{li:02d}.json',{'pass':True,'completed_block':li,'current_model_parameter_state':'all groups through this block committed; later blocks remain original M2','fixed_articles':[0,64,127],'tokens_each':2048,'native_full_forward_calls':3,'reference_uses_extractor':False,'comparisons':rows,'seconds':seconds})
 return seconds,rows

def main():
 global GPU_START,ORIG
 args='--resume' in sys.argv
 assert read('PREPARATION.json')['status']=='CPU_SOURCE_AUDIT_PASS' and read('PROTOCOL.json')['status']=='FROZEN_BEFORE_GPU' and read('BLOCK0_INHERITANCE.json')['pass']
 assert torch.__version__ == '2.6.0+cu124', f'v49 runtime mismatch: {torch.__version__}'
 assert read('SOURCE_AUDIT.json')['pass']
 assert (OUT/'R1/checkpoint/final').is_dir() and (OUT/'formal/groups').is_dir() and (OUT/'evaluation').is_dir()
 assert read('FROZEN_RUN_LOCK.json')['pass'] and read('V54_COST_PRECHECK.json')['go']
 for n,h in read('FROZEN_RUN_LOCK.json')['files'].items():assert sha(OUT/n)==h,f'frozen v54 source changed: {n}'
 assert sha(V53/'runtime_v53.py')==read('FROZEN_RUN_LOCK.json')['v53_runtime_sha256']
 assert sha(V53/'GPU_VALIDATION_RESULT.json')==read('FROZEN_RUN_LOCK.json')['v53_validation_result_sha256']
 assert sha(V48/'fast_accept.py')==read('PROTOCOL.json')['algorithm_source']['fast_accept_sha256']
 assert FAST_SHA==read('PROTOCOL.json')['v52_initialization_review']['core_sha256']
 assert sha(OUT/'initialization_review.py')==read('PROTOCOL.json')['v52_initialization_review']['review_sha256']
 assert read('NUMERICAL_VALIDATION.json')['pass'] and read('BLOCK0_INHERITANCE.json')['pass']
 assert sha(V48/'ACCEPTANCE_RULES.md')==V48_RULE_SHA
 budget.start('formal complete repair and deployment/evaluation')
 GPU_START=time.time()
 if (OUT/'GPU_START.json').exists():
  start=read('GPU_START.json');assert start['run_py_sha256']==sha(__file__) and start['protocol_sha256']==sha(OUT/'PROTOCOL.json')
 else:
  dump('GPU_START.json',dict(start_epoch=GPU_START,cap_seconds=CAP,pid=os.getpid(),run_py_sha256=sha(__file__),fast_accept_sha256=FAST_SHA,protocol_sha256=sha(OUT/'PROTOCOL.json'),binding='v53 strict decoder runtime + unchanged v52 initialization review and v49/v48 fast repair',requested_model_setting='Luna Max requested; interface cannot verify',prior_shared_charged_seconds=budget.seconds()[0],active_job_ledger='ACTIVE_JOB_LEDGER.json'))
 hist=read('RECOVERY_HISTORY.json') if (OUT/'RECOVERY_HISTORY.json').exists() else {'entries':[]}
 hist['entries'].append(dict(version='v54',start_unix=GPU_START,cumulative_elapsed_seconds=elapsed(),inherited_v51_block0_without_repair=True))
 dump('RECOVERY_HISTORY.json',hist)
 dump('RUN_STATUS.json',{'status':'RUNNING','category':'D','elapsed_seconds':elapsed(),'complete_model':False,'PPL_run':False,'resume':args})
 rt.batch_repair=bound_fast
 dump('RUNTIME_BINDING.json',{'pass':rt.batch_repair is bound_fast,'decoder_runtime':str(OUT/'runtime_v54.py'),'decoder_runtime_sha256':sha(OUT/'runtime_v54.py'),'runtime_dispatch':'v49.bound_fast','direct_delegate':'v48_fast_accept_v49.batch_repair','module':fast.batch_repair.__module__,'source_file':str(Path(fast.__file__).resolve()),'source_sha256':FAST_SHA,'terminal_check_inside_group_before_atomic_commit':True,'same_shared_search_helpers':True,'v42_legacy_dispatch_prevented':True,'v46_evaluator_key_prebound_to_audited_v18_manifests':True})
 assert rt.batch_repair is bound_fast and Path(fast.__file__).resolve()==(OUT/'fast_accept_v52.py').resolve()
 hashes=in_memory_eval_protocol();dump('EVALUATOR_PREFLIGHT.json',{'pass':True,'missing_v46_P_v18_key_supplied_in_memory':True,'manifest_hashes':hashes,'evaluator_source_sha256':sha(OUT/'runtime_v54.py'),'no_evaluator_code_changes':True})
 rt.WALL_START=GPU_START;rt.elapsed=elapsed;rt.START=time.perf_counter();rt.deadline=runtime_deadline;rt.MODE='preflight';rt.LAYER=-1;rt.MODULE=''
 def hardstop():
  try:dump('RUN_STATUS.json',{'status':'INCOMPLETE_HARD_CAP','category':'D','elapsed_seconds':elapsed(),'cap_seconds':CAP,'complete_model':False,'PPL_run':False,'last_committed_group':read('formal/COMMIT_STATUS.json') if (OUT/'formal/COMMIT_STATUS.json').exists() else None})
  finally:budget.finish();os._exit(124)
 watcher=threading.Timer(max(1,CAP-elapsed()),hardstop);watcher.daemon=True;watcher.start()
 try:
  with rt.timer('formal_warmup'):
   z=torch.zeros((64,64),device='cuda');z@z
  rt.MODE='formal';rt.batch_repair=bound_fast
  assert rt.batch_repair is bound_fast
  with rt.timer('formal_model_load'):
   model=AutoModelForCausalLM.from_pretrained(str(V18/'M2/checkpoint/final'),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()
  source_m2=V18/'M2/checkpoint/final/model.safetensors';assert sha(source_m2)==read('PROTOCOL.json')['initialization']['model_sha256']
  ids=rt.zero_model_check(model);hidden=rt.embed(model)
  for li in range(24):
   deadline(FINISH_RESERVE);ORIG[li]=torch.load(OUT/f'original_codes/layer_{li:02d}.pt',map_location='cpu',weights_only=True)
   if li>0:assert all((OUT/'formal/groups'/f'layer_{li-1:02d}_{g}.pt').exists() for g in GROUP_ORDER),'checkpoint gap before next block'
   _,have_groups=group_restore_audit(model,li,hidden)
   rt.LAYER=li;rt.MODE='formal';rt.batch_repair=bound_fast
   start=time.perf_counter();hidden=rt.run_layer(model,li,hidden);block_seconds=time.perf_counter()-start
   propagation_seconds,native_rows=verify_native_boundary(model,li,hidden)
   with (OUT/'BLOCK_TIMINGS.csv').open('a',encoding='utf8',newline='') as f:
    fields=['layer','seconds','elapsed_seconds','new_groups','accepted_updates','raw64_rechecks'];w=csv.DictWriter(f,fieldnames=fields)
    if f.tell()==0:w.writeheader()
    new_names={n for g,names in rt.GROUPS.items() if not have_groups[g] for n in names}
    w.writerow({'layer':li,'seconds':block_seconds,'elapsed_seconds':elapsed(),'new_groups':sum(not x for x in have_groups.values()),'accepted_updates':sum(int(r['accepted_updates']) for r in rt.STATS if int(r['layer'])==li and r['module'] in new_names),'raw64_rechecks':sum(int(r['FP64_rechecks']) for r in rt.STATS if int(r['layer'])==li and r['module'] in new_names)})
   hid=hashlib.sha256(hidden.tobytes()).hexdigest();dump(f'formal/PREFIX_STATUS_layer_{li:02d}.json',{'completed_layer':li,'hidden_sha256':hid,'elapsed_seconds':elapsed(),'group_order':GROUP_ORDER})
   group_inputs={}
   for g in GROUP_ORDER:
    p=OUT/'formal/groups'/f'layer_{li:02d}_{g}.pt';cp=torch.load(p,map_location='cpu',weights_only=False);assert_group(cp,li,g,ORIG[li])
    group_inputs[g]=cp['input_sha256'];key=(li,g)
    if key not in VERIFIED_GROUPS:
     r1=V46/'formal/groups'/f'layer_{li:02d}_{g}.pt';r1cp=torch.load(r1,map_location='cpu',weights_only=False)
     PROPAGATION.append({'layer':li,'group':g,'input_sha256':cp['input_sha256'],'C_sha256':cp['C_sha256'],'v46_R1_input_sha256':r1cp['input_sha256'],'v46_R1_C_sha256':r1cp['C_sha256'],'input_matches_R1':cp['input_sha256']==r1cp['input_sha256'],'C_matches_R1_if_input_same':(cp['C_sha256']==r1cp['C_sha256']) if cp['input_sha256']==r1cp['input_sha256'] else None,'restored_without_repair':False,'current_run_group_completed':True,'checkpoint_sha256':sha(p)})
     VERIFIED_GROUPS.add(key)
   PROPAGATION.append({'layer':li,'boundary_after_down':hid,'group_input_hashes':group_inputs})
   rt.csvwrite('PROPAGATION_RESTORE_CHECKS.csv',PROPAGATION)
   if li==0:
    verify_block0()
   timing_rows=list(csv.DictReader((V52/'BLOCK_TIMINGS.csv').open(encoding='utf8')))
   historical={int(x['layer']):float(x['seconds']) for x in timing_rows}
   measured_native=[float(x['seconds']) for x in csv.DictReader((OUT/'PROPAGATION_NATIVE_TIMINGS.csv').open(encoding='utf8'))]
   baseline_native=float(read(V53/'GPU_VALIDATION_TIMINGS.csv')[-1]['seconds']) if False else 9.246
   per_check=max(baseline_native, sum(measured_native)/len(measured_native))
   remaining_layers=list(range(li+1,24))
   remaining_repair=sum(historical[k] for k in remaining_layers)
   remaining_propagation=len(remaining_layers)*per_check
   predicted=remaining_repair+remaining_propagation+FINISH_RESERVE+UNCERTAINTY_RESERVE
   gate=elapsed()+predicted<=CAP
   dump('ONLINE_BUDGET_STATUS.json',{'after_completed_block':li,'elapsed_shared_seconds':elapsed(),'remaining_shared_seconds':CAP-elapsed(),'historical_same_initializer_v52_remaining_block_seconds':remaining_repair,'history_prefix_warning':'those timings used erroneous later prefix; prediction only, not validity evidence','v53_or_observed_per_block_native_comparison_seconds':per_check,'remaining_three_article_block_validation_seconds':remaining_propagation,'uncertainty_reserve_seconds':UNCERTAINTY_RESERVE,'finish_and_evaluation_reserve_seconds':FINISH_RESERVE,'predicted_required_seconds':predicted,'go':gate})
   if not gate:
    dump('RUN_STATUS.json',{'status':'BUDGET_STOP_AFTER_COMPLETE_BLOCK','category':'D','elapsed_seconds':elapsed(),'complete_model':False,'PPL_run':False,'last_complete_layer':li,'last_groups_restorable':True,'stop_reason':'remaining measured-history projection plus propagation and frozen 360s finish reserve exceeds shared budget'})
    return
   print('V54_BLOCK_COMPLETE',li,'seconds',round(block_seconds,3),'elapsed',round(elapsed(),3),'accepted',sum(int(r['accepted_updates']) for r in rt.STATS if int(r['layer'])==li),flush=True)
  assert len(rt.STATS)==168 and sum(int(r['rows']) for r in rt.STATS)==304128
  dump('FORMAL_REPAIR_COMPLETE.json',{'modules':len(rt.STATS),'rows':sum(int(r['rows']) for r in rt.STATS),'elapsed_seconds':elapsed(),'accepted_updates':sum(int(s['accepted_updates']) for s in rt.STATS),'modified_coordinates':sum(int(s['modified_coordinates']) for s in rt.STATS),'candidate_evaluations':sum(int(s['candidate_evaluations']) for s in rt.STATS),'rejected_candidates':sum(int(s['rejected_candidates']) for s in rt.STATS),'raw64_rechecks':sum(int(s['FP64_rechecks']) for s in rt.STATS)})
  assert elapsed()+FINISH_RESERVE<=CAP,'insufficient frozen finish reserve'
  rt.MODE='deployment';model=rt.final_save_check(model,ids)
  mcfg=V18/'M2/checkpoint/final/config.json';outcfg=OUT/'R1/checkpoint/final/config.json'
  cfg=json.loads(mcfg.read_text(encoding='utf8'));cfg2=json.loads(outcfg.read_text(encoding='utf8'));ignore={'_name_or_path','transformers_version'}
  r2cfg=json.loads((V45/'R/checkpoint/final/config.json').read_text(encoding='utf8'))
  config_equal=({k:v for k,v in cfg.items() if k not in ignore}=={k:v for k,v in cfg2.items() if k not in ignore}=={k:v for k,v in r2cfg.items() if k not in ignore});assert config_equal
  modelpath=OUT/'R1/checkpoint/final/model.safetensors';refpath=V45/'R/checkpoint/final/model.safetensors';m2path=V18/'M2/checkpoint/final/model.safetensors';diffs=[];diff_m2=[];comp_start=time.perf_counter()
  with safe_open(str(modelpath),framework='pt',device='cpu') as ours,safe_open(str(refpath),framework='pt',device='cpu') as r2,safe_open(str(m2path),framework='pt',device='cpu') as m2:
   assert set(ours.keys())==set(r2.keys())==set(m2.keys())
   for key in ours.keys():
    a=ours.get_tensor(key);b=r2.get_tensor(key);c=m2.get_tensor(key)
    if not torch.equal(a.view(torch.uint8),b.view(torch.uint8)):
     diffs.append({'key':key,'v52_tensor_sha256':tensor_hash(a),'original_R2_tensor_sha256':tensor_hash(b),'max_abs_v52_minus_R2':float((a.float()-b.float()).abs().max()),'unequal_bytes':int((a.view(torch.uint8)!=b.view(torch.uint8)).sum())})
    if not torch.equal(a.view(torch.uint8),c.view(torch.uint8)):
     diff_m2.append(key)
  tensor_key_count=0
  with safe_open(str(modelpath),framework='pt',device='cpu') as f:tensor_key_count=len(f.keys())
  rt.TIMINGS.append({'mode':'deployment','layer':-1,'module':'all_model_tensors','stage':'named_tensor_comparison_vs_original_R2','seconds':time.perf_counter()-comp_start,'elapsed_seconds':elapsed(),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'type':'measured CPU wall'})
  quant_rows=[]
  for li in range(24):
   src=ORIG.get(li) or torch.load(OUT/f'original_codes/layer_{li:02d}.pt',map_location='cpu',weights_only=True)
   for g in GROUP_ORDER:
    ck=torch.load(OUT/'formal/groups'/f'layer_{li:02d}_{g}.pt',map_location='cpu',weights_only=False);assert_group(ck,li,g,src)
    refcpgroup=torch.load(V45/'formal/groups'/f'layer_{li:02d}_{g}.pt',map_location='cpu',weights_only=False)
    for name,res in ck['modules'].items():
     refcp=refcpgroup['modules'][name]
     quant_rows.append({'layer':li,'group':g,'module':name,'codes_equal_original_R2':torch.equal(res['codes'],refcp['codes']),'scales_equal_M2_R2':torch.equal(res['scales'],refcp['scales']),'zeros_equal_M2_R2':torch.equal(res['zeros'],refcp['zeros']),'g_idx_equal_M2_R2':torch.equal(res['g_idx'],refcp['g_idx']),'FP16_grid_weights_equal_R2':torch.equal(res['weight'],refcp['weight']),'v52_input_sha256':ck['input_sha256'],'original_R2_input_sha256':refcpgroup['input_sha256'],'input_matches_original_R2':ck['input_sha256']==refcpgroup['input_sha256'],'v52_C_sha256':ck['C_sha256'],'original_R2_C_sha256':refcpgroup['C_sha256']})
  rt.csvwrite('reports/MODULE_VS_ORIGINAL_R2.csv',quant_rows);rt.csvwrite('reports/TENSOR_DIFFERENCES_VS_ORIGINAL_R2.csv',diffs)
  equal=(not diffs) and config_equal
  lock={'final_model_sha256':sha(modelpath),'config_sha256':sha(outcfg),'protocol_sha256':sha(OUT/'PROTOCOL.json'),'group_checkpoints':[{'path':str(p),'sha256':sha(p)} for p in sorted((OUT/'formal/groups').glob('*.pt'))],'modules_compared_with_original_R2':len(quant_rows),'named_model_tensor_differences_vs_original_R2':len(diffs),'inference_config_equal_original_R2':config_equal,'equivalent_to_original_R2':equal,'locked_before_evaluation_or_reuse_unix':time.time()}
  assert len(lock['group_checkpoints'])==96;dump('CHECKPOINT_LOCK.json',lock)
  deployment={'pass':True,'all_168_grid_pack_reload_and_sample_output_checks':True,'model_sha256':sha(modelpath),'config_sha256':sha(outcfg),'named_model_tensors_equal_original_R2':not diffs,'M2_quantized_target_tensors_different':len(diff_m2),'original_R2_model_sha256':sha(refpath),'M2_checkpoint_sha256':sha(m2path),'config_equal_original_R2':config_equal,'max_abs_diff_summary':max((d['max_abs_v52_minus_R2'] for d in diffs),default=0.0),'tensor_keys_compared':tensor_key_count,'completed_unix':time.time()};dump('DEPLOYMENT_COMPARISON.json',deployment)
  copies=[]
  rt.MODE='evaluation'
  assert rt.P['v18']['source_protocol']=={'v16_wikitext_manifest_sha256':EVAL_KEY_MAP['wikitext'],'v16_c4_manifest_sha256':EVAL_KEY_MAP['c4_fixed_budget']}
  assert not (OUT/'R1_wikitext.json').exists() and not (OUT/'R1_c4_fixed_budget.json').exists()
  rt.evaluate(model)
  for ds,key in [('wikitext','wikitext'),('c4_fixed_budget','c4_fixed_budget')]:
   srcres=OUT/f'R1_{ds}.json';srcwin=OUT/f'R1_{ds}_WINDOWS.csv'
   dstres=OUT/f'evaluation/v54_{ds}.json';dstwin=OUT/f'evaluation/v54_{ds}_WINDOWS.csv'
   assert srcres.exists() and srcwin.exists()
   res=json.loads(srcres.read_text(encoding='utf8'))
   assert res['model_sha256']==sha(modelpath) and res['manifest_sha256']==EVAL_KEY_MAP[key]
   srcres.replace(dstres);srcwin.replace(dstwin)
   res=json.loads(dstres.read_text(encoding='utf8'))
   copies.append({'dataset':ds,'result_sha256':sha(dstres),'windows_sha256':sha(dstwin),'PPL':res['ppl'],'NLL':res['nll'],'valid_tokens':res['scored_tokens'],'windows':res['windows'],'fresh_frozen_protocol_evaluation':True})
  rt.dump('QUALITY_EVALUATION_SOURCE.json',{'pass':len(copies)==2,'new_PPL_evaluations':2,'reuse_history':False,'P3_read':False,'evaluator_code_sha256':sha(OUT/'runtime_v54.py'),'source_manifests':hashes,'new_results':copies})
  if model is not None:del model
  blocks=list(csv.DictReader((OUT/'BLOCK_TIMINGS.csv').open(encoding='utf8')))
  status={'status':'COMPLETE','category':'COMPLETE_PENDING_AGGREGATION','complete_model':True,'modules':168,'rows':304128,'model_sha256':sha(modelpath),'equivalent_to_original_R2':equal,'PPL_run':True,'PPL_reused':False,'elapsed_seconds_from_gpu_start':elapsed(),'hard_cap_seconds':CAP,'blocks_completed':len({int(b['layer']) for b in blocks if int(b['new_groups'])>0}),'failed_or_resumed_groups':read('RECOVERY_HISTORY.json')['entries'] if (OUT/'RECOVERY_HISTORY.json').exists() else [],'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()};dump('RUN_STATUS.json',status)
  rt.csvwrite('STAGE_TIMINGS_THIS_PROCESS.csv',rt.TIMINGS)
  dump('COMPLETE.json',{'pass':True,'status_sha256':sha(OUT/'RUN_STATUS.json'),'all_96_committed_groups':True,'all_168_modules':True,'all_304128_rows':True,'equivalent_to_original_R2':equal,'model_ready_unix':time.time(),'full_job_elapsed_seconds':elapsed()})
 except BaseException as ex:
  tb=traceback.format_exc()
  try:
   with (OUT/'FAILED_ATTEMPTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'time_unix':time.time(),'elapsed_seconds':elapsed(),'error':repr(ex),'traceback':tb,'last_committed_group':read('formal/COMMIT_STATUS.json') if (OUT/'formal/COMMIT_STATUS.json').exists() else None},allow_nan=False)+'\n')
   dump('RUN_STATUS.json',{'status':'FAILED_STOP','category':'D','error':repr(ex),'traceback':tb,'elapsed_seconds':elapsed(),'complete_model':False,'PPL_run':False,'last_committed_group':read('formal/COMMIT_STATUS.json') if (OUT/'formal/COMMIT_STATUS.json').exists() else None})
  except Exception:pass
  raise
 finally:watcher.cancel();budget.finish()

if __name__=='__main__':
 try:main()
 except BaseException:
  if (OUT/'GPU_START.json').exists():
   p=OUT/'FAILED_ATTEMPTS.jsonl'
   if not p.exists():p.write_text(json.dumps({'error':'failure before inner runner handler','elapsed_since_process_start':time.time()-SCRIPT_START})+'\n',encoding='utf8')
  raise
