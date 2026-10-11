from pathlib import Path
import json,hashlib,difflib
O=Path(__file__).resolve().parent;B=O.parent;V51=B/'v51_m2_fast_integer_repair_fullmodel'
s=(V51/'run_v51.py').read_text(encoding='utf8');old=s
s=s.replace('v51_m2_fast_integer_repair_fullmodel','v52_m2_fast_integer_repair_fp64_initial_review').replace('v51','v52').replace('V51','V52')
s=s.replace("fast=loadmod('v48_fast_accept_v49',V48/'fast_accept.py')","sys.path.insert(0,str(OUT))\nimport budget\nfast=loadmod('v52_fast_accept',OUT/'fast_accept_v52.py')")
s=s.replace("FAST_SHA=sha(V48/'fast_accept.py')","FAST_SHA=sha(OUT/'fast_accept_v52.py')")
s=s.replace('def elapsed():return time.time()-GPU_START','def elapsed():return budget.elapsed()')
s=s.replace("t=time.perf_counter();r=fast.batch_repair(codes,scales,gidx,W,X,C,target)","fast.initialization_review.SOURCE_CHECK=lambda:initial_source_check(codes,scales,gidx,W,X,C,target)\n t=time.perf_counter();r=fast.batch_repair(codes,scales,gidx,W,X,C,target)")
s=s.replace("assert sha(V48/'fast_accept.py')==read('PROTOCOL.json')['algorithm_source']['fast_accept_sha256']","assert sha(V48/'fast_accept.py')==read('PROTOCOL.json')['algorithm_source']['fast_accept_sha256']\n assert FAST_SHA==read('PROTOCOL.json')['v52_initialization_review']['core_sha256']\n assert sha(OUT/'initialization_review.py')==read('PROTOCOL.json')['v52_initialization_review']['review_sha256']\n assert read('NUMERICAL_VALIDATION.json')['pass'] and read('INHERITANCE.json')['pass']")
lo=s.index(' if args:\n  start=read(');hi=s.index(" dump('RUN_STATUS.json',{'status':'RUNNING'",lo)
s=s[:lo]+''' budget.start('formal complete repair and deployment/evaluation')
 GPU_START=time.time()
 if (OUT/'GPU_START.json').exists():
  start=read('GPU_START.json');assert start['run_py_sha256']==sha(__file__) and start['protocol_sha256']==sha(OUT/'PROTOCOL.json')
 else:
  dump('GPU_START.json',dict(start_epoch=GPU_START,cap_seconds=CAP,pid=os.getpid(),run_py_sha256=sha(__file__),fast_accept_sha256=FAST_SHA,protocol_sha256=sha(OUT/'PROTOCOL.json'),binding='v52 local copy of exact v48 core plus only initial check fallback',requested_model_setting='Sol high requested; interface cannot verify',prior_v51_charged_seconds=657.305,active_job_ledger='ACTIVE_JOB_LEDGER.json'))
 hist=read('RECOVERY_HISTORY.json') if (OUT/'RECOVERY_HISTORY.json').exists() else {'entries':[]}
 hist['entries'].append(dict(version='v52',start_unix=GPU_START,cumulative_elapsed_seconds=elapsed(),inherited_v51_block0_without_repair=True))
 dump('RECOVERY_HISTORY.json',hist)
''' +s[hi:]
s=s.replace("Path(fast.__file__).resolve()==(V48/'fast_accept.py').resolve()","Path(fast.__file__).resolve()==(OUT/'fast_accept_v52.py').resolve()")
s=s.replace("finally:os._exit(124)","finally:budget.finish();os._exit(124)")
s=s.replace('finally:watcher.cancel()','finally:watcher.cancel();budget.finish()')
s=s.replace('rt.WALL_START=GPU_START;rt.START=', 'rt.WALL_START=GPU_START;rt.elapsed=elapsed;rt.START=')
s=s.replace('requested_model_setting\':\'Luna Max', 'requested_model_setting\':\'Sol high')
# Explicit provenance-only wrappers. No Gram cache; the original builder is called once on every request.
add='''
RUN_ID='v52-'+sha(OUT/'PROTOCOL.json')[:16]
PHASE='formal'
SOURCE_RECORDS={}
CURRENT_SOURCE_KEY=None
ORIGINAL_CAPTURE=rt.capture
ORIGINAL_BUILD_C=rt.build_C

def xhash(X):
 if torch.is_tensor(X):return tensor_hash(X)
 return hashlib.sha256(X.tobytes()).hexdigest()

def capture_with_source(model,li,hidden,group):
 global CURRENT_SOURCE_KEY
 X=ORIGINAL_CAPTURE(model,li,hidden,group)
 with rt.timer('v52_input_source_hash'):
  key=(RUN_ID,PHASE,int(li),str(group));CURRENT_SOURCE_KEY=key
  SOURCE_RECORDS[key]=dict(run_id=RUN_ID,phase=PHASE,block=int(li),dependency_group=str(group),input_sha256=xhash(X),input_shape=list(X.shape),input_dtype=str(X.dtype))
 return X

def build_C_with_source(X):
 key=CURRENT_SOURCE_KEY;assert key is not None
 with rt.timer('v52_input_pre_Gram_source_check'):
  m=SOURCE_RECORDS[key];assert m['input_sha256']==xhash(X) and m['input_shape']==list(X.shape) and m['input_dtype']==str(X.dtype), 'input source changed before original Gram builder'
 C=ORIGINAL_BUILD_C(X)
 with rt.timer('v52_Gram_source_hash'):
  m.update(C_sha256=tensor_hash(C),C_shape=list(C.shape),C_dtype=str(C.dtype),builder='unchanged v49 FP64 complete Gram; no C reuse')
  with (OUT/'SOURCE_RECORDS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(m,allow_nan=False)+'\\n')
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
'''
pos=s.index('\ndef record_module(');s=s[:pos]+add+s[pos:]
(O/'run_v52.py').write_text(s,encoding='utf8')
(O/'RUNNER_ADAPTER.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='read-only v51/run_v51.py',tofile='v52/run_v52.py')),encoding='utf8')
compile(s,str(O/'run_v52.py'),'exec')
# Inherited group tensor state is preserved. The new container explicitly records the original protocol/hash and inheritance.
import torch
ns={};v51p=hashlib.sha256((V51/'PROTOCOL.json').read_bytes()).hexdigest();v52p=hashlib.sha256((O/'PROTOCOL.json').read_bytes()).hexdigest();rows=[]
assert not list((V51/'formal/groups').glob('layer_01*'))
for g in ['qkv','o','up_gate','down']:
 p=V51/f'formal/groups/layer_00_{g}.pt';h=hashlib.sha256(p.read_bytes()).hexdigest();cp=torch.load(p,map_location='cpu',weights_only=False)
 assert cp['committed'] and cp['protocol_sha256']==v51p and cp['layer']==0 and cp['group']==g
 src=torch.load(O/'original_codes/layer_00.pt',map_location='cpu',weights_only=True)
 for n,r in cp['modules'].items():
  for k in ['scales','zeros','g_idx']:assert torch.equal(r[k],src[n][k])
  q=(r['scales'][:,r['g_idx'].long()]*(r['codes'].float()-8)).half();assert torch.equal(q.view(torch.int16),r['weight'].view(torch.int16))
  assert bool((r['codes']<=15).all() and (r['updates']<=32).all() and (r['evaluations']<=128*r['codes'].shape[1]).all())
  assert bool((r['terminal_final_raw64']<=r['terminal_initial_raw64']).all())
 cp['inherited_from']=dict(version='v51',path=str(p),sha256=h,original_protocol_sha256=v51p,repaired_under_v51=True,not_repaired_again_under_v52=True)
 cp['protocol_sha256']=v52p;dst=O/f'formal/groups/layer_00_{g}.pt';assert not dst.exists();torch.save(cp,dst)
 rows.append(dict(group=g,original_sha256=h,new_container_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),input_sha256=cp['input_sha256'],C_sha256=cp['C_sha256'],module_count=len(cp['modules']),inherited_without_repair=True))
(O/'INHERITANCE.json').write_text(json.dumps(dict(**{'pass':True},source_protocol_sha256=v51p,v52_protocol_sha256=v52p,groups=rows,layer1_qkv_has_no_inherited_state=True),indent=2)+'\n')
files=['run_v52.py','fast_accept_v52.py','initialization_review.py','budget.py','PROTOCOL.json']
(O/'IMPLEMENTATION_SOURCES.json').write_text(json.dumps({n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in files},indent=2)+'\n')
print('V52_RUNNER_AND_INHERITANCE_READY')

