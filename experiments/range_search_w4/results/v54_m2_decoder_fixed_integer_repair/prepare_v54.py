import csv,hashlib,json,shutil,time
from pathlib import Path
import torch

ROOT=Path(r'E:\quant-thesis')
BASE=ROOT/'experiments/range_search_w4/results'
OUT=BASE/'v54_m2_decoder_fixed_integer_repair'
V18=BASE/'range_w4_fullmodel_v18'
V46=BASE/'v46_m1_single_integer_repair_fullmodel'
V48=BASE/'v48_fast_numerical_acceptance'
V49=BASE/'v49_fullmodel_fast_integer_repair'
V51=BASE/'v51_m2_fast_integer_repair_fullmodel'
V52=BASE/'v52_m2_fast_integer_repair_fp64_initial_review'
V53=BASE/'v53_decoder_interface_propagation_audit'
GROUPS=['qkv','o','up_gate','down']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rd(p):return json.loads(Path(p).read_text(encoding='utf8'))
def wr(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
assert OUT.exists() and (OUT/'PREPARATION.json').exists() is False, 'target is already prepared; refusing overwrite'
for d in ['formal/groups','formal/rows','R1/checkpoint/final','evaluation','reports','diagnostics']:
 (OUT/d).mkdir(parents=True,exist_ok=True)
# Exact frozen algorithm modules and initialization fallback, copied to isolate v54 imports.
for n in ['fast_accept_v52.py','initialization_review.py','SOURCE_AUDIT.json','NUMERICAL_VALIDATION.json','CPU_TESTS.json']:
 src=V52/n; assert src.is_file();shutil.copy2(src,OUT/n);assert sha(src)==sha(OUT/n)
# Trusted pristine code maps from v52 provenance: these contain original M2 codes, never repaired weights/X/C.
srcaudit=rd(V52/'SOURCE_AUDIT.json')
assert srcaudit['pass'] and srcaudit['m2_model_sha256']==rd(V52/'PROTOCOL.json')['initialization']['model_sha256']
if not (OUT/'original_codes').exists():shutil.copytree(V52/'original_codes',OUT/'original_codes')
expected={int(x['layer']):x for x in rd(V52/'SOURCE_AUDIT.json')['m2_original_code_files']}
code_rows=[]
for li in range(24):
 src=V52/f'original_codes/layer_{li:02d}.pt';dst=OUT/f'original_codes/layer_{li:02d}.pt'
 assert sha(src)==expected[li]['v42_exact_code_file_sha256']==sha(dst)
 native=Path(expected[li]['m2_native_checkpoint'])
 assert sha(native)==expected[li]['native_checkpoint_sha256']
 code_rows.append({'layer':li,'original_code_file':str(dst),'sha256':sha(dst),'source_native_M2_layer':str(native),'source_sha256':sha(native),'role':'pristine M2 integer code and grid only; no repair parameters, calibration activations or C'})
# Runtime package: v53 corrected runtime source byte-for-byte, plus its exact decoder and path helper.
for n,src in [('runtime_v54.py',V53/'runtime_v53.py'),('decoder_interface.py',V53/'decoder_interface.py'),('audit_sources.py',V46/'audit_sources.py')]:
 shutil.copy2(src,OUT/n)
if sha(OUT/'runtime_v54.py')!=sha(V53/'runtime_v53.py'):raise RuntimeError('v53 runtime copy mismatch')
# New protocol carries v52 search/review fields, adds only corrected interface + required propagation gate.
p=rd(V52/'PROTOCOL.json')
p['version']='v54';p['status']='FROZEN_BEFORE_GPU'
p['requested_model_setting']='Luna Max requested; API cannot verify or switch'
p['objective']='Complete M2 fixed-grid repair from layer1 using v53 decoder interface and frozen v52 initialization review/v49 fast repair.'
p['v53_decoder_interface']={'source':str(V53/'runtime_v53.py'),'source_sha256':sha(V53/'runtime_v53.py'),'local_copy':'runtime_v54.py','local_copy_sha256':sha(OUT/'runtime_v54.py'),'interface_source_sha256':sha(OUT/'decoder_interface.py'),'strict_full_shape_and_dtype_before_cache_write':True,'supported_returns':['Tensor[B,T,H]','one-element tuple(Tensor[B,T,H])'],'unknown_type':'hard error'}
p['restore_boundary']={'base_model_sha256':rd(V53/'VALID_RESUME_BOUNDARY.json')['base_model_sha256'],'inherited_block0_groups':[{'group':g,'path':x['path'],'sha256':x['sha256']} for g,x in zip(GROUPS,rd(V53/'VALID_RESUME_BOUNDARY.json')['block0_groups_to_restore_without_repair'])],'repaired_from_layer':'layer1/qkv','v52_layer1_plus_parameters_inputs_and_C_forbidden':True,'v53_correct_layer1_qkv_input_sha256':rd(V53/'VALID_RESUME_BOUNDARY.json')['expected_boundary']['layer1_qkv_X_raw_content_sha256']}
p['propagation_validation']={'articles':[0,64,127],'tokens_each':2048,'after_each_block':True,'reference':'native complete model.forward plus independent next-block/qkv prehooks (final block uses final norm prehook)','reference_uses_v53_extractor':False,'comparison':'same v53 bitwise-first and rtol=1e-5, atol=1e-6 fallback','no_tolerance_changes':True,'all24_blocks':True}
p['budget_v54']={'shared_cap_seconds':10800,'inherited_shared_seconds':5677.428213768006,'starting_remaining_seconds':5122.571786231994,'finish_and_evaluation_reserve_seconds':360,'historical_M2_blocks1_23_seconds':4353.960454899992,'three_article_per_block_validation_baseline_seconds':9.246,'uncertainty_reserve_seconds':120,'projection_is_estimate':True}
wr(OUT/'PROTOCOL.json',p)
# Copy only authorized v51 block0 groups; wrap contents in v54 protocol containers and preserve source chain.
v53groups=rd(V53/'BLOCK0_GROUP_SOURCE_CHECKS.json')['groups']
v51prot=sha(V51/'PROTOCOL.json')
v54prot=sha(OUT/'PROTOCOL.json')
inherits=[]
for g in GROUPS:
 src=V51/f'formal/groups/layer_00_{g}.pt'
 dst=OUT/f'formal/groups/layer_00_{g}.pt'
 rec=next(x for x in rd(V53/'VALID_RESUME_BOUNDARY.json')['block0_groups_to_restore_without_repair'] if x['group']==g)
 assert sha(src)==rec['sha256']
 if dst.exists():
  cp=torch.load(dst,map_location='cpu',weights_only=False)
  assert cp.get('inherited_from_v53_approved_v51',{}).get('source_sha256')==rec['sha256']
 else:
  cp=torch.load(src,map_location='cpu',weights_only=False)
 assert cp['committed'] and cp['layer']==0 and cp['group']==g and (cp['protocol_sha256']==v51prot or cp.get('inherited_from_v53_approved_v51',{}).get('source_sha256')==rec['sha256'])
 grp=next(x for x in v53groups if x['group']==g)
 assert cp['input_sha256']==grp['native_reference_input_sha256']==grp['historical_source_input_sha256']
 cp['inherited_from_v53_approved_v51']=dict(version='v51',path=str(src),source_sha256=sha(src),source_protocol_sha256=v51prot,v53_independent_input_gate=True,not_repaired_again=True)
 cp['protocol_sha256']=v54prot
 torch.save(cp,dst)
 inherits.append({'group':g,'source_path':str(src),'source_sha256':sha(src),'v54_group_sha256':sha(dst),'input_sha256':cp['input_sha256'],'C_sha256_source_record_only':cp['C_sha256'],'modules':list(cp['modules']),'inherited_without_repair':True})
assert not list((OUT/'formal/groups').glob('layer_01_*'))
wr(OUT/'BLOCK0_INHERITANCE.json',{'pass':True,'v53_report_sha256':sha(V53/'REPORT.md'),'v53_resume_boundary_sha256':sha(V53/'VALID_RESUME_BOUNDARY.json'),'v53_gpu_result_sha256':sha(V53/'GPU_VALIDATION_RESULT.json'),'base_M2_sha256':p['initialization']['model_sha256'],'block0_groups':inherits,'v52_layer1_plus_loaded':False,'v52_activations_or_C_loaded':False})
# New runner is a narrow v52 runner copy: only runtime/output/protocol/cost gate and required native propagation checks change.
s=(V52/'run_v52.py').read_text(encoding='utf8')
s=s.replace("sys.path.insert(0,str(ROOT));sys.path.insert(0,str(V46));sys.path.insert(0,str(V48))", "sys.path.insert(0,str(ROOT));sys.path.insert(0,str(V46));sys.path.insert(0,str(V48));sys.path.insert(0,str(OUT))",1)
s=s.replace("OUT=BASE/'v52_m2_fast_integer_repair_fp64_initial_review';V46=", "OUT=BASE/'v54_m2_decoder_fixed_integer_repair';V52=BASE/'v52_m2_fast_integer_repair_fp64_initial_review';V53=BASE/'v53_decoder_interface_propagation_audit';V46=",1)
s=s.replace("CAP=10800;FINISH_RESERVE=600;", "CAP=10800;FINISH_RESERVE=360;UNCERTAINTY_RESERVE=120;",1)
s=s.replace("rt=loadmod('run_v42',V46/'runtime_v46.py')", "rt=loadmod('run_v42',OUT/'runtime_v54.py')",1)
s=s.replace("RUN_ID='v52-'+sha(OUT/'PROTOCOL.json')[:16]", "RUN_ID='v54-'+sha(OUT/'PROTOCOL.json')[:16]+'-'+str(os.getpid())",1)
s=s.replace("assert read('PREPARATION.json')['status']=='CPU_SOURCE_AUDIT_PASS' and read('PROTOCOL.json')['status']=='FROZEN_BEFORE_GPU'", "assert read('PREPARATION.json')['status']=='CPU_SOURCE_AUDIT_PASS' and read('PROTOCOL.json')['status']=='FROZEN_BEFORE_GPU' and read('BLOCK0_INHERITANCE.json')['pass']",1)
s=s.replace("rt=loadmod('run_v42',OUT/'runtime_v54.py');rt.OUT=OUT", "rt=loadmod('run_v42',OUT/'runtime_v54.py');rt.OUT=OUT",1)
s=s.replace("RT_PLACEHOLDER", "RT_PLACEHOLDER")
s=s.replace("dump('RUNTIME_BINDING.json',{'pass':rt.batch_repair is bound_fast,'runtime_dispatch':'v49.bound_fast'", "dump('RUNTIME_BINDING.json',{'pass':rt.batch_repair is bound_fast,'decoder_runtime':str(OUT/'runtime_v54.py'),'decoder_runtime_sha256':sha(OUT/'runtime_v54.py'),'runtime_dispatch':'v49.bound_fast'",1)
s=s.replace("'v46 evaluator key prebound", "'v46 evaluator key prebound")
s=s.replace("rt=loadmod('run_v42',OUT/'runtime_v54.py');rt.OUT=OUT", "rt=loadmod('run_v42',OUT/'runtime_v54.py');rt.OUT=OUT",1)
s=s.replace("dump('EVALUATOR_PREFLIGHT.json',{'pass':True,'missing_v46_P_v18_key_supplied_in_memory':True,'manifest_hashes':hashes,'evaluator_source_sha256':sha(V46/'runtime_v46.py')", "dump('EVALUATOR_PREFLIGHT.json',{'pass':True,'missing_v46_P_v18_key_supplied_in_memory':True,'manifest_hashes':hashes,'evaluator_source_sha256':sha(OUT/'runtime_v54.py')",1)
s=s.replace("dict(start_epoch=GPU_START,cap_seconds=CAP,pid=os.getpid(),run_py_sha256=sha(__file__),fast_accept_sha256=FAST_SHA,protocol_sha256=sha(OUT/'PROTOCOL.json'),binding='v52 local copy of exact v48 core plus only initial check fallback',requested_model_setting='Sol high requested; interface cannot verify',prior_v51_charged_seconds=657.305,active_job_ledger='ACTIVE_JOB_LEDGER.json')", "dict(start_epoch=GPU_START,cap_seconds=CAP,pid=os.getpid(),run_py_sha256=sha(__file__),fast_accept_sha256=FAST_SHA,protocol_sha256=sha(OUT/'PROTOCOL.json'),binding='v53 strict decoder runtime + unchanged v52 initialization review and v49/v48 fast repair',requested_model_setting='Luna Max requested; interface cannot verify',prior_shared_charged_seconds=budget.seconds()[0],active_job_ledger='ACTIVE_JOB_LEDGER.json')",1)
s=s.replace("version='v52',start_unix=GPU_START", "version='v54',start_unix=GPU_START",1)
# The calibration capture wrapper verifies the frozen v53 source hashes; it never uses id(X) and does not share/reuse C.
s=s.replace("CURRENT_SOURCE_KEY=None\nORIGINAL_CAPTURE", "CURRENT_SOURCE_KEY=None\nV53_GROUP_HASHES={x['group']:x['native_reference_input_sha256'] for x in read(V53/'BLOCK0_GROUP_SOURCE_CHECKS.json')['groups']}\nV53_BOUNDARY=read(V53/'VALID_RESUME_BOUNDARY.json')['expected_boundary']\nORIGINAL_CAPTURE",1)
s=s.replace("with rt.timer('v52_input_source_hash'):\n  key=(RUN_ID,PHASE,int(li),str(group));CURRENT_SOURCE_KEY=key\n  SOURCE_RECORDS[key]=dict(run_id=RUN_ID,phase=PHASE,block=int(li),dependency_group=str(group),input_sha256=xhash(X),input_shape=list(X.shape),input_dtype=str(X.dtype))", "with rt.timer('v54_input_source_hash'):\n  xh=xhash(X)\n  if li==0:assert xh==V53_GROUP_HASHES[group],f'v53 approved block0 input mismatch: {group}'\n  if li==1 and group=='qkv':assert xh==V53_BOUNDARY['layer1_qkv_X_raw_content_sha256'], 'corrected full-token layer1/qkv boundary mismatch'\n  key=(RUN_ID,PHASE,int(li),str(group));CURRENT_SOURCE_KEY=key\n  SOURCE_RECORDS[key]=dict(run_id=RUN_ID,phase=PHASE,block=int(li),dependency_group=str(group),input_sha256=xh,input_shape=list(X.shape),input_dtype=str(X.dtype),v53_block0_source_match=(li==0),v53_layer1_boundary_match=(li==1 and group=='qkv'))",1)
s=s.replace("assert m['C_sha256']==tensor_hash(C)", "assert m['C_sha256']==tensor_hash(C)",1)
s=s.replace("'input_hash', 'stage'", "'input_hash', 'stage'")
# Add the independently hooked, complete native-forward check.
marker="\ndef main():\n global GPU_START,ORIG"
helper=r'''
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
'''
assert marker in s
s=s.replace(marker,'\n'+helper+marker,1)
# Validate new local implementation and unchanged numeric code/runtime binding before job start.
s=s.replace("assert (OUT/'R1/checkpoint/final').is_dir() and (OUT/'formal/groups').is_dir() and (OUT/'evaluation').is_dir()", "assert (OUT/'R1/checkpoint/final').is_dir() and (OUT/'formal/groups').is_dir() and (OUT/'evaluation').is_dir()\n assert read('FROZEN_VALIDATION_LOCK.json')['pass'] and read('V54_COST_PRECHECK.json')['go']\n for n,h in read('FROZEN_VALIDATION_LOCK.json')['files'].items():assert sha(OUT/n)==h,f'frozen v54 source changed: {n}'\n assert sha(V53/'runtime_v53.py')==read('FROZEN_VALIDATION_LOCK.json')['v53_runtime_sha256']\n assert sha(V53/'GPU_VALIDATION_RESULT.json')==read('FROZEN_VALIDATION_LOCK.json')['v53_validation_result_sha256']",1)
# Preserve v52 initialization review and frozen shared core.
# Capture block native check after each repaired block, before it is counted as a completed block.
oldline="start=time.perf_counter();hidden=rt.run_layer(model,li,hidden);block_seconds=time.perf_counter()-start"
newline="start=time.perf_counter();hidden=rt.run_layer(model,li,hidden);block_seconds=time.perf_counter()-start\n   propagation_seconds,native_rows=verify_native_boundary(model,li,hidden)"
assert oldline in s;s=s.replace(oldline,newline,1)
s=s.replace("dump('BLOCK0_GATE.json',{'pass':True,'role':'formal block, repaired once'", "dump('BLOCK0_GATE.json',{'pass':True,'role':'v51 block0 restored after v53 gate; not repaired again in v54'",1)
# Replace too-conservative v52/M1 projected gate with frozen M2 block timing + v53 measured propagation reserve.
lo=s.index("   if li==0:\n    verify_block0();")
hi=s.index("   print('V49_BLOCK_COMPLETE'",lo)
costgate=r'''   if li==0:
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
'''
s=s[:lo]+costgate+s[hi:]
# Always actually evaluate this new model; do not take v52's equivalent-R2 PPL reuse branch.
start=s.index("  audit=read(V46/'PPL_SOURCE_AUDIT.json');copies=[]")
end=s.index("  if model is not None:del model",start)
evalblock=r'''  copies=[]
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
'''
s=s[:start]+evalblock+s[end:]
s=s.replace("'PPL_run':not equal,'PPL_reused':equal", "'PPL_run':True,'PPL_reused':False",1)
s=s.replace("print('V49_BLOCK_COMPLETE'", "print('V54_BLOCK_COMPLETE'",1)
s=s.replace("'status':'RUNNING'", "'status':'RUNNING'",1)
# Stamp all new RUN_STATUS / COMPLETE version details where sensible.
s=s.replace("dict(version='v52',start_unix", "dict(version='v54',start_unix")
# Keep local legacy R1 output naming explicit; report explains this path alias is internal to reused frozen evaluator.
(OUT/'run_v54.py').write_text(s,encoding='utf8')
compile(s,str(OUT/'run_v54.py'),'exec')
# Budget carries v51+v52+v53 exact original ledger; only v54 receives a new segment.
shutil.copy2(V53/'ACTIVE_JOB_LEDGER.json',OUT/'ACTIVE_JOB_LEDGER.json')
prior=sha(V53/'ACTIVE_JOB_LEDGER.json')
b="""import atexit,json,os,time\nfrom pathlib import Path\nO=Path(__file__).resolve().parent;P=O/'ACTIVE_JOB_LEDGER.json';ACTIVE=None;CAP=10800.;VERSION='v54'\ndef load():return json.loads(P.read_text())\ndef save(x):\n t=P.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\\\n');t.replace(P)\ndef seconds():\n x=load();vals=[(s.get('version'),s.get('charged_seconds',max(0,time.time()-s['start_epoch']))) for s in x['segments']];return x['v51_charged_seconds']+sum(v for _,v in vals),sum(v for k,v in vals if k==VERSION)\ndef start(label):\n global ACTIVE\n x=load();assert all('charged_seconds' in s for s in x['segments']);total,local=seconds();assert total<CAP\n ACTIVE=len(x['segments']);x['segments'].append(dict(version=VERSION,label=label,pid=os.getpid(),start_epoch=time.time()));save(x);atexit.register(finish)\ndef finish():\n global ACTIVE\n if ACTIVE is None:return\n x=load();s=x['segments'][ACTIVE]\n if 'charged_seconds' not in s:s.update(end_epoch=time.time());s['charged_seconds']=s['end_epoch']-s['start_epoch'];save(x)\n ACTIVE=None\n"""
(OUT/'budget.py').write_text(b.replace('\n','\n'),encoding='utf8')
# Update protocol after exact source copies and generated runner exist.
p=rd(OUT/'PROTOCOL.json')
p['budget_v54']['parent_ledger_path']=str(V53/'ACTIVE_JOB_LEDGER.json')
p['budget_v54']['parent_ledger_sha256']=prior
p['budget_v54']['finish_reserve_seconds']=360
p['budget_v54']['uncertainty_reserve_seconds']=120
p['budget_v54']['same_initializer_layer_timings_source']=str(V52/'BLOCK_TIMINGS.csv')
p['budget_v54']['block_native_check_overhead_source']='v53 STAGE_TIMINGS fixed_complete_articles_all24_layers; measured synchronized wall 9.246 s'
p['budget_v54']['new_job_hard_stop_seconds']=10800
wr(OUT/'PROTOCOL.json',p)
# Rebind inherited v51 group containers to final v54 protocol hash after the protocol is finalized.
newph=sha(OUT/'PROTOCOL.json')
for g in GROUPS:
 q=OUT/f'formal/groups/layer_00_{g}.pt';cp=torch.load(q,map_location='cpu',weights_only=False);cp['protocol_sha256']=newph;torch.save(cp,q)
inher=rd(OUT/'BLOCK0_INHERITANCE.json')
for rec in inher['block0_groups']:
 q=OUT/f"formal/groups/layer_00_{rec['group']}.pt";rec['v54_group_sha256']=sha(q)
wr(OUT/'BLOCK0_INHERITANCE.json',inher)
# Keep source-origin / output checkpoints from v53 and all audited history unchanged.
wr(OUT/'INITIALIZATION_REVIEW_REUSE.json',{'pass':True,'source_v52_protocol_sha256':sha(V52/'PROTOCOL.json'),'fast_core_source':str(V52/'fast_accept_v52.py'),'fast_core_sha256':sha(V52/'fast_accept_v52.py'),'review_source':str(V52/'initialization_review.py'),'review_sha256':sha(V52/'initialization_review.py'),'numerical_validation_sha256':sha(V52/'NUMERICAL_VALIDATION.json'),'candidate_order_grid_acceptance_and_budget_unchanged':True,'v53_runtime_source_sha256':sha(V53/'runtime_v53.py'),'note':'reuse exact frozen initialization fallback; historical row122 fixture is only a regression record, new formal inputs are checked at live run'})
# Exact PPL source hashes and model/revision protocol.
hist=rd(V53/'HISTORICAL_CHECKPOINT_SOURCES.json')
wr(OUT/'HISTORICAL_SOURCE_LOCK.json',{'v18_M2_checkpoint_sha256':p['initialization']['model_sha256'],'v18_M1_checkpoint_sha256':hist['M1']['sha256'],'v45_R2_checkpoint_sha256':hist['v45_original_R2']['sha256'],'v46_R1_checkpoint_sha256':hist['v46_R1']['sha256'],'v49_fast_R1_checkpoint_sha256':hist['v49_fast_R1']['sha256'],'calibration_manifest_sha256':p['calibration']['manifest_sha256'],'v53_report_sha256':sha(V53/'REPORT.md'),'v53_boundary_sha256':sha(V53/'VALID_RESUME_BOUNDARY.json'),'v53_runtime_sha256':sha(V53/'runtime_v53.py'),'old_historical_return_type_classification':'v45/v46/v49 UNKNOWN; retain measured PPL as actual checkpoint measurements only'})
# CPU preparation and budget gate derived solely from same-initializer and frozen validation timing records.
with (V52/'BLOCK_TIMINGS.csv').open(encoding='utf-8-sig',newline='') as f: bt=list(csv.DictReader(f))
repair=sum(float(x['seconds']) for x in bt if int(x['layer'])>=1)
with (V53/'GPU_VALIDATION_TIMINGS.csv').open(encoding='utf-8-sig',newline='') as f: st=list(csv.DictReader(f))
checks=[float(x['synchronized_seconds']) for x in st if x['stage']=='fixed_complete_articles_all24_layers']
assert len(checks)==1
prop=checks[0]*(24-1)
restore=17.0 # v52 measured committed_group_prefix_restore_validation for block0, rounded up.
uncertainty=120.;finish=360.
budget=rd(V53/'ACTIVE_JOB_LEDGER.json');start=budget['v51_charged_seconds']+sum(x['charged_seconds'] for x in budget['segments'])
remaining=10800-start
projected=repair+prop+restore+uncertainty+finish
pre={'pass':True,'go':projected<=remaining,'historical_same_initializer_repair_layers1_23_seconds':repair,'measured_v53_three_article_full24_propagation_seconds_per_state':checks[0],'remaining_23_state_propagation_projection_seconds':prop,'historical_block0_restore_input_validation_seconds':restore,'uncertainty_reserve_seconds':uncertainty,'finalization_and_two_PPL_reserve_seconds':finish,'projected_total_required_seconds':projected,'starting_shared_elapsed_seconds':start,'starting_shared_remaining_seconds':remaining,'projected_headroom_seconds':remaining-projected,'projection_not_measurement':True,'method':'same M2 initialization v52 measured layer1-23 block wall + v53 full24 fixed-article reference timer for 23 later repaired states + measured/rounded restore + historic120s reserve + required360s finish reserve','limitation':'v52 repair timing used invalid later inputs; it is an execution estimate only. Online gate replaces each projected term with current measured wall when larger.'}
wr(OUT/'V54_COST_PRECHECK.json',pre)
assert pre['go'], 'conservative preflight says the 360-second finish reserve cannot be kept; no GPU started'
wr(OUT/'PREPARATION.json',{'status':'CPU_SOURCE_AUDIT_PASS','version':'v54','directories_precreated_and_writable':True,'original_M2_and_calibration_hashes_verified_inherited':True,'m2_original_code_cache_files':len(code_rows),'m2_original_code_cache':code_rows,'v53_runtime_byte_equal':True,'only_four_v51_block0_groups_present':True,'v52_layer1plus_params_inputs_C_not_copied':True,'CPU_only_preparation':True,'cpu_seconds':None})
# Freeze source/runtime/algorithm and new runner hashes before GPU. Note runtime is byte-identical to v53.
freeze_files=['PROTOCOL.json','run_v54.py','runtime_v54.py','decoder_interface.py','audit_sources.py','fast_accept_v52.py','initialization_review.py','budget.py','INITIALIZATION_REVIEW_REUSE.json','BLOCK0_INHERITANCE.json','V54_COST_PRECHECK.json','PREPARATION.json']
wr(OUT/'FROZEN_RUN_LOCK.json',{'pass':True,'files':{n:sha(OUT/n) for n in freeze_files},'v53_runtime_sha256':sha(V53/'runtime_v53.py'),'v53_validation_result_sha256':sha(V53/'GPU_VALIDATION_RESULT.json'),'v53_boundary_sha256':sha(V53/'VALID_RESUME_BOUNDARY.json'),'v52_core_sha256':sha(V52/'fast_accept_v52.py'),'v52_initialization_review_sha256':sha(V52/'initialization_review.py'),'v52_acceptance_rules_sha256':sha(V48/'ACCEPTANCE_RULES.md'),'algorithm_and_tolerance_change':False,'no_v50_gram':True,'no_held_or_P3':True})
# No v52 layer1+ activation/C/repair state is materialized in the new version.
assert {x.name for x in (OUT/'formal/groups').glob('*.pt')}=={f'layer_00_{g}.pt' for g in GROUPS}
# Manifest covers the prepared setup; later records/checkpoints append in v54.
manifest=[]
for q in sorted(OUT.rglob('*')):
 if q.is_file() and q.name not in ['DELIVERY_MANIFEST.json','prepare.log'] and '__pycache__' not in q.parts:
  manifest.append({'path':str(q.relative_to(OUT)),'sha256':sha(q),'bytes':q.stat().st_size})
wr(OUT/'DELIVERY_MANIFEST.json',{'status':'PREPARED_FROZEN','files':manifest})
print(json.dumps({'prepared':True,'precheck':pre,'block0':inherits,'v53_parent_ledger_sha256':prior},ensure_ascii=False))
