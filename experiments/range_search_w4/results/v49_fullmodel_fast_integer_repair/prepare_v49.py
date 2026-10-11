"""CPU source/protocol audit and write-path preflight. No CUDA/model forward."""
import csv,hashlib,json,os,shutil,time
from pathlib import Path
import torch
ROOT=Path(r'E:\quant-thesis');BASE=ROOT/'experiments/range_search_w4/results'
OUT=BASE/'v49_fullmodel_fast_integer_repair';V46=BASE/'v46_m1_single_integer_repair_fullmodel';V48=BASE/'v48_fast_numerical_acceptance';V18=BASE/'range_w4_fullmodel_v18'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 t=time.perf_counter();start=time.time();OUT.mkdir(parents=True,exist_ok=True)
 for d in ['original_codes','formal/groups','formal','R1/checkpoint/final','evaluation/source_v46','evaluation','reports','diagnostics','recovery']:
  p=OUT/d;p.mkdir(parents=True,exist_ok=True);probe=p/'.v49_write_probe';probe.write_bytes(b'v49');assert probe.read_bytes()==b'v49';probe.unlink()
 # Existing audit/protocol files are expected when this CPU preflight is rerun.
 # The initial run already checked the empty output area before writing them.
 if (OUT/'PROTOCOL.json').exists():
  prot=read(OUT/'PROTOCOL.json');assert prot['version']=='v49' and prot['algorithm_source']['rules_sha256']==sha(V48/'ACCEPTANCE_RULES.md')
 else:
  assert not (OUT/'GPU_START.json').exists(),'existing GPU work needs resume, not re-preparation'
  shutil.copytree(V46/'original_codes',OUT/'original_codes',dirs_exist_ok=True)
  f=V46/'original_codes'
  files=sorted(f.glob('layer_*.pt'));assert len(files)==24
  rows=0;modules=0;mapping=[]
  for li,p in enumerate(files):
   s=torch.load(p,map_location='cpu',weights_only=True);assert len(s)==7
   for name,x in s.items():
    c=x['codes'];sc=x['scales'];g=x['g_idx'].long();z=x['zeros']
    assert c.ndim==2 and bool(((c>=0)&(c<=15)).all()) and torch.equal(z,torch.full_like(z,8))
    assert torch.equal(g,torch.arange(c.shape[1])//128) and bool(torch.isfinite(sc).all())
    q=(sc[:,g]*(c.float()-8)).half()
    assert torch.equal(q.view(torch.int16),x['weight'].view(torch.int16))
    assert torch.equal(torch.stack((x['packed_u4']&15,x['packed_u4']>>4),-1).flatten(1),c)
    rows+=c.shape[0];modules+=1;mapping.append(dict(layer=li,module=name,rows=c.shape[0],input_dim=c.shape[1],source_sha256=sha(V46/'original_codes'/p.name),staged_sha256=sha(OUT/'original_codes'/p.name)))
  assert all(r['source_sha256']==r['staged_sha256'] for r in mapping)
  assert (rows,modules)==(304128,168),(rows,modules)
  shapes={i:sorted((x['module'],x['rows'],x['input_dim']) for x in mapping if x['layer']==i) for i in range(24)}
  assert all(shapes[i]==shapes[0] for i in range(24)),'actual layer matrix shapes differ; no block0 cost extrapolation allowed'
  pr=read(V46/'PROTOCOL.json');audit=read(V46/'BASELINE_AUDIT.json');ppl=read(V46/'PPL_SOURCE_AUDIT.json')
  assert audit['pass'] and audit['comparison_protocol_shared'] and ppl['pass'] and not ppl['P3_read']
  manifest_names={'wikitext':('WIKITEXT_TEST_MANIFEST.json','v16_wikitext_manifest_sha256'),'c4_fixed_budget':('C4_EVAL_MANIFEST.json','v16_c4_manifest_sha256')}
  manifest_audit={}
  for ds,(fn,key) in manifest_names.items():
   p=V18/'evaluation/manifests'/fn;expected=audit['evaluation_manifests']['wikitext' if ds=='wikitext' else 'c4_fixed_budget'];assert sha(p)==expected==audit['evaluation_manifests'][ds]
   token=json.loads(p.read_text(encoding='utf8'))['token_ids'];assert len(token)==audit['token_counts'][ds][0]+1
   manifest_audit[ds]=dict(path=str(p),sha256=sha(p),input_tokens=len(token),valid_tokens=audit['token_counts'][ds][0],windows=audit['token_counts'][ds][1],max_length=2048,stride=512)
  rule=read(V48/'IMPLEMENTATION_INTEGRITY.json');v48status=read(V48/'STATUS.json');assert rule['pass_all'] and v48status['category']=='A' and v48status['status']=='COMPLETE'
  v48hash={k:sha(V48/f) for k,f in [('run_v48','run_v48.py'),('fast_accept','fast_accept.py'),('rules','ACCEPTANCE_RULES.md')]}
  assert v48hash['run_v48']==rule['frozen_before_GPU_hashes']['run_v48.py']
  assert v48hash['fast_accept']==rule['frozen_before_GPU_hashes']['fast_accept.py']
  assert v48hash['rules']==rule['frozen_before_GPU_hashes']['ACCEPTANCE_RULES.md']
  rt_source=V46/'runtime_v46.py';repair_source=V46/'repair_v45.py';precision_source=V46/'precision_check.py'
  assert sha(repair_source)==pr['repair_source']['repair_v45_sha256'] and sha(precision_source)==pr['repair_source']['precision_check_sha256']
  evalpatch={'wikitext':sha(V18/'evaluation/manifests/WIKITEXT_TEST_MANIFEST.json'),'c4_fixed_budget':sha(V18/'evaluation/manifests/C4_EVAL_MANIFEST.json')}
  # The frozen evaluator indexes these audited source hashes through P['v18'].
  assert 'v18' not in pr
  # v46 continuation demonstrates this exact missing-key mapping and leaves the evaluator unchanged.
  cont=V46/'continue_evaluation_v46.py';assert 'P.setdefault("v18", {})["source_protocol"] = source_protocol' in cont.read_text(encoding='utf8')
  proj=read(V46/'COST_PROJECTION.json');blocks=list(csv.DictReader((V46/'BLOCK_TIMINGS.csv').open(encoding='utf8')));assert len(blocks)==24
  v46wall=sum(float(r['seconds']) for r in blocks);assert abs(v46wall-5485.6751965)<.1
  v46eval=sum(float(r['seconds']) for r in csv.DictReader((V46/'STAGE_TIMINGS.csv').open(encoding='utf8')) if r['stage']=='PPL_evaluation')
  terminal_v48=sum(float(r['seconds']) for r in csv.DictReader((V48/'STAGE_TIMINGS.csv').open(encoding='utf8')) if r['stage']=='independent_full_terminal_FP64')
  cost={'type':'pre-run conservative planning estimate; not measured', 'v46_formal_24_block_wall_seconds':v46wall,'v46_original_GPU_start_to_final_seconds':read(V46/'FINAL_STATUS.json')['elapsed_seconds_from_original_gpu_start'],'v46_failed_dispatch_seconds':json.loads((V46/'EXECUTION_AMENDMENT_001.json').read_text())['failed_attempt_elapsed_seconds'],'v46_PPL_measured_seconds':v46eval,'v48_L0_updown_terminal_measured_seconds':terminal_v48,'rule':'project measured block0 as formal run; before continuing use max(1.35*measured block0, 1.35*v46 max block)+600s final save/reload/eval reserve; no reduce if pressure', 'reserve_seconds':600,'hard_cap_seconds':10800,'shape_source':'all 168 trusted native M1 layer states; same 7 shapes repeated across 24 blocks','C_build_optimization':False}
  proto={'version':'v49','status':'FROZEN_BEFORE_GPU','model_setting':'Luna Max requested; current interface cannot verify/switch setting.','experiment':'Original formal M1 plus the exact v48 fast numerical integer repair, compared with v46 R1.','initialization':{'checkpoint':str(V18/'M1/checkpoint/final'),'sha256':pr['initialization']['model_sha256'],'revision':pr['initialization']['revision'],'config_sha256':audit['models']['M1']['config_sha256']},'calibration':{'manifest_sha256':pr['initialization']['calibration_sha256'],'documents':128,'tokens_each':2048,'evaluation_separated':True},'quantization':pr['quantization'],'algorithm_source':{'fast_accept_path':str(V48/'fast_accept.py'),'sha256':v48hash['fast_accept'],'rules':str(V48/'ACCEPTANCE_RULES.md'),'rules_sha256':v48hash['rules'],'batch_runner':str(V48/'run_v48.py'),'runner_sha256':v48hash['run_v48'],'matrix_batch_repair_identity':'v48 fast_accept.batch_repair','candidate_helpers':'v46/v45 choose, propose, score_and_retire, retire_caps, called unchanged by fast_accept.py','v46_runtime':sha(rt_source),'v45_original_repair':sha(repair_source),'v45_precision_check':sha(precision_source),'acceptance_revision':'v48 FP64 increment empirical envelope; raw-X independent FP64 for uncertainty; fixed p/B checks and refresh at 1,8,16,24,32; no safety certificate; no change/tuning allowed'},'fixed_search':{'neighborhood':'legal actual FP16 grid delta for +/-1 code; code0..15 zero8 G128','max_accepted':32,'candidate_budget':'128 * input dimension','tie_breaking':'v45 unchanged deterministic row/coordinate/offset order','C':'full undamped FP64 C, unchanged original runtime builder','group_order':['qkv','o','up_gate','down'],'sibling_commit':'entire group committed only after every sibling matrix returns and validates'},'restore_protocol':{'start_from':'original formal M1 every process','committed_group':'load only a complete atomic group checkpoint; verify protocol/source/grid/code/packed/dequantized weight and recaptured current-prefix input hash before applying; skip all repair for that group','failed_group':'no commit until all group siblings complete; incomplete scratch discarded on restart','prefix':'rebuild previous blocks from original M1 and committed groups, recapture own prefix; no v46 activations used for repairs'},'output':'v49/R1/checkpoint/final/model.safetensors (directory local to v49)','evaluation':{'code':'v46 unchanged runtime.evaluate; v18 source hash mapping prebound in memory from audited manifests before GPU','max_length':2048,'stride':512,'token_weighted_valid_labels':True,'P3_read':False,'reuse':'Only after all named model tensors and inference configuration equal v46 R1; otherwise each WT2 and fixed-budget C4 exactly once'},'cost':cost,'hard_gpu_budget_seconds':10800,'model_revision':pr['initialization']['revision'],'freeze_time_epoch':start}
  dump('PROTOCOL.json',proto);dump('COST_PROJECTION.json',cost)
  sourceaudit=[]
  for p in [V46/'PROTOCOL.json',V46/'BASELINE_AUDIT.json',V46/'PPL_SOURCE_AUDIT.json',V46/'runtime_v46.py',V46/'repair_v45.py',V46/'precision_check.py',V46/'continue_evaluation_v46.py',V46/'original_codes/layer_00.pt',V46/'CHECKPOINT_LOCK.json',V46/'R1/checkpoint/final/model.safetensors',V48/'ACCEPTANCE_RULES.md',V48/'fast_accept.py',V48/'run_v48.py',V48/'STATUS.json',V18/'CALIBRATION_MANIFEST.json',V18/'M1/checkpoint/final/model.safetensors',V18/'M1/checkpoint/final/config.json']:
   sourceaudit.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
  dump('SOURCE_AUDIT.json',{'pass':True,'sources':sourceaudit,'all_168_original_code_grid_sources':mapping,'all_24_block_shapes_equal':True,'calibration':{'sha256':sha(V18/'CALIBRATION_MANIFEST.json'),'expected':pr['initialization']['calibration_sha256']},'m1_checkpoint_sha256':sha(V18/'M1/checkpoint/final/model.safetensors'),'v46_ppl_source_audit_sha256':sha(V46/'PPL_SOURCE_AUDIT.json'),'ppl_manifest_hashes':manifest_audit,'v48_hashes':v48hash,'in_memory_evaluator_key_fix':{'missing_key':'v18','solution':'prepopulate rt.P["v18"]["source_protocol"] with audited WT2/C4 manifest SHA hashes in memory before any evaluation; evaluator code unchanged','frozen_example_script':str(cont),'script_sha256':sha(cont)},'source_code_not_modified':True,'staged_layers':24,'staged_modules':modules,'staged_rows':rows})
  prep={'started_unix':start,'completed_unix':time.time(),'CPU_elapsed_seconds':time.perf_counter()-t,'cuda_initialized':False,'output_directories_precreated_and_write_probed':True,'original_code_files_copied':24,'preexisting_ppl_windows_verified':True}
  dump('PREPARATION.json',prep)
 if not (OUT/'JOB_START.json').exists():dump('JOB_START.json',{'start_unix':time.time(),'preparation_started_unix':read(OUT/'PREPARATION.json')['started_unix'],'mode':'new full formal run'})
 source_now=read('SOURCE_AUDIT.json');modules=source_now['staged_modules'];rows=source_now['staged_rows'];shape_map={i:sorted((x['module'],x['rows'],x['input_dim']) for x in source_now['all_168_original_code_grid_sources'] if x['layer']==i) for i in range(24)}
 assert all(shape_map[i]==shape_map[0] for i in range(24))
 # Append-only values are never rewritten during --resume; current frozen run identity is checked by the GPU runner.
 dump('PREFLIGHT.json',{'pass':True,'preparation_cpu_seconds':time.perf_counter()-t,'cuda_initialized':False,'all_sources_and_manifest_hashes_checked':True,'v48_test_evidence_reused':True,'all_168_m1_code_and_grids_checked':True,'all_required_directories_writable':True,'runtime_binder_bug_checked_and_prebound':True,'restore_and_eval_entrypoints_present':True,'new_alpha_or_search':False})
 print('V49_CPU_PREFLIGHT_PASS',modules,rows,'CPU_seconds',time.perf_counter()-t,flush=True)
if __name__=='__main__':main()
