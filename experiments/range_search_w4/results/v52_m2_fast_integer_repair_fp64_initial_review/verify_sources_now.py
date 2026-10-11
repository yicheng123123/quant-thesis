import hashlib,json,time
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parent;A=json.loads((O/'SOURCE_AUDIT.json').read_text());t=time.perf_counter()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
checks=[]
def check(p,h):
 got=sha(p);assert got==h,str(p);checks.append(dict(path=str(p),expected_sha256=h,actual_sha256=got,unchanged=True))
check(B/'range_w4_fullmodel_v18/M2/checkpoint/final/model.safetensors',A['m2_model_sha256'])
check(B/'range_w4_fullmodel_v18/M2/checkpoint/final/config.json',A['m2_config_sha256'])
check(B/'range_w4_fullmodel_v18/CALIBRATION_MANIFEST.json',A['calibration_manifest_sha256'])
for q in A['pristine_fp_weight_files']:check(q['path'],q['sha256'])
m={'v49_run_v49_sha256':B/'v49_fullmodel_fast_integer_repair/run_v49.py','v48_fast_accept_sha256':B/'v48_fast_numerical_acceptance/fast_accept.py','v48_acceptance_rules_sha256':B/'v48_fast_numerical_acceptance/ACCEPTANCE_RULES.md','v48_runner_sha256':B/'v48_fast_numerical_acceptance/run_v48.py','v46_runtime_sha256':B/'v46_m1_single_integer_repair_fullmodel/runtime_v46.py','v45_repair_helpers_sha256':B/'v46_m1_single_integer_repair_fullmodel/repair_v45.py'}
for k,p in m.items():check(p,A['source_files'][k])
for i in A['m2_original_code_files']:
 check(i['m2_native_checkpoint'],i['native_checkpoint_sha256']);check(O/f"original_codes/layer_{i['layer']:02d}.pt",i['v51_staged_sha256'])
(O/'SOURCE_REVERIFICATION.json').write_text(json.dumps({'pass':True,'checks':checks,'CPU_wall_seconds':time.perf_counter()-t,'GPU_calls':0,'no_source_or_state_changed':True,'copied_v51_source_audit_reverified_not_assumed':True},indent=2)+'\n')
print('CURRENT_SOURCE_REVERIFICATION_PASS',len(checks),time.perf_counter()-t)
