"""CPU-only trusted checkpoint/source audit and isolated staging."""
import json,shutil,time
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent;V44=OUT.parent/'v44_m2_single_integer_repair_fullmodel';V43=OUT.parent/'v43_m2_single_integer_repair_fullmodel'
def main():
 t=time.perf_counter()
 for d in ['original_codes','formal/groups','reports','validation','evidence/v43','evidence/v44','R/checkpoint/final','evaluation/windows','evaluation/logs']:
  p=OUT/d;p.mkdir(parents=True,exist_ok=True);f=p/'.write_probe';f.write_bytes(b'v45');assert f.read_bytes()==b'v45';f.unlink()
 for fn in ['run_v42.py','audit_sources.py','SOURCE_AUDIT.json','SOURCE_MODULES.csv','BASELINE_PROVENANCE_AUDIT.json']:shutil.copy2(V44/fn,OUT/fn)
 from audit_sources import sha,V18
 from precision_check import POLICY
 read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
 def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 assert read(V43/'RUN_STATUS.json')['status']=='NUMERICAL_STOP_FROZEN_RULE'
 assert read(V44/'RUN_STATUS.json')['status']=='FAILED_STOP'
 audit=read(V44/'reports/OUTPUT_AUDIT.json');assert audit['pass'] and audit['modules']==27 and audit['rows']==49792
 source=read(V44/'SOURCE_AUDIT.json')
 for s in source['sources']:
  assert sha(s['source'])==s['source_sha256'] and sha(s['derived'])==s['derived_sha256']
  shutil.copy2(s['derived'],OUT/'original_codes'/Path(s['derived']).name)
 assert sha(V18/'M2/checkpoint/final/model.safetensors')==source['model_sha256'] and sha(V18/'CALIBRATION_MANIFEST.json')==source['calibration_sha256']
 watched=[]
 for v,files in [(V43,['PROTOCOL.json','RUN_STATUS.json','NUMERICAL_FAILURE_formal_3_mlp_up_proj.pt','NUMERICAL_DIAGNOSIS.json','FAILED_ATTEMPTS.jsonl']), (V44,['PROTOCOL.json','RUN_STATUS.json','CRITICAL_FAILURE_formal_3_mlp_down_proj.pt','NUMERICAL_DIAGNOSIS.json','FAILED_ATTEMPTS.jsonl','DECISION.json','run_v44.py','rejection_repair.py'])]:
  for fn in files:
   f=v/fn;watched.append({'path':str(f),'sha256':sha(f)});shutil.copy2(f,OUT/'evidence'/v.name[:3]/fn)
 inherited=[]
 for rec in audit['checkpoints']:
  path=Path(rec['path']);assert sha(path)==rec['sha256'];ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['committed']
  orig=torch.load(OUT/f'original_codes/layer_{ck["layer"]:02d}.pt',map_location='cpu',weights_only=True)
  for name,s in ck['modules'].items():
   for k in ['scales','zeros','g_idx']:assert torch.equal(s[k],orig[name][k])
   assert bool((s['codes']<=15).all())
   assert torch.equal(torch.stack((s['packed_u4']&15,s['packed_u4']>>4),-1).flatten(1),s['codes'])
   assert torch.equal((s['scales'][:,s['g_idx'].long()]*(s['codes'].float()-8)).half().view(torch.int16),s['weight'].view(torch.int16))
  if ck.get('reused_without_new_repair'):assert sha(ck['source_v43_path'])==ck['source_v43_sha256']
  inherited.append({'source_path':str(path),'source_sha256':sha(path),'layer':ck['layer'],'group':ck['group'],'modules':len(ck['modules']),'input_sha256':ck['input_sha256'],'C_sha256':ck['C_sha256']})
 assert len(inherited)==15 and sum(r['modules'] for r in inherited)==27
 p=read(V44/'PROTOCOL.json');diag=read(V44/'NUMERICAL_DIAGNOSIS.json')
 p.update(experiment='v45 frozen same-precision increment verification plus FP32 strict decline',status='NUMERICAL_POLICY_PREDECLARED_BEFORE_GPU',A='Predict original quadratic argmin and threshold. FP32 full-row direct non-decrease rejects and restores row. Direct decrease plus old discrepancy pass accepts. Direct decrease plus old discrepancy fail triggers independent raw-X FP64 increment and cache/formula check; accepts only check pass and direct64<0. Otherwise rejects FP64 non-decrease, or stops on same-precision/cache correctness error. No widened old cross-precision tolerance.',numerical_revision='Explicit v45 revision; not stepwise equivalent to v43/v44. Never accepts FP32 tie/rise using FP64. Nonfinite/grid/code/pollution/FP64 cache or formula failures are hard errors.',initialization='OriginalM2 plus27 trusted committed modules in15 v44 groups. Never repair committed groups again. Failed block3 down restarts originalM2 codes; failure tentative state is validation-only.',validation='First inspect v44 full failure state with raw-X a=X@e,d=X@actual_delta, mean(2*a*d+d*d), independent of C/p. Validate v43 rejection, v44 new branch, corrupted cache/formula, blacklist and budgets plus batched/row reference. No full pilot rerun.',numerical_policy=POLICY,FP64_increment_tolerance='gamma(k)=k*u/(1-k*u), u=2^-53, k=d+N+ceil(N/4096). rtol=32*gamma(k); atol=32*gamma(k)*mean(2*(abs(X)@abs(e))*(abs(X)@abs(delta))+(abs(X)@abs(delta))^2)+64*u*mean((X@e)^2+(X@delta)^2). Match abs(pred-direct64)<=atol+rtol*max(abs(pred),abs(direct64)). This is empirical and scale-aware, not a strict error theorem.',FP64_cache_tolerance='Elementwise abs(p-e@C)<=[64*gamma(d+32)+64*u]*sum(abs(e))*max_over_input(abs(C_column)). Same criterion at original drift checkpoints and every FP64 increment recheck.',recovery_boundary_input_sha256=diag['input_sha256'],recovery_boundary_C_sha256=diag['C_sha256'],gpu_hard_cap_seconds=10800,time_scope='New180min globalclock includes initial diagnosis, checks, formal recovery, failures, save/reload/eval. No restart of clock.',claims='Existing discrete repair application; explicit verification revision, not new paper method, not officialReQuant reproduction.',implementation_reuse='Byte-identical run_v42/audit_sources runtime; separatev45 engine retains FP32 direct function/shape/4096chunk and search rule. New independent FP64 verification and acceptance-flow branches only.',pilot='No full pilot; inherit v42 evidence and validated v44 groups. Formal block0 only restored, not repaired.',actual_budget='Count every actual legal eligible candidate increment evaluation including repeated neighborhoods,128*d cap,32 accepted updates. Blacklist current row-state rejects; clear only next acceptance.')
 dump('PROTOCOL.json',p)
 for r in inherited:
  ck=torch.load(r['source_path'],map_location='cpu',weights_only=False);ck.update(protocol_sha256=sha(OUT/'PROTOCOL.json'),source_v44_path=r['source_path'],source_v44_sha256=r['source_sha256'],reused_without_new_repair=True)
  torch.save(ck,OUT/'formal/groups'/Path(r['source_path']).name)
 dump('SOURCE_REUSE_AUDIT.json',{'pass':True,'inherited_groups':inherited,'historical_watched':watched,'original_M2_sha256':source['model_sha256'],'calibration_sha256':source['calibration_sha256'],'legacy_runtime_sha256':sha(OUT/'run_v42.py'),'original_failed_group_not_imported':True,'writable_directories_verified':True,'CPU_seconds':time.perf_counter()-t})
 d44=read(V44/'DECISION.json');cp=read(V44/'COST_PROJECTION.json')
 # Extra numerical verification allowance is separate, not a smaller safety factor.
 total=21*cp['v42_measured_block0_seconds']*1.35+30+360+180+1200
 dump('COST_PROJECTION.json',{'type':'prediction, not measurement','remaining_fullblock_equivalent_upper_bound':21,'v42_measured_block0_seconds':cp['v42_measured_block0_seconds'],'safety_multiplier':1.35,'setup_seconds_prediction':30,'finish_reserve_seconds':360,'diagnostic_validation_seconds_prediction':180,'new_precision_recheck_seconds_allowance_prediction':1200,'conservative_total_projection_seconds':total,'hard_cap_seconds':10800,'go':total<=10800,'historical_v42_GPU_budget_wall_seconds':cp['historical_v42_GPU_budget_wall_seconds'],'historical_v42_pilot_block_seconds':cp['historical_v42_pilot_block_seconds'],'historical_v43_GPU_budget_wall_seconds':cp['historical_v43_GPU_budget_wall_seconds'],'historical_v44_GPU_budget_wall_seconds':d44['v44_gpu_budget_global_wall_seconds_including_diagnosis'],'historical_M2_build_seconds':cp['historical_M2_build_seconds'],'precision_cost_uncertainty':'1200s incremental allowance is a planning assumption, not a measurement; online conservative block cost guard retains1.35 and360s finish reserve.'})
 print('PREPARATION_PASS',total,flush=True)
if __name__=='__main__':main()
