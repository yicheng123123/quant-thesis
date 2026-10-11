"""CPU preparation only; preserves all v42/v43 evidence and writes v44 only."""
import ast,csv,hashlib,json,shutil,time
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent;V43=OUT.parent/'v43_m2_single_integer_repair_fullmodel';V42=OUT.parent/'v42_m2_single_integer_repair_fullmodel';V18=OUT.parent/'range_w4_fullmodel_v18'
torch.set_num_threads(4)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
 t=time.perf_counter();assert not (OUT/'GPU_START.json').exists()
 for d in ['original_codes','formal/groups','reports','validation','evidence','R/checkpoint/final','evaluation/windows','evaluation/logs']:
  p=OUT/d;p.mkdir(parents=True,exist_ok=True);f=p/'.write_probe';f.write_bytes(b'v44');assert f.read_bytes()==b'v44';f.unlink()
 status=read(V43/'RUN_STATUS.json');assert status['status']=='NUMERICAL_STOP_FROZEN_RULE'
 audit=read(V43/'reports/OUTPUT_AUDIT.json');assert audit['pass'] and audit['modules']==25 and audit['rows']==40064
 for name in ['run_v42.py','audit_sources.py','SOURCE_AUDIT.json','SOURCE_MODULES.csv','BASELINE_PROVENANCE_AUDIT.json']:
  shutil.copy2(V43/name,OUT/name);assert sha(V43/name)==sha(OUT/name)
 src=read(V43/'SOURCE_AUDIT.json')
 for s in src['sources']:
  assert sha(s['source'])==s['source_sha256'] and sha(s['derived'])==s['derived_sha256']
  p=Path(s['derived']);dst=OUT/'original_codes'/p.name;shutil.copy2(p,dst);assert sha(dst)==s['derived_sha256']
 assert sha(V18/'CALIBRATION_MANIFEST.json')==src['calibration_sha256'] and sha(V18/'M2/checkpoint/final/model.safetensors')==src['model_sha256']
 inherited=[]
 for s in audit['checkpoints']:
  p=Path(s['path']);assert sha(p)==s['sha256'];ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['committed']
  o=torch.load(OUT/f'original_codes/layer_{ck["layer"]:02d}.pt',map_location='cpu',weights_only=True)
  for name,u in ck['modules'].items():
   c=u['codes'];assert bool((c<=15).all())
   for k in ['scales','zeros','g_idx']:assert torch.equal(u[k],o[name][k])
   unpack=torch.stack((u['packed_u4']&15,u['packed_u4']>>4),-1).flatten(1);assert torch.equal(unpack,c)
   assert torch.equal((u['scales'][:,u['g_idx'].long()]*(c.float()-8)).half().view(torch.int16),u['weight'].view(torch.int16))
  inherited.append({'source_path':str(p),'source_sha256':sha(p),'layer':ck['layer'],'group':ck['group'],'input_sha256':ck['input_sha256'],'C_sha256':ck['C_sha256'],'modules':len(ck['modules'])})
 assert len(inherited)==14 and sum(r['modules'] for r in inherited)==25
 watched=[V43/'PROTOCOL.json',V43/'RUN_STATUS.json',V43/'NUMERICAL_FAILURE_formal_3_mlp_up_proj.pt',V43/'NUMERICAL_DIAGNOSIS.json',V43/'FAILED_ATTEMPTS.jsonl',V43/'run_v42.py',V43/'run_v43.py']
 p=read(V43/'PROTOCOL.json');p.update({'experiment':'v44 explicit state-specific rejection revision and inherited-prefix full-model completion','status':'FROZEN_BEFORE_GPU','A':'Keep original predictor threshold and FP32 full-row direct scorer/tolerances. Accept only strict direct decrease plus discrepancy pass. Direct tie/rise with discrepancy within tolerance rejects only that coordinate/offset for the unchanged row state, restores code/E and keeps p/loss/accept-count. Clear rejected set only after that row next accepts. Material discrepancy/nonfinite/grid/restore errors still stop.','budget_execution':'Count every actually evaluated eligible candidate, including repeated evaluations of other candidates. Rejected coordinates not evaluated/selected again in unchanged state. If remaining budget cannot complete a neighborhood, score only its ascending prefix within budget and terminate that row without accepting an incomplete-neighborhood minimum. Never reset128*d budget; accept cap32.','initialization':'Reconstruct original M2 and load25 modules in14 trusted committed v43 groups. Never repair those groups again. Uncommitted block3 up/gate restarts at original M2 parameters; partial tentative failure codes used only for validation.','numerical_revision':'Not stepwise equivalent to v43. Independent row accept/reject transaction; rejection cannot undo other rows. OriginalFP32 scorer/function/chunk4096/fulloutput-row shape and original tolerances retained. No single-row orFP64 direct acceptance fallback.','validation':'CPU deterministic transaction fixtures + saved v43 failure-code fixture with all4864 output rows and exact current-prefix input. Fixture accept cap1 only for validation; formal remains32. Verify original3 rejects, state bitwise restore, legal independent accepts, blacklist/no-repeat/budget/termination and final direct recomputation.','gpu_hard_cap_seconds':10800,'time_scope':'New global wall clock begins first validation CUDA warmup and includes validation, failures, formal continuation, saving/reload/eval; historical v42/v43 costs separate and cumulative.','model_setting':'Current interface cannot confirm model setting; previous Sol high request retained as preference.','claims':'Existing discrete repair application validation, explicit numerical-flow revision; not new paper method or officialReQuant reproduction. No stepwise equivalence claim to v43.','inherited_v43_protocol_sha256':sha(V43/'PROTOCOL.json'),'recovery_boundary_input_sha256':read(V43/'NUMERICAL_DIAGNOSIS.json')['input_sha256'],'recovery_boundary_C_sha256':read(V43/'NUMERICAL_DIAGNOSIS.json')['C_sha256']})
 dump('PROTOCOL.json',p)
 for r in inherited:
  ck=torch.load(r['source_path'],map_location='cpu',weights_only=False);ck.update(protocol_sha256=sha(OUT/'PROTOCOL.json'),source_v43_path=r['source_path'],source_v43_sha256=r['source_sha256'],reused_without_new_repair=True)
  torch.save(ck,OUT/'formal/groups'/Path(r['source_path']).name)
 for f in watched:shutil.copy2(f,OUT/'evidence'/f.name)
 dump('SOURCE_REUSE_AUDIT.json',{'pass':True,'CPU_seconds':time.perf_counter()-t,'inherited_groups':inherited,'historical_v43_watched':[{'path':str(f),'sha256':sha(f)} for f in watched],'source_v42_protocol_sha256':sha(V42/'PROTOCOL.json'),'legacy_runtime_sha256':sha(OUT/'run_v42.py'),'calibration_sha256':src['calibration_sha256'],'original_M2_sha256':src['model_sha256'],'recovery_boundary_input_sha256':p['recovery_boundary_input_sha256'],'recovery_boundary_C_sha256':p['recovery_boundary_C_sha256'],'failed_tentative_state_not_imported':True,'writable_directories_verified':True})
 projection=read(V43/'COST_PROJECTION.json');forecast=21*projection['v42_measured_block0_seconds']*1.35+30+360+120
 dump('COST_PROJECTION.json',{'type':'prediction, not measurement','remaining_full_block_equivalent_upper_bound':21,'v42_measured_block0_seconds':projection['v42_measured_block0_seconds'],'unchanged_safety_multiplier':1.35,'setup_projection_seconds':30,'finish_reserve_seconds':360,'validation_projection_seconds':120,'conservative_total_projection_seconds':forecast,'hard_cap_seconds':10800,'go':forecast<=10800,'historical_v42_GPU_budget_wall_seconds':read(V42/'RUN_STATUS.json')['elapsed_seconds'],'historical_v42_pilot_block_seconds':projection['v42_prior_pilot_block_seconds'],'historical_v43_GPU_budget_wall_seconds':status['elapsed_seconds'],'historical_v43_repair_attempt_wall_seconds':status['repair_attempt_elapsed_seconds'],'historical_v43_formal_stage_seconds':read(V43/'DECISION.json')['formal_all_attempt_stage_seconds'],'historical_v43_diagnostic_stage_seconds':read(V43/'DECISION.json')['diagnostic_measured_stage_seconds'],'historical_M2_build_seconds':2115.9353341000096})
 for f in ['rejection_repair.py']:ast.parse((OUT/f).read_text())
 print('V44_PREPARATION_PASS',forecast,'seconds forecast',flush=True)
if __name__=='__main__':main()
