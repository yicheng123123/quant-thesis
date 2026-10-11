"""CPU-only v42 evidence reuse, immutable runtime copy and 180-minute gate."""
import ast,csv,hashlib,json,shutil,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;V42=OUT.parent/'v42_m2_single_integer_repair_fullmodel';V18=OUT.parent/'range_w4_fullmodel_v18'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
 assert not (OUT/'GPU_START.json').exists();t=time.perf_counter()
 directories=['original_codes','formal/groups','R/checkpoint/final','reports','evaluation/windows','evaluation/logs','evidence']
 probes=[]
 for d in directories:
  p=OUT/d;p.mkdir(parents=True,exist_ok=True);f=p/'.v43_write_probe';f.write_bytes(b'v43-write-test');assert f.read_bytes()==b'v43-write-test';f.unlink();probes.append(str(p))
 source=load(V42/'SOURCE_AUDIT.json');audit=load(V42/'OUTPUT_AUDIT.json');repro=load(V42/'V41_A_REPRODUCTION.json');lock=load(V42/'GPU_START.json')
 assert audit['pass'] and repro['pass'] and sha(V42/'run_v42.py')==lock['implementation_sha256'] and sha(V42/'PROTOCOL.json')==lock['protocol_sha256']
 assert len(repro['trajectories'])==96 and all(r['codes_equal'] and r['all5_budget_selection_states_equal'] for r in repro['trajectories'])
 for s in source['sources']:assert sha(s['source'])==s['source_sha256'];assert sha(s['derived'])==s['derived_sha256']
 pilot=[]
 for r in audit['records']:
  if 'checkpoint' in r:
   assert sha(r['checkpoint'])==r['sha256'];pilot.append(r)
 assert len(pilot)==4
 for name in ['run_v42.py','audit_sources.py','SOURCE_AUDIT.json','SOURCE_MODULES.csv','V41_A_REPRODUCTION.json','BASELINE_PROVENANCE_AUDIT.json']:
  shutil.copy2(V42/name,OUT/name);assert sha(V42/name)==sha(OUT/name)
 derived=[]
 for s in source['sources']:
  p=Path(s['derived']);dst=OUT/'original_codes'/p.name;shutil.copy2(p,dst);assert sha(dst)==s['derived_sha256'];derived.append({'path':str(dst),'sha256':sha(dst),'read_only_source':str(p)})
 v42_protocol=load(V42/'PROTOCOL.json');cal=V18/'CALIBRATION_MANIFEST.json';assert sha(cal)==source['calibration_sha256']
 caldata=load(cal);assert len(caldata['samples'])==128
 for s in caldata['samples']:
  assert len(s['token_ids'])==2048 and hashlib.sha256(json.dumps(s['token_ids']).encode()).hexdigest()==s['token_sha256']
 assert sha(V18/'M2/checkpoint/final/model.safetensors')==source['model_sha256']
 # Verify historical baseline records/manifests/windows/model-file hashes without scoring.
 for b in load(V42/'BASELINE_PROVENANCE_AUDIT.json')['baselines']:
  assert sha(b['raw_result_path'])==b['raw_result_sha256'];r=load(b['raw_result_path'])
  for wf in r['model_weight_files']:assert sha(Path(r['model_source'])/wf['name'])==wf['sha256']
  mf=V18/'evaluation/manifests'/('WIKITEXT_TEST_MANIFEST.json' if b['dataset']=='wikitext' else 'C4_EVAL_MANIFEST.json');assert sha(mf)==r['evaluation_manifest_sha256']
 projection=load(V42/'COST_PROJECTION.json');cost=list(csv.DictReader((V42/'COST_BILL.csv').open()))
 stages={r['stage']:float(r['measured_seconds']) for r in cost if r['mode']=='pilot'}
 excluded=['model_load','initial_embedding_capture','unpack_pack_zero_commit','zero_model_output']
 block_stage_sum=sum(v for k,v in stages.items() if k not in excluded)
 shapes=list(csv.DictReader((V42/'SOURCE_MODULES.csv').open()));shape0=sorted((r['module'],int(r['rows']),int(r['input_dim'])) for r in shapes if r['layer']=='0')
 assert all(sorted((r['module'],int(r['rows']),int(r['input_dim'])) for r in shapes if r['layer']==str(i))==shape0 for i in range(24))
 predicted=projection['measured_complete_block0_seconds']*24*1.35+30;assert abs(predicted-projection['predicted_formal_repair_seconds'])<1e-9
 eval_seconds=sum(load(V18/f'evaluation/raw/M2_{d}.json')['evaluation_wall_seconds'] for d in ['wikitext','c4_fixed_budget'])
 reserve=360;assert eval_seconds<reserve
 gate={'type':'projection, not measurement','v42_measured_block0_seconds':projection['measured_complete_block0_seconds'],'v42_block_stage_seconds':{k:v for k,v in stages.items() if k not in excluded},'v42_block_stage_sum_seconds':block_stage_sum,'v42_unitemized_block_wall_seconds':projection['measured_complete_block0_seconds']-block_stage_sum,'actual_shapes_rows_by_inputs':shape0,'identical_blocks':24,'unmargined_24_block_projection_seconds':projection['measured_complete_block0_seconds']*24,'unchanged_safety_multiplier':1.35,'fixed_setup_projection_seconds':30,'formal_repair_conservative_projection_seconds':predicted,'historical_M2_two_corpora_evaluation_seconds':eval_seconds,'save_reload_eval_reserved_seconds':reserve,'new_hard_cap_seconds':10800,'conservative_total_projection_seconds':predicted+reserve,'go':predicted+reserve<=10800,'v42_prior_pilot_block_seconds':projection['measured_complete_block0_seconds'],'v42_prior_total_GPU_budget_wall_seconds':load(V42/'RUN_STATUS.json')['elapsed_seconds']}
 dump('COST_PROJECTION.json',gate)
 p=dict(v42_protocol);p.update({'experiment':'v43 budget extension of v42 frozen M2 single integer repair','status':'FROZEN_BEFORE_FORMAL_GPU','inherited_v42_protocol_sha256':sha(V42/'PROTOCOL.json'),'inherited_runtime_sha256':sha(V42/'run_v42.py'),'gpu_hard_cap_seconds':10800,'pilot':'Reuse read-only completed v42 block0 pilot and v41 reproduction; do not rerun either. Formal block0 starts from original M2 and is repaired once.','cost_gate':'Inherited measured block0 x24 x1.35+30sec; unchanged safety1.35. Reserve360sec minimum. Dynamic cost evidence retains1.35; stop if remaining conservative forecast does not fit.','time_scope':'New wall timer begins at first v43 warmup;10800sec includes failures/recovery/formal/save/reload/PPL. Prior v42 cost separate.','implementation_reuse':'run_v42.py and audit_sources.py copied byte-for-byte. New run_v43.py coordinator changes only budget/output staging and skips historical checks. No cache/search/candidate/update changes.','evaluation':'Inherited frozen WT2 and fixed-budget C4 each once, only after packed code/grid/config/reload output checks and deployment lock.','model_setting':'Sol high requested; interface cannot confirm or switch.'})
 dump('PROTOCOL.json',p)
 functions={}
 for n in ast.parse((OUT/'run_v42.py').read_text()).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):functions[n.name]=hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
 dump('SOURCE_REUSE_AUDIT.json',{'pass':True,'CPU_seconds':time.perf_counter()-t,'source_v42':str(V42),'runtime_byte_identical':True,'runtime_sha256':sha(OUT/'run_v42.py'),'algorithm_function_AST_sha256':functions,'v42_protocol_sha256':sha(V42/'PROTOCOL.json'),'v42_pilot_checkpoints':pilot,'original_codes_copies':derived,'calibration_sha256':sha(cal),'v42_input_evidence':'128x2048 frozen token hashes and persisted per-group input/C hashes verified against v42 pilot checkpoint hash records. Raw pilot X was not saved; no new forward or pilot recomputation.','writable_directories_verified':probes,'historical_baseline_hashes_rechecked':True})
 for name in ['OUTPUT_AUDIT.json','COST_PROJECTION.json','COST_BILL.csv','RUN_STATUS.json','PROTOCOL.json','IMPLEMENTATION_LOCK.json']:shutil.copy2(V42/name,OUT/'evidence'/f'v42_{name}')
 print(json.dumps(gate,ensure_ascii=False,indent=2));assert gate['go'],'cost projection exceeds180 minutes; stop'
if __name__=='__main__':main()
