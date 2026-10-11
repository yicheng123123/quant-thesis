"""Final pre-GPU reconciliation of v36 forwards, correctness work and hashes."""
import csv,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];RESULTS=ROOT/'experiments/range_search_w4/results'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def stable(p,obj):
 text=json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
 Path(p).write_text(text,encoding='utf-8',newline='\n')
vpath=HERE/'manifests/FROZEN_PROTOCOL.json';protocol=json.loads((RESULTS/'range_w4_awq_scaling_gptq_compat_v34/NEXT_EXPERIMENT_PROTOCOL.json').read_text(encoding='utf-8'))
protocol['protocol_id']='v36_l12_awq_selector_dynamic_gptq_32_new_articles'
protocol['schema_version']='1.0-v36'
protocol['correctness'] += [
 'FP64 linear-transform max abs <= 1e-9; FP32 max abs <= 1e-3 (inherited from v32 correctness gate); FP16 fold drift is finite and fully reported, with no post-result tolerance adjustment',
 'full unquantized selected-D L12 MLP FP16 fold drift measured before GPTQ candidates on search and after lock on held',
 'alpha=1 checked before any non-native alpha candidate in each full-Linear bank; group-entry weights, W, scale, zero, g_idx and direct row scores compare exactly',
 'selected row-wise W/scale/zero/g_idx stitched and copied into actual L12 modules with hash check before the selection lock'
]
protocol['workload']['forwards']['fold_error_measurement'] = 2
protocol['workload']['forwards']['forward_count_note'] = 'Two additional FP16 unquantized selected-D fold-error MLP forwards (one per split, on held after lock) are explicitly added to v34 count; the two reference MLP forwards are held unquantized-target references.'
protocol['workload']['correctness_gptq_calls']['per_split'] = {'alpha1_gate_up_native_parity':4,'alpha1_down_native_parity':4,'down_isolation_replays':2}
protocol['held_gate']['fold_error_interpretation'] = 'The v34 phrase is applied literally as final G11 full-MLP held MSE >= 5x the same-split FP16 unquantized selected-D fold-error MSE; this is a quantization-signal floor check, not an improvement threshold.'
protocol['held_gate']['wtl_tie_rule'] = 'Descriptive W/T/L tables use exact equality only; no new epsilon is used to modify any quality gate.'
protocol['execution_status']={'gpu_experiment':'FROZEN_NOT_STARTED','harness':'run_v36.py','runtime':True}
stable(vpath,protocol)

# Extend the v34-derived table with real forward/capture and folded-reference tasks.
jobpath=HERE/'JOB_TABLE.csv'
with jobpath.open(newline='',encoding='utf-8') as f:
 rows=list(csv.DictReader(f))
base_tasks={'AWQ_RTN_D_selector_ratio','main_GPTQ_candidate_bank','main_GPTQ_down_candidate',
            'native alpha1 parity','down candidate isolation replay'}
rows=[r for r in rows if r['task_type'] in base_tasks]
old_fields=['split','phase','task_type','arm','linear','shape','candidate_count','candidate_group_count',
            'scored_rows','tokens_per_candidate','row_alpha_score_slots','score_tensor_elements',
            'cache_reuse','additional_gptq_calls','detail']
fields=old_fields+['full_mlp_forwards','activation_capture_sets','l13_boundary_captures','prefix_decoder_block_forwards',
                  'full_mlp_score_elements','projection_row_score_elements','parity_score_row_slots',
                  'parity_score_tensor_elements','replay_score_row_slots','replay_score_tensor_elements']
for r in rows:
 r['full_mlp_forwards']='1' if r['task_type']=='AWQ_RTN_D_selector_ratio' else '0'
 r['activation_capture_sets']='0';r['l13_boundary_captures']='0';r['prefix_decoder_block_forwards']='0'
 r['full_mlp_score_elements']='0';r['projection_row_score_elements']='0';r['parity_score_row_slots']='0';r['parity_score_tensor_elements']='0';r['replay_score_row_slots']='0';r['replay_score_tensor_elements']='0'
extra=[]
def add(split,phase,task,arm,linear,cand,rows_scored,tokens,mlp,cap=0,boundary=0,detail='',prefix_blocks=0,mlp_elements=0,projection_elements=0):
 extra.append({'split':split,'phase':phase,'task_type':task,'arm':arm,'linear':linear,'shape':'[8,2048,896] or frozen L12 MLP',
  'candidate_count':cand,'candidate_group_count':cand,'scored_rows':rows_scored,'tokens_per_candidate':tokens,
  'row_alpha_score_slots':0,'score_tensor_elements':0,'cache_reuse':'frozen whole-article group; common v18 M2 L12 prefix',
  'additional_gptq_calls':0,'detail':detail,'full_mlp_forwards':mlp,'activation_capture_sets':cap,'l13_boundary_captures':boundary,
  'prefix_decoder_block_forwards':prefix_blocks,'full_mlp_score_elements':mlp_elements,
  'projection_row_score_elements':projection_elements,'parity_score_row_slots':0,'parity_score_tensor_elements':0,
  'replay_score_row_slots':0,'replay_score_tensor_elements':0})
for split,search,held in [('S0','P0','P1'),('S1','P2','P3')]:
 add(split,'prefix_capture','common_L12_search_capture','all_arms','L0..L12 prefix; postnorm/pre-norm/hidden',1,0,8*2048,0,1,0,f'{search}: 8 whole articles; held is not read',prefix_blocks=8)
 add(split,'prefix_capture_after_lock','common_L12_held_capture','all_arms','L0..L12 prefix; postnorm/pre-norm/hidden',1,0,8*2048,0,1,0,f'{held}: 8 whole articles; capture only after selection lock',prefix_blocks=8)
 add(split,'search','selected_D_FP16_fold_error_reference','correctness','complete L12 MLP',1,896,8*2048,1,0,0,'unquantized D-fold drift measured before held access')
 add(split,'held_after_lock','unquantized_target_reference','all_arms','complete L12 MLP',1,896,8*2048,1,0,0,f'{held}: one reference full MLP pass after lock')
 add(split,'held_after_lock','held_quantized_arm_evaluation','G00/G01/G10/G11','complete L12 block + MLP',4,896,8*2048,4,0,4,'four whole-arm forwards after lock; capture actual L13 boundary once per arm',mlp_elements=4*8*2048*896,projection_elements=4*8*2048*(4864+4864+896))
 for arm in ('G00','G01','G10','G11'):
  add(split,'search','gate_up_to_down_actual_input_refresh','arm_specific',f'{arm} gate/up -> down input/H',1,896,8*2048,0,0,0,'whole committed gate/up projections; one actual downstream input per arm')
for r in extra:
 rows.append(r)
# The down-state .8 replay score is a correctness score, separate from the
# frozen main row-alpha candidate slots.
for r in rows:
 if r['task_type']=='native alpha1 parity':
  # This aggregate row covers four full gate/up and four full down parity calls per split.
  r['parity_score_row_slots']=4*4864+4*896
  r['parity_score_tensor_elements']=(4*4864+4*896)*8*2048
 if r['task_type']=='down candidate isolation replay':
  r['replay_score_row_slots']=896
  r['replay_score_tensor_elements']=896*8*2048
with jobpath.open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

main=sum(int(r['candidate_count']) for r in rows if r['task_type'].startswith('main_GPTQ'))
slots=sum(int(r['row_alpha_score_slots']) for r in rows)
correct=sum(int(r['additional_gptq_calls']) for r in rows)
selector=sum(int(r['candidate_count']) for r in rows if r['task_type']=='AWQ_RTN_D_selector_ratio')
mlp_fwd=sum(int(r['full_mlp_forwards']) for r in rows)
captures=sum(int(r['activation_capture_sets']) for r in rows)
l13=sum(int(r['l13_boundary_captures']) for r in rows)
refresh=sum(1 for r in rows if r['task_type']=='gate_up_to_down_actual_input_refresh')
prefix_blocks=sum(int(r['prefix_decoder_block_forwards']) for r in rows)
mlp_score_elems=sum(int(r['full_mlp_score_elements']) for r in rows)
proj_score_elems=sum(int(r['projection_row_score_elements']) for r in rows)
replay_slots=sum(int(r['replay_score_row_slots']) for r in rows)
replay_elems=sum(int(r['replay_score_tensor_elements']) for r in rows)
gptq_score_elems=sum(int(r['score_tensor_elements']) for r in rows if r['task_type'].startswith('main_GPTQ'))
selector_score_elems=sum(int(r['score_tensor_elements']) for r in rows if r['task_type']=='AWQ_RTN_D_selector_ratio')
parity=sum(int(r['candidate_count']) for r in rows if r['task_type']=='native alpha1 parity')
parity_slots=sum(int(r['parity_score_row_slots']) for r in rows)
parity_elems=sum(int(r['parity_score_tensor_elements']) for r in rows)
replays=sum(int(r['candidate_count']) for r in rows if r['task_type']=='down candidate isolation replay')
if (main,slots,correct,selector,mlp_fwd,captures,l13,refresh,parity,replays)!=(124,428544,20,40,52,4,8,8,16,4):
 raise RuntimeError(f'v36 count mismatch {(main,slots,correct,selector,mlp_fwd,captures,l13,refresh,parity,replays)}')
audit=json.loads((HERE/'JOB_TABLE_AUDIT.json').read_text(encoding='utf-8'))
audit['counts'].update({'main_full_Linear_GPTQ_executions':main,'alpha1_parity_GPTQ_calls':parity,
 'down_isolation_replay_calls':replays,'total_full_Linear_GPTQ_if_all_run':main+correct,
 'row_alpha_score_slots':slots,'full_MLP_forwards_total':mlp_fwd,
 'full_MLP_forwards_breakdown':{'AWQ_RTN_selector':40,'held_unquantized_references':2,
                                 'selected_D_fold_error_reference':2,'held_quantized_arms':8},
 'activation_capture_sets':captures,'search_gate_up_to_down_refreshes':refresh,'L13_boundary_captures':l13,
 'correctness_replay_row_score_slots':4*896,
 'correctness_replay_row_score_slots_separate_from_main_slots':True,
 'alpha1_parity_direct_score_row_slots':parity_slots,
 'alpha1_parity_direct_score_elements':parity_elems,
 'GPTQ_local_score_output_elements':gptq_score_elems,
 'GPTQ_candidate_residual_elements':gptq_score_elems,
 'AWQ_selector_residual_elements':selector_score_elems,
 'alpha1_parity_direct_score_residual_elements':parity_elems,
 'correctness_replay_residual_elements':replay_elems,
 'held_full_MLP_residual_elements':mlp_score_elems,
 'held_projection_row_residual_elements':proj_score_elems,
 'prefix_decoder_block_forwards_by_article':prefix_blocks,
 'full_scored_residual_elements_including_diagnostics':gptq_score_elems+selector_score_elems+parity_elems+replay_elems+mlp_score_elems+proj_score_elems})
audit['job_table'].update({'rows':len(rows),'full_MLP_forwards':mlp_fwd,'activation_capture_sets':captures,
                           'search_down_refreshes':refresh,'L13_boundary_captures':l13,
                           'prefix_decoder_block_forwards':prefix_blocks,'GPTQ_residual_elements':gptq_score_elems,
                           'AWQ_selector_residual_elements':selector_score_elems,'correctness_replay_residual_elements':replay_elems,
                           'held_MLP_residual_elements':mlp_score_elems,'held_projection_residual_elements':proj_score_elems,
                           'replay_row_score_slots':replay_slots,'parity_row_score_slots':parity_slots,
                           'parity_score_residual_elements':parity_elems})
audit['forward_accounting']='Per split: prefix captures search/held once each; 20 AWQ ratio candidates; 4 gate/up-to-down propagations; 1 held reference MLP; 1 selected-D fold-error reference; 4 held full L12 arm forwards and 4 L13 boundary captures.'
run_report=HERE/'raw/FINAL_RUN_REPORT.json'
if run_report.exists():
 runtime=json.loads(run_report.read_text(encoding='utf-8-sig'))
 audit['status']='PLAN_COUNT_RECONCILIATION_PASS; EXECUTION_PARTIAL_STOPPED'
 audit['execution_status']={'runner_status':runtime.get('status'),'held_splits_written':runtime.get('held_results_written',[]),
                           'gpu_seconds_charged':runtime.get('gpu_seconds_used'),'ppl':runtime.get('ppl','NOT_RUN'),
                           'full_model_quantization':runtime.get('full_model_quantization','NOT_RUN'),
                           'actual_execution_audit':'raw/ACTUAL_EXECUTION_AUDIT.json'}
 audit['quality_gate_status']='NOT_PASS_OR_UNEVALUABLE: S0 primary threshold missed; S1 unmeasured; fold-error reference invalid by code audit; projection-tail gate unavailable after stop.'
with (HERE/'JOB_TABLE_AUDIT.json').open('w',encoding='utf-8',newline='\n') as f:json.dump(audit,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

sources=[
 ('experiments/range_search_w4/results/range_w4_awq_scaling_gptq_compat_v34/NEXT_EXPERIMENT_PROTOCOL.json'),
 ('experiments/range_search_w4/results/range_w4_awq_scaling_gptq_compat_v34/REPORT.md'),
 ('experiments/range_search_w4/results/range_w4_awq_gptq_v35/NEW_ARTICLE_MANIFEST.json'),
 ('experiments/range_search_w4/results/range_w4_awq_gptq_v35/FROZEN_PROTOCOL_V35.json'),
 ('experiments/range_search_w4/results/range_w4_awq_gptq_v35/JOB_TABLE.csv'),
 ('experiments/range_search_w4/results/range_w4_awq_gptq_v35/PRECHECK.json'),
 ('experiments/range_search_w4/results/range_w4_awq_gptq_v35/REPORT.md'),
 ('experiments/range_search_w4/results/range_w4_fullmodel_v18/PROTOCOL.json'),
 ('experiments/range_search_w4/results/range_w4_fullmodel_v18/run_formal.py'),
 ('experiments/range_search_w4/results/range_w4_fullmodel_v18/FORMAL_TIMING.csv'),
 ('experiments/range_search_w4/results/range_w4_fullmodel_v18/CALIBRATION_MANIFEST.json'),
 ('experiments/range_search_w4/results/range_w4_fullmodel_v18/M2/RUN_REPORT.json'),
 ('experiments/range_search_w4/results/range_w4_fullmodel_v18/M2/checkpoint/final/model.safetensors'),
 ('experiments/range_search_w4/results/range_w4_m2_runtime_v17_3/optimized_native_m2.py'),
 ('.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors'),
 ('.hf_cache/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet'),
]
# Use the actual v35 parquet path and exact local artifacts.
sources[-1]='.hf_cache/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet'
local=['prepare_v36.py','split_v36_token_cache.py','run_v36.py','finalize_v36_freeze.py','analyze_partial_v36.py','build_output_manifest.py',
       'manifests/ARTICLE_MANIFEST.json','manifests/GROUP_TOKEN_CACHE_MANIFEST.json','manifests/FROZEN_PROTOCOL.json',
       'JOB_TABLE.csv','JOB_TABLE_AUDIT.json','raw/RUN_STATE.json','raw/BUDGET_LEDGER.csv',
       'cache/P0_tokens.json','cache/P1_tokens.json','cache/P2_tokens.json','cache/P3_tokens.json']
allhash={}
for rel in sources:
 p=ROOT/rel
 if not p.is_file():raise FileNotFoundError(p)
 allhash[rel]=sha(p)
for rel in local:
 p=HERE/rel
 if not p.is_file():raise FileNotFoundError(p)
 allhash[(p.relative_to(ROOT)).as_posix()]=sha(p)
src={'git_head':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip(),
     'worktree_dirty':True,'v35_status_preserved':'STOP_INSUFFICIENT_DISJOINT_ARTICLES',
     'official_awq':protocol['provenance']['official_upstream'],'paths':allhash}
with (HERE/'manifests/SOURCE_HASHES.json').open('w',encoding='utf-8',newline='\n') as f:json.dump(src,f,indent=2,ensure_ascii=False);f.write('\n')
state_path=HERE/'raw/RUN_STATE.json';state=json.loads(state_path.read_text(encoding='utf-8'))
ledger=HERE/'raw/BUDGET_LEDGER.csv'
if ledger.exists():
 with ledger.open(newline='',encoding='utf-8') as f: existing_ledger=list(csv.DictReader(f))
 cumulative=float(existing_ledger[-1]['cumulative_gpu_wall_seconds']) if existing_ledger else 5.2
else:
 cumulative=5.2
state.update({'protocol_sha256':sha(vpath),'article_manifest_sha256':sha(HERE/'manifests/ARTICLE_MANIFEST.json'),
 'job_table_sha256':sha(jobpath),'group_token_cache_manifest_sha256':sha(HERE/'manifests/GROUP_TOKEN_CACHE_MANIFEST.json'),
 'job_audit_sha256':sha(HERE/'JOB_TABLE_AUDIT.json'),
 'external_cuda_probe_wall_seconds':5.2,'gpu_seconds_used':cumulative,
 'runtime_status':('CORRECTNESS_HARNESS_FAIL_RESUMABLE' if state.get('last_stage')=='RUNTIME_ERROR' else state.get('runtime_status','FROZEN_BEFORE_GPU; correctness-first runner ready')),
 'held_access':state.get('held_access','P1/P3 token files unopened by model runtime until split lock'),
 'hard_gpu_budget_seconds':1800,'stop_compute_at_seconds':1620,'finalization_reserve_seconds':180})
with state_path.open('w',encoding='utf-8',newline='\n') as f:json.dump(state,f,indent=2,ensure_ascii=False);f.write('\n')
probe=['','cuda_device_probe','torch.cuda.is_available/device_count/device_name','5.200000','5.200000','','','PASS','Conservative process wall charged before runtime; this was only device availability/name probe, not model/GPTQ']
if ledger.exists():
 with ledger.open(newline='',encoding='utf-8') as f: ledger_rows=list(csv.reader(f))
 if len(ledger_rows)<2 or ledger_rows[0]!=['split','stage','task','wall_seconds','cumulative_gpu_wall_seconds','peak_allocated_bytes','peak_reserved_bytes','status','detail'] or ledger_rows[1]!=probe:
  raise RuntimeError('existing GPU budget ledger does not match frozen preflight probe')
else:
 with ledger.open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['split','stage','task','wall_seconds','cumulative_gpu_wall_seconds','peak_allocated_bytes','peak_reserved_bytes','status','detail']);w.writerow(probe)
print(json.dumps({'protocol_sha256':sha(vpath),'job_table_sha256':sha(jobpath),'job_table_rows':len(rows),
 'main_gptq':main,'correctness_gptq':correct,'row_alpha_slots':slots,'full_mlp_forwards':mlp_fwd,
 'prefix_capture_sets':captures,'search_down_refreshes':refresh,'l13_captures':l13,
 'frozen_before_gpu':True,'external_cuda_probe_seconds_charged':5.2},indent=2))
