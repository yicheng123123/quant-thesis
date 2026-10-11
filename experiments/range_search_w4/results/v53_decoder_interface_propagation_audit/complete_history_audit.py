"""Read-only CPU inventory and provenance audit. Current replay is not historical environment proof."""
import ast,csv,hashlib,json,math,re,time
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parent;R=Path('E:/quant-thesis');t=time.perf_counter()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def j(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def csvout(n,rs):
 with (O/n).open('w',newline='',encoding='utf8') as f:
  cols=list(dict.fromkeys(k for r in rs for k in r));w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rs)
locations=list(csv.DictReader((O/'CALLSITE_INVENTORY.csv').open(encoding='utf8')));excluded=list(csv.DictReader((O/'NON_DECODER_DOUBLE_INDEX.csv').open(encoding='utf8')))
source_paths=[R/'.venv/Lib/site-packages/transformers/models/qwen2/modeling_qwen2.py',R/'.venv/Lib/site-packages/transformers/integrations/sdpa_attention.py',R/'.venv-gptq/Lib/site-packages/transformers/models/qwen2/modeling_qwen2.py',R/'.venv-gptq/Lib/site-packages/transformers/__init__.py']
dump('SUPPORTED_LIBRARY_SOURCE_EVIDENCE.json',{'files':[{'path':str(p),'sha256':sha(p)} for p in source_paths],'current_library':'5.16.1 Tensor hidden_states','local_legacy_library':'4.49.0 one-element tuple under output_attentions=False','local_legacy_source_is_NOT_proof_it_was_used_in_every_historical_job':True})
# PPL source audit compares the files actually measured, every window and valid-token weighting; does not infer calibration validity.
V18=B/'range_w4_fullmodel_v18';V45=B/'v45_m2_single_integer_repair_fullmodel';V46=B/'v46_m1_single_integer_repair_fullmodel';V49=B/'v49_fullmodel_fast_integer_repair';V52=B/'v52_m2_fast_integer_repair_fp64_initial_review'
audit=j(V46/'PPL_SOURCE_AUDIT.json')['historical_source_records'];v49recs=j(V52/'SOURCE_AUDIT.json')['v49_r1_ppl_sources'];quality=[];checked_models={}
for label,D,sub in [('M1',V18,'M1/checkpoint/final'),('M2',V18,'M2/checkpoint/final'),('v45_original_R2',V45,'R/checkpoint/final'),('v46_R1',V46,'R1/checkpoint/final'),('v49_fast_R1',V49,'R1/checkpoint/final')]:
 mp=D/sub/'model.safetensors';mh=sha(mp);checked_models[label]={'path':str(mp),'sha256':mh,'config':str(D/sub/'config.json'),'config_sha256':sha(D/sub/'config.json'),'saved_config_transformers_version':j(D/sub/'config.json').get('transformers_version')}
 for ds in ['wikitext','c4_fixed_budget']:
  if label in ['M1','M2','v45_original_R2']:
   key={'M1':'M1','M2':'M2','v45_original_R2':'R2'}[label];rec=audit[f'{key}_{ds}'];rp=Path(rec['result_path']);wp=Path(rec['windows_path']);assert sha(rp)==rec['result_sha256'] and sha(wp)==rec['windows_sha256']
  elif label=='v46_R1':
   rec=j(V46/'PPL_SOURCE_AUDIT.json')['R1_evaluations'][ds];rp=Path(rec['result_path']);wp=Path(rec['windows_path']);assert sha(rp)==rec['result_sha256'] and sha(wp)==rec['windows_sha256']
  else:
   rec=v49recs[ds];rp=Path(rec['result_path']);wp=Path(rec['windows_path']);assert sha(rp)==rec['result_sha256'] and sha(wp)==rec['windows_sha256']
  res=j(rp)
  if 'model_sha256' in res:assert res['model_sha256']==mh
  else:assert any(f['sha256']==mh for f in res['model_weight_files'])
  ww=list(csv.DictReader(wp.open(encoding='utf8')));tokens=sum(int(w['valid_tokens']) for w in ww);nll=sum(float(w['window_nll']) for w in ww)/tokens
  assert abs(nll-res.get('nll',res.get('average_nll')))<1e-10 and abs(math.exp(nll)-res.get('ppl',res.get('perplexity')))/res.get('ppl',res.get('perplexity'))<1e-10
  manifest=V18/'evaluation/manifests'/('WIKITEXT_TEST_MANIFEST.json' if ds=='wikitext' else 'C4_EVAL_MANIFEST.json');man=j(manifest);manifest_hash=sha(manifest)
  assert res.get('manifest_sha256',res.get('evaluation_manifest_sha256'))==manifest_hash
  assert tokens==len(man['token_ids'])-1 and res['max_length']==2048 and res['stride']==512
  assert len(ww)==582 if ds=='wikitext' else len(ww)==253
  # Check no windows omitted, overlap/valid-token accounting intact.
  for ix,w in enumerate(ww):assert int(w.get('window_index',w.get('window_idx',ix)))==ix
  quality.append({'model':label,'dataset':ds,'checkpoint_sha256':mh,'result_path':str(rp),'result_sha256':sha(rp),'windows_path':str(wp),'windows_sha256':sha(wp),'manifest_sha256':manifest_hash,'windows':len(ww),'valid_tokens':tokens,'NLL':nll,'PPL':res.get('ppl',res.get('perplexity')),'independent_actual_checkpoint_measurement_verified':True,'does_NOT_prove_calibration_prefix_correctness':True})
csvout('HISTORICAL_PPL_PROVENANCE.csv',quality);dump('HISTORICAL_CHECKPOINT_SOURCES.json',checked_models)
# Actual call relationships and type evidence are distinct fields. Unknown is retained where direct historical type was not logged.
impact=[]
def add(version,route,index,type_ev,historical_type,error_branch,affected,calibration,ppl,quality_cost,action,evidence):
 impact.append(dict(version=version,actual_call_route=route,same_decoder_index_present=index,historical_return_evidence=type_ev,historical_return_type=historical_type,actually_executed_erroneous_Tensor_branch=error_branch,affected_scope=affected,calibration_protocol_validity=calibration,checkpoint_provenance='source hash audited where a final model exists; group/failed state retained otherwise',historical_PPL=ppl,quality_and_timing_interpretation=quality_cost,action=action,evidence=evidence))
add('v18','run_formal.refresh_after_down/replay_completed_prefix',True,'FORMAL_RUNTIME explicitly records4.49.0; old source declares tuple; no per-call type trace','tuple API supported by historical recorded runtime; direct per-call type UNKNOWN','NOT_EVIDENCED','would affect layer1+ if Tensor; no evidence it ran Tensor','SUPPORTED_BY_RECORDED4.49_RUNTIME; no exhaustive historical input revalidation','M1/M2 both actual checkpoint measurements verified','retain original baseline/cost with documented4.49 scope; current5.x rerun unsafe','retain M1/M2 initializer and measured PPL; future runtime use corrected interface','v18/FORMAL_RUNTIME.json;v18/run_formal.py;local4.49source')
for v,route in [('v42','run_v42.advance_block pilot'),('v43','run_v42.advance_block formal'),('v44','run_v42.advance_block formal'),('v45','run_v45 -> run_v42.run_layer -> advance_block')]:
 add(v,route,True,'v45 final save emits4.49 metadata; v42-44 no direct return trace. v45 inherits prior submitted groups','UNKNOWN for full historical chain','UNKNOWN','potential downstream input/boundary scope; current5.x replay alone cannot assign historical failure','UNKNOWN_FULL_CHAIN; no blanket invalidation','v45 R2 windows and saved checkpoint verified;42-44 no final PPL claim','PPL describes measured R2; frozen calibration causal/cost claim requires historical interface evidence','keep artifacts/PPL as measurements; qualify full-protocol conclusions; rebuild only if historical Tensor misuse established',f'{v} runtime source;v45 saved config/generation metadata;v45 source reuse chain')
for v,route in [('v46','run_v46_retry -> runtime_v46.run_layer -> advance_block'),('v49','run_v49 -> loadmod(runtime_v46) -> run_layer -> advance_block')]:
 add(v,route,True,'saved config/generation emits4.49; complete stage logs exist; per-call return type not logged','UNKNOWN (4.49 tuple strongly suggested by save metadata)','UNKNOWN','potential layer1+ if Tensor; not proved by current5.x source replay','UNKNOWN_HISTORICAL_INTERFACE; no blanket invalidation','two independent actual-checkpoint evaluations verified','retain PPL observations and measured work; do not treat current5.x failure as proof old runs failed; protocol-comparison validity qualified','no PPL rerun for unchanged measured checkpoint solely from this finding; seek original interface/runtime record for calibration conclusion',f'{v} run entry/stage logs; checkpoint config; PPL result+windows+model hashes')
add('v51','run_v51 -> runtime_v46 -> advance_block',True,'RUNTIME_PREFLIGHT explicitly5.16.1 + saved boundary content matches broadcast and independent full sequence differs','Tensor strongly supported; direct historical type not recorded','CONFIRMED_HISTORICAL_BOUNDARY_CONTENT','block0->layer1 boundary and layer1 failure input; block0 group validity independently checked in v53','boundary INVALID; block0 conditional v53 independent gate','none; D preserved','657.305s remains failed/diagnostic cost; not quality verification','reuse only block0 if v53 native group+boundary gate proves it; discard old boundary/failure input as production source','v51/RUNTIME_PREFLIGHT.json;v51/PREFIX_STATUS_layer00;v52/diagnosis;v53 native reference')
add('v52','run_v52 -> runtime_v46 -> advance_block',True,'runtime source and actual diagnostic Tensor[1,2048,896] recorded; historical saved boundary hashes corroborate','Tensor','CONFIRMED','layer1-23 inputs/168 execution quality protocol; block0 four inherited groups conditional; diagnostic final model INVALID','INVALID_REAL_PREFIX_AFTER_BLOCK0','WT2 incomplete and interrupted; C4 not started; no complete PPL','all work/cost retained; local MSE on erroneous inputs not full-model quality evidence','never resume layer1+ groups or diagnostic saved model; rebuild downstream after corrected boundary','v52/PREFIX_COMPATIBILITY_DIAGNOSIS.json;v52/MODEL_VALIDITY.json')
add('v47','profile_v46_l0_matrices -> runtime_v46.capture(L0)',False,'imports risky runtime but L0 timing path does not call advance_block','NOT_APPLICABLE for L0 timing path','NO_ON_THIS_PATH','local L0 up/down capture; v46 source calibration qualification separate','local L0 profiling not exposed to double decoder extraction','no new PPL','retain local measured bottleneck and overhead with source/workload scope; original v46 full calibration status separately qualified','retain profiling as local implementation timing, not proof of a valid whole-model repair','v47/profile_v46_l0_matrices.py')
add('v48','failure-state replay advances toL3; local full matrices captureL0',True,'L3 replay relies risky runtime; actual historical type UNKNOWN; L0 path only capture','UNKNOWN for L3 replay, NOT_APPLICABLE for L0 capture','UNKNOWN for L3; NO for L0 capture','failure/rejection replay could be affected if Tensor; L0 numerical mechanism independent','local L0 source/target tests retained; historical L3 replay qualification UNKNOWN','no new PPL','retain numerical results on their logged X; no full-model correctness claim from shared replay','preserve both failure and positive tests; corrected forward for future replay','v48/run_v48.py lines60,76,173')
add('v50','run_v50 -> runtime_v46.capture(L0)',False,'no advance_block in experiment production path','NOT_APPLICABLE','NO_ON_THIS_PATH','local mixed-Gram completeL0 inputs from locked v49 model','local numerical timing not invalidated by decoder output extraction; v49 quality qualification separate','no PPL','retain scoped local Gram numerical/cost evidence; not wholemodel quality proof','no Gram path adoption in v53; preserve existing local study','v50/run_v50.py capture0')
csvout('HISTORICAL_IMPACT.csv',impact)
dump('CPU_AUDIT_SUMMARY.json',{'cpu_wall_seconds':time.perf_counter()-t,'GPU_calls':0,'actual_decoder_index_sites':len(locations),'hook_double_indices_separate':len(excluded),'historical_checkpoint_models_hashed':len(checked_models),'historical_corpus_results_verified':len(quality),'UNKNOWN_rows_intentionally_retained':True,'current_replay_not_used_as_sole_historical_type_proof':True})
print('V53_CPU_HISTORY_AUDIT_COMPLETE',len(locations),'sites',len(quality),'PPL records',time.perf_counter()-t,flush=True)
