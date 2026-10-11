import json, hashlib, os, shutil, time
from pathlib import Path
import torch
from transformers import AutoConfig
OUT=Path(__file__).resolve().parent; BASE=OUT.parent; ROOT=OUT.parents[3]; V18=BASE/'range_w4_fullmodel_v18'; V55=BASE/'v55_m2_decoder_fixed_integer_repair'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def jwrite(n,v): (OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def main():
 t=time.perf_counter()
 for d in ['formal/groups','formal/rows','original_codes','evaluation','reports','diagnostics','M1_REPAIRED/checkpoint/final']: (OUT/d).mkdir(parents=True,exist_ok=True)
 # Start the standalone cap with a conservative CPU-only allowance for the read-only preflight already completed before this ledger existed.
 ledger_path=OUT/'ACTIVE_JOB_LEDGER.json'
 if not ledger_path.exists():
  ledger={'version':'v56','cap_seconds':9000.0,'preflight_charged_seconds':120.0,'preflight_charge_basis':'Conservative CPU-only allowance for the read-only preflight already completed before this ledger existed; no model/GPU calls.','segments':[],'historical_shared_v55_seconds_separate':11037.312983,'historical_shared_v55_not_in_v56_cap':True,'idle_intervals_excluded':True}
  ledger_path.write_text(json.dumps(ledger,indent=2)+'\n',encoding='utf8')
 else:
  ledger=json.loads(ledger_path.read_text(encoding='utf8')); assert ledger['version']=='v56' and ledger['cap_seconds']==9000.0
 # Verify empty output destinations and writability.
 for d in ['formal/groups','original_codes','evaluation','reports','M1_REPAIRED/checkpoint/final']:
  q=OUT/d/'.write_probe'; q.write_text('v56 writable\n',encoding='utf8'); q.unlink()
 # The M1 source is the official v18 artifact; scale/zero/g_idx are loaded from its native layer files.
 m1=V18/'M1/checkpoint/final';m2=V18/'M2/checkpoint/final';runtime=json.loads((V18/'FORMAL_RUNTIME.json').read_text(encoding='utf8')); v18p=json.loads((V18/'PROTOCOL.json').read_text(encoding='utf8'))
 m1sha=sha(m1/'model.safetensors');m2sha=sha(m2/'model.safetensors')
 assert m1sha==runtime['m1']['final_model_sha256'] and m2sha==runtime['m2']['final_model_sha256']
 assert sha(V18/'CALIBRATION_MANIFEST.json')==v18p['source_protocol']['v16_calibration_manifest_sha256']
 assert v18p['model']['revision']=='7ae557604adf67be50417f59c2c2f167def9a775'
 cfgsha=sha(m1/'config.json')
 proto=json.loads((V55/'PROTOCOL.json').read_text(encoding='utf8'))
 proto['name']='v56 M1 initialization + v55 frozen fast integer repair, correct-prefix full-model validation'
 proto['status']='FROZEN_BEFORE_GPU'
 proto['version']='v56'
 proto['model']={'id':v18p['model']['id'],'revision':v18p['model']['revision'],'dtype':'FP16','device':'NVIDIA GeForce RTX 4060 Laptop GPU','attention':'SDPA','use_cache':False,'tf32':False}
 proto['initialization']={'method':'M1 native GPTQ initialization','checkpoint':str(m1),'model_sha256':m1sha,'config_sha256':cfgsha,'model_revision':v18p['model']['revision'],'quantization_metadata_sha256':sha(V18/'FORMAL_RUNTIME.json'),'alpha_selection_source':'already completed formal v18 M1; all selected_alpha=1.0; no search or regeneration','native_layer_source_pattern':str(V18/'checkpoints/M1/layer_XX.pt'),'integer_code_source':'unique exact enumeration against native FP16 M1 dense weights and that same source layer scales/zero/g_idx; no v46 parameters or M2 codes'}
 proto['comparison']={'arms':['M1','M2','v56_M1_plus_repair','v55_M2_plus_repair'],'primary_metric':'valid-token-weighted mean NLL','historic_M1_M2_v55_evaluations':'reuse after source/manifest/checkpoint audit','v56_evaluations':'one fresh WT2 test and one fresh fixed-budget C4 validation','interaction':'descriptive only'}
 proto['evaluation']['reuse_rule']='Run v56 checkpoint once on each frozen manifest. Reuse only audited historical M1, M2 and v55 results; never reuse v46 repair quality.'
 proto['evaluation']['manifest_sha256']={'wikitext':'acd46b20a6b84b534d5625bfab505d599001679a775d619960a40e8da4f62f46','c4_fixed_budget':'0b0d70d70d3c1e0ba62126c81a3f3e97531e58d6701236fb8b3315450b7188d4'}
 proto['budget_v56']={'cap_seconds':9000,'finish_reserve_seconds':360,'prior_v55_shared_ledger_seconds_separate':11037.312983,'not_a_reset':True,'no_v55_cost_added_to_v56_cap':True}
 proto['provenance']={'decoder_interface_source':'v53 validated strict decoder interface, copied from v55 frozen runtime','fast_repair_source':'v55 runtime and v52/v49/v48 implementation unchanged','M1_start_layer':0,'reused_repair_groups':0,'reused_historical_activations_or_C':False,'read_P3':False}
 (OUT/'PROTOCOL.json').write_text(json.dumps(proto,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 # Reuse exact frozen implementation files copied before this script. Verify byte identities against v55.
 for f in ['runtime_v54.py','decoder_interface.py','fast_accept_v52.py','initialization_review.py']:
  assert sha(OUT/f)==sha(V55/f),f'implementation copy mismatch: {f}'
 # Source audit performs unique code recovery from official M1 parameters and checks PPL baselines.
 import sys
 sys.path.insert(0,str(OUT)); import audit_sources
 audit_sources.main()
 # Record the exact effective model config equality through the same installed loader.
 c1=AutoConfig.from_pretrained(str(m1),local_files_only=True).to_dict();c2=AutoConfig.from_pretrained(str(m2),local_files_only=True).to_dict()
 drop={'_name_or_path','transformers_version','_commit_hash'}
 normalize=lambda c:{k:v for k,v in c.items() if k not in drop}
 eff_equal=normalize(c1)==normalize(c2)
 assert eff_equal,'M1 and M2 effective inference config differs'
 jwrite('INITIALIZATION_PREFLIGHT.json',{'pass':True,'M1_model_sha256':m1sha,'M2_model_sha256':m2sha,'M1_config_sha256':cfgsha,'M2_config_sha256':sha(m2/'config.json'),'M1_M2_effective_config_equal':eff_equal,'M1_native_grid_code_recovery':'unique exact FP16 grid match, all 168 modules','reused_v55_groups':0,'reused_v46_repair_parameters':False,'grid_roundtrip_test':'all M1 native rows uniquely recovered, dequantized bitwise and packed/unpacked','all_required_output_dirs_precreated_writable':True,'torch_version':torch.__version__,'GPU_calls':0,'preparation_script_seconds':time.perf_counter()-t})
 jwrite('NUMERICAL_VALIDATION.json',json.loads((V55/'NUMERICAL_VALIDATION.json').read_text(encoding='utf8')))
 jwrite('CPU_TESTS.json',json.loads((V55/'CPU_TESTS.json').read_text(encoding='utf8')))
 # Separate historical ledger provenance; it is never summed into v56's 9000 seconds.
 v55ledger=json.loads((V55/'ACTIVE_JOB_LEDGER.json').read_text(encoding='utf8'))
 hist={'historical_v55_shared_total_seconds':11037.312983,'source_path':str(V55/'FINAL_BUDGET_ACCOUNTING.json'),'source_sha256':sha(V55/'FINAL_BUDGET_ACCOUNTING.json'),'source_ledger_sha256':sha(V55/'ACTIVE_JOB_LEDGER.json'),'cap_seconds':12600.0,'not_in_v56_budget':True,'source_ledger_completed_segments':len(v55ledger['segments'])}
 jwrite('HISTORICAL_COST_SEPARATE.json',hist)
 # Explicitly audit v55 result provenance; source module mismatch in its unused copied audit helper is disclosed, not treated as run evidence.
 qsrc=json.loads((V55/'QUALITY_EVALUATION_SOURCE.json').read_text(encoding='utf8'))
 assert qsrc['pass'] and qsrc['new_PPL_evaluations']==2
 jwrite('V55_ARM_SOURCE_AUDIT.json',{'pass':True,'quality_evaluation_source_sha256':sha(V55/'QUALITY_EVALUATION_SOURCE.json'),'reported_final_model_sha256':'0fa87105e36deac2ead4c871e6e0a802590418e6ccacd616929cdf91ecc8ec0c','historical_quality_measured_fresh':True,'completed_group_scale_audit':'v55 formal group grid matched official M2 scales/zero/g_idx; no v56 recovery from these groups','unused_stale_audit_helper':'v55 local audit_sources.py currently describes M1 source; v55 prepare_v55 copied original_codes from v54 M2 and formal committed groups match M2 grid. We do not use that helper or any v55/v46 group parameters for v56.'})
 # Add exact CPU preparation wall charge, measured from this script only.
 import budget
 budget.charge_cpu('v56 CPU-only protocol/source audit and exact M1 code recovery',time.perf_counter()-t,modules=168,rows=304128)
 prep={'status':'CPU_SOURCE_AUDIT_PASS','version':'v56','cap_seconds':9000.0,'M1_model_hash_verified':True,'calibration_manifest_verified':True,'M1_native_grid_recovery_passed':True,'all_168_modules_and_304128_rows_recovered':True,'runtime_fast_accept_initialization_review_decoder_sources_copied_byte_equal':True,'no_v55_or_v46_repair_groups_loaded':True,'all_directories_writable':True,'GPU_calls':0,'torch_version':torch.__version__,'cpu_seconds':time.perf_counter()-t,'preflight_prior_allowance_seconds':120.0}
 jwrite('PREPARATION.json',prep)
 print('PREPARATION_PASS',json.dumps(prep,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
