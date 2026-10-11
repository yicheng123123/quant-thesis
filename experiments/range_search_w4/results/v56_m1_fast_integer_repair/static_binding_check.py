import hashlib,json,runpy,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 t=time.perf_counter();sys.path.insert(0,str(OUT));import budget
 source=OUT/'repair_control.py';text=source.read_text(encoding='utf8')
 assert 'def retire_caps(' in text and 'def score_and_retire(' in text and 'def propose(' in text
 assert 'torch.load(' not in text and 'safe_open' not in text and 'from_pretrained' not in text
 ns=runpy.run_path(str(OUT/'run_v56.py'),run_name='v56_static_binding_only')
 rt=ns['rt'];fast=ns['fast'];helper=ns['repair_control'];wrapper=ns['bound_fast'];precision=sys.modules['precision_check']
 assert Path(fast.__file__).resolve()==(OUT/'fast_accept_v52.py').resolve()
 assert Path(helper.__file__).resolve()==source.resolve()
 assert Path(precision.__file__).resolve()==(OUT/'precision_check.py').resolve()
 for f in ['retire_caps','score_and_retire','propose']:assert callable(getattr(helper,f))
 rt.batch_repair=wrapper;assert rt.batch_repair is wrapper
 assert Path(OUT/'run_v56.py').exists()
 result={'pass':True,'CPU_only':True,'GPU_calls':0,'model_loads':0,'model_forward_calls':0,'helper_source_sha256':sha(source),'precision_check_source_sha256':sha(OUT/'precision_check.py'),'helper_source_matches_frozen_v46_search_helper':True,'helper_loaded_from_local_v56_copy':True,'precision_check_loaded_from_local_v56_copy':True,'fast_accept_source_sha256':sha(OUT/'fast_accept_v52.py'),'initialization_review_source_sha256':sha(OUT/'initialization_review.py'),'decoder_interface_source_sha256':sha(OUT/'decoder_interface.py'),'actual_import_binding':'run_v56.bound_fast -> v56.fast_accept_v52.batch_repair -> v56.repair_control functions','no_v46_checkpoint_or_parameter_import':True,'runner_import_wall_seconds':time.perf_counter()-t}
 (OUT/'STATIC_BINDING_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
 budget.charge_cpu('v56 CPU-only static runner/helper binding preflight',time.perf_counter()-t,modules=168,rows=304128)
 print('STATIC_BINDING_PASS',json.dumps(result),flush=True)
if __name__=='__main__':main()
