from pathlib import Path
import ast,hashlib,json,time
O=Path(__file__).resolve().parent
for n,d in [('DEPLOYMENT_ADAPTER_LOCK.json','diagnostics/DEPLOYMENT_ADAPTER_LOCK_ATTEMPT2.json'),('finish_v52.py','diagnostics/finish_v52_attempt2.py'),('DEPLOYMENT_FINISH_FAILURE.json','diagnostics/DEPLOYMENT_FINISH_FAILURE_ATTEMPT2.json'),('deployment_finish.log','diagnostics/deployment_finish_attempt2.log')]:
 assert not (O/d).exists();(O/d).write_bytes((O/n).read_bytes())
p=O/'finish_v52.py';s=p.read_text()
needle="   rt.MODE='evaluation';assert rt.P['v18']"
assert s.count(needle)==1
s=s.replace(needle,"   rt.MODE='deployment'\n   with rt.timer('finish_evaluation_model_reload'):\n    model=AutoModelForCausalLM.from_pretrained(str(OUT/'R1/checkpoint/final'),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()\n   assert isinstance(model,torch.nn.Module) and next(model.parameters()).is_cuda, 'real locked model required before evaluator'\n   rt.MODE='evaluation';assert rt.P['v18']")
p.write_text(s);ast.parse(s)
# No earlier attempt executed a model forward or produced a result/window file.
assert not any((O/f'R1_{ds}{suffix}').exists() for ds in ['wikitext','c4_fixed_budget'] for suffix in ['.json','_WINDOWS.csv'])
assert not list((O/'evaluation').glob('M2_FAST*'))
lock=json.loads((O/'DEPLOYMENT_ADAPTER_LOCK.json').read_text());lock['frozen_before_deployment_continue_unix']=time.time();lock['files']['finish_v52.py']=hashlib.sha256(p.read_bytes()).hexdigest();lock['attempt3_after_IO_adapter_bugs_preserved']=True
(O/'DEPLOYMENT_ADAPTER_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
(O/'DEPLOYMENT_ADAPTER_DIAGNOSTICS.json').write_text(json.dumps({'attempt1':'undefined outcfg before lock; no PPL invoked','attempt2':'model reload insertion missing from generated tail; stopped before first forward, zero PPL windows','repair_function_untouched':True,'all_failed_active_seconds_in_ledger':True,'attempt3_reloads_same_locked_checkpoint_and_checks_model_before_unchanged_evaluator':True,'no_evaluation_results_deleted':True},indent=2)+'\n')
print('RELOAD_INSERTION_PRESENT_FROZEN_NO_PRIOR_PPL_WINDOWS')
