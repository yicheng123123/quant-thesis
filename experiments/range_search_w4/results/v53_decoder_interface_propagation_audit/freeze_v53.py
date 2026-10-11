from pathlib import Path
import hashlib,json,time
O=Path(__file__).resolve().parent
files=['PROTOCOL.json','decoder_interface.py','runtime_v53.py','validate_cpu.py','DETERMINISTIC_TESTS.json','budget_v53.py','validate_propagation.py','UNCHANGED_ALGORITHM_AUDIT.json','INTERFACE_CONTRACT.md']
assert json.loads((O/'DETERMINISTIC_TESTS.json').read_text())['pass']
lock={'frozen_before_GPU_unix':time.time(),'files':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in files},'no_tolerance_or_algorithm_adjustment':True}
(O/'FROZEN_VALIDATION_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
# Conservative prediction; native complete decoder execution with only final lm_head position retained.
(O/'GPU_BUDGET_PRECHECK.json').write_text(json.dumps({'prediction_not_measurement':True,'native_full_forward_calls':643,'seconds_per_reference_call_assumed':.75,'reference_prediction_seconds':643*.75,'extra_capture_IO_and_setup_reserve_seconds':70,'predicted_total_seconds':643*.75+70,'new_cap_seconds':600,'shared_remaining_seconds':10800-json.loads((O/'PROTOCOL.json').read_text())['prior_shared_charged_seconds'],'basis':'Prior actual block0 full128 refresh about2.8s;24blocks gives about0.525s/document;0.75s/reference plus70s extra is conservative extrapolation, not guarantee. Head logits_to_keep=1 removes unneeded projection only; decoder graph and tokens unchanged.','go':True},indent=2)+'\n')
print('V53_FROZEN_GO',643*.75+70)
