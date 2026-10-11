import json,datetime,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;p=O/'ACTIVE_JOB_LEDGER.json';d=json.loads(p.read_text());s=d['segments'][-1];assert 'charged_seconds' not in s
# Identity was recorded immediately before Stop-Process. Entire command wall is a conservative upper bound after that timestamp.
end=(O/'diagnostics/STOPPED_PROCESS_IDENTITY.json').stat().st_mtime+0.678968
s.update(end_epoch=end,charged_seconds=max(0,end-s['start_epoch']),forced_stop=True,charged_end_is_conservative_upper_bound=True,end_basis='identity file mtime immediately before Stop-Process + total measured stop-command wall0.678968s')
p.write_text(json.dumps(d,indent=2)+'\n')
stop={'status':'FAILED_STOP','category':'D','complete_model':False,'PPL_run':True,'PPL_complete':False,'PPL_reused':False,'reason':'Frozen tuple-style advance_block indexing is incompatible with installed Transformers5.16.1 Tensor return; per-document first token is broadcast as next-block hidden states','all168_matrices_terminal_local_checks_passed':True,'correct_real_prefix_propagation':False,'model_must_not_be_used_as_valid_v52_quality_result':True,'original_config_stop_preserved':'diagnostics/CONFIG_ASSERTION_STOP.json','partial_evaluation_log_preserved':'deployment_finish.log','remaining_budget_not_a_reason_to_resume_invalid_protocol':True}
(O/'RUN_STATUS.json').write_text(json.dumps(stop,indent=2)+'\n');(O/'PREFIX_COMPATIBILITY_STOP.json').write_text(json.dumps(stop,indent=2)+'\n')
print('FORCED_STOP_LEDGER_CLOSED',s['charged_seconds'],'shared',d['v51_charged_seconds']+sum(x['charged_seconds'] for x in d['segments']))
