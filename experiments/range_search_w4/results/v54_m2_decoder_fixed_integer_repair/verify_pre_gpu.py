import csv,hashlib,importlib.util,json
from pathlib import Path
import torch
O=Path(__file__).resolve().parent;B=O.parent;R=O.parent
V18=R/'range_w4_fullmodel_v18';V51=R/'v51_m2_fast_integer_repair_fullmodel';V52=R/'v52_m2_fast_integer_repair_fp64_initial_review';V53=R/'v53_decoder_interface_propagation_audit';V48=R/'v48_fast_numerical_acceptance';V46=R/'v46_m1_single_integer_repair_fullmodel'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rd(p):return json.loads(Path(p).read_text(encoding='utf8'))
lock=rd(O/'FROZEN_RUN_LOCK.json');assert lock['pass']
for n,h in lock['files'].items():assert sha(O/n)==h,(n,sha(O/n),h)
assert sha(V53/'runtime_v53.py')==lock['v53_runtime_sha256']
assert sha(V53/'GPU_VALIDATION_RESULT.json')==lock['v53_validation_result_sha256']
assert rd(V53/'GPU_VALIDATION_RESULT.json')['pass']
assert sha(O/'runtime_v54.py')==sha(V53/'runtime_v53.py')
assert sha(V52/'fast_accept_v52.py')==lock['v52_core_sha256']
assert sha(V52/'initialization_review.py')==lock['v52_initialization_review_sha256']
assert sha(V48/'ACCEPTANCE_RULES.md')==lock['v52_acceptance_rules_sha256']
assert rd(O/'V54_COST_PRECHECK.json')['go']
assert sha(V53/'ACTIVE_JOB_LEDGER.json')==rd(O/'PROTOCOL.json')['budget_v54']['parent_ledger_sha256']
ledger=rd(O/'ACTIVE_JOB_LEDGER.json')
assert all('charged_seconds' in x for x in ledger['segments'])
total=ledger['v51_charged_seconds']+sum(x['charged_seconds'] for x in ledger['segments'])
assert abs(total-rd(O/'V54_COST_PRECHECK.json')['starting_shared_elapsed_seconds'])<1e-8
assert abs(10800-total-rd(O/'V54_COST_PRECHECK.json')['starting_shared_remaining_seconds'])<1e-8
assert total>=5677.428213768006
assert any(x.get('version')=='v54' and x.get('charged_seconds',0)>0 for x in ledger['segments'])
m2=V18/'M2/checkpoint/final/model.safetensors';m2sha=sha(m2)
assert m2sha==rd(O/'PROTOCOL.json')['initialization']['model_sha256']
cal=V18/'CALIBRATION_MANIFEST.json';assert sha(cal)==rd(O/'PROTOCOL.json')['calibration']['manifest_sha256']
calj=rd(cal);assert len(calj['samples'])==128 and all(len(s['token_ids'])==2048 for s in calj['samples'])
for ds,n in [('wikitext','WIKITEXT_TEST_MANIFEST.json'),('c4_fixed_budget','C4_EVAL_MANIFEST.json')]:
 assert sha(V18/'evaluation/manifests'/n)==rd(V46/'BASELINE_AUDIT.json')['evaluation_manifests'][ds]
groups=rd(O/'BLOCK0_INHERITANCE.json')['block0_groups']
assert len(groups)==4 and rd(O/'BLOCK0_INHERITANCE.json')['v52_layer1_plus_loaded'] is False
assert {x.name for x in (O/'formal/groups').glob('*.pt')}=={f'layer_00_{g}.pt' for g in ['qkv','o','up_gate','down']}
orig={i:torch.load(O/f'original_codes/layer_{i:02d}.pt',map_location='cpu',weights_only=True) for i in range(24)}
checked=[]
for g in groups:
 p=V51/f"formal/groups/layer_00_{g['group']}.pt";dst=O/f"formal/groups/layer_00_{g['group']}.pt"
 assert sha(p)==g['source_sha256'] and sha(dst)==g['v54_group_sha256']
 cp=torch.load(dst,map_location='cpu',weights_only=False)
 assert cp['committed'] and cp['layer']==0 and cp['group']==g['group'] and cp['protocol_sha256']==sha(O/'PROTOCOL.json')
 assert cp['input_sha256']==g['input_sha256']
 for n,s in cp['modules'].items():
  z=orig[0][n]
  for k in ['scales','zeros','g_idx']:assert torch.equal(s[k],z[k])
  codes=s['codes'].long();assert bool(((codes>=0)&(codes<=15)).all())
  assert torch.equal(torch.stack((s['packed_u4']&15,s['packed_u4']>>4),dim=-1).flatten(1).long(),codes)
  q=(s['scales'][:,s['g_idx'].long()]*(codes.float()-8)).half()
  assert torch.equal(q.view(torch.int16),s['weight'].half().view(torch.int16))
  assert torch.equal(s['weight'].view(torch.int16),((z['scales'][:,z['g_idx'].long()]*(codes.float()-8)).half()).view(torch.int16))
 checked.append({'group':g['group'],'source_hash_pass':True,'protocol_hash_pass':True,'codes_grid_packing_pass':True,'module_count':len(cp['modules']),'input_sha256':cp['input_sha256']})
files=list(O.rglob('*.pt'));formal=[p.name for p in (O/'formal/groups').glob('*.pt')]
assert len(formal)==4 and not list(O.rglob('*.npy'))
compile((O/'run_v54.py').read_text(encoding='utf8'),str(O/'run_v54.py'),'exec')
compile((O/'budget.py').read_text(encoding='utf8'),str(O/'budget.py'),'exec')
assert all((O/p).is_dir() for p in ['formal/groups','formal/rows','R1/checkpoint/final','evaluation','reports'])
result={'pass':True,'GPU_started':False,'model_loads':0,'inherited_shared_elapsed_seconds':total,'remaining_shared_seconds':10800-total,'M2_checkpoint_sha256':m2sha,'calibration_manifest_sha256':sha(cal),'v53_runtime_unchanged':True,'frozen_fast_and_init_review_hashes_pass':True,'block0_groups_verified':checked,'formal_group_checkpoints_present':formal,'v52_layer1plus_params_inputs_or_C_copied':False,'V52_original_code_maps_only':len(orig)==24,'evaluation_manifests_locked':True,'runner_syntax':True,'finish_reserve_seconds':360,'precheck_go':True}
(O/'PRE_GPU_CHECK.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
