import ast,csv,difflib,hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(r'E:\quant-thesis');B=ROOT/'experiments/range_search_w4/results';O=B/'v52_m2_fast_integer_repair_fp64_initial_review';V51=B/'v51_m2_fast_integer_repair_fullmodel';V48=B/'v48_fast_numerical_acceptance'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):
 (O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
for d in ['formal/groups','formal/rows','R1/checkpoint/final','evaluation','reports','diagnostics','original_codes']: (O/d).mkdir(parents=True,exist_ok=True)
p=O/'.write_probe';p.write_text('ok');assert p.read_text()=='ok';p.unlink()
original=(V48/'fast_accept.py').read_text(encoding='utf8')
new=original.replace('from precision_check import gamma, cache_check, verify','from precision_check import gamma, cache_check, verify\nimport initialization_review')
oldline="    initial=rt.direct(X,E);assert bool(torch.isclose(initial,initial_quad,rtol=2e-5,atol=1e-12).all())"
assert original.count(oldline)==1
new=new.replace(oldline,"    initial=rt.direct(X,E);initialization_check=initialization_review.review(st,SC,W,X,C,initial,initial_quad,target)")
new=new.replace("acceptance_revision='v48 empirical FP64 envelope and independent raw-X recheck')","acceptance_revision='v48 empirical FP64 envelope and independent raw-X recheck',initialization_check=initialization_check)")
(O/'fast_accept_v52.py').write_text(new,encoding='utf8')
(O/'LOCAL_REPAIR.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='read-only v48/fast_accept.py',tofile='v52/fast_accept_v52.py')),encoding='utf8')
a=ast.parse(original);b=ast.parse(new)
for name in ['envelope','settle']:
 assert ast.dump(next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name==name))==ast.dump(next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name==name))
# Restoring the three explicitly allowed edits must restore the exact historical source.
assert new.replace('\nimport initialization_review','').replace('    initial=rt.direct(X,E);initialization_check=initialization_review.review(st,SC,W,X,C,initial,initial_quad,target)',oldline).replace(',initialization_check=initialization_check)',' )').replace("recheck' )","recheck')")==original
p=json.loads((V51/'PROTOCOL.json').read_text())
p['version']='v52';p['requested_model_setting']='Sol high requested; current interface cannot verify or switch settings.'
p['objective']='Continue M2 fast repair with only state-preserving independent raw-X FP64 fallback for the initial consistency check.'
p['v52_initialization_review']=dict(original_rtol=2e-5,original_atol=1e-12,failed_rows_only=True,independent_reference='raw X FP64 @ actual FP16-grid error, chunk4096, FP64 SSE/N',original_fp32_initial_and_quadratic_loss_retained=True,original_candidate_threshold_and_settle_unchanged=True,no_theoretical_certificate=True,source_key='run_id, phase, block, dependency_group; actual X hash/shape/dtype and C hash; no id(X) key or Gram reuse',core_sha256=sha(O/'fast_accept_v52.py'),review_sha256=sha(O/'initialization_review.py'),budget_sha256=sha(O/'budget.py'))
p['shared_budget']=dict(cap_seconds=10800,v51_measured_seconds=657.3046090602875,v51_conservative_charged_seconds=657.305,v52_initial_remaining_seconds=10142.695,active_job_segments_only=True,application_closed_wait_excluded=True,CPU_environment_diagnostics_separate=True,finish_reserve_seconds=600)
p['inherited_groups']=dict(source_version='v51',block=0,groups=['qkv','o','up_gate','down'],must_not_repair_again=True,uncommitted_layer1_qkv_start='original formal M2')
dump('PROTOCOL.json',p)
dump('PREPARATION.json',dict(status='CPU_SOURCE_AUDIT_PASS',writable_output_checked=True,version='v52'))
shutil.copy2(V51/'SOURCE_AUDIT.json',O/'SOURCE_AUDIT.json')
manifest=[]
for src in sorted((V51/'original_codes').glob('*.pt')):
 dst=O/'original_codes'/src.name;shutil.copy2(src,dst);assert sha(src)==sha(dst);manifest.append(dict(path=str(src),sha256=sha(src),copy=str(dst)))
for src in [V51/'PROTOCOL.json',V51/'REPORT.md',V51/'COST_LEDGER.csv',V51/'diagnostics/INITIAL_QUAD_FAILURE.json',V51/'formal/PREFIX_STATUS_layer_00.json',V51/'formal/COMMIT_STATUS.json',V48/'fast_accept.py',V48/'ACCEPTANCE_RULES.md']:
 manifest.append(dict(path=str(src),sha256=sha(src)))
for src in sorted((V51/'formal/groups').glob('*.pt')):manifest.append(dict(path=str(src),sha256=sha(src)))
dump('HISTORY_IMMUTABILITY_BEFORE.json',manifest)
shutil.copy2(V51/'formal/PREFIX_STATUS_layer_00.json',O/'formal/PREFIX_STATUS_layer_00.json')
rows=list(csv.DictReader((V51/'BLOCK_TIMINGS.csv').open()))
with (O/'BLOCK_TIMINGS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(r for r in rows if int(r['layer'])==0 and int(r['new_groups'])==4)
dump('INFRASTRUCTURE_DIAGNOSTICS.json',dict(status='普通沙盒仍失败，已获授权的沙盒外执行入口可用',app_version='26.1002.52244',app_build='13536',PowerShell='7.6.5',cause='setup refresh runtime ACL update fails with OS sharing violation32 on loaded node_repl/Swift runtime DLL',no_global_permission_changes=True,no_further_helper_termination=True,not_GPU_experiment_time=True,scope='CPU tools diagnosis, separated from active GPU job ledger'))
print('V52_CPU_PREPARED',sha(O/'PROTOCOL.json'),len(manifest))
