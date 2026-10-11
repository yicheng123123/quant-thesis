"""CPU-only v53 delivery aggregation. No model forward, repair or PPL."""
import csv,hashlib,json,time
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parent;t=time.perf_counter()
def j(n):return json.loads((O/n).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def readcsv(n):return list(csv.DictReader((O/n).open(encoding='utf8')))
def csvout(n,rs):
 with (O/n).open('w',newline='',encoding='utf8') as f:
  cols=list(dict.fromkeys(k for r in rs for k in r));w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rs)
for n,h in j('FROZEN_VALIDATION_LOCK.json')['files'].items():assert sha(O/n)==h,f'frozen validation changed{n}'
history=[]
for r in j('HISTORY_BEFORE.json'):
 h=sha(Path(r['path']));assert h==r['sha256'];history.append(dict(path=r['path'],before_sha256=r['sha256'],after_sha256=h,unchanged=True))
# Verify additional audited historical source/checkpoint/result files against their live source audit.
for r in readcsv('CALLSITE_INVENTORY.csv'):assert sha(Path(r['file']))==r['source_sha256']
for r in readcsv('HISTORICAL_PPL_PROVENANCE.csv'):
 assert sha(Path(r['result_path']))==r['result_sha256'];assert sha(Path(r['windows_path']))==r['windows_sha256']
for r in j('HISTORICAL_CHECKPOINT_SOURCES.json').values():assert sha(Path(r['path']))==r['sha256'] and sha(Path(r['config']))==r['config_sha256']
dump('HISTORY_IMMUTABILITY_AFTER.json',{'pass':True,'snapshots':history,'audited_extra_sources_and_checkpoints_unchanged':True,'historical_reports_not_rewritten':True})
r=j('GPU_VALIDATION_RESULT.json');assert r['pass'];assert j('DETERMINISTIC_TESTS.json')['pass'];L=j('ACTIVE_JOB_LEDGER.json');assert all('charged_seconds' in s for s in L['segments']);new=sum(s['charged_seconds'] for s in L['segments'] if s.get('version')=='v53');total=L['v51_charged_seconds']+sum(s['charged_seconds'] for s in L['segments']);assert new<=600 and total<=10800
assert sha(Path(L['prior_ledger_read_only_path']))==L['prior_ledger_sha256'];assert abs(total-(L['v53_start_shared_charged_seconds']+new))<1e-8
# Validation reference/cache byte identity is separately recorded from .npy header file hash.
import numpy as np
rawrecords=[]
for p in sorted((O/'validation_cache').glob('*.npy')):
 a=np.load(p,mmap_mode='r');h=hashlib.sha256()
 for k in range(len(a)):h.update(np.ascontiguousarray(a[k]).tobytes())
 rawrecords.append({'path':str(p),'file_sha256':sha(p),'raw_tensor_content_sha256':h.hexdigest(),'shape':list(a.shape),'dtype':str(a.dtype),'role':'independent validation evidence; future formal run must replay its own model prefix'})
dump('VALIDATION_INPUT_LOCK.json',{'pass':True,'source_protocol_sha256':sha(O/'PROTOCOL.json'),'inputs':rawrecords,'C_cache_built_or_reused':False})
assert all(g['historical_source_input_matches_native'] for g in j('BLOCK0_GROUP_SOURCE_CHECKS.json')['groups'])
cp_sources=[{'group':g,'path':str(B/'v51_m2_fast_integer_repair_fullmodel/formal/groups'/f'layer_00_{g}.pt'),'sha256':sha(B/'v51_m2_fast_integer_repair_fullmodel/formal/groups'/f'layer_00_{g}.pt')} for g in ['qkv','o','up_gate','down']]
boundary=j('BLOCK0_BOUNDARY_INDEPENDENT_CHECK.json')
resume={'decision':'ALLOW_BLOCK0_ONLY_WITH_CORRECTED_OWN_PREFIX_REPLAY','base_model':str(B/'range_w4_fullmodel_v18/M2/checkpoint/final'),'base_model_sha256':boundary['source_model_checkpoint_sha256'],'block0_groups_to_restore_without_repair':cp_sources,'source_v52_containers_verified_tensors_equal_v51':True,'independent_full128_group_input_validation_pass':True,'frozen_grid_code_pack_and_terminal_source_checks_pass':True,'first_unrepaired_dependency_group':'layer1/qkv','layer1_and_all_later_parameters':'original formal M2, not v52 repaired parameters','forbidden_sources':['v52/formal/groups/layer_01_* through layer_23_*','v52/R1/checkpoint/final diagnostic model','v51/v52 old broadcast hidden boundary and layer1 failure X','all incomplete tentative state'],'next_run_must_replay_prefix_with':str(O/'runtime_v53.py'),'expected_boundary':{'run_context':'originalM2 + exactly listed committedblock0 groups','block':0,'dependency_group':'down->layer1/qkv','hidden_shape':[128,2048,896],'hidden_dtype':'float16','hidden_raw_content_sha256':boundary['correct_hidden_sha256'],'layer1_qkv_X_shape':[262144,896],'layer1_qkv_X_dtype':'float16','layer1_qkv_X_raw_content_sha256':boundary['correct_layer1_qkv_X_sha256']},'v53_validation_cache_not_auto_production_activation_cache':True,'unknown_grid_parameters_or_any_source_mismatch':'stop; do not reuse block0; next authorized formal run must start all groups from originalM2','no_next_model_or_PPL_authorized_by_this_manifest':True}
dump('VALID_RESUME_BOUNDARY.json',resume)
usage=[{'artifact':'formal v18 M1/M2 checkpoints and original PPL','use':'retain initializer and measured baseline','rebuild_or_reevaluate_required':'not from this evidence; v18 recorded4.49 supports tuple compatibility','qualification':'do not run unadapted legacy driver under5.x'}, {'artifact':'v45 R2/v46 R1/v49 fast R1 checkpoint PPL observations','use':'retain actual saved-model PPL and raw windows','rebuild_or_reevaluate_required':'no automatic rerun of same checkpoints; seek stronger historical interface evidence','qualification':'full calibration return type/protocol UNKNOWN; descriptive quality difference is not frozen-method causal proof'}, {'artifact':'v51/v52 inherited block0 four committed groups','use':'allowed after v53 independent native-forward gate','rebuild_or_reevaluate_required':'no repeat block0 repair; replay corrected fulltoken boundary','qualification':'only exact listed M2/grid/source states; all128 group and boundary inputs checked'}, {'artifact':'v52 layer1-23 repaired groups and diagnostic final checkpoint','use':'failure/diagnosis only','rebuild_or_reevaluate_required':'rebuild layer1/qkv onwards from originalM2; evaluate a new valid locked fullmodel after future authorized repair','qualification':'cannot resume or reuse its unfinished PPL'}, {'artifact':'v51/v52 old boundary/row122 failure input','use':'numerical regression fixture only','rebuild_or_reevaluate_required':'replace production boundary by actualownprefix replay and corrected hash','qualification':'matching old hash did not prove correct native input'}, {'artifact':'v47/v48 L0/v50 local timing/numerical evidence','use':'retain with matrix/source scope','rebuild_or_reevaluate_required':'not solely due decoder-return extraction on those capture-only paths','qualification':'v48 L3 failure replay historical type UNKNOWN; no fullmodel/PPL claims'}, {'artifact':'v28/v29/v40 old-runtime records','use':'retain old4.49 scoped artifacts and observations','rebuild_or_reevaluate_required':'no blanket invalidation; current5.x legacy calls require adaptation','qualification':'PPL for these expanded versions not re-audited here'}, {'artifact':'v36/v37 legacy prefix/boundary entries','use':'preserve evidence','rebuild_or_reevaluate_required':'UNKNOWN until their own runtime/interface evidence is sufficient','qualification':'no new held/P3 data read or evaluation'}]
csvout('USE_DECISIONS.csv',usage)
scan=j('diagnostics/CPU_SCAN_ATTEMPT1.json');cpu=j('CPU_AUDIT_SUMMARY.json')
cost=[dict(item='v51+v52 carried shared charge',seconds=L['v53_start_shared_charged_seconds'],basis='exact original ledger; user display5278.453 preserved'),dict(item='v53 added GPU validation active-job wall',seconds=new,basis='includes modelload, warmup, source IO, comparisons and all validation; cap600'),dict(item='new shared cumulative charge',seconds=total,basis='prior+v53; cap10800'),dict(item='shared remaining',seconds=10800-total,basis='not reset'),dict(item='CPU initial broad static scan wall to stop request',seconds=scan['stop_epoch']-scan['start_epoch'],basis='stopped own CPU audit task, not a helper; request timing only, original log retained'),dict(item='CPU initial static scan process CPU time',seconds=scan['process_CPU_seconds'],basis='process cumulative CPU seconds; not wall and not GPU'),dict(item='CPU completed narrowed history audit wall',seconds=cpu['cpu_wall_seconds'],basis='sourceinventory+checkpoint/window hashing; zero GPU calls; overlaps GPU job wall'),dict(item='CPU deterministic tests',seconds=j('DETERMINISTIC_TESTS.json')['CPU_seconds'],basis='21cases; zero GPU calls'),dict(item='CPU preparation',seconds=j('PREPARATION.json')['cpu_seconds'],basis='sourcecopy/ASTdiff/protocol/snapshot'),dict(item='CPU environment/tool diagnosis independent total',seconds=None,basis='NOT_MEASURED; used authorized entry, no environment repair'),dict(item='CPU supplementary inventory/report/package audit',seconds=None,basis='partial per-tool wall exists; not claimed as independent exclusive total')]
csvout('COST_LEDGER.csv',cost)
dump('CPU_DIAGNOSTIC_EVENTS.json',{'GPU_calls':0,'events':[{'event':'broad static scan deliberately stopped and narrowed','record':'diagnostics/CPU_SCAN_ATTEMPT1.json','no_helper_process_terminated':True},{'event':'v18 old result field differences handled as average_nll/perplexity','original_values_unchanged':True},{'event':'CPU resume builder initially used invalid encoding name utf8-sig; corrected to utf-8-sig','GPU_cost':0,'independent_failure_time':'NOT_MEASURED','no_historical_mutation':True}],'all_costs_separate_from_gpu_budget':True})
result={'status':'COMPLETE','interface_and_independent_propagation':'PASS','deterministic_cases':21,'fixed_articles':[0,64,127],'full_24_layer_input_comparison_articles':3,'all128_fulltoken_block0_group_and_boundary_validation':'PASS bitwise','block0_reuse_allowed':True,'layer1plus_reuse_allowed':False,'next_run_initialization':'originalM2 + verified inherited block0, corrected ownprefix replay, layer1/qkv originalM2','historical_calibration_interface_unknown':['v42-45 full recovery chain','v46','v49','v36','v37','v48 L3 diagnostic replay'],'historical_PPL_actual_checkpoint_measurements_verified':10,'new_PPL_or_fullmodel_repair_runs':0,'v53_added_charged_seconds':new,'shared_cumulative_charged_seconds':total,'shared_remaining_seconds':10800-total,'old_ledger_and_history_read_only':True,'unchanged_algorithm':True,'requested_model_setting':'Sol high; interface cannot verify','no_next_version_started':True}
dump('DECISION.json',result)
(O/'CORRECTION_NOTICE.md').write_text('''# v53 historical correction (historical reports remain unchanged)

v52's initialFP64 fallback regression is valid for the exact saved numerical fixture; it was not proof that that fixture came from a correct model prefix. All168 local matrix executions and grid/reload checks did not establish fullmodel protocol validity. The actual current5.16 decoder returns Tensor; old tuple extraction broadcasted the first token. Historical v51/v52 boundary content corroborates this error. v52 layer1+ groups and its diagnostic final model are forbidden for future recovery, and its incomplete WT2 run is not PPL evidence.

v53 independently verifies native fullforward/hook inputs for all128 block0 dependency states and the true layer1 boundary. OriginalM2 is loaded, each committedblock0 group is reconstructed in the original order, and no v52 layer1+ parameters are loaded. Their exact source inputs match the historical group hashes and their grids/codes/weights/terminal records remain intact. This establishes the limited block0 reuse boundary; it does not rehabilitate later v52 artifacts. Future formal recovery must replay full tokens and use the corrected boundary, not the historical broadcast hash.

Current library replay alone does not establish old execution types. v18/v28/v29/v40 explicitly record4.49; that API declares tuple output. v45/v46/v49 saved config metadata supports4.49, but direct historical calibration return types were not logged across the full chain: UNKNOWN is retained. No blanket invalidation of those calibrations is asserted. Ten old PPL result/window/checkpoint/hash records are independently source-audited and continue to describe the models actually measured; this does not certify their intended calibration algorithm or justify speed/causal claims across an unknown protocol boundary.

Historical timing remains measured work with its original scope. Invalid v51/v52 work and failures stay in the cost ledger; do not compare their reduced-update erroneous-input runs with a future corrected run as an algorithm speedup. Local L0 capture-only studies v47/v48/v50 do not consume decoder output and are not directly exposed to this extraction bug; v48's L3 diagnostic replay uses advance_block and retains a separate historical uncertainty.

This is an interface/propagation correctness correction. It does not alter Gram/p/code/grid/search/budgets/numeric acceptance, create a new quantized model, or establish quality/PPL/new-method novelty. Next full repair and PPL require a separate explicit task.
''',encoding='utf8')
(O/'REPORT.md').write_text(f'''# v53 REPORT

**接口修复与独立传播验证通过。block0可限定复用；v52 layer1以后必须重建。**

## 修复与验证

新实现为独立`runtime_v53.py`及`decoder_interface.py`。仅支持冻结flags下的Tensor[B,T,H]和合法单元素tuple；未知类型明确报错。提取返回原Tensor，不改内容/dtype；所有embedding、qkv、hook和block刷新缓存先验证完整shape/dtype严格相等，再写入，禁止广播。

最小diff仅改变embed/capture/advance_block。其他函数/类AST保持一致；C/p、固定整数网格、候选顺序、32更新上限和评价预算、v52初始化FP64复核与接受规则未改变。相关源码与验证协议已在GPU前冻结，之后hash未变。

21个确定性用例通过，含可区分token的Tensor/tuple、首token误取、shape/dtype错误、未知返回、非有限值和拒绝后缓存不变。

真实参考由模型原生`model.forward`和输入prehook独立取得，参考路径不调用新提取/缓存函数。每次保留完整2048 token和全部24 Transformer；仅lm_head输出选择最后一个位置，减少不影响层输入的验证开销，不计算loss/PPL。

- 正式128篇，按原M2→qkv→o→up/gate→down重建block0状态：4组全部输入、sibling共享输入和历史source hash逐字节一致。
- 原M2＋4个已提交block0组，正式128篇block0→layer1：完整hidden及qkv输入与原生forward逐字节一致；旧广播边界被拒绝。
- 预冻结文章0/64/127，各完整2048 token，24层hidden和qkv输入对照全部逐字节一致。该24层逐输入比较覆盖3篇；不能改写成128篇×24层全部逐输入对照。
- 原code/grid/packed和实际FP16权重与v51/v52继承状态一致。仅在内存重建锁定参数；block0以外参数版本未变。没有新模型checkpoint、C构建、修复、held或PPL。

## 历史影响

`CALLSITE_INVENTORY.csv`列出{cpu['project_callsite_risk_locations']}个项目危险取法（含分步索引和经norm别名的等价路径），archive角色单列；无调用关系的vendor及hook tuple/batch取法另表，不混为decoder错误。

v51/v52广播有历史输入/边界内容和环境证据，layer1后传播无效。v18/v28/v29/v40记录4.49，支持tuple接口，未发现同类Tensor误用证据。v45/v46/v49保存元数据支持4.49，但历史直接返回类型缺少日志，全链校准有效性仍为UNKNOWN，不能仅凭当前重放认定有效/无效。v36/v37和v48 L3 replay相应不确定性列入影响表。

5个历史模型的10份旧PPL核验通过：实际checkpoint SHA、result/window SHA、连续窗口、manifest/token与valid-token加权NLL一致。可继续作为对应真实模型的测量值使用，不自动证明其校准传播符合算法协议。不要求仅因本次发现重评同一旧checkpoint；也不使用旧PPL替代未来新模型评测。

## 可执行恢复边界

从**原正式M2**重建完整模型，仅装入`VALID_RESUME_BOUNDARY.json`列出的v51 block0四组，跳过重复修复；用v53新接口重放本次完整prefix。期望正确hidden raw hash：`{boundary['correct_hidden_sha256']}`；正确layer1 qkv raw hash：`{boundary['correct_layer1_qkv_X_sha256']}`。

随后从**layer1/qkv，原M2 code/scale/zero/g_idx**开始。不得载入v52 layer1–23的已修复组、诊断最终模型、旧广播边界或失败tentative状态。若来源/网格/新边界不匹配，停止，后续从原M2全部重新开始。v53验证缓存只是对照证据，不自动成为新生产activation缓存；未来仍从自身模型重放。旧boundary预期必须由独立正确记录替换，并在下一任务冻结，不改历史报告。

本轮未启动下一正式运行。剩余时间不等于正确输入下完整修复一定能容纳；必须依据正确传播另行作保守成本检查，不用v52错误输入上的执行时间承诺速度。

## 成本与交付

- 原共同账本精确携带：{L['v53_start_shared_charged_seconds']:.6f}s（用户显示5278.453）。
- v53新增验证活跃作业：**{new:.3f}s / 600s**，包含预热、载入、IO、来源、全部比较；实际643次原生完整forward，不作PPL。
- 共同累计：**{total:.3f}s / 10800s**；剩余**{10800-total:.3f}s**。未重置旧657.305s或任何失败。应用等待不计入。
- CPU首次扫描到停止请求{scan['stop_epoch']-scan['start_epoch']:.3f}s，进程CPU{scan['process_CPU_seconds']:.3f}s；收窄后完成审计{cpu['cpu_wall_seconds']:.3f}s，确定性测试{j('DETERMINISTIC_TESTS.json')['CPU_seconds']:.3f}s，另列。前两者与验证wall可能重叠，不相加冒充GPU时间。只停止本轮CPU审计任务，未结束自动重启的辅助进程。
- 阶段实测`GPU_VALIDATION_TIMINGS.csv`，内部嵌套打点`STAGE_TIMINGS.csv`；不要把嵌套表相加冒充exclusive total。预测552.25s与实测分开。

交付：最小diff/接口契约、冻结lock、确定性和独立传播记录、30处callsite与历史影响/PPL来源表、纠正说明、恢复清单、输入hash/shape/dtype锁、原始阶段与预算、复现命令。历史来源前后hash均核验，报告和checkpoint保持只读。

沿用已授权可用执行入口；本轮未重测普通沙盒或扩大环境修复。Sol high是请求设置，接口不能确认实际设置。完成后停止，不自动新版本/整模型/PPL，也不声明质量提升或理论保证。
''',encoding='utf8')
(O/'REPRODUCE.md').write_text('''# v53 reproduction (PowerShell at E:/quant-thesis)

All current validations have completed. No selection/rerun is performed. Historical files remain read-only; all new output remains inside this directory.

CPU deterministic tests (21 cases; no CUDA):
    .\\.venv\\Scripts\\python.exe -B experiments/range_search_w4/results/v53_decoder_interface_propagation_audit/validate_cpu.py

Executed single bounded native propagation validator (already passed; do not silently reset/rerun budget):
    .\\.venv\\Scripts\\python.exe -B experiments/range_search_w4/results/v53_decoder_interface_propagation_audit/validate_propagation.py
It refuses a second completed run. budget_v53.py carries the immutable v52 ledger into the new local ledger; newGPU cap600, sharedcap10800. Native reference hooks do not use decoder_interface.py. Calibrated prefix uses originalM2 plus four verified block0 states only.

CPU history audit sequence (does not execute historical models):
    .\\.venv\\Scripts\\python.exe -B experiments/range_search_w4/results/v53_decoder_interface_propagation_audit/audit_history.py
    .\\.venv\\Scripts\\python.exe -B experiments/range_search_w4/results/v53_decoder_interface_propagation_audit/expand_callsite_impact.py
    .\\.venv\\Scripts\\python.exe -B experiments/range_search_w4/results/v53_decoder_interface_propagation_audit/complete_norm_alias_inventory.py

CPU final delivery/source verification:
    .\\.venv\\Scripts\\python.exe -B experiments/range_search_w4/results/v53_decoder_interface_propagation_audit/finalize_v53.py

No full-model repair or PPL command is supplied/started. Future authorized driver must load the corrected local runtime as run_v42, then unchanged v52 fast core/initialization review, retain full terminal checks/provenance, restore only the listedblock0 groups, replay its own fulltoken prefix, and start originalM2 layer1/qkv. See VALID_RESUME_BOUNDARY.json and LEGACY_CALLSITE_HANDLING.md. Do not execute archived legacy drivers unadapted under5.x.
''',encoding='utf8')
dump('CPU_FINALIZATION.json',{'cpu_wall_seconds_before_manifest':time.perf_counter()-t,'GPU_calls':0,'history_after_hash_checks':len(history),'report_and_protocol_fidelity_pass':True})
manifest=[]
for p in sorted(O.rglob('*')):
 if p.is_file() and p.name not in ['DELIVERY_MANIFEST.json','finalize.log'] and '__pycache__' not in p.parts:manifest.append({'path':str(p.relative_to(O)),'sha256':sha(p),'bytes':p.stat().st_size})
dump('DELIVERY_MANIFEST.json',{'status':'COMPLETE','files':manifest,'no_quantization_or_PPL':True,'history_read_only':True})
print('V53_DELIVERY_COMPLETE',json.dumps(result),flush=True)
