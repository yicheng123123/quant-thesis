"""Finalize evidence, run the declared artifact gates, and write reports."""
from pathlib import Path
import ast, csv, hashlib, json

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
V16=OUT.parent/'range_w4_fullmodel_baseline_v16';V17=OUT.parent/'range_w4_fullmodel_smoke_v17';V171=OUT.parent/'range_w4_m2_parity_v17_1'
def sha_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads((OUT/p).read_text(encoding='utf-8'))
def dump(p,x):(OUT/p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    history=[]
    for root in [V16,V171]:
        manifest=json.loads((root/'ARTIFACT_SHA256.json').read_text(encoding='utf-8'))
        for rel,expected in manifest.items():
            actual=sha_file(root/rel);history.append({'path':str(root/rel),'expected_sha256':expected,'actual_sha256':actual,'pass':actual==expected})
    fps=json.loads((V16/'HISTORICAL_FINGERPRINTS.json').read_text(encoding='utf-8'))
    for rel in ['experiments/range_search_w4/dynamic_core.py','.venv-gptq/Lib/site-packages/gptqmodel/quantization/gptq.py']:
        actual=sha_file(ROOT/rel);history.append({'path':str(ROOT/rel),'expected_sha256':fps[rel],'actual_sha256':actual,'pass':actual==fps[rel]})
    oldmap=json.loads((V171/'SOURCE_MAP.json').read_text(encoding='utf-8'))
    for key in ['old_external_hinv','old_custom']:
        item=oldmap[key];actual=sha_file(item['path']);history.append({'path':item['path'],'expected_sha256':item['file_sha256'],'actual_sha256':actual,'pass':actual==item['file_sha256']})
    cert=read('CERTIFIED_PATH_CHECK.json');actual=sha_file(cert['certified_path'])
    history.append({'path':cert['certified_path'],'expected_sha256':cert['certified_source_sha256'],'actual_sha256':actual,'pass':actual==cert['certified_source_sha256']})
    dump('HISTORICAL_INTEGRITY_CHECK.json',{'pass':all(x['pass'] for x in history),'check_count':len(history),'checks':history,
        'scope':'v16 and v17.1 artifact manifests; v16 frozen source fingerprints; v17 custom source and caller; certified native_m2 implementation. All writes remain in v17.2.'})

    raw=read('M2_SMOKE_RAW.json');timing=read('TIMING_SUMMARY.json');landscape=list(csv.DictReader((OUT/'M2_SEARCH_LANDSCAPES.csv').open(encoding='utf-8',newline='')))
    assert len(landscape)==320 and len({(int(r['row']),float(r['alpha'])) for r in landscape})==320
    assert all(float(r['search_mse'])>=0 and float(r['weight_mse'])>=0 for r in landscape)
    assert raw['status']=='COMPLETE' and raw['candidate_rows_completed']==320 and raw['expected_candidate_rows']==320
    alpha1=read('M2_EMBEDDED_ALPHA1_PARITY.json')['candidate_alpha1']
    isolation=read('CANDIDATE_ISOLATION.json');commit=read('M2_COMMIT_CORRECTNESS.json')
    selfp=read('SELF_PREFIX_PROPAGATION.json');transition=read('MODULE_TRANSITION_SMOKE.json');grid=read('M2_GRID_EQUIVALENCE.json')
    qkv=read('QKV_INPUT_CHECK.json');capture=read('CAPTURE_SEMANTICS.json');cost=read('M2_COST_ESTIMATE.json')
    flags={'certified_path':read('CERTIFIED_PATH_CHECK.json')['pass'],'alpha1_tiny':read('ALPHA1_PREFLIGHT_PARITY.json')['pass'],
        'capture_semantics':capture['pass'],'qkv_inputs_exact':qkv['pass'],'embedded_alpha1':alpha1['pass'],
        'candidate_count_320':len(landscape)==320,'candidate_order_invariant':isolation['pass'],
        'selected_commit_exact':commit['pass'],'self_prefix_propagation':selfp['status']=='PASS',
        'module_transition':transition['pass'],'grid_equivalence':grid['pass'],
        'calibration_manifest_unchanged':read('PROTOCOL.json')['calibration_manifest_sha256']==sha_file(V16/'CALIBRATION_MANIFEST.json'),
        'historical_v8_v17_1_integrity':all(x['pass'] for x in history),
        'conservative_cost_le_48h':cost['conservative_hours']<=48}
    ready=all(flags.values());verdict='M2_FULLMODEL_SMOKE_READY' if ready else 'M2_FULLMODEL_SMOKE_NOT_READY'
    dump('correctness_checks.json',{'verdict':verdict,'checks':flags,'gate_decision':cost['status'],
        'M3_smoke_allowed':ready,'formal_full_model_search_allowed':False,'PPL_allowed':False})

    selected=raw['selected_alpha_by_row'];baseline=raw['candidates']['1.0']['search_mse']
    selected_loss=[raw['candidates'][str(selected[str(r)])]['search_mse'][r] for r in range(32)]
    relative=[(baseline[r]-selected_loss[r])/baseline[r]*100 if baseline[r] else 0.0 for r in range(32)]
    summary={'verdict':verdict,'m2_cost_gate':cost['status'],'candidate_row_alpha_scores':320,'full_shape_alpha_executions':10,
        'alpha_counts':raw['selected_count_by_alpha'],'search_subset_only':True,'heldout_evaluated':False,'ppl_evaluated':False,
        'search_mse_mean_alpha1':sum(baseline)/len(baseline),'search_mse_mean_selected':sum(selected_loss)/len(selected_loss),
        'mean_row_relative_search_mse_gain_percent':sum(relative)/len(relative),
        'rows_with_positive_search_gain':sum(v>0 for v in relative),'rows_with_zero_search_gain':sum(v==0 for v in relative),
        'min_row_relative_search_gain_percent':min(relative),'max_row_relative_search_gain_percent':max(relative),
        'native_candidate_seconds':timing['candidate_per_call_ms'],'per_10alpha_row_ms':timing['per_10alpha_row_ms'],
        'prefix_drift':read('SELF_PREFIX_PROPAGATION.json'),'cost_estimate_hours':cost['cost_scenarios'],
        'history_integrity_checks':len(history),'source_manifest_integrity_pass':all(x['pass'] for x in history)}
    dump('SUMMARY.json',summary)

    warning={'observed':'Transformers emitted: Sliding Window Attention is enabled but not implemented for sdpa.',
        'checkpoint_config':{k:json.loads((ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/config.json').read_text())[k] for k in ['sliding_window','use_sliding_window','max_position_embeddings']},
        'protocol':'v16 mandates SDPA and use_cache=False. Checkpoint has use_sliding_window=false; the smoke sequences have 64 tokens.',
        'classification':'nonblocking environment warning; recorded for reproducibility'}
    dump('ENVIRONMENT_WARNINGS.json',warning)

    with (OUT/'PATCH_MANIFEST.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['path','change','historical_files_modified'])
        w.writerow(['prepare_smoke.py','Certified source check, frozen data loading, true Linear input hooks, preflight parity','NO'])
        w.writerow(['run_m2_smoke.py','320 row-alpha smoke, order isolation, direct hook comparison, row commits, prefix transition and cost estimate','NO'])
        w.writerow(['finalize_v17_2.py','Artifact/checksum verification, summary and report generation','NO'])
        w.writerow(['v17.2 result artifacts','New isolated smoke evidence','NO'])

    issues=[
      {'stage':'source_preflight','issue':'decorated GPTQ.quantize initially resolved to PyTorch contextlib wrapper','resolution':'inspect.unwrap used; actual loaded GPTQ source hash matched frozen v17.1 source map','effect':'no candidates ran before source gate passed'},
      {'stage':'capture_preflight','issue':'first capture shape declaration used projection weight geometry for up/gate/down instead of forward-input geometry','resolution':'expected activation shapes corrected; actual q/k/v and up/gate sibling inputs verified bitwise equal','effect':'no candidates ran before capture gate passed'},
      {'stage':'candidate_runner','issue':'first smoke invocation stopped on CPU/GPU tensor equality after preflight parity','resolution':'comparison devices normalized'},
      {'stage':'candidate_runner','issue':'second smoke invocation completed all320 scores then stopped on g_idx equality using mismatched devices','resolution':'g_idx expected vector moved to CPU; resumed from persisted full-alpha candidate cache; subsequent parity gates passed'}]
    dump('EXECUTION_ISSUES.json',{'issues':issues,'scientific_candidate_failure':False,'completed_alpha_grid_rerun':False,
        'note':'The completed alpha grid is preserved in the local progress cache; the successful resume reused it and re-executed isolation/reference controls.'})

    counts=raw['selected_count_by_alpha'];timing=read('TIMING_SUMMARY.json')
    rows=['# Updated M2 compute budget','',
      'Frozen scale: 24 decoder blocks, 168 Linear modules, 304,128 output rows and 3,041,280 row-alpha scores.',
      'The output-shape inventory separates 48 k/v modules at 128x896, 48 q/o modules at 896x896, 48 up/gate modules at 4864x896, and 24 down modules at 896x4864.',
      'Each full-target alpha call produces scores for every output row of that Linear. The projected 1,680 GPTQ executions are not 3,041,280 separate GPU launches. Candidate executions rebuild native Hinv; shared Hessian construction is estimated separately.','',
      '| Scenario | GPTQ candidates | Input capture | Shared Hessian | Commit/refresh | Selection/export | Checkpoints | Total |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in cost['cost_scenarios'].items():
        rows.append(f"| {name} | {s['core_candidate_seconds']/3600:.2f} h | {s['sequential_capture_seconds']/3600:.2f} h | {s['shared_Hessian_build_seconds']/3600:.2f} h | {s['commit_refresh_seconds']/3600:.2f} h | {s['selection_export_seconds']/3600:.2f} h | {s['checkpoint_seconds']/3600:.3f} h | {s['total_hours']:.2f} h |")
    rows+=['',f"Conservative estimate: **{cost['conservative_hours']:.2f} hours**; registered gate is **{cost['status']}** (48 hours maximum). The gate fails, so M2 is not ready for formal full-model search.",
      '',f"Certified measurements: k/v 128x896 median {cost['measured_core_seconds']['k_v_128x896_median']:.6f}s; q/o 896x896 representative q {cost['measured_core_seconds']['q_o_896x896_transition_q']:.6f}s; up/gate 4864x896 one run {cost['measured_core_seconds']['up_4864x896_single']:.6f}s; down 896x4864 one run {cost['measured_core_seconds']['down_896x4864_single']:.6f}s. Large-shape timing is a single pilot, and all candidate timing includes native Hinv reconstruction.",
      '',f"Shared-H estimate scales the frozen v15 32-H build time 2.053s by 336 to {cost['shared_Hessian_build_central_seconds']:.3f}s. Sequential capture uses frozen 133.076s full-prefix timing scaled from 16x128 to 128x2048 tokens; refreshed prefixes use four GPTQ module groups per layer. Those terms dominate and are extrapolated. Checkpoint writes assume two 942.3 MiB snapshots at 180 MiB/s. No cost assumption was changed to cross the gate.",
      '',f"Calibration smoke alpha counts for 32 k_proj rows: {json.dumps(counts,ensure_ascii=False)}. These are search-subset outcomes only, not held-out or PPL results.",'']
    (OUT/'UPDATED_M2_COMPUTE_BUDGET.md').write_text('\n'.join(rows),encoding='utf-8')

    report=f'''# v17.2 M2 full-model smoke resume\n\nVerdict: **{verdict}**.\n\nThe certified implementation path and SHA matched v17.1. The alpha=1 preflight exactly matched native GPTQ for weight, scale, zero, g_idx and output MSE. Actual pre-hooks on all seven L0 projections captured the Linear inputs. q/k/v inputs were bitwise equal, and the GPTQ order `[k,v,q] -> [o] -> [up,gate] -> [down]` is documented separately from the forward dependency graph.\n\nThe L0 k_proj experiment completed 320 row-alpha scores from the frozen v16 calibration samples 0 and 1, first 64 tokens each. Ten full 128x896 native-core calls scored the 32 fixed rows. Alpha counts were `{json.dumps(counts,ensure_ascii=False)}`. Candidate-order reversal, embedded alpha=1, direct native range-hook comparison on four rows across all ten alphas, selected commit hashes, and absence of nonselected-state leakage passed.\n\nSelf-prefix passed. Committing row 0 at alpha 0.9 changed the downstream o_proj input: relative L2 drift {selfp['relative_L2_drift']:.6g}, maximum absolute change {selfp['max_abs_diff']:.6g}. The module transition committed the k/v/q group, captured a refreshed o_proj input, built its Hessian from that activation, committed native alpha=1 o_proj, then captured current up/gate/down inputs.\n\n**Cost gate: {cost['status']}, conservative estimate {cost['conservative_hours']:.2f} hours.** This exceeds the frozen 48-hour limit. The run therefore ends at `{verdict}`; no M3 smoke, full-model search, or PPL was run. See `UPDATED_M2_COMPUTE_BUDGET.md` for measured timings and explicit capture/Hessian/refresh/checkpoint assumptions.\n\nNo held-out or final-model improvement is claimed; the landscape uses only the frozen search subset. Runner issues encountered before successful completion are itemized in `EXECUTION_ISSUES.json`.\n'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')

    required=['PROTOCOL.json','CERTIFIED_PATH_CHECK.json','CAPTURE_SEMANTICS.md','CAPTURE_SEMANTICS.json','SEQUENTIAL_QUANT_ORDER.json','FORWARD_DEPENDENCY.json','QKV_INPUT_CHECK.json','M2_EMBEDDED_ALPHA1_PARITY.json','CANDIDATE_ISOLATION.json','M2_SMOKE_RAW.json','M2_SEARCH_LANDSCAPES.csv','M2_COMMIT_CORRECTNESS.json','SELF_PREFIX_PROPAGATION.json','MODULE_TRANSITION_SMOKE.json','M2_GRID_EQUIVALENCE.json','M2_TIMING_SMOKE.csv','UPDATED_M2_COMPUTE_BUDGET.md','correctness_checks.json','REPORT.md','SUMMARY.json','PATCH_MANIFEST.csv']
    assert all((OUT/name).is_file() for name in required)
    for p in OUT.glob('*.json'):json.loads(p.read_text(encoding='utf-8'))
    for p in OUT.glob('*.csv'):
        with p.open(encoding='utf-8',newline='') as f:list(csv.DictReader(f))
    for p in OUT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    dump('ARTIFACT_VALIDATION.json',{'required_artifacts_present':True,'landscape_rows':len(landscape),'historical_hashes_pass':all(x['pass'] for x in history),
        'json_csv_python_parse':'PASS','verdict':verdict,'M3_started':False})
    files={p.name:sha_file(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['ARTIFACT_SHA256.json','finalize.log']}
    dump('ARTIFACT_SHA256.json',files)
    print(json.dumps({'verdict':verdict,'checks':flags,'selected_counts':counts,'cost_gate':cost['status'],'conservative_hours':cost['conservative_hours'],'history_files_checked':len(history)},indent=2),flush=True)

if __name__=='__main__':main()
