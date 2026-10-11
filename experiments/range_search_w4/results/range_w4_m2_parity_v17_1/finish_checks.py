"""Bounded final evidence checks; no search or model evaluation."""
import ast, csv, json, time
from pathlib import Path
import torch
from run_audit import OUT, ROOT, V16, V17, cmp, compare_result, dump, sha, sha_file, load_core
from native_m2 import native_dynamic_gptq

def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))

def main():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    state=torch.load(OUT/'repro_state.pt',map_location='cuda',weights_only=False)
    w=state['W']; x=torch.cat([d[0] for d in state['docs']]).float()
    reference=state['native_output']; legacy=load_core()
    assert sha(reference[0])==read('NATIVE_REPEATABILITY.json')['reference_hashes']['weight']
    torch.cuda.synchronize(); started=time.perf_counter()
    # Reproduce the old 128-row diagnostic without comparing compensation
    # operators from unequal Hinv. The operator audit uses common native Hinv.
    old=legacy.dynamic_gptq(w.float(),state['custom']['Hinv'],alpha=1.,trace=True)
    restored=native_dynamic_gptq(w,state['H'],nsamples=2,alpha=1.,trace=True)
    torch.cuda.synchronize(); elapsed=time.perf_counter()-started
    original={
        'scope':'exact old v17 128-row diagnostic; unequal Hinv end-result reproduction only',
        'weight':cmp(reference[0],old.weight_fp32.half()),
        'scale':cmp(reference[1],old.scales),
        'reported_row17_col513':{'native':float(reference[0][17,513]),'custom':float(old.weight_fp32.half()[17,513])},
        'local_output_mse_native_first32':float((x@(reference[0][:32].float()-w[:32].float()).T).square().mean()),
        'local_output_mse_custom_first32':float((x@(old.weight_fp32[:32].half().float()-w[:32].float()).T).square().mean()),
        'groups':[{'group':i,'entry':cmp(restored.group_inputs[i],old.group_inputs[i]),'scale':cmp(reference[1][:,i],old.scales[:,i])} for i in range(7)],
        'supplemental_two_call_seconds_including_trace':elapsed,
    }
    prior=json.loads((V17/'M2_PARITY_DIAGNOSTIC.json').read_text())
    original['matches_v17_diagnostic']=(original['weight']['num_diff']==prior['weight_diff_count_fp16'] and original['scale']['num_diff']==prior['scale_diff_count'] and original['reported_row17_col513']['native']==prior['first_weight_mismatch']['native'] and original['reported_row17_col513']['custom']==prior['first_weight_mismatch']['custom'])
    dump('ORIGINAL_FAILURE_REPRODUCTION.json',original)

    level1=compare_result(w,x,reference,restored,rows=[17],cols=slice(512,640))
    level1.update(level=1,scope='reported failing row17 group4; original full target geometry preserved',
        group0_control=compare_result(w,x,reference,restored,rows=[17],cols=slice(0,128)))
    dump('PARITY_LEVEL1.json',level1)
    gidx=cmp(reference[3],restored.g_idx)
    expected=torch.arange(896,device=restored.g_idx.device,dtype=restored.g_idx.dtype)//128
    dump('GIDX_CHECK.json',{'native_vs_restored':gidx,'expected_group_boundaries':cmp(expected,restored.g_idx),'pass':gidx['exact'] and torch.equal(expected,restored.g_idx)})

    # Read-only integrity checks against existing frozen fingerprints.
    integrity=[]
    baseline=json.loads((V16/'ARTIFACT_SHA256.json').read_text())
    for rel,expected_hash in baseline.items():
        actual=sha_file(V16/rel)
        integrity.append({'path':str(V16/rel),'expected':expected_hash,'actual':actual,'pass':actual==expected_hash,'provenance':'v16 frozen artifact manifest'})
    fingerprints=json.loads((V16/'HISTORICAL_FINGERPRINTS.json').read_text())
    for rel in ['experiments/range_search_w4/dynamic_core.py','.venv-gptq/Lib/site-packages/gptqmodel/quantization/gptq.py','.venv-gptq/Lib/site-packages/transformers/models/qwen2/modeling_qwen2.py']:
        actual=sha_file(ROOT/rel)
        integrity.append({'path':str(ROOT/rel),'expected':fingerprints[rel],'actual':actual,'pass':actual==fingerprints[rel],'provenance':'v16 frozen historical fingerprint'})
    for info in read('SOURCE_MAP.json').values():
        if isinstance(info,dict) and 'file_sha256' in info:
            actual=sha_file(info['path'])
            integrity.append({'path':info['path'],'expected':info['file_sha256'],'actual':actual,'pass':actual==info['file_sha256'],'provenance':'loaded source map captured during this audit'})
    dump('HISTORICAL_INTEGRITY_CHECK.json',{'pass':all(i['pass'] for i in integrity),'checks':integrity,'scope_note':'v16 manifest and selected source fingerprints verified; v17 result files were read only, but no pre-turn v17 result hash manifest was available.'})

    required=['PROTOCOL.json','SOURCE_MAP.md','START_STATE_CHECK.json','FIRST_DIVERGENCE.json','operator_trace_native.json','operator_trace_custom.json','DTYPE_TIMELINE.csv','HINV_AUDIT.json','QUANTIZER_AUDIT.json','COMPENSATION_AUDIT.json','NATIVE_REPEATABILITY.json','CUSTOM_REPEATABILITY.json','PATCH_MANIFEST.csv','REPORT.md']+[f'PARITY_LEVEL{i}.json' for i in range(1,6)]
    assert all((OUT/name).is_file() for name in required)
    json_count=csv_count=0
    for p in OUT.glob('*.json'):json.loads(p.read_text(encoding='utf-8'));json_count+=1
    for p in OUT.glob('*.csv'):
        with p.open(encoding='utf-8',newline='') as f:list(csv.DictReader(f))
        csv_count+=1
    for p in OUT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    def event_keys(name):
        return {(e['operation'],e['column'],t) for e in read(name)['events'] for t in e['tensors']}
    coverage=(event_keys('operator_trace_native.json')==event_keys('operator_trace_custom.json'))
    comp=read('COMPENSATION_AUDIT.json');nr=read('NATIVE_REPEATABILITY.json');cr=read('CUSTOM_REPEATABILITY.json')
    flags={
        'corrected_start':read('START_STATE_CHECK.json')['corrected_start_pass'],
        'trace_coverage_complete':coverage,
        'trace_exact':comp['all_compared_group0_operations_exact'],
        'instrumentation_exact':all(c['exact'] for c in comp['instrumentation_did_not_change_results'].values()),
        'native_repeat5':nr['all_bitwise_repeatable'] and nr['runs']==5,
        'custom_repeat5':cr['restored_native_core_full_shape']['all_bitwise_repeatable'] and cr['restored_native_core_full_shape']['runs']==5,
        'native_vs_custom':cr['native_vs_restored']['pass'],
        'ladder':all(read(f'PARITY_LEVEL{i}.json')['pass'] for i in range(1,6)),
        'g_idx':read('GIDX_CHECK.json')['pass'],
        'old_failure_reproduced':original['matches_v17_diagnostic'],
        'history_integrity':all(i['pass'] for i in integrity),
    }
    verdict='M2_NATIVE_PARITY_RESTORED' if all(flags.values()) else 'M2_NATIVE_PARITY_NOT_RESTORED'
    summary=read('SUMMARY.json');summary['verdict']=verdict;summary['final_checks']=flags
    summary['legacy32_corrected_scale_mismatch']=read('LEGACY_REPAIR_CONTROL.json')['scale_first32']
    dump('SUMMARY.json',summary)
    report=(OUT/'REPORT.md').read_text(encoding='utf-8')
    report=report.replace('group0/row17, complete row17','reported failing group4/row17 (plus group0 control), complete row17')
    report+='\n## Final artifact checks\n\n'
    report+=f'Original full128-row failure reproduced against v17: {original["matches_v17_diagnostic"]}. Original FP16 mismatches: {original["weight"]["num_diff"]}; scale mismatches: {original["scale"]["num_diff"]}. The row17/col513 values match the saved v17 diagnostic. ORIGINAL_FAILURE_REPRODUCTION.json includes group entry/scale checks and the two original local MSEs.\n\n'
    report+='Correcting damping alone yields exact full128-row weight and scale, while the separate legacy32-row execution still differs at 24 scale elements (max 7.450580596923828e-09). The restored interface retains full native projection geometry and does not qualify independent row slicing.\n\n'
    report+=f'Reported failing group4/row17 Level1: {level1["pass"]}; group0 control: {level1["group0_control"]["pass"]}; g_idx and boundaries: {read("GIDX_CHECK.json")["pass"]}. Native/custom trace tensor coverage matches in both directions: {coverage}. v16 and recorded source checksum checks pass: {flags["history_integrity"]}. No pre-turn hash manifest exists for v17 result files; they were read only in this audit. Final verdict after every required check: **{verdict}**. STOP.\n\n'
    report+='Supplemental verification command: `.\\.venv-gptq\\Scripts\\python.exe experiments/range_search_w4/results/range_w4_m2_parity_v17_1/finish_checks.py`. It runs two bounded alpha=1 calls on the same k_proj and validates existing artifacts; it does not resume any experiment.\n'
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    dump('ARTIFACT_VALIDATION.json',{'pass':all(flags.values()),'checks':flags,'json_files_parsed':json_count,'csv_files_parsed':csv_count,'python_ast':'PASS','required_artifacts_present':True})
    with (OUT/'PATCH_MANIFEST.csv').open('a',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['run_audit.py / upstream_probe.py / finish_checks.py','New bounded reproduction and evidence checks','Preserve historical callers; identify reduction mismatch and verify strict parity','NO (audit only)','NO'])
    dump('ARTIFACT_SHA256.json',{p.name:sha_file(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['ARTIFACT_SHA256.json','finish_checks.log']})
    print(json.dumps({'verdict':verdict,'checks':flags,'original_reproduction':original['reported_row17_col513']},indent=2),flush=True)
    if not all(flags.values()):raise RuntimeError('Final evidence checks failed; STOP')

if __name__=='__main__':main()
