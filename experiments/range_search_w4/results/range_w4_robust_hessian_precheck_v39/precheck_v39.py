"""Standard-library-only artifact precheck. No tensor deserialization or model imports.

Writes only beside this script; reads an explicit source allowlist, never P3 data.
Use: .venv-gptq/Scripts/python.exe -B <this file>
"""
import csv
import hashlib
import io
import json
import math
import sys
import time
from collections import defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RES = OUT.parent
V30 = RES / 'range_w4_weight_error_regularization_v30'
V24 = RES / 'range_w4_structured_group_e2_v24'
V25 = RES / 'range_w4_dense_scalar_audit_v25'
V38 = RES / 'range_w4_evidence_synthesis_v38'
EARLY = RES / 'range_w4_omniquant_full_v21_1' / 'M3'
M3 = RES / 'range_w4_omniquant_resume_v21_4' / 'M3'
FINAL = RES / 'range_w4_omniquant_finalize_v21_5'
T0, CPU0 = time.perf_counter(), time.process_time()
SOURCES, CHECKS, PHASES = {}, [], []


def budget():
    if time.process_time() - CPU0 >= 300 or time.perf_counter() - T0 >= 300:
        raise TimeoutError('300s CPU / conservative wall execution stop')


def digest(path):
    path = Path(path).resolve()
    budget()
    assert path.is_relative_to(ROOT), path
    # No dataset/capture contents are read. PPL files below are old evaluated windows.
    assert 'P3' not in path.name and 'held' not in path.as_posix().lower(), path
    if path in SOURCES:
        return SOURCES[path]['sha256']
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            budget()
            h.update(chunk)
    SOURCES[path] = {'path': path.relative_to(ROOT).as_posix(),
                     'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
    return h.hexdigest()


def read_json(path):
    digest(path)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    digest(path)
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def check(name, value, detail=None):
    CHECKS.append({'name': name, 'pass': bool(value), 'detail': detail})


def phase(name, start):
    PHASES.append({'phase': name, 'wall_seconds': time.perf_counter() - start})


def m3_audit():
    start = time.perf_counter()
    for p in [V38/'EVIDENCE_TABLE.csv', V38/'FINAL_ROUTE_STATUS.md',
              V25/'audit_dense_v25.py', V25/'REPORT.md', FINAL/'REPORT.md']:
        digest(p)
    early = read_json(EARLY/'M3_PARTIAL_SUMMARY.json')
    old_lock = read_json(EARLY/'FORMAL_PROTOCOL_LOCK.json')
    lock = read_json(M3/'FORMAL_PROTOCOL_LOCK.json')
    summary = read_json(M3/'M3_TRAINING_SUMMARY.json')
    resume = read_json(M3/'RESUME_STATE_VALIDATION.json')
    early_pointer = read_json(EARLY/'checkpoints/LATEST.json')
    early_block = EARLY/'checkpoints'/early_pointer['checkpoint_dir']
    early_manifest = read_json(early_block/'CHECKPOINT_MANIFEST.json')
    check('M3_resume_source_manifest', digest(early_block/'CHECKPOINT_MANIFEST.json') == early_pointer['manifest_sha256'] == resume['historical_source_manifest_sha256'])
    for k in ['state','rng']:
        check('M3_resume_source_'+k, digest(early_block/early_manifest[k+'_file']) == early_manifest[k+'_file_sha256'])
        check('M3_isolated_prefix_'+k, digest(M3/'checkpoints/block_20'/early_manifest[k+'_file']) == early_manifest[k+'_file_sha256'])
    digest(M3/'OMNIQUANT_PATCH.diff')
    read_json(M3/'OMNIQUANT_SOURCE_AUDIT.json')
    pointer = read_json(M3/'checkpoints/LATEST.json')
    block = M3/'checkpoints'/pointer['checkpoint_dir']
    manifest = read_json(block/'CHECKPOINT_MANIFEST.json')
    check('M3_latest_manifest', digest(block/'CHECKPOINT_MANIFEST.json') == pointer['manifest_sha256'])
    for k in ['state', 'rng']:
        check('M3_checkpoint_'+k, digest(block/manifest[k+'_file']) == manifest[k+'_file_sha256'])
    check('M3_completed_24', manifest['completed_layer_count'] == 24)
    config = RES/'range_w4_omniquant_kaggle_v20_1/kaggle_bundle/configs/m3_preflight.json'
    check('M3_same_training_config', digest(config) == lock['config_sha256'] == old_lock['config_sha256'])
    for k in ['model_revision','model_weight_sha256','calibration_manifest_sha256','ordered_token_hashes_sha256']:
        check('M3_same_'+k, old_lock[k] == lock[k])
    check('M3_same_quantization', old_lock['quantization'] == lock['quantization'])
    export_manifest = read_json(M3/'DENSE_EXPORT_MANIFEST.json')
    check('M3_export_168', export_manifest['status'] == 'PASS' and export_manifest['linear_modules_replaced'] == 168)
    for entry in export_manifest['files']:
        check('M3_export_file_'+entry['path'], digest(M3/'dense_export'/entry['path']) == entry['sha256'])
    export_sha = digest(M3/'dense_export/model.safetensors')
    eval_protocol = read_json(M3/'EVALUATION_PROTOCOL.json')
    v18 = read_json(RES/'range_w4_fullmodel_v18/PROTOCOL.json')
    reuse = read_csv(V25/'BASELINES_REUSED.csv')
    evaluations = []
    for label, stem, dataset in [('WT2','WT2_M3_FINAL.json','wikitext'), ('C4','C4_M3_FINAL.json','c4')]:
        final = read_json(FINAL/stem)
        rawpath, windowpath = Path(final['raw_source']), Path(final['window_source'])
        raw, windows = read_json(rawpath), read_csv(windowpath)
        check(label+'_raw_hash', digest(rawpath) == final['raw_json_sha256'])
        check(label+'_window_hash', digest(windowpath) == final['window_csv_sha256'])
        check(label+'_model_hash', all(f['sha256'] == export_sha for f in raw['model_weight_files']))
        check(label+'_status', raw['status'] == final['status'] == 'PASS')
        check(label+'_window_indices', [int(r['window_index']) for r in windows] == list(range(raw['window_count'])))
        check(label+'_window_raw_match', all(all(float(r[k]) == float(j[k]) for k in ['begin','end','new_targets','valid_tokens','window_loss','window_nll']) for r,j in zip(windows, raw['window_results'])))
        targets = sum(int(r['valid_tokens']) for r in windows)
        nll = math.fsum(float(r['window_nll']) for r in windows)
        ppl = math.exp(nll/targets)
        check(label+'_recompute', targets == final['targets'] and abs(ppl-final['ppl']) <= 1e-10)
        check(label+'_window_settings', raw['max_length'] == 2048 and raw['stride'] == 512 and not raw['add_special_tokens'] and not raw['chat_template'])
        manifest_path = M3/Path(raw['evaluation_manifest'])
        check(label+'_evaluation_manifest', digest(manifest_path) == raw['evaluation_manifest_sha256'] == v18['source_protocol']['v16_'+('wikitext' if label=='WT2' else 'c4')+'_manifest_sha256'])
        baseline = next(r for r in reuse if r['method']=='M3' and r['dataset']==('wikitext' if label=='WT2' else 'c4_fixed_budget'))
        check(label+'_v25_reuse', abs(float(baseline['ppl'])-ppl) <= 1e-10 and baseline['evaluation_token_ids_sha256']==raw['evaluation_token_ids_sha256_i64le'])
        evaluations.append({'dataset':dataset,'targets':targets,'windows':len(windows),'mean_nll':nll/targets,'ppl':ppl,'raw_sha256':digest(rawpath),'window_sha256':digest(windowpath),'checkpoint_sha256':export_sha,'token_ids_sha256':raw['evaluation_token_ids_sha256_i64le'],'evaluation_started_utc':raw['started_utc'],'evaluation_completed_utc':raw['completed_utc'],'new_evaluation':False})
    result = {'resolution':'SUMMARY_OMISSION_OF_COMPLETED_CONTINUATION',
              'early_protocol_id':old_lock['protocol_id'], 'early_status':early['status'],
              'early_attempt_id':resume['historical_step_log_audit']['attempt_id'],
              'completed_protocol_id':lock['protocol_id'], 'completed_attempt_id':summary['attempt_id'],
              'source_block20':resume['historical_source_checkpoint'],
              'source_block20_manifest_sha256':resume['historical_source_manifest_sha256'],
              'final_checkpoint_manifest_sha256':digest(block/'CHECKPOINT_MANIFEST.json'),
              'completed_blocks':manifest['completed_layer_count'],
              'quantization':lock['quantization'], 'model_revision':lock['model_revision'],
              'training_config_sha256':digest(config),
              'source_audit_changed':old_lock['omniquant_source_audit_sha256'] != lock['omniquant_source_audit_sha256'],
              'historical_prefix_successful_steps_directly_observed':False,
              'no_training_or_evaluation_rerun':True,'evaluations':evaluations}
    write_json('M3_SOURCE_RECONCILIATION.json', result)
    write_csv('M3_EVALUATION_RECOMPUTE.csv', evaluations)
    phase('M3_artifact_hash_and_window_reaggregation', start)
    return result


def fraction(x):
    # Match v30 float -> shortest decimal conversion, then exact rational arithmetic.
    return F(str(float(x)))


def decimal_string(x):
    if x is None:
        return 'infinity'
    with localcontext() as ctx:
        ctx.prec = 60
        return str(Decimal(x.numerator)/Decimal(x.denominator))


def selection_space():
    start = time.perf_counter()
    candidates = read_csv(V30/'SEARCH_CANDIDATE_TABLE.csv')
    old_paths = read_csv(V30/'SELECTION_PATH.csv')
    old_boundaries = read_csv(V30/'SELECTION_PATH_BOUNDARIES.csv')
    full_intervals = read_csv(V30/'FULL_PATH_INTERVAL_CHOICES_SEARCH_ONLY.csv')
    full_boundaries = read_csv(V30/'FULL_PATH_BOUNDARY_CHOICES_SEARCH_ONLY.csv')
    kappas = read_csv(V30/'KAPPA_NORMALIZATION.csv')
    for p in [V30/'analyze_v30.py',V30/'RUN_SUMMARY.json', V24/'run_e2_v24.py', V24/'PROTOCOL.json',
              ROOT/'.venv-gptq/Lib/site-packages/gptqmodel/quantization/gptq.py']:
        digest(p)
    by_unit = defaultdict(list)
    checked_stems = set()
    for r in candidates:
        by_unit[r['unit_id']].append(r)
        check('MSE_SSE_normalization_'+r['unit_id']+'_'+r['alpha'], math.isclose(float(r['search_output_mse'])*32768,float(r['search_output_sse']),rel_tol=1e-14))
        check('weight_normalization_'+r['unit_id']+'_'+r['alpha'], math.isclose(float(r['weight_mse_e2_over_input_dim'])*int(r['input_dim']),float(r['weight_squared_norm_e2']),rel_tol=1e-14))
        stem = r['source_candidate_stem']
        meta_path = V24/'raw/full_candidates'/f'{stem}.json'
        if stem not in checked_stems:
            check('candidate_file_'+stem, digest(meta_path.with_suffix('.pt')) == r['candidate_tensor_file_sha256'])
            check('candidate_meta_'+stem, digest(meta_path) == r['candidate_metadata_file_sha256'])
            checked_stems.add(stem)
        meta = read_json(meta_path)
        check('candidate_score_'+r['unit_id']+'_'+r['alpha'], float(meta['search_mse_by_row'][r['row']]) == float(r['search_output_mse']))
        check('candidate_H_'+r['unit_id']+'_'+r['alpha'], meta['hessian_sha256'] == r['hessian_tensor_sha256'])
    check('bank_complete', len(candidates)==480 and len(by_unit)==48 and len(checked_stems)==120)
    curves, units, discrepancies = [], [], []
    kmap = {(r['layer'],r['projection'],r['split_id']):r for r in kappas}
    for uid, bank in sorted(by_unit.items()):
        budget()
        check('grid_'+uid, sorted(float(r['alpha']) for r in bank)==[i/10 for i in range(1,11)])
        b = {r['alpha']:fraction(r['search_output_mse']) for r in bank}
        m = {r['alpha']:fraction(r['weight_squared_norm_e2']) for r in bank}
        baseline = max((a for a in b if b[a]==min(b.values())),key=float)
        original = next(r['alpha'] for r in bank if r['is_original_search_selected']=='True')
        check('rho0_'+uid,float(baseline)==float(original))
        common = [a for a in b if b[a]==min(b.values()) and m[a]==min(m.values())]
        active = []
        for a in sorted(b,key=float):
            lo, hi, possible = F(0), None, True
            # Intersect all affine inequalities; exact ties prefer larger alpha.
            for other in b:
                dm, db = m[a]-m[other], b[other]-b[a]
                if dm==0:
                    if db<0 or (db==0 and float(other)>float(a)):
                        possible=False
                elif dm>0:
                    bound = db/dm
                    hi = bound if hi is None else min(hi,bound)
                else:
                    lo = max(lo,db/dm)
            possible = possible and (hi is None or hi>=lo)
            def owns(point):
                if point is None or point<0:
                    return False
                scores={z:b[z]+point*m[z] for z in b}
                return a==max((z for z in b if scores[z]==min(scores.values())),key=float)
            left_in = possible and owns(lo)
            right_in = possible and hi is not None and owns(hi)
            possible = possible and (hi is None or hi>lo or (left_in and right_in))
            if possible:
                active.append(a)
            kr = kmap[(bank[0]['layer'],bank[0]['projection'],bank[0]['split_id'])]
            kappa = fraction(kr['kappa_trace_over_input_dim'])
            curves.append({'unit_id':uid,'layer':bank[0]['layer'],'projection':bank[0]['projection'],
                           'row':bank[0]['row'],'split_id':bank[0]['split_id'],'alpha':a,
                           'search_mse':str(b[a]),'weight_norm2':str(m[a]),
                           'active_for_nonnegative_rho':possible,'rho_lower':decimal_string(lo) if possible else '',
                           'rho_upper':decimal_string(hi) if possible else '',
                           'lower_inclusive':left_in if possible else '', 'upper_inclusive':right_in if possible else '',
                           'rho_lower_exact':str(lo) if possible else '', 'rho_upper_exact':str(hi) if possible and hi is not None else '',
                           'equivalent_tau_lower':decimal_string(lo/kappa) if possible else '',
                           'equivalent_tau_upper':decimal_string(hi/kappa) if possible and hi is not None else ('infinity' if possible else '')})
        legacy=[r for r in old_paths if r['unit_id']==uid]
        exact_active=[r for r in curves if r['unit_id']==uid and r['active_for_nonnegative_rho']]
        legacy_sequence=[float(r['alpha']) for r in legacy]
        exact_sequence=[float(r['alpha']) for r in sorted(exact_active,key=lambda r:F(r['rho_lower_exact']))]
        check('v30_envelope_sequence_'+uid,exact_sequence==legacy_sequence)
        for interval in legacy:
            rho=F(interval['rho_test'])
            best=max((a for a in b if b[a]+rho*m[a]==min(b[z]+rho*m[z] for z in b)),key=float)
            check('v30_interval_choice_'+uid+'_'+interval['segment_index'], float(best)==float(interval['alpha']))
        for boundary in (r for r in old_boundaries if r['unit_id']==uid):
            left=next(a for a in b if float(a)==float(boundary['left_alpha']))
            right=next(a for a in b if float(a)==float(boundary['right_alpha']))
            root=(b[right]-b[left])/(m[left]-m[right])
            exact_best=max((a for a in b if b[a]+root*m[a]==min(b[z]+root*m[z] for z in b)),key=float)
            if float(exact_best)!=float(boundary['boundary_alpha_by_larger_alpha_tie_rule']):
                discrepancies.append({'unit_id':uid,'rho_exact':str(root),'rho_v30_decimal':boundary['rho'],
                                      'v30_boundary_alpha':boundary['boundary_alpha_by_larger_alpha_tie_rule'],
                                      'exact_boundary_alpha':exact_best,'cause':'rounded Decimal crossing is not exact affine equality; no quality re-evaluation'})
        units.append({'unit_id':uid,'layer':bank[0]['layer'],'projection':bank[0]['projection'],'row':bank[0]['row'],
                      'split_id':bank[0]['split_id'],'rho0_alpha':baseline,'common_minimizer_alphas':';'.join(common),
                      'lower_envelope_alphas':';'.join(exact_best['alpha'] for exact_best in sorted(exact_active,key=lambda r:F(r['rho_lower_exact']))),
                      'envelope_candidates':len(active),'can_change_for_rho_nonnegative':len(active)>1,
                      'kappa':kr['kappa_trace_over_input_dim'],'normalization':'rho=tau*kappa_matrix'})
    # Preserve complete old search-only paths verbatim, not their held summaries.
    for name in ['SELECTION_PATH.csv','SELECTION_PATH_BOUNDARIES.csv','FULL_PATH_INTERVAL_CHOICES_SEARCH_ONLY.csv','FULL_PATH_BOUNDARY_CHOICES_SEARCH_ONLY.csv']:
        (OUT/('V30_REUSED_'+name)).write_bytes((V30/name).read_bytes())
    write_csv('CANDIDATE_SELECTION_INTERVALS.csv',curves)
    write_csv('UNIT_SELECTION_SPACE.csv',units)
    write_json('BOUNDARY_NUMERIC_AUDIT.json',{'differences':discrepancies,'tie_rule_changed':False,'exact_fraction_intervals_are_mathematical_audit_not_new_selector':True})
    matrix_summary=[]
    matrix_paths=[]
    common_tau_roots=set()
    for key, kr in sorted(kmap.items()):
        us=[u for u in units if (u['layer'],u['projection'],u['split_id'])==key]
        relevant=[r for r in curves if (r['layer'],r['projection'],r['split_id'])==key and r['active_for_nonnegative_rho']]
        roots=sorted({F(r['rho_lower_exact']) for r in relevant if F(r['rho_lower_exact'])>0})
        kappa=fraction(kr['kappa_trace_over_input_dim'])
        common_tau_roots.update(r/kappa for r in roots)
        for i in range(len(roots)+1):
            lo=roots[i-1] if i else F(0)
            hi=roots[i] if i<len(roots) else None
            test=(lo+hi)/2 if hi is not None else lo+max(F(1),lo+1)
            choices={}
            for u in us:
                bank=by_unit[u['unit_id']]
                vals={r['alpha']:fraction(r['search_output_mse'])+test*fraction(r['weight_squared_norm_e2']) for r in bank}
                choices[u['row']]=max((a for a in vals if vals[a]==min(vals.values())),key=float)
            matrix_paths.append({'layer':key[0],'projection':key[1],'split_id':key[2],'segment':i,
                                 'rho_lower_exclusive':decimal_string(lo),'rho_upper_exclusive':decimal_string(hi),
                                 'row_alpha_map':json.dumps(choices,sort_keys=True),
                                 'scope':'one rho shared by four diagnostic rows; not all output rows'})
        matrix_summary.append({'layer':key[0],'projection':key[1],'split_id':key[2],
                               'fixed_rows':len(us),'changing_rows':sum(u['can_change_for_rho_nonnegative'] for u in us),
                               'common_minimizer_rows':sum(bool(u['common_minimizer_alphas']) for u in us),
                               'search_documents':kr['search_documents'],'kappa':kr['kappa_trace_over_input_dim'],
                               'matrix_rho_open_interval_configs':len(roots)+1})
    write_csv('MATRIX_SELECTION_SPACE.csv',matrix_summary)
    write_csv('MATRIX_RHO_PATH.csv',matrix_paths)
    summary={'candidate_rows':len(candidates),'units':len(units),'bank_states':len(checked_stems),
             'changing_units':sum(u['can_change_for_rho_nonnegative'] for u in units),
             'invariant_units':sum(not u['can_change_for_rho_nonnegative'] for u in units),
             'common_minimizer_units':sum(bool(u['common_minimizer_alphas']) for u in units),
             'active_candidate_intervals':sum(r['active_for_nonnegative_rho'] for r in curves),
             'legacy_full_intervals':len(set(r['path_interval_id'] for r in full_intervals)),
             'legacy_full_boundaries':len(set(r['selector'] for r in full_boundaries)),
             'boundary_rounding_discrepancies':len(discrepancies),
             'independent_matrix_rho_open_interval_configs':math.prod(r['matrix_rho_open_interval_configs'] for r in matrix_summary),
             'common_tau_open_interval_configs':len(common_tau_roots)+1,
             'maximum_raw_H_input_kappa_relative_gap':max(float(r['relative_gap_H_vs_input']) for r in kappas),
             'held_read_or_rescored':False,'new_tau_experiments':0,
             'tensor_values_deserialized':False,'matrix_statistics_built':0,'eigendecompositions':0}
    write_json('SELECTION_SPACE_SUMMARY.json',summary)
    phase('search_only_exact_affine_envelope_and_provenance',start)
    return summary


def main():
    write_json('PROTOCOL.json',{'version':'v39','scope':'CPU artifact audit and mathematical feasibility only',
                              'order':['M3 source conflict','v30 search-only affine selection space','math/statistics/literature decision'],
                              'cpu_seconds_cap':300,'conservative_execution_wall_cap_seconds':300,
                              'new_candidates':0,'GPU':False,'model_forward':False,'held_read':False,'P3_read':False,'PPL':False,
                              'new_rho_quality_rule':None,'historical_code_and_artifacts':'READ_ONLY',
                              'input_C':'undamped per-search-token Gram; no new construction',
                              'tie_rule':'rho0 reuse original; positive exact objective ties prefer larger alpha',
                              'literature':'arXiv:2605.05693v2; source research is separate read-only wall, no model computation'})
    m3=m3_audit()
    space=selection_space()
    budget()
    # Verify small/medium source files stayed unchanged; 1GB export read once with identity receipt.
    for path, entry in SOURCES.items():
        if entry['bytes'] < 32*1024*1024:
            check('source_unchanged_'+entry['path'],hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'])
    write_csv('SOURCE_MANIFEST.csv',list(SOURCES.values()))
    write_json('CORRECTNESS.json',{'status':'PASS' if all(c['pass'] for c in CHECKS) else 'FAIL','checks':CHECKS,
                                 'scope':'artifact identity, normalization, window reaggregation and search-only affine envelope; not GPU parity'})
    budget_info={'cpu_seconds':time.process_time()-CPU0,'wall_seconds':time.perf_counter()-T0,
                 'cpu_cap_seconds':300,'phases':PHASES,'new_gpu_seconds':0,'new_model_forwards':0,
                 'new_candidate_executions':0,'new_held_scores':0,'new_ppl_runs':0,'P3_contents_read':False,
                 'scope':'this standard-library executable, including hashes and window reaggregation; interactive reading/research wall separate'}
    write_json('BUDGET.json',budget_info)
    result={'status':'COMPLETE' if all(c['pass'] for c in CHECKS) else 'AUDIT_FAILURE',
            'verdict':'B_DIFFERENT_SELECTION_SPACE_STATISTICAL_RULE_NOT_ESTABLISHED',
            'M3_conflict':m3['resolution'],'selection_space':space,
            'checks_passed':sum(c['pass'] for c in CHECKS),'checks_total':len(CHECKS),'budget':budget_info}
    write_json('SUMMARY.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        write_json('STOP.json',{'exception':repr(exc),'cpu_seconds':time.process_time()-CPU0,'wall_seconds':time.perf_counter()-T0})
        raise
