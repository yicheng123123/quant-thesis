"""CPU-only v22 artifact analysis. No model, quantizer, forward, or optimizer runs."""
from pathlib import Path
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = OUT.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csvdump(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def stats(x):
    x = x.detach().float().flatten()
    assert torch.isfinite(x).all()
    q = torch.quantile(x, torch.tensor([0., .05, .25, .5, .75, .95, 1.]))
    return dict(count=x.numel(), mean=x.double().mean().item(), std_population=x.double().std(unbiased=False).item(),
                **dict(zip(['min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'], q.tolist())))


def describe(records):
    lo = torch.cat([r['lo'].flatten() for r in records])
    hi = torch.cat([r['hi'].flatten() for r in records])
    row_std = torch.cat([((r['lo'] + r['hi']) / 2).std(dim=1, unbiased=False) for r in records])
    row_span = torch.cat([((r['lo'] + r['hi']) / 2).amax(1) - ((r['lo'] + r['hi']) / 2).amin(1) for r in records])
    m2 = torch.cat([r['alpha'] for r in records])
    # Endpoint average is only a descriptive statistic, NOT effective width or M2 alpha.
    row_mean = torch.cat([((r['lo'] + r['hi']) / 2).mean(1) for r in records])
    pearson = torch.corrcoef(torch.stack([m2.double(), row_mean.double()]))[0, 1].item()
    return dict(groups=lo.numel(), rows=m2.numel(), lower_multiplier=stats(lo), upper_multiplier=stats(hi),
                absolute_endpoint_multiplier_difference=stats((lo-hi).abs()),
                row_group_endpoint_average_std=stats(row_std), row_group_endpoint_average_span=stats(row_span),
                fraction_rows_group_span_above_0_05=(row_span > .05).double().mean().item(),
                fraction_groups_endpoint_difference_above_0_05=((lo-hi).abs() > .05).double().mean().item(),
                fraction_endpoint_multipliers_below_0_9=torch.cat([lo,hi]).lt(.9).double().mean().item(),
                fraction_endpoint_multipliers_above_0_99=torch.cat([lo,hi]).gt(.99).double().mean().item(),
                m2_mean_alpha=m2.double().mean().item(),
                descriptive_pearson_m2_alpha_vs_m3_row_endpoint_average=pearson)


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    checkpoint = RESULTS / 'range_w4_omniquant_resume_v21_4/M3/checkpoints/block_23/omni_parameters.pth'
    alpha_csv = RESULTS / 'range_w4_fullmodel_v18/M2/selected_alpha.csv'
    quality = RESULTS / 'range_w4_omniquant_finalize_v21_5/QUALITY_COMPARISON.json'
    reports = [
        'range_w4_96row_v11/FULL10_RUN_REPORT.md', 'range_w4_gate_v12/VALIDATION_REPORT.md',
        'range_w4_failure_mechanism_v13/REPORT.md', 'range_w4_calibration_coverage_v14/REPORT.md',
        'range_w4_quantized_prefix_v15/REPORT.md', 'range_w4_fullmodel_v18/REPORT.md',
        'range_w4_fullmodel_review_v19/REPORT.md', 'range_w4_fullmodel_review_v19/LOCAL_TO_GLOBAL_ASSESSMENT.md',
        'range_w4_omniquant_finalize_v21_5/REPORT.md',
        'range_w4_fullmodel_v18/PROTOCOL.json', 'range_w4_fullmodel_v18/M2/runtime_breakdown.csv',
        'range_w4_omniquant_resume_v21_4/M3/FORMAL_PROTOCOL_LOCK.json',
        'range_w4_omniquant_resume_v21_4/M3/run_formal_m3.py',
        'range_w4_omniquant_resume_v21_4/M3/official_port/quantize/quantizer.py',
        'range_w4_omniquant_resume_v21_4/M3/official_port/quantize/omniquant.py',
        'range_w4_omniquant_resume_v21_4/M3/official_port/quantize/utils.py',
        'range_w4_m2_runtime_v17_3/optimized_native_m2.py']
    inputs = [checkpoint, alpha_csv, quality] + [RESULTS / p for p in reports]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    rows = defaultdict(list)
    with alpha_csv.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows[r['module']].append(r)
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert sorted(data) == list(range(24))
    records, examples = [], []
    changed = total = 0
    dtypes = Counter()
    for layer, state in sorted(data.items()):
        assert len(state) == 14 and all('bound_factor' in n for n in state)
        for name, upper in state.items():
            if not name.endswith('upbound_factor'):
                continue
            lower = state[name.replace('upbound_factor', 'lowbound_factor')]
            assert upper.shape == lower.shape and upper.ndim == 2 and upper.shape[1] == 1
            projection = name.split('.')[2]
            branch = name.split('.')[1]
            module = f'model.layers.{layer}.{branch}.{projection}'
            rr = sorted(rows[module], key=lambda r: int(r['row']))
            assert [int(r['row']) for r in rr] == list(range(len(rr)))
            groups = upper.numel() // len(rr)
            assert groups * len(rr) == upper.numel() and groups == (38 if projection == 'down_proj' else 7)
            for t in [upper, lower]:
                assert t.device.type == 'cpu' and torch.isfinite(t).all()
                total += t.numel()
                changed += int((t != 4.).sum())
                dtypes[str(t.dtype)] += t.numel()
            # FP32 sigmoid used for stable descriptive summaries, not replay of FP16 quantization.
            lo = lower.float().sigmoid().reshape(len(rr), groups)
            hi = upper.float().sigmoid().reshape(len(rr), groups)
            alpha = torch.tensor([float(r['selected_alpha']) for r in rr], dtype=torch.float64)
            records.append(dict(layer=layer, projection=projection, module=module, lo=lo, hi=hi, alpha=alpha))
            for row in sorted({0, len(rr)//4, len(rr)//2, 3*len(rr)//4, len(rr)-1}):
                examples.append(dict(module=module, row=row, m2_alpha=float(rr[row]['selected_alpha']),
                                     m3_lower_multipliers=lo[row].tolist(), m3_upper_multipliers=hi[row].tolist()))
    assert len(records) == 168 and sum(len(r['alpha']) for r in records) == 304128
    module_summaries = [dict(module=r['module'], layer=r['layer'], projection=r['projection'], **describe([r])) for r in records]
    projection_summaries = {p: describe([r for r in records if r['projection'] == p]) for p in sorted({r['projection'] for r in records})}
    layer_summaries = {str(i): describe([r for r in records if r['layer'] == i]) for i in range(24)}
    m2_all = [r for rr in rows.values() for r in rr]
    histogram = Counter(r['selected_alpha'] for r in m2_all)
    m2_mean_gain = math.fsum(float(r['relative_output_gain']) for r in m2_all) / len(m2_all)
    diagnostic = dict(semantic_note='Sigmoid endpoint coefficients on pristine group min/max in asymmetric OmniQuant. Not M2 alpha, not effective width, not actual scale/zero; FP32 descriptive sigmoid.',
                      global_summary=describe(records), projection_summaries=projection_summaries, layer_summaries=layer_summaries,
                      module_summaries=module_summaries, scalar_lwc_parameter_count=total, raw_logits_different_from_initial_4_count=changed,
                      checkpoint_dtypes=dict(dtypes), M2_alpha_histogram=dict(sorted(histogram.items())), M2_mean_search_row_relative_gain=m2_mean_gain)
    dump('M3_RANGE_STATISTICS.json', diagnostic)
    dump('M3_PRESELECTED_ROW_EXAMPLES.json', dict(selection='rows 0,floor(n/4),floor(n/2),floor(3n/4),n-1 of all modules, independent of values', examples=examples))
    flat = []
    for r in module_summaries:
        flat.append(dict(module=r['module'], layer=r['layer'], projection=r['projection'], groups=r['groups'], rows=r['rows'],
                         m2_mean_alpha=r['m2_mean_alpha'], lower_mean=r['lower_multiplier']['mean'], upper_mean=r['upper_multiplier']['mean'],
                         lower_p05=r['lower_multiplier']['p05'], upper_p05=r['upper_multiplier']['p05'],
                         endpoint_abs_difference_mean=r['absolute_endpoint_multiplier_difference']['mean'],
                         row_group_span_median=r['row_group_endpoint_average_span']['median'],
                         row_group_span_p95=r['row_group_endpoint_average_span']['p95'],
                         rows_group_span_above_005_fraction=r['fraction_rows_group_span_above_0_05']))
    csvdump('M2_M3_MODULE_RANGE_SUMMARY.csv', flat)
    q = json.loads(quality.read_text(encoding='utf-8'))
    gap = {}
    for dataset, v in q['datasets'].items():
        base, m3 = v['baseline_values'], v['M3']
        m2gain = base['M1']['nll'] - base['M2']['nll']
        extra = base['M2']['nll'] - m3['nll']
        gap[dataset] = dict(m2_nll_gain_over_m1=m2gain, m3_extra_nll_gain_over_m2=extra,
                            extra_gap_over_existing_m2_nll_gain=extra/m2gain,
                            m2_ppl_recovery_percent=100*(base['M1']['ppl']-base['M2']['ppl'])/(base['M1']['ppl']-base['M0']['ppl']),
                            m3_ppl_recovery_percent=100*(base['M1']['ppl']-m3['ppl'])/(base['M1']['ppl']-base['M0']['ppl']))
    dump('QUALITY_GAP_RECALC.json', gap)
    after = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    assert before == after
    dump('INPUT_PROVENANCE.json', dict(inputs=before, historical_input_hashes_unchanged=True))
    dump('CORRECTNESS.json', dict(status='PASS', scope='artifact analysis only, not new quantization/model correctness',
                                blocks=24, modules=168, rows=304128, all_parameters_finite=True, grouping_and_pairing_checked=True,
                                historical_inputs_hashes_unchanged=True, new_gptq_executions=0, new_model_forwards=0,
                                new_optimizer_steps=0, new_ppl_runs=0, cpu_analysis_wall_seconds=time.perf_counter()-started))
    print(json.dumps(dict(global_summary=diagnostic['global_summary'], projections=projection_summaries, parameters=total, quality_gap=gap), indent=2))


if __name__ == '__main__':
    main()
