from __future__ import annotations

import torch

from audit_common import HERE, context, read_json, write_json
from replay_basic import run_step
from trace_backward import TraceObserver


def load_fixture(name):
    return torch.load(HERE / f'fixtures/{name}.pt', map_location='cpu', weights_only=False)


def main():
    context()
    if read_json(HERE / 'FAILURE_REPRODUCIBILITY.json')['status'] != 'PASS':
        raise RuntimeError('reproducibility gate has not passed')
    fixture = load_fixture('actual_l21')
    runs = {}
    for mode in ['D1', 'D2', 'D3']:
        path = HERE / f'DIAGNOSTIC_{mode}.json'
        if path.exists():
            result = read_json(path)
        else:
            result = run_step(fixture, mode=mode, native_call=(mode == 'D1'), observer=TraceObserver(mode))
            write_json(path, result)
        runs[mode] = result
        print(f"{mode} loss={result['loss']} grad_finite={result['all_lwc_gradients_finite']}", flush=True)
    comparisons = {'diagnostic_only': True, 'same_frozen_input_weight_target_values': True,
                   'D2_uses_FP32_math_and_same_65536_loss_scale': True,
                   'D3_uses_same_autocast_forward_and_unit_backward_scale': True,
                   'runs': {k: {field: r[field] for field in ['mode','loss','all_lwc_gradients_finite',
                          'nonfinite_parameter_names','amp','peak_allocated_bytes','peak_reserved_bytes','timing_instrumented_seconds']} for k,r in runs.items()}}
    write_json(HERE / 'AMP_DIAGNOSTIC_COMPARISON.json', comparisons)
    sample_runs = [read_json(HERE / 'REPLAY_1.json')]
    for sample in [1, 2, 3]:
        path = HERE / f'SAMPLE_{sample}_DIAGNOSTIC.json'
        result = read_json(path) if path.exists() else run_step(fixture, sample=sample, native_call=False)
        if not path.exists():
            write_json(path, result)
        sample_runs.append(result)
        print(f"SAMPLE {sample} loss={result['loss']} grad_finite={result['all_lwc_gradients_finite']}", flush=True)
    write_json(HERE / 'SAMPLE_SPECIFICITY.json', {'diagnostic_only': True, 'fresh_state_for_each_sample': True,
        'sample_order': [0,1,2,3], 'committed_updates': 0,
        'runs': [{k:r[k] for k in ['sample','loss','all_lwc_gradients_finite','nonfinite_parameter_names','input','forward_output','amp']} for r in sample_runs]})
    layer_runs = []
    for source in ['actual_l20','actual_l21','fp_l20','fp_l21','fp_l22','fp_l23']:
        path = HERE / f'LAYER_{source}_DIAGNOSTIC.json'
        if source == 'actual_l21':
            result = read_json(HERE / 'REPLAY_1.json')
        else:
            result = read_json(path) if path.exists() else run_step(load_fixture(source), native_call=False)
            if not path.exists():
                write_json(path, result)
        layer_runs.append(result)
        print(f"LAYER {source} loss={result['loss']} grad_finite={result['all_lwc_gradients_finite']}", flush=True)
    write_json(HERE / 'LAYER_SPECIFICITY.json', {'diagnostic_only': True, 'sample_index': 0,
        'fresh_official_LWC_initialization': True, 'committed_updates': 0,
        'L22_L23_formal_prefix_available': False,
        'limitation': 'FP-prefix comparisons share a common native-prefix condition; actual L20/L21 use learned quantized prefixes. Missing trained L21+ states prevent actual formal L22/L23 inference.',
        'runs': [{k:r[k] for k in ['layer','dependency','loss','all_lwc_gradients_finite','nonfinite_parameter_names','input','forward_output','amp']} for r in layer_runs]})
    write_json(HERE / 'DIAGNOSTICS_COMPLETE.json', {'status':'PASS', 'formal_training_continued':False, 'committed_updates':0})


if __name__ == '__main__':
    main()
