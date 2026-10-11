"""Disposable one-sample replays. Never save optimizer updates to the formal run."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time

import torch

from audit_common import HERE, DEVICE, context, formal, new_layer, tree, stats, sha_tensor, read_json, write_json
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters, smooth_and_quant_temporary
from utils import NativeScalerWithGradNormCount, ampscaler_get_grad_norm


def gradient_map(layer):
    rows = []
    for name, param in layer.named_parameters():
        if 'bound_factor' not in name:
            continue
        projection = name.split('.')[-3]
        rows.append({'module_name': name.rsplit('.', 1)[0], 'projection': projection,
                     'parameter_name': name, 'parameter_shape': list(param.shape),
                     'parameter_dtype': str(param.dtype), 'parameter_value': stats(param),
                     'gradient': stats(param.grad),
                     'nonfinite_indices': (~torch.isfinite(param.grad)).nonzero().detach().cpu().tolist() if param.grad is not None else None,
                     'gradient_sha256': sha_tensor(param.grad) if param.grad is not None else None,
                     'nonfinite_mask_sha256': sha_tensor(~torch.isfinite(param.grad)) if param.grad is not None else None})
    return rows


def run_step(fixture, sample=0, mode='D1', native_call=True, observer=None):
    layer, args = new_layer(fixture, fp32_input=(mode == 'D2_FULL_FP32'))
    is_fp32 = mode in ('D2','D2_FULL_FP32')
    use_scaler = mode != 'D3'
    inp = tree(fixture['inputs'][sample:sample+1], DEVICE, is_fp32)
    target = tree(fixture['targets'][sample:sample+1], DEVICE, is_fp32)
    kwargs = tree(fixture['kwargs'][sample], DEVICE, is_fp32)
    optimizer = torch.optim.AdamW([{'params': let_parameters(layer, False), 'lr': args.let_lr},
                                  {'params': lwc_parameters(layer), 'lr': args.lwc_lr}], weight_decay=args.wd)
    initial_values = {n: p.detach().clone() for n, p in layer.named_parameters() if 'bound_factor' in n}
    wrapper = NativeScalerWithGradNormCount()
    state_before = wrapper.state_dict()
    step_calls = []
    optimizer.register_step_pre_hook(lambda *a: step_calls.append('optimizer.step'))
    phase = {}
    base_unscale = wrapper._scaler.unscale_

    def observed_unscale(opt):
        phase['scaled_before_unscale'] = gradient_map(layer)
        value = base_unscale(opt)
        phase['unscaled_before_optimizer'] = gradient_map(layer)
        return value

    wrapper._scaler.unscale_ = observed_unscale
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()
    start = time.perf_counter()
    if observer:
        observer.install(layer)
    with torch.cuda.amp.autocast(enabled=not is_fp32):
        smooth_and_quant_temporary(layer, args, False)
        if observer:
            observer.after_quantization(layer, inp, target)
        out = layer(inp, **kwargs)[0]
        loss = torch.nn.MSELoss()(target, out)
    if observer:
        observer.after_forward(layer, inp, target, out, loss)
    loss_value = float(loss.detach())
    optimizer.zero_grad()
    scale = float(wrapper._scaler.get_scale()) if use_scaler else 1.0
    scaled_loss_value = loss_value * scale
    if observer:
        observer.install_graph(loss)
    if use_scaler and native_call:
        norm = wrapper(loss, optimizer, parameters=get_omni_parameters(layer, False)).detach()
    elif use_scaler:
        wrapper._scaler.scale(loss).backward()
        wrapper._scaler.unscale_(optimizer)
        norm = ampscaler_get_grad_norm(get_omni_parameters(layer, False)).detach()
    else:
        loss.backward()
        norm = ampscaler_get_grad_norm(get_omni_parameters(layer, False)).detach()
        phase['unscaled_before_optimizer'] = gradient_map(layer)
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    grads = gradient_map(layer)
    changed = [n for n, p in layer.named_parameters() if n in initial_values and not torch.equal(initial_values[n], p.detach())]
    result = {'layer': fixture['layer'], 'sample': sample, 'mode': mode, 'dependency': fixture['dependency'],
              'loss': loss_value, 'loss_finite': math.isfinite(loss_value),
              'forward_output': stats(out), 'input': stats(inp), 'teacher_target': stats(target),
              'gradients': grads,
              'all_lwc_gradients_finite': all(r['gradient'].get('all_finite', False) for r in grads),
              'nonfinite_parameter_names': [r['parameter_name'] for r in grads if not r['gradient'].get('all_finite', False)],
              'optimizer': {'initial_state_entries': 0, 'step_call_count': len(step_calls),
                            'state_entries_after': len(optimizer.state), 'changed_lwc_parameter_names': changed,
                            'state_dict_after': tree(optimizer.state_dict()), 'updates_committed': 0},
              'amp': {'autocast_enabled': not is_fp32, 'autocast_dtype': 'torch.float16' if not is_fp32 else None,
                      'parameter_dtype': 'torch.float32', 'grad_scaler_used': use_scaler,
                      'state_before': state_before if use_scaler else None,
                      'state_after': wrapper.state_dict() if use_scaler else None,
                      'scale_factor': scale, 'scaled_loss': scaled_loss_value,
                      'native_scaler_call_with_step_and_update': bool(native_call and use_scaler)},
              'gradient_phases': phase,
              'gradient_norm': float(norm) if bool(torch.isfinite(norm)) else None,
              'gradient_norm_nonfinite': not bool(torch.isfinite(norm)),
              'timing_instrumented_seconds': duration,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'formal_quality_result': False}
    # A failing scaler should skip the optimizer; no nonfinite numbers leak into JSON.
    result['optimizer'].pop('state_dict_after')
    result['optimizer']['state_tensor_audit'] = {
        str(key): {name: stats(value) if isinstance(value, torch.Tensor) else value for name, value in state.items()}
        for key, state in optimizer.state.items()}
    if observer:
        observer.finish(result)
    del layer, optimizer, wrapper, inp, target, out, loss, norm, initial_values
    gc.collect()
    torch.cuda.empty_cache()
    return result


def flatten_gradients(results, path):
    fields = ['replay', 'phase', 'module_name', 'projection', 'parameter_name', 'shape', 'dtype',
              'finite_count', 'nan_count', 'posinf_count', 'neginf_count', 'max_finite_abs', 'mean_finite_abs',
              'parameter_min', 'parameter_max', 'parameter_mean', 'parameter_std', 'parameter_finite', 'gradient_sha256']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, result in enumerate(results):
            phases = dict(result['gradient_phases'])
            phases['after_native_scaler_call'] = result['gradients']
            for phase, rows in phases.items():
                for row in rows:
                    g, p = row['gradient'], row['parameter_value']
                    writer.writerow({'replay': i+1, 'phase': phase,
                        **{key: row[key] for key in ['module_name', 'projection', 'parameter_name', 'gradient_sha256']},
                        'shape': json.dumps(row['parameter_shape']), 'dtype': row['parameter_dtype'],
                        **{key: g[key] for key in ['finite_count', 'nan_count', 'posinf_count', 'neginf_count', 'max_finite_abs', 'mean_finite_abs']},
                        'parameter_min': p['min'], 'parameter_max': p['max'], 'parameter_mean': p['mean'],
                        'parameter_std': p['std'], 'parameter_finite': p['all_finite']})


def main():
    context()
    fixture = torch.load(HERE / 'fixtures/actual_l21.pt', weights_only=False, map_location='cpu')
    results = []
    for repeat in range(3):
        cache = HERE / f'REPLAY_{repeat+1}.json'
        result = read_json(cache) if cache.exists() else run_step(fixture)
        if not cache.exists():
            write_json(cache, result)
        results.append(result)
        print(f"REPLAY {repeat+1} loss={result['loss']} finite={result['all_lwc_gradients_finite']} bad={result['nonfinite_parameter_names']}", flush=True)
    stable = all(r['loss'] == 0.15326565504074097 and not r['all_lwc_gradients_finite'] for r in results)
    identical = all([g['gradient_sha256'] for g in r['gradients']] == [g['gradient_sha256'] for g in results[0]['gradients']] for r in results)
    same_masks = all([g['nonfinite_mask_sha256'] for g in r['gradients']] == [g['nonfinite_mask_sha256'] for g in results[0]['gradients']] for r in results)
    bad_hashes = [[g['gradient_sha256'] for g in r['gradients'] if not g['gradient']['all_finite']] for r in results]
    bad_identical = all(hashes == bad_hashes[0] for hashes in bad_hashes)
    write_json(HERE / 'FAILURE_REPRODUCIBILITY.json', {
        'status': 'PASS' if stable and same_masks and bad_identical else 'FAIL', 'repeats': 3,
        'reference_loss': 0.15326565504074097, 'replayed_losses': [r['loss'] for r in results],
        'loss_bitwise_equal_to_historical_scalar': stable,
        'all_replays_nonfinite': all(not r['all_lwc_gradients_finite'] for r in results),
        'replay_gradients_bitwise_equal': identical,
        'nonfinite_masks_bitwise_equal': same_masks,
        'nonfinite_parameter_gradient_tensors_bitwise_equal': bad_identical,
        'finite_parameter_gradient_hash_differences': [g['parameter_name'] for i,g in enumerate(results[0]['gradients']) if len({r['gradients'][i]['gradient_sha256'] for r in results}) > 1],
        'reproducibility_scope': 'exact loss and stable named nonfinite tensors/counts/positions; finite gradient bitwise equality is separately reported and is not the failure-identity requirement',
        'prior_strict_hash_check': 'FAILURE_REPRODUCIBILITY_STRICT_GRAD_HASH_CHECK.json preserves the extra all-gradient byte-equality check; only finite q_proj gradient bytes differed, while every nonfinite tensor matched bitwise',
        'same_named_nonfinite_parameters_across_repeats': all(r['nonfinite_parameter_names'] == results[0]['nonfinite_parameter_names'] for r in results),
        'nonfinite_parameter_names': results[0]['nonfinite_parameter_names'],
        'historical_per_parameter_comparison': 'not available: v21.1 saved only aggregate gradient flags, not named gradient tensors',
        'optimizer_and_scaler_restoration': 'fresh per-block AdamW (empty state) and fresh native GradScaler, following official loop; checkpoint RNG restored',
        'committed_updates': 0})
    flatten_gradients(results, HERE / 'LWC_GRADIENT_FAILURE_MAP.csv')
    write_json(HERE / 'AMP_STATE.json', results[0]['amp'])
    write_json(HERE / 'OPTIMIZER_STATE_AUDIT.json', results[0]['optimizer'])
    if not stable or not same_masks or not bad_identical:
        raise RuntimeError('reproducibility gate failed; stop causal diagnosis')


if __name__ == '__main__':
    main()
