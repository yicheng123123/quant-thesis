"""Direct numeric checks on the first overflowing native half GEMM."""
from __future__ import annotations

import torch

from audit_common import HERE, DEVICE, context, stats, read_json, write_json
from replay_basic import run_step
from trace_backward import TraceObserver


def main():
    context()
    fixture = torch.load(HERE / 'fixtures/actual_l21.pt', map_location='cpu', weights_only=False)
    result_path = HERE / 'DIAGNOSTIC_D1_OPERAND.json'
    if not result_path.exists():
        result = run_step(fixture, observer=TraceObserver('D1_OPERAND'))
        write_json(result_path, result)
    result = read_json(result_path)
    replay = read_json(HERE / 'REPLAY_1.json')
    equivalence = result['loss'] == replay['loss'] and all(
        g['nonfinite_mask_sha256'] == h['nonfinite_mask_sha256'] for g,h in zip(result['gradients'],replay['gradients']))
    if not equivalence:
        raise RuntimeError('instrumented replay changed the failure mask/loss')
    tensors = torch.load(HERE / 'FIRST_GEMM_OPERANDS_D1_OPERAND.pt', weights_only=False, map_location='cpu')
    x = tensors['input'].to(DEVICE)
    go = tensors['incoming_grad_output'].to(DEVICE)
    native = tensors['native_grad_weight_transpose'].to(DEVICE)
    bad = ~torch.isfinite(native)
    indices = bad.nonzero()
    # Exact same HALF input values, but diagnostic FP32 accumulation/output.
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        fp32 = x.float().T @ go.float()
        half_recomputed = x.T @ go
    finite_limit = torch.finfo(torch.float16).max
    exceed = fp32.abs() > finite_limit
    coordinates = []
    for i,j in indices.cpu().tolist():
        xv = x[:,i].double()
        gv = go[:,j].double()
        dot64 = (xv * gv).sum()
        coordinate = {'transpose_input_channel':i,'output_channel':j,
                      'weight_row':j,'weight_column':i,
                      'fp32_dot':float(fp32[i,j]), 'fp64_dot_same_fp16_operands':float(dot64),
                      'fp16_finite_limit':float(finite_limit),
                      'exceeds_fp16_limit':bool(dot64.abs()>finite_limit),
                      'unscaled_equivalent_fp64_dot':float(dot64/65536),
                      'native_gradient_class':'+Inf' if bool(torch.isposinf(native[i,j])) else '-Inf' if bool(torch.isneginf(native[i,j])) else 'NaN'}
        coordinates.append(coordinate)
    write_json(HERE / 'GEMM_OVERFLOW_VERIFICATION.json', {
        'status':'PASS' if bool(torch.equal(bad,~torch.isfinite(half_recomputed))) and all(c['exceeds_fp16_limit'] for c in coordinates) else 'FAIL',
        'instrumented_replay_loss_and_nonfinite_masks_match':equivalence,
        'operation':tensors['operation'], 'projection':tensors['tags'],
        'gradient_formula':'grad_W_transpose = X_transpose @ incoming_scaled_grad_output',
        'native_weight_gradient':stats(native), 'fp32_same_operand_gemm':stats(fp32),
        'fp32_over_fp16_limit_count':int(exceed.sum()),
        'over_limit_positions_equal_native_nonfinite_positions':bool(torch.equal(exceed,bad)),
        'same_fp16_gemm_recomputed_bad_mask_equal':bool(torch.equal(bad,~torch.isfinite(half_recomputed))),
        'overflow_coordinates':coordinates,
        'scope':'numeric backward boundary confirmation only; no changed baseline or optimizer updates'})
    print(coordinates,flush=True)


if __name__ == '__main__':
    main()
