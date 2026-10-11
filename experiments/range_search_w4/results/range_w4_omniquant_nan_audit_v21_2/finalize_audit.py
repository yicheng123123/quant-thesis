"""Consolidate existing diagnostic evidence; no forward/backward is run."""
from __future__ import annotations

import csv

import torch

from audit_common import HERE, read_json, write_json, stats, sha_file, protected_hashes


def main():
    repro = read_json(HERE/'FAILURE_REPRODUCIBILITY.json')
    gemm = read_json(HERE/'GEMM_OVERFLOW_VERIFICATION.json')
    sources = read_json(HERE/'SOURCE_NUMERICAL_AUDIT.json')
    rng = read_json(HERE/'RNG_RESTORATION_CHECK.json')
    protection = protected_hashes() == read_json(HERE/'PROTECTED_HASHES.json')
    trace = read_json(HERE/'FIRST_NONFINITE_BACKWARD_D1.json')
    tensor_trace = read_json(HERE/'BACKWARD_TENSOR_TRACE_D1.json')['events']
    canonical_trace = dict(trace)
    canonical_trace.update({'GEMM_numeric_verification':'GEMM_OVERFLOW_VERIFICATION.json',
        'operation_boundary_verified':gemm['status']=='PASS',
        'first_nonfinite_phase':'scaled backward, before unscale and before optimizer',
        'gradient_dtype_at_first_overflow':'torch.float16',
        'LWC_gradient_dtype':'torch.float32',
        'overflow_coordinates':gemm['overflow_coordinates'],
        'FP16_accumulator_internal_dtype_claimed':False,
        'subsequent_independent_finite_to_nonfinite_transition':'gate_proj MmBackward0 weight gradient contains two -Inf entries',
        'quantizer_propagation':[e for e in tensor_trace if not e['gradient']['all_finite']]})
    write_json(HERE/'FIRST_NONFINITE_BACKWARD.json',canonical_trace)
    forward = read_json(HERE/'FORWARD_NUMERICS_D1.json')
    l20 = torch.load(HERE/'fixtures/actual_l20.pt',weights_only=False,map_location='cpu')
    forward['tensors']['FP_teacher_input_hidden_states'] = stats(l20['targets'][0:1])
    forward['teacher_input_provenance']='FP output after block20 from the same actual-prefix reconstruction'
    write_json(HERE/'FORWARD_NUMERICS.json',forward)
    quant = read_json(HERE/'QUANTIZER_NUMERICS_AUDIT_D1.json')
    quant.update({'first_nonfinite_origin':'F.linear weight-gradient half GEMM, upstream of the quantizer derivative',
        'no_scale_clipmin_hits':all(q['scale_at_clipmin_count']==0 for q in quant['quantizers'].values()),
        'all_forward_quantizer_observables_finite':all(v.get('all_finite',True) for q in quant['quantizers'].values() for v in q.values() if isinstance(v,dict)),
        'LWC_parameters_initial_value':4.0,
        'bad_gradient_flat_indices':[3969,24402],
        'group_mapping':'rows 567 and 3486, input group0 containing channel62; 7 input groups per gate/up output row',
        'derivative_limit':'Inf enters the affine derivative from dequantized-weight grad; NaN appears in combined scale/zero gradients. Node hooks do not isolate the internal engine accumulation of multiple nonfinite contributions.'})
    write_json(HERE/'QUANTIZER_NUMERICS_AUDIT.json',quant)
    comparison=read_json(HERE/'AMP_DIAGNOSTIC_COMPARISON.json')
    full32=read_json(HERE/'DIAGNOSTIC_D2_FULL_FP32.json')
    full32_trace=read_json(HERE/'FIRST_NONFINITE_BACKWARD_D2_FULL_FP32.json')
    for mode in ('D1','D2','D3'):
        nodes=read_json(HERE/f'BACKWARD_NODE_TRACE_{mode}.json')['events']
        comparison['runs'][mode]['all_observed_backward_gradients_finite'] = all(
            g.get('all_finite',True) for e in nodes for field in ('grad_inputs','grad_outputs') for g in e[field])
    comparison['runs']['D2']['limitation']='LWC gradients finite, but original unoptimized FP16 weight leaf receives Inf at terminal ToCopyBackward0; not a fully FP32 graph.'
    comparison['runs']['D2_FULL_FP32']={k:full32[k] for k in (
        'mode','loss','all_lwc_gradients_finite','nonfinite_parameter_names','amp',
        'peak_allocated_bytes','peak_reserved_bytes','timing_instrumented_seconds')}
    comparison['runs']['D2_FULL_FP32'].update({
        'all_observed_backward_gradients_finite':full32_trace['all_observed_backward_gradients_finite'],
        'graph_leaf_dtypes':full32_trace['graph_leaf_dtypes'],
        'diagnostic_precision_change':'Construct original native weight leaves in FP32 before QuantLinear buffers; all frozen numerical values unchanged.'})
    comparison.update({'baseline_changed':False,
        'D2_not_equal_to_a_full_deactive_amp_formal_run':'D2 uses the frozen actual-prefix HALF values promoted to FP32 and the frozen HALF teacher targets promoted to FP32. Official deactive_amp would change whole prefix/cache dtype and leaves teacher autocast enabled.',
        'timing_not_a_calibration_speed_benchmark':'instrumentation and node/tensor statistics are included; do not extrapolate these diagnostic times to full-model runtime'})
    write_json(HERE/'AMP_DIAGNOSTIC_COMPARISON.json',comparison)
    sources.update({'deactive_amp_W4_end_to_end_certified':False,
        'deactive_amp_source_only_handoff_caveat':'Official quantize/omniquant.py:277,284 and formal port:468,475 unconditionally cast qlayer.half() before final propagation under traincast(); with deactive_amp=True traincast is nullcontext and cache is FP32. This dtype handoff requires a separate end-to-end check; not an executed/confirmed bug and not the cause of the audited AMP failure.',
        'FP32_fixture_diagnostic_is_not_official_flag_end_to_end_validation':True})
    write_json(HERE/'SOURCE_NUMERICAL_AUDIT.json',sources)
    results=[read_json(HERE/name) for name in ['REPLAY_1.json','REPLAY_2.json','REPLAY_3.json',
        'DIAGNOSTIC_D1.json','DIAGNOSTIC_D2.json','DIAGNOSTIC_D3.json',
        'SAMPLE_1_DIAGNOSTIC.json','SAMPLE_2_DIAGNOSTIC.json','SAMPLE_3_DIAGNOSTIC.json',
        'LAYER_actual_l20_DIAGNOSTIC.json','LAYER_fp_l20_DIAGNOSTIC.json','LAYER_fp_l21_DIAGNOSTIC.json',
        'LAYER_fp_l22_DIAGNOSTIC.json','LAYER_fp_l23_DIAGNOSTIC.json',
        'DIAGNOSTIC_D1_OPERAND.json','FAILURE_PARAMETER_COORDINATES.json','DIAGNOSTIC_D2_FULL_FP32.json']]
    checks={'three_replicas_exact_historical_loss':repro['loss_bitwise_equal_to_historical_scalar'],
        'three_replicas_same_nonfinite_tensors_and_masks':repro['nonfinite_parameter_gradient_tensors_bitwise_equal'] and repro['nonfinite_masks_bitwise_equal'],
        'RNG_restoration':rng['status']=='PASS',
        'first_finite_to_nonfinite_node_identified':trace['first_finite_to_nonfinite_node']['operation']=='MmBackward0',
        'same_operand_fp64_overflow_verification':gemm['status']=='PASS' and gemm['over_limit_positions_equal_native_nonfinite_positions'],
        'instrumentation_preserves_loss_and_failure_masks':gemm['instrumented_replay_loss_and_nonfinite_masks_match'],
        'all_observed_forward_stages_finite':forward['all_observed_tensors_finite'],
        'FP32_diagnostic_all_LWC_gradients_finite':comparison['runs']['D2']['all_lwc_gradients_finite'],
        'full_FP32_diagnostic_all_observed_backward_gradients_finite':full32_trace['all_observed_backward_gradients_finite'],
        'full_FP32_diagnostic_has_only_FP32_leaves':full32_trace['graph_leaf_dtypes']==['torch.float32'],
        'autocast_unit_scale_diagnostic_all_LWC_gradients_finite':comparison['runs']['D3']['all_lwc_gradients_finite'],
        'native_math_sources_match_official':all(v['identical'] for v in sources['protected_algorithm_sources'].values()),
        'no_actual_optimizer_step_or_commit':all(r['optimizer']['step_call_count']==0 and r['optimizer']['updates_committed']==0 for r in results),
        'historical_sources_checkpoints_logs_reports_unchanged':protection}
    if not all(checks.values()):
        raise RuntimeError(f'final audit gate failed: {checks}')
    write_json(HERE/'correctness_checks.json',{'status':'PASS','scope':'root-cause audit correctness only; v21.1 formal M3 remains stopped',
        'checks':checks,'limitations':['all-gradient bitwise equality fails only for finite q_proj gradient values',
                                     'historical per-parameter gradients were not saved',
                                     'SDPA internal scores not separately observed',
                                     'L22/L23 controls use native FP prefixes, not unavailable actual formal prefixes',
                                     'original D2 has unused FP16 leaf cast overflow; separate D2_FULL_FP32 is finite',
                                     'samples 0..3 share first token284; broader data specificity remains untested',
                                     'T4 not tested']})
    allocation_elements=128*2048*896
    costs={'prefix_reconstruction_seconds':read_json(HERE/'CAPTURE_RECONSTRUCTION.json')['actual_prefix_replay_seconds'],
        'single_sample_backward_replays':len(results),'actual_AdamW_step_calls':sum(r['optimizer']['step_call_count'] for r in results),
        'formal_committed_updates':0,
        'sum_instrumented_replay_seconds':sum(r['timing_instrumented_seconds'] for r in results),
        'single_layer_diagnostic_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in results),
        'single_layer_diagnostic_peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in results),
        'FP16_two_full_calibration_hidden_buffers_bytes':allocation_elements*2*2,
        'FP32_two_full_calibration_hidden_buffers_bytes':allocation_elements*4*2,
        'cost_scope':'partial diagnostic accounting; excludes model loading, FP-prefix capture, CPU checks and file IO; hook timings are not full calibration throughput'}
    write_json(HERE/'AUDIT_COST.json',costs)
    write_json(HERE/'FINAL_STATUS.json',{'verdict':'M3_NUMERICAL_ROOT_CAUSE_IDENTIFIED',
        'root_cause_category':'OFFICIAL_NUMERICAL_INSTABILITY_REPRODUCED',
        'primary_category':'C: AMP/loss-scaled FP16 weight-gradient output overflow under the audited official LWC/AMP semantics',
        'scope':'Qwen2.5-0.5B-Instruct native architecture port, frozen sample/state and RTX4060 Laptop; not a general claim about every official OmniQuant model',
        'first_operation':'native_layer.mlp.up_proj F.linear / MmBackward0 weight-gradient output',
        'reproducibility':'3/3 exact loss; same four nonfinite LWC tensors and masks',
        'implementation_bug_confirmed':False,'port_bug_confirmed':False,
        'T4_executed':False,'device_specificity_established':False,
        'formal_training_continued':False,'formal_export_or_evaluation_executed':False,
        'correctness': 'PASS', 'next_action':'STOP; a later human-reviewed numerical protocol decision is needed before any baseline restart'})
    # Final code/data provenance, without hashing mutable summaries into themselves.
    write_json(HERE/'CODE_HASHES.json',{p.name:sha_file(p) for p in HERE.glob('*.py')})
    write_json(HERE/'FIXTURE_MANIFEST.json',{'files':[{'path':str(p.relative_to(HERE)), 'bytes':p.stat().st_size,'sha256':sha_file(p)} for p in (HERE/'fixtures').glob('*.pt')],
        'purpose':'minimal per-layer diagnostic state; not a complete quantized model or baseline export'})
    print('FINAL AUDIT PASS',flush=True)


if __name__ == '__main__':
    main()
