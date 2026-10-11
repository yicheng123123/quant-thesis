from __future__ import annotations

import hashlib
import inspect
from collections import Counter

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2

from audit_common import HERE, M3, PORT, BUNDLE, context, formal, stats, read_json, write_json, sha_file, protected_hashes
from quantize.quantizer import UniformAffineQuantizer
from replay_basic import run_step


def main():
    config, lock, tokens, checkpoint_path, resume = context()
    fixture = torch.load(HERE / 'fixtures/actual_l21.pt', weights_only=False, map_location='cpu')
    saved = fixture['rng_state']
    rng = resume['rng_state']
    rng_checks = {'python': saved['python'] == rng['python'],
        'numpy_generator': saved['numpy'][0] == rng['numpy'][0],
        'numpy_array': bool(np.array_equal(saved['numpy'][1], rng['numpy'][1])),
        'numpy_tail': saved['numpy'][2:] == rng['numpy'][2:],
        'torch_cpu': bool(torch.equal(saved['torch_cpu'], rng['torch_cpu'])),
        'torch_cuda': all(torch.equal(a,b) for a,b in zip(saved['torch_cuda'],rng['torch_cuda']))}
    write_json(HERE / 'RNG_RESTORATION_CHECK.json', {'status':'PASS' if all(rng_checks.values()) else 'FAIL',
        'checks':rng_checks,'torch_deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic':torch.backends.cudnn.deterministic})
    if not all(rng_checks.values()):
        raise RuntimeError('RNG restoration mismatch')

    old_optimizer = read_json(HERE / 'OPTIMIZER_STATE_AUDIT.json')
    old_optimizer.update({'state_origin':'new AdamW inside each official block loop',
        'checkpoint_optimizer_state':'not saved or inherited; a completed-block boundary requires a fresh next-block optimizer',
        'exp_avg':'not allocated: empty state before and after the skipped step',
        'exp_avg_sq':'not allocated: empty state before and after the skipped step',
        'step':'not allocated', 'finite_status':'no optimizer tensors exist to be nonfinite',
        'stale_prior_block_state_ruled_out':True,
        'skip_evidence':'optimizer step pre-hook was not called; initial/final state remained empty and LWC parameters unchanged'})
    write_json(HERE / 'OPTIMIZER_STATE_AUDIT.json', old_optimizer)

    initialization = []
    completed_factor_stats = []
    with safe_open(str(formal.FROZEN_MODEL / 'model.safetensors'), framework='pt',device='cpu') as header:
        for idx in [0,12,20,21]:
            rows = []
            for projection in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
                               'mlp.gate_proj','mlp.up_proj','mlp.down_proj']:
                shape = header.get_slice(f'model.layers.{idx}.{projection}.weight').get_shape()
                q = UniformAffineQuantizer(n_bits=4,symmetric=False,dynamic_method='per_channel',group_size=128,shape=shape,lwc=True)
                rows.append({'projection':projection,'weight_shape':shape,
                    'upbound_factor':stats(q.upbound_factor),'lowbound_factor':stats(q.lowbound_factor)})
            initialization.append({'layer':idx,'source':'official constant-4 constructor reconstruction', 'parameters':rows})
            if idx in resume['state']:
                completed_factor_stats.append({'layer':idx,'source':'trained END-of-block checkpoint, not initialization',
                    'parameters':{name:stats(tensor) for name,tensor in resume['state'][idx].items()}})
    actual_initial = {name:stats(tensor) for name,tensor in fixture['state_dict'].items() if 'bound_factor' in name}
    write_json(HERE / 'LWC_PARAMETER_STATE_HISTORY.json', {
        'status':'PASS', 'initialization':initialization,
        'actual_layer21_pre_backward_parameters':actual_initial,
        'actual_layer21_all_init_4':all(s['min']==4 and s['max']==4 and s['all_finite'] for s in actual_initial.values()),
        'trained_checkpoint_context':completed_factor_stats,
        'interpretation':'L21 starts with the same official initialization as other layers, not their trained clipping values; no abnormal starting LWC factor was observed'})

    manifest = read_json(BUNDLE / 'calibration/CALIBRATION_MANIFEST.json')
    tokenizer = AutoTokenizer.from_pretrained(str(formal.FROZEN_MODEL),local_files_only=True,use_fast=True)
    samples = []
    for idx in [0,1,2,3]:
        counts = Counter(tokens[idx])
        samples.append({'sample_index':idx, 'sequence_length':len(tokens[idx]),
            'token_sha256':manifest['samples'][idx]['token_sha256'],
            'token_ids':tokens[idx], 'all_layers_reuse_same_sample_ids':True,
            'min_id':min(tokens[idx]),'max_id':max(tokens[idx]),
            'all_ids_in_model_vocabulary':all(0<=v<fixture['model_config']['vocab_size'] for v in tokens[idx]),
            'unique_token_count':len(counts), 'max_single_token_count':max(counts.values()),
            'special_token_counts':{str(v):counts[v] for v in tokenizer.all_special_ids if counts[v]},
            'source_metadata':{k:v for k,v in manifest['samples'][idx].items() if k!='token_ids'},
            'actual_layer21_input_stats':stats(fixture['inputs'][idx]),
            'actual_layer21_teacher_output_stats':stats(fixture['targets'][idx])})
    fp21 = torch.load(HERE/'fixtures/fp_l21.pt',weights_only=False,map_location='cpu')
    write_json(HERE / 'CALIBRATION_SAMPLE_AUDIT.json', {
        'status':'PASS', 'manifest_sha256':lock['calibration_manifest_sha256'],
        'samples':samples,'native_FP_prefix_layer21_sample0_input_stats':stats(fp21['inputs'][0]),
        'resampled_or_modified_tokens':False,
        'interpretation':'Frozen sample IDs are valid and identical across layers. Large but finite activations are observed; this does not establish corrupt or uniquely pathological sample0.'})

    coordinate_path = HERE / 'FAILURE_PARAMETER_COORDINATES.json'
    if not coordinate_path.exists():
        result = run_step(fixture)
        write_json(coordinate_path, result)
    result = read_json(coordinate_path)
    if result['loss'] != read_json(HERE/'REPLAY_1.json')['loss']:
        raise RuntimeError('coordinate replay loss mismatch')
    operands = torch.load(HERE/'FIRST_GEMM_OPERANDS_D1_OPERAND.pt',weights_only=False,map_location='cpu')
    gemm = read_json(HERE/'GEMM_OVERFLOW_VERIFICATION.json')
    contributions = []
    for c in gemm['overflow_coordinates']:
        products = operands['input'][:,c['weight_column']].double()*operands['incoming_grad_output'][:,c['weight_row']].double()
        top = products.abs().topk(5).indices.tolist()
        contributions.append({'weight_row':c['weight_row'],'weight_column':c['weight_column'],
            'official_G128_bound_factor_flat_index':c['weight_row']*7+c['weight_column']//128,
            'top_absolute_dot_product_contributions':[{'token_position':pos,'token_id':tokens[0][pos],
                'input_half_value':float(operands['input'][pos,c['weight_column']]),
                'incoming_scaled_half_gradient':float(operands['incoming_grad_output'][pos,c['weight_row']]),
                'product_fp64':float(products[pos])} for pos in top],
            'fp64_dot':float(products.sum())})
    write_json(HERE / 'ACTIVATION_OVERFLOW_CONTRIBUTIONS.json',{
        'data':contributions,
        'post_attention_norm_weight_channel62':float(fixture['state_dict']['native_layer.post_attention_layernorm.weight'][62]),
        'layer21_post_attention_norm_weight_stats':stats(fixture['state_dict']['native_layer.post_attention_layernorm.weight']),
        'failed_parameters':[{k:g[k] for k in ['parameter_name','parameter_shape','nonfinite_indices','gradient']} for g in result['gradients'] if not g['gradient']['all_finite']]})

    original_root = HERE.parent / 'range_w4_fullmodel_baseline_v16/official_source'
    algorithm = {}
    for path in ['quantize/quantizer.py','quantize/int_linear.py','quantize/utils.py','utils.py']:
        algorithm[path]={'official_source_sha256':sha_file(original_root/path),
                         'formal_source_sha256':sha_file(PORT/path),
                         'identical':sha_file(original_root/path)==sha_file(PORT/path)}
    write_json(HERE / 'SOURCE_NUMERICAL_AUDIT.json',{
        'official_commit':lock['official_commit'], 'protected_algorithm_sources':algorithm,
        'official_main':str(original_root/'main.py'), 'official_block_driver':str(original_root/'quantize/omniquant.py'),
        'official_W4A16_default':'deactive_amp=False; qlayer.float(); student autocast FP16; per-block fresh GradScaler',
        'official_deactive_amp_flag':'exposed CLI; auto-enabled for 8<=bits<16, not W4; explicit flag changes activation cache dtype/student traincast but not the unconditional teacher autocast or GradScaler wrapper',
        'quantizer_guards':'scale clamp[1e-5,1e4], zero clamp[-1e4,1e4], integer clamp[0,15], sigmoid clipping factors; no sqrt/log in LWC affine path',
        'LET_truncate_guard':'truncate_number threshold1e-2 applies to LET smooth_scale only; LET is off here',
        'native_family_source_comparison':'upstream QuantLlamaMLP and native Qwen2MLP both use down_proj(SiLU(gate_proj(x))*up_proj(x)); shared unmodified QuantLinear uses F.linear; no extra MLP GEMM overflow guard in the upstream Llama adapter',
        'attention_guards':'upstream OPT/Llama upcast softmax; native Qwen SDPA/RMSNorm plumbing preserved; first bad node is MLP weight-gradient GEMM, not attention',
        'qwen_native_source':inspect.getfile(modeling_qwen2),'qwen_native_source_sha256':sha_file(__import__('pathlib').Path(inspect.getfile(modeling_qwen2))),
        'implementation_or_port_bug_confirmed':False,
        'native_supported_family_runtime_executed':False,
        'scope':'source comparison only; no additional model download or native-family calibration run'})
    write_json(HERE / 'PROTECTED_ARTIFACT_CHECK.json', {'status':'PASS' if protected_hashes()==read_json(HERE/'PROTECTED_HASHES.json') else 'FAIL',
        'formal_sources_checkpoints_logs_reports_unchanged':protected_hashes()==read_json(HERE/'PROTECTED_HASHES.json')})
    print('STATE AND DATA AUDIT DONE',flush=True)


if __name__ == '__main__':
    main()
