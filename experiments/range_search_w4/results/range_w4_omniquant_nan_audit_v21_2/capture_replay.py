"""Replay completed prefix without optimizer calls; stop at L21 optimizer init."""
from __future__ import annotations

import copy
import gc
import inspect
import time
from types import SimpleNamespace

import torch

from audit_common import HERE, M3, PARENT, DEVICE, formal, context, tree, write_json, sha_file, sha_tensor, protected_hashes
from models.int_qwen2_layer import QuantQwen2DecoderLayer
from quantize.omniquant import omniquant
from run_local_preflight import load_exact_model


class AuditBoundary(Exception):
    pass


class Logger:
    def info(self, message):
        print(message, flush=True)


def fixture(layer_id, module, model_config, inputs, targets, kwargs, rng, dependency):
    return {'layer': layer_id, 'model_config': model_config.to_dict(),
            'state_dict': tree(module.state_dict()),
            'requires_grad': {name: p.requires_grad for name, p in module.named_parameters()},
            'inputs': tree(inputs), 'targets': tree(targets), 'kwargs': tree(kwargs),
            'rng_state': tree(rng), 'dependency': dependency,
            'input_hashes': [sha_tensor(x) for x in inputs],
            'target_hashes': [sha_tensor(x) for x in targets]}


def main():
    config, lock, tokens, checkpoint_path, resume = context()
    HERE.mkdir(parents=True, exist_ok=True)
    fixtures = HERE / 'fixtures'
    fixtures.mkdir(exist_ok=True)
    if (fixtures / 'actual_l21.pt').exists():
        raise RuntimeError('capture fixture exists; use its audited cache rather than overwriting')
    write_json(HERE / 'PROTECTED_HASHES.json', protected_hashes())
    write_json(HERE / 'PROTOCOL.json', {
        'name': 'v21.2 OmniQuant layer21 non-finite gradient root-cause audit',
        'scope': 'diagnostic only; no formal optimizer updates, full-model export, or PPL',
        'source_parent': str(PARENT), 'official_commit': lock['official_commit'],
        'frozen_model_revision': lock['model_revision'],
        'parent_lock_sha256': sha_file(M3 / 'FORMAL_PROTOCOL_LOCK.json'),
        'checkpoint': str(checkpoint_path.parent),
        'calibration_manifest_sha256': lock['calibration_manifest_sha256'],
        'replicas': 3, 'failure_loss_reference': 0.15326565504074097,
        'replica_loss_absolute_tolerance': 0.0,
        'sample_diagnostics': [0, 1, 2, 3], 'layer_diagnostics': [20, 21, 22, 23],
        'comparisons': ['D1 formal autocast FP16 + native GradScaler default',
                        'D2 FP32 forward/backward with the same starting weights, input values, targets and scale',
                        'D3 loss-scaling audit: formal autocast forward with unscaled backward only'],
        'layer_diagnostic_dependencies': {
            'actual_L20_L21': 'reconstructed learned W4 prefix at their actual pre-update boundary',
            'FP_prefix_L20_L21_L22_L23': 'native FP16 prefix; fresh official LWC initialization; diagnostic comparison only',
            'limitation': 'No completed L21 checkpoint exists, so actual formal L22/L23 prefix states cannot be reconstructed.'},
        'forbidden': ['continue formal training', 'change frozen optimizer/LR/epsilon/quantizer',
                      'skip sample/layer', 'complete L21', 'export full model', 'PPL/C4', 'formal Kaggle rerun'],
        'writes': 'only this v21.2 directory',
        'expected_cost': 'one read-only 21-block x128-sample teacher/quantized prefix replay; <=20 one-sample backward diagnostics plus bounded trace controls; no epoch loops',
    })
    model, identity = load_exact_model(formal.MODEL_ID, formal.MODEL_REVISION, str(formal.FROZEN_MODEL), DEVICE, HERE)
    args = formal.make_args(config, checkpoint_path, 21, HERE / 'FORBIDDEN_TRAIN_OUTPUT')
    args.v21_restore_rng = lambda: formal.restore_rng_state(resume['rng_state'])
    before20 = {}
    capture_info = {'optimizer_calls': 0, 'prefix_replayed_blocks': [], 'captures': []}

    def callback(event, layer_id, module):
        frame = inspect.currentframe().f_back
        local = frame.f_locals
        if event == 'replayed_prefix':
            capture_info['prefix_replayed_blocks'].append(layer_id)
            if layer_id == 19:
                before20['inputs'] = tree(local['quant_inps'][:4])
            if layer_id == 20:
                fresh = QuantQwen2DecoderLayer(model.config, local['layer'], args).float().eval()
                f = fixture(20, fresh, model.config, before20['inputs'], local['fp_inps'][:4],
                            local['cache']['layer_kwargs'][:4], resume['rng_state'],
                            'actual learned W4 prefix 0..19; fresh L20 LWC=4 initialization')
                torch.save(f, fixtures / 'actual_l20.pt')
                capture_info['captures'].append({'path': 'fixtures/actual_l20.pt', 'sha256': sha_file(fixtures / 'actual_l20.pt')})
                del fresh
        elif event == 'optimizer_init':
            if layer_id != 21:
                raise RuntimeError('unexpected optimizer initialization before target boundary')
            optimizer = local['optimizer']
            if optimizer.state:
                raise RuntimeError('target optimizer is not fresh')
            f = fixture(21, module, model.config, local['quant_inps'][:4], local['fp_inps'][:4],
                        local['cache']['layer_kwargs'][:4], formal.current_rng_state(),
                        'actual learned W4 prefix 0..20; fresh L21 LWC=4 initialization')
            f['optimizer_initial_state'] = optimizer.state_dict()
            torch.save(f, fixtures / 'actual_l21.pt')
            capture_info['captures'].append({'path': 'fixtures/actual_l21.pt', 'sha256': sha_file(fixtures / 'actual_l21.pt')})
            capture_info['target_optimizer_state_entries'] = len(optimizer.state)
            raise AuditBoundary('captured L21 before any backward/optimizer call')
        else:
            raise RuntimeError(f'forbidden training event {event}')

    args.v21_layer_callback = callback
    args.v21_step_check_callback = lambda *a: (_ for _ in ()).throw(RuntimeError('forbidden formal step'))
    data = [(torch.tensor(ids, dtype=torch.long).view(1, -1), i) for i, ids in enumerate(tokens)]
    start = time.perf_counter()
    try:
        omniquant(SimpleNamespace(model=model, device=DEVICE, _device=DEVICE, seqlen=2048), args, data, None, None, Logger())
        raise RuntimeError('audit boundary was not reached')
    except AuditBoundary as exc:
        print(str(exc), flush=True)
    torch.cuda.synchronize()
    capture_info['actual_prefix_replay_seconds'] = time.perf_counter() - start
    capture_info['status'] = 'PASS'
    capture_info['target_boundary'] = 'L21 after float conversion and fresh optimizer construction, before native scaler construction'
    capture_info['state_reconstruction'] = 'pristine exact model + block20 saved LWC factors + official half/inplace W4 prefix quantization + full128 sample replay'
    capture_info['optimizer_state_restoration'] = 'fresh per-block state (empty), as required by official source; no prior optimizer state is inherited'
    write_json(HERE / 'CAPTURE_RECONSTRUCTION.json', capture_info)
    del model, callback, before20
    gc.collect()
    torch.cuda.empty_cache()

    # A separate common FP-prefix control, without substituting it for actual L21 inputs.
    model, _ = load_exact_model(formal.MODEL_ID, formal.MODEL_REVISION, str(formal.FROZEN_MODEL), DEVICE, HERE)
    held = {}
    handles = []
    for idx in [20, 21, 22, 23]:
        def pre(module, inputs, kwargs, idx=idx):
            held[idx] = {'inputs': tree(inputs[0] if inputs else kwargs['hidden_states']), 'kwargs': tree(kwargs)}
        def post(module, inputs, output, idx=idx):
            held[idx]['targets'] = tree(output[0])
        handles.append(model.model.layers[idx].register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(model.model.layers[idx].register_forward_hook(post))
    with torch.no_grad(), torch.cuda.amp.autocast():
        model.model(torch.tensor(tokens[0], dtype=torch.long, device=DEVICE).view(1, -1), use_cache=False)
    for h in handles:
        h.remove()
    for idx in [20, 21, 22, 23]:
        fresh = QuantQwen2DecoderLayer(model.config, model.model.layers[idx], args).float().eval()
        record = held[idx]
        record['kwargs'].pop('hidden_states', None)
        f = fixture(idx, fresh, model.config, record['inputs'], record['targets'],
                    [record['kwargs']], resume['rng_state'], 'native FP16 prefix; fresh LWC=4; diagnostic only')
        torch.save(f, fixtures / f'fp_l{idx}.pt')
        del fresh
    write_json(HERE / 'FP_PREFIX_CAPTURE.json', {'status': 'PASS', 'sample_index': 0,
                                               'layers': [20, 21, 22, 23], 'formal_state_equivalence_claim': False})
    print('CAPTURE DONE', flush=True)


if __name__ == '__main__':
    main()
