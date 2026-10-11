"""Read-only access to frozen M3 inputs; all new writes stay in v21.2."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import torch

# Avoid writing import caches into frozen parent/source directories.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'range_w4_omniquant_full_v21_1'
M3 = PARENT / 'M3'
PORT = M3 / 'official_port'
BUNDLE = HERE.parent / 'range_w4_omniquant_kaggle_v20_1' / 'kaggle_bundle'
DEVICE = torch.device('cuda:0')
sys.path.insert(0, str(PORT))
sys.path.insert(0, str(PARENT / 'LOCAL_PREFLIGHT'))
if importlib.util.find_spec('termcolor') is None:
    color = types.ModuleType('termcolor')
    color.colored = lambda text, *args, **kwargs: text
    sys.modules['termcolor'] = color
spec = importlib.util.spec_from_file_location('frozen_formal', M3 / 'run_formal_m3.py')
formal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formal)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha_file(path: Path) -> str:
    return formal.sha_file(path)


def sha_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def tree(value: Any, device='cpu', fp32=False) -> Any:
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device).clone()
        return result.float() if fp32 and result.is_floating_point() else result
    if isinstance(value, dict):
        return {key: tree(item, device, fp32) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(tree(item, device, fp32) for item in value)
    if isinstance(value, list):
        return [tree(item, device, fp32) for item in value]
    return value


def stats(value: torch.Tensor | None) -> dict[str, Any]:
    if value is None:
        return {'present': False}
    t = value.detach()
    flat = t.flatten()
    finite = torch.isfinite(flat)
    good = flat[finite].double()
    result = {'present': True, 'shape': list(t.shape), 'dtype': str(t.dtype),
              'numel': t.numel(), 'finite_count': int(finite.sum()),
              'nan_count': int(torch.isnan(flat).sum()),
              'posinf_count': int(torch.isposinf(flat).sum()),
              'neginf_count': int(torch.isneginf(flat).sum()),
              'all_finite': bool(finite.all())}
    result.update({'min': float(good.min()) if good.numel() else None,
                   'max': float(good.max()) if good.numel() else None,
                   'mean': float(good.mean()) if good.numel() else None,
                   'std': float(good.std(unbiased=False)) if good.numel() else None,
                   'max_finite_abs': float(good.abs().max()) if good.numel() else None,
                   'mean_finite_abs': float(good.abs().mean()) if good.numel() else None})
    return result


def context():
    formal.write_runtime_environment()
    lock = read_json(M3 / 'FORMAL_PROTOCOL_LOCK.json')
    source = read_json(M3 / 'OMNIQUANT_SOURCE_AUDIT.json')
    for relative, info in source['files'].items():
        expected = info.get('patched_sha256') or info['official_sha256']
        if sha_file(PORT / relative) != expected:
            raise RuntimeError(f'protected source hash mismatch: {relative}')
    config_path = BUNDLE / 'configs/m3_preflight.json'
    calibration_path = BUNDLE / 'calibration/CALIBRATION_MANIFEST.json'
    if sha_file(config_path) != lock['config_sha256'] or sha_file(calibration_path) != lock['calibration_manifest_sha256']:
        raise RuntimeError('frozen config/calibration changed')
    state_path, completed, resume = formal.verify_latest_checkpoint(lock)
    if completed != 21:
        raise RuntimeError('expected completed prefix 0..20')
    calibration = read_json(calibration_path)
    tokens = [record['token_ids'] for record in calibration['samples']]
    for ids, record in zip(tokens, calibration['samples']):
        if len(ids) != 2048 or hashlib.sha256(json.dumps(ids).encode()).hexdigest() != record['token_sha256']:
            raise RuntimeError('token identity mismatch')
    return read_json(config_path), lock, tokens, state_path, resume


def new_layer(fixture, fp32_input=False):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from models.int_qwen2_layer import QuantQwen2DecoderLayer
    from quantize.utils import set_quant_state
    config = Qwen2Config.from_dict(fixture['model_config'])
    config._attn_implementation = 'sdpa'
    args = formal.make_args(read_json(BUNDLE / 'configs/m3_preflight.json'), None, 0, HERE / 'FORBIDDEN_TRAIN_OUTPUT')
    native = Qwen2DecoderLayer(config, layer_idx=fixture['layer'])
    native = native.float() if fp32_input else native.half()
    layer = QuantQwen2DecoderLayer(config, native, args).float().eval().to(DEVICE)
    layer.load_state_dict(fixture['state_dict'], strict=True)
    for name, param in layer.named_parameters():
        param.requires_grad_(fixture['requires_grad'][name])
    layer.let = False
    set_quant_state(layer, weight_quant=False, act_quant=True)
    formal.restore_rng_state(fixture['rng_state'])
    return layer, args


def protected_hashes():
    paths = list(PORT.rglob('*.py')) + list((M3 / 'checkpoints').rglob('*'))
    paths += list((M3 / 'logs').glob('*'))
    paths += [PARENT / n for n in ['PROTOCOL.json', 'REPORT.md', 'FINAL_STATUS.json',
                                  'correctness_checks.json', 'ARTIFACT_VALIDATION.json', 'LOCAL_MODEL_MANIFEST.json']]
    paths += [M3 / n for n in ['run_formal_m3.py', 'FORMAL_PROTOCOL_LOCK.json', 'STOP_DIAGNOSTIC.json']]
    return {str(p): sha_file(p) for p in paths if p.is_file()}
