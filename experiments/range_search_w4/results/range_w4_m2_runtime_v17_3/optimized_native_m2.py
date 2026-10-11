"""v17.3 semantics-preserving native GPTQ wrapper.

The only cached numerical object is the block inverse of the damped H
Cholesky factor. It depends on H and the frozen GPTQ settings, not alpha.
The final Cholesky that creates native Hinv remains inside GPTQ.quantize().
All mutable candidate state is created afresh for every alpha.
"""
from dataclasses import dataclass
from types import MethodType

import torch
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


@dataclass
class PreparedNativeState:
    h_sha256: str
    hessian: torch.Tensor
    dead_mask: torch.Tensor
    damped_hessian: torch.Tensor
    cholesky: torch.Tensor
    inverse_before_final_cholesky: torch.Tensor
    nsamples: int
    group_size: int
    block_size: int
    damp_percent: float


@dataclass
class NativeM2Result:
    weight_fp16: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    g_idx: torch.Tensor


def _cfg():
    return QuantizeConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        static_groups=False,
        true_sequential=True,
        damp_percent=0.1,
        damp_auto_increment=0,
        mse=0,
    )


def _sha256(tensor):
    import hashlib

    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def prepare_native_state(hessian, *, nsamples):
    """Prepare the exact alpha-independent input consumed by native GPTQ."""
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1] or hessian.dtype != torch.float32:
        raise ValueError("Expected square FP32 native Hessian")
    if nsamples <= 0:
        raise ValueError("nsamples must be positive")

    config = _cfg()
    h = hessian.detach().clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    damp = config.damp_percent * torch.mean(torch.diag(h))
    diag = torch.arange(h.shape[0], device=h.device)
    h[diag, diag] += damp
    chol = torch.linalg.cholesky(h)
    # This is the same installed native method GPTQ.quantize invokes. Its
    # output is immutable and independent of alpha.
    inverse_before_final_cholesky = GPTQ.block_cholesky_inverse(
        None, chol, block_size=h.shape[0]
    )
    return PreparedNativeState(
        h_sha256=_sha256(hessian),
        hessian=hessian.detach().clone(),
        dead_mask=dead.detach().clone(),
        damped_hessian=h.detach().clone(),
        cholesky=chol.detach().clone(),
        inverse_before_final_cholesky=inverse_before_final_cholesky.detach().clone(),
        nsamples=int(nsamples),
        group_size=128,
        block_size=128,
        damp_percent=float(config.damp_percent),
    )


@torch.inference_mode()
def optimized_dynamic_gptq(weight, prepared, *, alpha=1.0, check_cholesky=False):
    """Run the installed native quantize() using an immutable cached H inverse.

    GPTQ.quantize() itself is unchanged. The instance method replacement only
    returns the precomputed output of its native block_cholesky_inverse call.
    """
    if weight.ndim != 2 or weight.dtype != torch.float16:
        raise ValueError("Frozen M2 requires FP16 [out, in] weight")
    if weight.device != prepared.hessian.device:
        raise ValueError("Weight and prepared Hessian must share a device")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    # The module parameter aliases pristine W. GPTQ._clone_module() makes the
    # candidate-local FP32 working copy before any in-place compensation.
    module = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, device="meta", dtype=weight.dtype
    )
    module.weight = torch.nn.Parameter(weight.detach(), requires_grad=False)
    core = GPTQ(module, qcfg=_cfg())
    core.quantizer.configure(perchannel=True, bits=4, sym=True)
    core.H = prepared.hessian.clone()
    core.nsamples = prepared.nsamples

    if check_cholesky:
        def cached_inverse(this, lower, upper=False, block_size=512):
            if not torch.equal(lower, prepared.cholesky):
                raise RuntimeError("Native candidate Cholesky differs from prepared immutable state")
            return prepared.inverse_before_final_cholesky
    else:
        def cached_inverse(this, lower, upper=False, block_size=512):
            return prepared.inverse_before_final_cholesky
    core.block_cholesky_inverse = MethodType(cached_inverse, core)

    if alpha != 1.0:
        original_find_params = core.quantizer.find_params
        calls = 0

        def range_hook(current, weight=False):
            nonlocal calls
            original_find_params(current, weight=weight)
            # Preserve native first whole-tensor find_params. Apply alpha only
            # at the native dynamic G128 group estimates, as in v17.2.
            if calls > 0:
                core.quantizer.scale = core.quantizer.scale * alpha
            calls += 1

        core.quantizer.find_params = range_hook

    quantized, scale, zero, g_idx, *_ = core.quantize(blocksize=128)
    return NativeM2Result(quantized, scale, zero, g_idx)
