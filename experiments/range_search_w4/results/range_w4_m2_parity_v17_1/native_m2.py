"""Frozen W4/G128 M2 interface using the actual GPTQModel core.

Input is pristine FP16 current-target W plus undamped native FP32 H.
Alpha=1 calls the native quantizer without a range modification. A non-unit
alpha only scales the freshly estimated dynamic group scale. This module
contains no search loop; the v17.1 audit runs alpha=1 only.
"""
from dataclasses import dataclass
import copy
import torch
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ

@dataclass
class NativeM2Result:
    weight_fp16: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    g_idx: torch.Tensor
    group_inputs: list[torch.Tensor]

def native_dynamic_gptq(weight, hessian, *, nsamples, alpha=1.0, trace=False):
    if weight.ndim!=2 or weight.dtype!=torch.float16:
        raise ValueError('Frozen M2 requires original FP16 weight [rows, columns]')
    if hessian.shape!=(weight.shape[1],weight.shape[1]) or hessian.dtype!=torch.float32:
        raise ValueError('Supply undamped native FP32 Hessian of matching geometry')
    if weight.device!=hessian.device or not (0<alpha<=1) or nsamples<=0:
        raise ValueError('Invalid frozen M2 input')
    config=QuantizeConfig(bits=4,group_size=128,sym=True,desc_act=False,
        static_groups=False,true_sequential=True,damp_percent=.1,damp_auto_increment=0,mse=0)
    # Preserve original target geometry. Arbitrarily slicing output rows can
    # select different GEMM arithmetic; callers slice results after execution.
    module=torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,device='meta',dtype=weight.dtype)
    module.weight=torch.nn.Parameter(weight.detach().clone(),requires_grad=False)
    core=GPTQ(module,qcfg=config)
    core.quantizer.configure(perchannel=True,bits=4,sym=True)
    core.H=hessian.detach().clone();core.nsamples=int(nsamples)
    entries=[]
    if alpha!=1.0 or trace:
        original=core.quantizer.find_params
        calls=0
        def range_hook(current,weight=False):
            nonlocal calls
            original(current,weight=weight)
            # Native calls find_params on all W once, then at each dynamic group.
            if calls>0:
                if trace:entries.append(current.detach().cpu().clone())
                if alpha!=1.0:core.quantizer.scale=core.quantizer.scale*alpha
            calls+=1
        core.quantizer.find_params=range_hook
    quantized,scale,zero,gidx,*_=core.quantize(blocksize=128)
    return NativeM2Result(quantized,scale,zero,gidx,entries)
