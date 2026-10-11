"""Explicit Qwen2 decoder contract under use_cache=False/output_attentions=False."""
import numpy as np
import torch

def extract_hidden(result, *, expected_shape):
    # Supported native APIs: new Tensor, legacy one-element tuple.
    if torch.is_tensor(result):
        hidden=result
    elif type(result) is tuple and len(result)==1 and torch.is_tensor(result[0]):
        hidden=result[0]
    else:
        raise TypeError(f'unsupported decoder result type={type(result).__name__}; expected Tensor or (Tensor,)')
    expected=tuple(expected_shape)
    if hidden.ndim!=3 or tuple(hidden.shape)!=expected:
        raise ValueError(f'decoder hidden shape {tuple(hidden.shape)} != full [batch,tokens,hidden] {expected}')
    if not hidden.is_floating_point() or not bool(torch.isfinite(hidden).all()):
        raise ValueError('decoder hidden is not finite floating-point')
    return hidden  # Identity and bytes retained; no cast, clone, slicing or reshape.

def write_document(cache, document, hidden):
    if not isinstance(cache,np.ndarray) or cache.ndim!=3:
        raise ValueError('cache must have [documents,tokens,features] shape')
    if not (0<=document<len(cache)):raise IndexError('document outside cache')
    expected=tuple(cache[document:document+1].shape)
    if not torch.is_tensor(hidden) or hidden.ndim!=3 or tuple(hidden.shape)!=expected:
        raise ValueError(f'cache source shape {getattr(hidden,"shape",None)} != target {expected}; broadcasting forbidden')
    source=hidden.detach().cpu().numpy()
    target=cache[document:document+1]
    if source.shape!=target.shape or source.dtype!=target.dtype:
        raise ValueError(f'cache source shape/dtype {source.shape}/{source.dtype} != target {target.shape}/{target.dtype}')
    np.copyto(target,source,casting='no')  # Shape equality already established.
