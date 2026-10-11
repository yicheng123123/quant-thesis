def dynamic_gptq(original_weight: torch.Tensor, hinv: torch.Tensor, *, alpha: float | Sequence[float]=1.0, group_size: int=128, block_size: int=128, trace: bool=False) -> DynamicResult:
    """Use the same block updates and group-entry W as GPTQModel.

    Native find_params reads global W at the group boundary, while the current
    block uses W1 for column compensation. Keep those states distinct.
    """
    if original_weight.ndim != 2 or hinv.ndim != 2:
        raise ValueError('expected W [rows, columns] and square Hinv')
    rows, cols = original_weight.shape
    if hinv.shape != (cols, cols) or group_size <= 0 or block_size <= 0:
        raise ValueError('incompatible GPTQ geometry')
    ratios = [float(alpha)] * math.ceil(cols / group_size) if isinstance(alpha, (int, float)) else list(alpha)
    if len(ratios) != math.ceil(cols / group_size) or any((not 0 < a <= 1 for a in ratios)):
        raise ValueError('one valid alpha per group is required')
    work = original_weight.float().clone()
    qweight = torch.zeros_like(work)
    scales = []
    zeros = []
    entries = []
    current_scale = None
    current_zero = None
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        local = work[:, i1:i2].clone()
        _audit_emit('working_block', {'value': local, 'Hinv_block': hinv[i1:i2, i1:i2]}, i1, 85)
        errors = torch.zeros_like(local)
        local_inv = hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j
            if col % group_size == 0:
                entering = work[:, col:min(col + group_size, cols)]
                if trace:
                    entries.append(entering.detach().cpu().clone())
                current_scale, current_zero = dynamic_params(entering, ratios[col // group_size])
                _audit_emit('group_entry', {'value': entering, 'scale': current_scale, 'zero': current_zero}, col, 94)
                scales.append(current_scale.clone())
                zeros.append(current_zero.clone())
            assert current_scale is not None and current_zero is not None
            current = local[:, j:j + 1]
            _audit_emit('pre_quant_value', {'value': current.flatten()}, col, 98)
            _audit_emit('divisor', {'value': local_inv[j, j]}, col, 98)
            q = (torch.round(current / current_scale) + current_zero).clamp(0, 15)
            _audit_emit('rounded_integer_q', {'value': q.flatten()}, col, 99)
            q = (q - current_zero) * current_scale
            _audit_emit('dequantized_q', {'value': q.flatten()}, col, 100)
            qweight[:, col] = q.flatten()
            error = (current.flatten() - q.flatten()) / local_inv[j, j]
            _audit_emit('error_vector', {'value': error}, col, 102)
            _observed_product = error[:, None] @ local_inv[j, j:].unsqueeze(0)
            _audit_emit('compensation_product', {'value': _observed_product}, col, 103)
            local[:, j:] -= _observed_product
            _audit_emit('updated_remaining_W', {'value': local[:, j:]}, col, 103)
            errors[:, j] = error
        if i2 < cols:
            _observed_transition = errors @ hinv[i1:i2, i2:]
            _audit_emit('transition_product', {'value': _observed_transition}, i1, 106)
            work[:, i2:] -= _observed_transition
            _audit_emit('group_exit_remaining_W', {'value': work[:, i2:]}, i1, 106)
    return DynamicResult(qweight, torch.cat(scales, dim=1), torch.cat(zeros, dim=1), entries)
