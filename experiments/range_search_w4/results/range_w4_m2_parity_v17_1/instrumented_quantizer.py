def quantize(x, scale, zero, maxq, requires_groupwise_processing: bool):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    if requires_groupwise_processing:
        q = torch.clamp(torch.round(x / scale), -maxq, maxq)
        _rec.emit('rounded_integer_q', {'value': q.flatten()}, _rec.column, 33)
        return scale * q
    else:
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        _rec.emit('rounded_integer_q', {'value': q.flatten()}, _rec.column, 36)
        return scale * (q - zero)
