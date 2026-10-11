def quantize(self, blocksize=128):
    start = time.time()
    for inp in self.fwd_inputs_buffered_data:
        self.process_batch(inp)
    del self.fwd_inputs_buffered_data
    if sys.platform == 'darwin' and os.getenv('PYTORCH_ENABLE_MPS_FALLBACK') != '1':
        raise RuntimeError('For MacOS you must set env `PYTORCH_ENABLE_MPS_FALLBACK=1` before running quantization.')
    if self.module_copy is None:
        W = self._clone_module()
    else:
        W = self.module_copy
        self.module_copy = None
    self.quantizer.find_params(W, weight=True)
    H = self.H
    del self.H
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    scale = []
    zero = []
    now_idx = 1
    if self.qcfg.static_groups:
        import copy
        groups = []
        for i in range(0, self.columns, self.qcfg.group_size):
            quantizer = copy.deepcopy(self.quantizer)
            quantizer.find_params(W[:, i:i + self.qcfg.group_size], weight=True)
            scale.append(quantizer.scale)
            zero.append(quantizer.zero)
            groups.append(quantizer)
    if self.qcfg.desc_act:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)
    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)
    damp_percent = self.qcfg.damp_percent
    while 1 > damp_percent > 0:
        try:
            damp = damp_percent * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.device)
            H[diag, diag] += damp
            with lock:
                H = torch.linalg.cholesky(H)
                try:
                    H = self.block_cholesky_inverse(H, block_size=self.columns)
                except torch.OutOfMemoryError:
                    H = self.block_cholesky_inverse(H, block_size=self.columns // 2)
                    log.warn('Quantization: OOM bypassed via low memory math at a cost of lower accuracy: `cholesky_inverse`')
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H
            break
        except torch._C._LinAlgError as e:
            if self.qcfg.damp_auto_increment != 0:
                log.warn(f'Quantization: Current `damp_percent = {damp_percent:.5f}` is too low, auto-incrementing by `{self.qcfg.damp_auto_increment:.5f}`')
                damp_percent += self.qcfg.damp_auto_increment
            else:
                log.warn('Quantization: Please increase damp or nsamples for calibration data to avoid the following quant error: current damp_percent=`{damp_percent:.5f}`')
                raise e
    if not 0 < damp_percent < 1:
        raise ValueError(f'Quantization: `damp_percent` must between 0 and 1. current is {damp_percent}')
    for i1 in range(0, self.columns, blocksize):
        i2 = min(i1 + blocksize, self.columns)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        _audit_emit('working_block', {'value': W1, 'Hinv_block': Hinv[i1:i2, i1:i2]}, i1, 380)
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]
            _audit_emit('pre_quant_value', {'value': w}, i1 + i, 387)
            d = Hinv1[i, i]
            _audit_emit('divisor', {'value': d}, i1 + i, 388)
            if self.qcfg.group_size != -1:
                if not self.qcfg.static_groups:
                    if (i1 + i) % self.qcfg.group_size == 0:
                        self.quantizer.find_params(W[:, i1 + i:i1 + i + self.qcfg.group_size], weight=True)
                        _audit_emit('group_entry', {'value': W[:, i1 + i:i1 + i + self.qcfg.group_size], 'scale': self.quantizer.scale, 'zero': self.quantizer.zero}, i1 + i, 393)
                    if (i1 + i) // self.qcfg.group_size - now_idx == -1:
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        now_idx += 1
                else:
                    idx = i1 + i
                    if self.qcfg.desc_act:
                        idx = perm[idx]
                    self.quantizer = groups[idx // self.qcfg.group_size]
            q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
            _audit_emit('dequantized_q', {'value': q}, i1 + i, 406)
            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2
            err1 = (w - q) / d
            _audit_emit('error_vector', {'value': err1}, i1 + i, 410)
            _observed_product = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            _audit_emit('compensation_product', {'value': _observed_product}, i1 + i, 411)
            W1[:, i:] -= _observed_product
            _audit_emit('updated_remaining_W', {'value': W1[:, i:]}, i1 + i, 411)
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2
        _observed_transition = Err1.matmul(Hinv[i1:i2, i2:])
        _audit_emit('transition_product', {'value': _observed_transition}, i1, 417)
        W[:, i2:] -= _observed_transition
        _audit_emit('group_exit_remaining_W', {'value': W[:, i2:]}, i1, 417)
    torch_sync(self.device)
    avg_loss = torch.sum(Losses).item() / self.nsamples
    if math.isnan(avg_loss):
        print('Losses sum item:', torch.sum(Losses).item())
        raise ValueError(f'Quantization: Failed due to `NaN` loss for `{self.name}`')
    group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns
    if self.qcfg.static_groups and self.qcfg.desc_act:
        g_idx = [perm[i] // group_size for i in range(self.columns)]
    else:
        g_idx = [i // group_size for i in range(self.columns)]
    g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
    if self.qcfg.desc_act:
        Q = Q[:, invperm]
        g_idx = g_idx[invperm]
    if isinstance(self.module, transformers.Conv1D):
        Q = Q.t()
    if Q.shape != self.module.weight.shape:
        Q = Q.reshape(self.module.weight.shape).type_as(self.module.weight.data)
    else:
        Q = Q.type_as(self.module.weight.data)
    Q = Q.to(device=self.device)
    if scale == []:
        scale.append(self.quantizer.scale)
        zero.append(self.quantizer.zero)
    scale = torch.cat(scale, dim=1)
    zero = torch.cat(zero, dim=1)
    duration = time.time() - start
    return (Q, scale, zero, g_idx, duration, avg_loss, damp_percent, self.nsamples)
