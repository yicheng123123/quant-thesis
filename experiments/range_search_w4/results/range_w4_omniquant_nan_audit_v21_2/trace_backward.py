"""Observation-only tensor/node hooks around the unchanged native math."""
from __future__ import annotations

import types

import torch

from audit_common import HERE, stats, write_json
from quantize.int_linear import QuantLinear
import quantize.quantizer as quantizer_source


def gradient_stats(value):
    if value is None:
        return {'present': False}
    t = value.detach().float()
    finite = torch.isfinite(t)
    result = {'present': True, 'shape': list(t.shape), 'dtype': str(value.dtype), 'numel': t.numel(),
              'all_finite': bool(finite.all()), 'finite_count': int(finite.sum()),
              'nan_count': int(torch.isnan(t).sum()), 'posinf_count': int(torch.isposinf(t).sum()),
              'neginf_count': int(torch.isneginf(t).sum()),
              'max_finite_abs': float(torch.where(finite, t.abs(), 0).max()) if t.numel() else None}
    return result


class TraceObserver:
    def __init__(self, suffix='D1'):
        self.suffix = suffix
        self.handles = []
        self.method_restores = []
        self.forward = {}
        self.quantizers = {}
        self.tags = {}
        self.tensor_events = []
        self.node_events = []
        self.nodes = []
        self.current_quantizer = None
        self.first_transition = None
        self.leaf_metadata = []

    def tag(self, label, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        self.forward[label] = stats(tensor)
        if tensor.grad_fn is not None:
            self.tags.setdefault(tensor.grad_fn, []).append(label)
        if tensor.requires_grad:
            def hook(grad, label=label):
                self.tensor_events.append({'event_index': len(self.tensor_events), 'tensor': label, 'gradient': gradient_stats(grad)})
            self.handles.append(tensor.register_hook(hook))

    def install(self, layer):
        for name, p in layer.named_parameters():
            if 'bound_factor' in name:
                self.tag(name, p)
        for name, module in layer.named_modules():
            if isinstance(module, QuantLinear) or name in ['native_layer.input_layernorm', 'native_layer.post_attention_layernorm',
                    'native_layer.self_attn', 'native_layer.mlp', 'native_layer.mlp.act_fn', 'native_layer']:
                def hook(module, inputs, output, name=name):
                    out = output[0] if isinstance(output, tuple) else output
                    self.tag(name + '.output', out)
                    if isinstance(module, QuantLinear) and out.grad_fn is not None:
                        node = out.grad_fn
                        while node.name().startswith(('ViewBackward', 'UnsafeViewBackward', 'ReshapeAliasBackward')):
                            options = [n for n, _ in node.next_functions if n is not None]
                            if len(options) != 1:
                                break
                            node = options[0]
                        self.tags.setdefault(node, []).append(name + '.linear_operation')
                self.handles.append(module.register_forward_hook(hook))
            if name in ['native_layer.post_attention_layernorm', 'native_layer.mlp.down_proj']:
                def pre(module, inputs, name=name):
                    self.tag('residual_after_attention' if name.endswith('post_attention_layernorm') else 'swiglu_product', inputs[0])
                self.handles.append(module.register_forward_pre_hook(pre))
            if isinstance(module, QuantLinear):
                self.tag(name + '.unoptimized_original_weight_buffer', module.weight)
                q = module.weight_quantizer
                base = q.fake_quant
                def observed_fake(x, scale, rounded_zero, name=name, base=base, q=q):
                    self.current_quantizer = name
                    self.tag(name + '.scale', scale)
                    self.tag(name + '.rounded_zero', rounded_zero)
                    with torch.no_grad():
                        w = x.reshape(-1, q.group_size)
                        raw_min, raw_max = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
                        low = q.sigmoid(q.lowbound_factor) * raw_min
                        high = q.sigmoid(q.upbound_factor) * raw_max
                        norm = w / scale
                        rounded = norm.round()
                        clamp_input = rounded + rounded_zero
                        self.quantizers[name] = {
                            'raw_weight': stats(w), 'raw_min': stats(raw_min), 'raw_max': stats(raw_max),
                            'lower_clipping_bound': stats(low), 'upper_clipping_bound': stats(high),
                            'lowbound_factor': stats(q.lowbound_factor), 'upbound_factor': stats(q.upbound_factor),
                            'sigmoid_low_factor': stats(q.sigmoid(q.lowbound_factor)),
                            'sigmoid_up_factor': stats(q.sigmoid(q.upbound_factor)),
                            'scale': stats(scale), 'rounded_zero_point': stats(rounded_zero),
                            'scale_at_clipmin_count': int((scale == quantizer_source.CLIPMIN).sum()),
                            'inverse_scale_absmax': float((1.0 / scale).abs().max()),
                            'normalized_weight_and_round_input': stats(norm), 'round_output': stats(rounded),
                            'clamp_input': stats(clamp_input), 'clamp_output': stats(clamp_input.clamp(q.qmin, q.qmax)),
                            'intermediate_provenance': 'bounds/clamp intermediates recomputed for diagnostics with identical no-grad formulas; scale/rounded_zero/dequant are actual native forward tensors'}
                    result = base(x, scale, rounded_zero)
                    self.tag(name + '.dequantized_weight', result)
                    self.quantizers[name]['dequant_output'] = stats(result)
                    self.current_quantizer = None
                    return result
                q.fake_quant = observed_fake
                self.method_restores.append((q, 'fake_quant', base))
        self.base_round = quantizer_source.round_ste
        def observed_round(x):
            name = self.current_quantizer or 'unknown_quantizer'
            self.tag(name + '.normalized_weight_round_input', x)
            result = self.base_round(x)
            self.tag(name + '.round_ste_output', result)
            return result
        quantizer_source.round_ste = observed_round

    def after_quantization(self, layer, inp, target):
        self.tag('quant_student_input', inp)
        self.tag('FP_teacher_block_output', target)

    def after_forward(self, layer, inp, target, out, loss):
        self.tag('student_block_output', out)
        self.tag('reconstruction_loss', loss)

    def saved_operands(self, node):
        result = {}
        for name in dir(node):
            if not name.startswith('_saved_'):
                continue
            try:
                value = getattr(node, name)
                if isinstance(value, torch.Tensor):
                    result[name] = stats(value)
                elif isinstance(value, (int, float, bool, str, type(None))):
                    result[name] = value
                elif isinstance(value, (tuple, list)) and all(isinstance(v, (int, float, bool, str)) for v in value):
                    result[name] = value
            except RuntimeError as exc:
                result[name] = {'unavailable': str(exc)}
        return result

    def install_graph(self, loss):
        visited = set()
        stack = [loss.grad_fn]
        while stack:
            node = stack.pop()
            if node is None or node in visited:
                continue
            visited.add(node)
            self.nodes.append(node)
            stack.extend(n for n, _ in node.next_functions if n is not None)
        for node_id, node in enumerate(self.nodes):
            if hasattr(node, 'variable'):
                self.leaf_metadata.append({'node_id':node_id,'shape':list(node.variable.shape),
                                           'dtype':str(node.variable.dtype),'requires_grad':node.variable.requires_grad})
            def callback(grad_inputs, grad_outputs, node=node, node_id=node_id):
                ins = [gradient_stats(v) for v in grad_inputs]
                outs = [gradient_stats(v) for v in grad_outputs]
                event = {'execution_index': len(self.node_events), 'graph_node_id': node_id,
                         'operation': node.name(), 'tags': self.tags.get(node, []),
                         'grad_outputs': outs, 'grad_inputs': ins}
                has_output = any(v.get('present') for v in outs)
                finite_outputs = all(v.get('all_finite', True) for v in outs)
                bad_inputs = any(not v.get('all_finite', True) for v in ins)
                event['finite_to_nonfinite_transition'] = has_output and finite_outputs and bad_inputs
                if event['finite_to_nonfinite_transition'] and self.first_transition is None:
                    event['saved_forward_operands'] = self.saved_operands(node)
                    self.first_transition = event
                    if node.name() == 'MmBackward0' and len(grad_inputs) == 2:
                        # Diagnostic replay operands; no model or optimizer state is committed.
                        torch.save({'operation': node.name(), 'tags': event['tags'],
                            'input': node._saved_self.detach().cpu().clone(),
                            'incoming_grad_output': grad_outputs[0].detach().cpu().clone(),
                            'native_grad_weight_transpose': grad_inputs[1].detach().cpu().clone()},
                            HERE / f'FIRST_GEMM_OPERANDS_{self.suffix}.pt')
                self.node_events.append(event)
            self.handles.append(node.register_hook(callback))

    def finish(self, result):
        for h in self.handles:
            h.remove()
        for q, name, base in self.method_restores:
            setattr(q, name, base)
        quantizer_source.round_ste = self.base_round
        bad_tensors = [e for e in self.tensor_events if not e['gradient']['all_finite']]
        forward_bad = [key for key, value in self.forward.items() if not value['all_finite']]
        write_json(HERE / f'FORWARD_NUMERICS_{self.suffix}.json', {
            'mode': self.suffix, 'all_observed_tensors_finite': not forward_bad,
            'nonfinite_forward_tensors': forward_bad, 'tensors': self.forward,
            'attention_scores': 'not materialized: preserve native SDPA backend; projection/attention/residual outputs and autograd node are observed'})
        write_json(HERE / f'QUANTIZER_NUMERICS_AUDIT_{self.suffix}.json', {'mode': self.suffix, 'quantizers': self.quantizers,
            'derivative_formula': 'scale=(sigmoid(up)*amax(W)-sigmoid(low)*amin(W))/15, clamp[1e-5,1e4]; STE round(W/scale), affine clamp[0,15], dequant=(q-zero)*scale; zero.round has native zero derivative',
            'quantizer_source_unchanged': True})
        write_json(HERE / f'BACKWARD_NODE_TRACE_{self.suffix}.json', {'node_count': len(self.nodes), 'events': self.node_events})
        write_json(HERE / f'BACKWARD_TENSOR_TRACE_{self.suffix}.json', {'events': self.tensor_events})
        write_json(HERE / f'FIRST_NONFINITE_BACKWARD_{self.suffix}.json', {
            'mode': self.suffix, 'node_count': len(self.nodes), 'backward_events': len(self.node_events),
            'first_finite_to_nonfinite_node': self.first_transition,
            'first_nonfinite_key_tensor': bad_tensors[0] if bad_tensors else None,
            'forward_all_observed_finite': not forward_bad,
            'all_observed_backward_gradients_finite': all(g.get('all_finite',True) for e in self.node_events for g in e['grad_inputs']+e['grad_outputs']),
            'graph_leaf_dtypes':sorted({r['dtype'] for r in self.leaf_metadata}),
            'graph_leaf_metadata':self.leaf_metadata,
            'loss': result['loss'], 'scale_factor': result['amp']['scale_factor'],
            'interpretation_scope': 'first observed native autograd-node gradient boundary; an opaque GEMM/SDPA kernel internal arithmetic is not separately instrumented'})
