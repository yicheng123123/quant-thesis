# L0 activation capture semantics

Each input is captured at the target Linear forward pre-hook, from `args[0]`. The two frozen v16 calibration samples use their first 64 token IDs. Shapes below concatenate the two actual invocations.

| Target | Actual input source | Preceding computation | Sibling relation | Shape |
|---|---|---|---|---:|
| `self_attn.k_proj` | decoder layer input_layernorm output | RMSNorm(hidden_states); k projection then rotary embedding | shares pre-projection tensor with q_proj and v_proj | `[128, 896]` |
| `self_attn.v_proj` | decoder layer input_layernorm output | RMSNorm(hidden_states); value projection | shares pre-projection tensor with q_proj and k_proj | `[128, 896]` |
| `self_attn.q_proj` | decoder layer input_layernorm output | RMSNorm(hidden_states); q projection then rotary embedding | shares pre-projection tensor with k_proj and v_proj | `[128, 896]` |
| `self_attn.o_proj` | self-attention context after Q/K/V, RoPE, causal attention and head reshape | attention weighted values are merged across heads | not a Q/K/V sibling | `[128, 896]` |
| `mlp.up_proj` | post-attention residual after post_attention_layernorm | downstream of self_attn.o_proj and residual addition; MLP pre_feedforward norm | shares normalized MLP input with gate_proj | `[128, 896]` |
| `mlp.gate_proj` | post-attention residual after post_attention_layernorm | same MLP input; gate branch before SiLU and multiplication | shares normalized MLP input with up_proj | `[128, 896]` |
| `mlp.down_proj` | Silu(gate_proj(x)) * up_proj(x) | MLP down projection after gated activation product | no sibling projection input | `[128, 4864]` |

q/k/v receive bitwise identical normalized hidden-state inputs in this forward. up/gate likewise share their normalized MLP input. GPTQ grouping/order is not a forward dependency; see `SEQUENTIAL_QUANT_ORDER.json` and `FORWARD_DEPENDENCY.json`.
