# SDPA sliding-window 提醒

`v18/logs/formal_m1_m2.log` 和 `v18/logs/eval_v18.log` 均出现 Transformers 的 “Sliding Window Attention is enabled but not implemented for sdpa” 提醒。v19 检查当前 `transformers 4.49.0` Qwen2 代码：`Qwen2DecoderLayer.__init__` 仅以 `config.sliding_window` 非空触发该 warning；实际 attention 代码要求 `use_sliding_window`，而 mask 构造即使检查 `sliding_window`，也只在距离超过 32768 时添加屏蔽。本模型三个 config 的 `use_sliding_window=false`、`sliding_window=32768`，评测窗口最多 2048 tokens、`use_cache=False`。

M0/M1/M2 的正式评测入口都以相同 `AutoModelForCausalLM.from_pretrained(..., torch_dtype=float16, attn_implementation="sdpa")`、TF32 关闭、相同 token stream/window 参数加载各自 checkpoint；checkpoint 重载后模型路径被用于 PPL，而非内存中校准对象。故没有看到三者因该 warning 使用不同 forward 实现的证据。当前上下文短于 32768，warning 本身不表明窗口截断实际发生。

这一点主要是**绝对 PPL 的环境/实现 caveat**；对 M1 vs M2 相对差异形成系统性偏差的可能性较低，但不能从同一路径自动证明数值交互完全为零。若将来换更长上下文或不同 Transformers/attention backend，应重新核验协议，不能沿用本次 PPL 对比。
