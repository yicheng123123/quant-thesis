# Decoder/cache contract

For the actual frozen use_cache=False/output_attentions=False calls: support Tensor[B,T,H] and legacy plain one-element tuple containing Tensor[B,T,H]. Unknown result/container types, extra tuple items, rank/shape mismatch, nonfloating/nonfinite hidden states fail explicitly. No content changes or dtype conversion in extraction; returned Tensor identity is retained.

Cache writes use a B=1 document slice [1,T,H]. Source and target rank, every dimension, and dtype must exactly agree before copy; [H] and [T,H] cannot enter this path. Embedding, qkv, sibling-hook and block-refresh cache writes use the same strict write policy. Reference native forward does not use this extraction/write implementation.

Only embed/capture/advance_block change in isolated runtime_v53.py. Remaining function/class ASTs, C builder, candidate scoring, p updates, integer grid, tie order, budgets, acceptance and evaluator are unchanged. v52 initialization review and fast repair sources are read-only, not exercised as a new repair.

Future integration must explicitly load runtime_v53.py as the shared runtime and retain v52 initialization_review and fast_accept_v52 unchanged. Do not run historical entry points which import the old runtime. This audit does not authorize a new full repair or PPL.
