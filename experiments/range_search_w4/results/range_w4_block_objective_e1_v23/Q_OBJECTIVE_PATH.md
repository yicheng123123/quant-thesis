# q_proj：block目标路径

输入是该文档经过 frozen v18 M2 quantized prefix 后的 block-0/12/23 hidden state。q/k/v projection共享block入口，但当前仅替换预注册的一个 q_proj output row；候选张量由该行所在q_proj的full Linear dynamic GPTQ产生。

候选进入真实 Qwen2 decoder-layer forward：q_proj → reshape/rotary → attention logits与softmax → attention value聚合 → o_proj → attention residual/RMSNorm → gate/up与SwiGLU → down_proj → MLP residual。最终比较完整decoder block输出与同一quantized-prefix输入下 pristine FP16 block输出的MSE。整个路径保留其他q/k/v/o/MLP为frozen v18 M2 weights；只变目标row alpha。Local selector仍计算同一candidate row对原始FP target Linear的直接X error。

因此block metric并非q_proj输出MSE重命名；它穿过attention、projection与block余下路径。FP teacher input由manifest hash固定、对候选不变。held block losses仅评价被冻结的search alpha及完整landscape供regret/ranking诊断。
