# up_proj：block目标路径

输入是同一full v18 M2 quantized prefix 下的目标block hidden state。up_proj候选只替换一个预注册output row，其他up rows先固定为search data上选出的local-M2 stitch，gate_proj和down_proj保持frozen v18 M2量化权重。

每候选走原生Qwen2 decoder-layer：post-attention residual/RMSNorm → gate_proj与up_proj → SiLU(gate)×up → down_proj → MLP residual → block output。比较量是整块输出对固定FP16 teacher的MSE，teacher基于同一个M2-prefix入口；并非up Linear输出MSE。目标candidate只变该row，故与local selector共享相同GPTQ权重和量化元数据。

held只评分。不会根据held block输出重选alpha，也不会把held最优row写回/stitch。
