# v57 统一运行时四模型正式质量对照 — 预检报告

## 状态

**正式评测未启动，0/8 项完成。** 本轮仅完成 CPU 侧来源、哈希、配置和历史窗口算术核查。没有完整模型的 CUDA 加载或前向，没有量化、整数码修复、新校准，也没有读取 P3。历史文件保持只读。

## 预算判断

v56 的阶段表中 WT2 同步评测计时为 2146.696 秒，C4 为 132.287 秒，阶段计时合计 2278.982 秒。v56 成功评测 continuation 的进程 wall 为 2283.576 秒。阶段计时和进程 wall 是不同口径，分别记录。

按每个模型一对数据的实测 eval-only wall 外推，四模型约需 9134.3 秒，约为 v57 1800 秒上限的 5.07 倍，且还未计入 v57 的加载和来源检查。仅 WT2 阶段本身就需要 2146.7 秒，已经超出整轮上限。因此没有启动任何一个会留下半套窗口的 GPU 评测。已计时 CPU 检查下界为 50.162 秒，其中最终可复现预检脚本为 19.231 秒；另有少量只读路径定位未累计计时，明细见 CPU_DIAGNOSTIC_LEDGER.csv。v57 GPU、完整模型加载和正式评测均为 0 秒。

## 环境和 checkpoint 来源

当前 Python 环境包版本实测为 Python 3.12.10、PyTorch 2.6.0+cu124、CUDA runtime 12.4、Transformers 5.16.1、tokenizers 0.23.2。冻结协议指定 Qwen2.5-0.5B-Instruct revision 7ae557604adf67be50417f59c2c2f167def9a775，FP16、CUDA、SDPA、TF32 off、FP32 matmul highest、eval 模式、use_cache=false、max_length=2048、stride=512、不添加 special tokens、不套 chat template，并以 valid_tokens 加权。记录证明请求了 SDPA，但没有记录 PyTorch 最终采用的底层 attention kernel。

四个 safetensors 文件的 SHA 与各自历史来源记录均匹配；按当前 Transformers AutoConfig loader 归一化后，四个有效 config 指纹一致。第一次比较原始 config JSON 时发现序列化别名差异：旧配置使用 rope_theta/torch_dtype，新配置使用 rope_parameters/dtype，并带有 layer_types；按当前 AutoConfig 统一解析后，这些配置映射到相同有效指纹。这是已记录并修正的 CPU 预检判断，不涉及权重修改。v57 在 CPU 上计算了四个 checkpoint 的全部具名 tensor SHA，并验证 v55/v56 已有 named-hash 清单与文件张量吻合；M1/M2 没有 v18 named-tensor 清单，本轮补算并保存在 v57。完整 Hugging Face model loader 没有在本轮对四个模型实际执行，所以本轮不能声称四个 checkpoint 均经过统一运行时加载通过。历史 M1/M2 测量使用 PyTorch 2.6.0+cu124、Transformers 4.49.0；M1 修复历史评测使用 PyTorch 2.6.0+cu124、Transformers 5.16.1；M2 修复历史评测使用 PyTorch 2.5.1+cu121，Transformers 版本未记录。静态配置与 checkpoint 格式相符，但 R2 在当前 runtime 的完整加载仍未实测。

## 输入和历史窗口

两个冻结 manifest 文件摘要核对通过，tokenizer revision 与模型 revision 相同。WT2 token-ID hash 为 8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3，C4 为 bc9a7575527abbc41352352a0439c6bee062ea265b7bd03120704e83f4af7053。八组历史 windows CSV 已用窗口 NLL 总和除以 valid token 总数独立复算，并用 exp(weighted NLL) 复算 PPL。结果及来源哈希在 HISTORICAL_WINDOW_RECALC.csv。此项是历史文件算术审计，不是 v57 新评测。

由于没有 v57 窗口，统一四臂差异、交互量和本轮相对历史的环境/执行差异均不可计算。待算交互量定义为 (NLL(M2+repair)-NLL(M2)) - (NLL(M1+repair)-NLL(M1))，单位是 nats/valid token；负值表示固定本轮测量下 M2 臂修复的 NLL 降幅较大。当前不填入跨环境历史估计。

## 交付文件

ENVIRONMENT_LOCK.json、CHECKPOINT_SOURCE_AUDIT.csv、NAMED_TENSOR_HASHES.csv、INPUT_SOURCE_AUDIT.csv、HISTORICAL_WINDOW_RECALC.csv、V57_EIGHT_EVAL_STATUS.csv、UNIFIED_COMPARISON_PENDING.csv、HISTORICAL_VS_V57_PENDING.csv、COST_PRECHECK.json、CPU_DIAGNOSTIC_LEDGER.csv、RUN_STATUS.json、preflight_v57.py 与 REPRODUCE.md 均保存在本独立目录。

## 结论

本轮没有统一 runtime 的 PPL 数据，不能回答 scalar10 修复前后优势或交互量。按 v56 的现有同运行时实测，原定八项评测需约 2.54 小时的 eval-only wall，之后还要留出加载和核验时间；1800 秒预算不足。历史传播有效性 UNKNOWN 继续保留，统一重评也不能替代校准输入传播审计。
