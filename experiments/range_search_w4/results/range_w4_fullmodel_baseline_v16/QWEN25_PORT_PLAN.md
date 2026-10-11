# Qwen2.5 最小 architecture port 设计

目标 checkpoint/revision见PROTOCOL；metadata模型名Qwen2ForCausalLM/model_type=qwen2。官方 `omniquant()` 分支只有 llama/opt/falcon/mixtral，未知家族直接ValueError，因此**没有原生Qwen2.5支持**。AutoModel能加载Qwen不代表quantization/calibration/evaluate支持。直接把net改名llama不合法：旧QuantLlamaAttention读取 attention内rotary_emb，并按旧seq_len/tuple-cache API调用，而本机Qwen2使用model级生成的 position_embeddings 和Cache.update。

## 固定映射

24个 `model.model.layers`，每层是Qwen2DecoderLayer；embedding=`model.model.embed_tokens`，final norm=`model.model.norm`，lm_head与embedding tied，保持FP16。每层q/o=(896,896)，k/v=(128,896)，gate/up=(4864,896)，down=(896,4864)，全部从safetensors header核验。q/k/v有bias，其他四个无bias；所有bias保留。RMSNorm eps1e-6、SiLU门控、残差路径保留。

14 query heads、2 KV heads、head_dim64、GQA repeat factor7；RoPE theta1e6，max positions32768，use_sliding_window=false。v4.49 Qwen2Attention forward接收position_embeddings=(cos,sin)、attention_mask、past_key_value(Cache)、cache_position，decoder返回tuple；model forward负责生成RoPE和mask。完整kwargs必须捕获/转移/重放，不能只有position_ids。训练/评测均use_cache=False；cache compatibility只做微型generation等价检查，不增加KV量化。

## 拟修改文件 / 类：全部先在新port副本中实施

1. 新增 `models/int_qwen2_layer.py::make_quant_qwen2_decoder(config, ori_layer, args)`：deepcopy原生Qwen2DecoderLayer，严格只替换上述七个nn.Linear为**未修改的官方QuantLinear**，返回仍具有原生forward的block。使用与官方Mixtral weight-only分支相同的替换思想；不得照搬QuantLlamaAttention。保留Qwen2Attention、Qwen2MLP、Qwen2RMSNorm、RoPE、GQA、bias、Cache API和SDPA分派；不新写attention计算。工厂签名兼容官方DecoderLayer(config,layer,args)。LET=false，故无需LET plumbing和OmniNorm。
2. `quantize/omniquant.py::omniquant`：新增基于config.model_type=='qwen2'的discovery/move-to-device分支，prefix=model.layers；DecoderLayer绑定上述工厂。扩展Catcher缓存每个sample的attention_mask/position_ids/cache_position/position_embeddings和必要kwargs，teacher/student重放保持它们。只替换旧接口的调用参数，不改teacher/student来源、MSELoss、AMP/AdamW、LWC、循环或inplace quant。Qwen不应借用旧is_llama旋转代码；LET=false时flag不影响smooth逻辑。
3. `main.py` 或新 `run_omni_qwen25.py`：准确model_type分派、固定local checkpoint/revision、读取共同calibration manifest，不调用随机采样或旧缓存key；只负责配置/数据适配。不要将旧net_choices伪装成llama。
4. `datautils.py` 的新manifest loader边界：直接构造128个 [1,2048] long tokens 和官方tar(-100除末token)接口；不修改优化目标。缓存key包含manifest/tokenizer/commit/config hash。
5. `models/LMClass.py`：固定local checkpoint/tokenizer，FP16，指定SDPA；保留nativeAutoModelForCausalLM，不改模型层。或者极小duck-typed LM容器绕过无关lm_eval dependencies，记录适配差异。
6. 新 `export_qwen25_dense.py` / evaluator bridge：官方inplace quant完成后，将QuantLinear的已量化FP16weight/bias转回原生nn.Linear，移除只为训练存在的quantizer参数，导出原生架构dense checkpoint；另存Omni factors、参数、scales/zeros与hash。shared PPL evaluator接收同一nativeQwen模型对象/固定IDs。真实packed导出单独审核，不加入主quality协议。

## 不变性与停止点

上述是**拟patch，不是已实现/已验证适配**。不修改 `quantize/quantizer.py`、`quantize/int_linear.py`、LWC参数数目/init/AdamW/AMP/MSE/epochs。每block激活233k左右LWC scalar参数（精确232,960），全模型5,591,040；原权重/norm/bias梯度必须None，只有bound_factor更新。官方临时量化和inplace量化保留。若必须改objective/quantizer/optimizer/参数化，则中止M3命名，先记录新方法定义；不悄悄patch成完整baseline。

主M3是官方weight-only配置，LET关闭。将来如需完整LET扩展，Qwen GQA下q/k宽度、v/o缩放和biased projection的等价变换需要另行证明，不在本轮port边界内。native无量化forward与port关量化forward先exact/allclose；GQA/RoPE/cache检查不过就停止，不跑formal calibration。
