"""Generate v16 audit/design deliverables from frozen evidence; never run models."""
from pathlib import Path
import json, csv, math, statistics, hashlib, subprocess
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
SHA='feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14'
BASE=f'https://github.com/OpenGVLab/OmniQuant/blob/{SHA}'
def j(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def save(name,text):
    p=OUT/name
    if p.exists():raise RuntimeError('refuse overwrite: '+name)
    p.write_text(text.strip()+'\n',encoding='utf-8',newline='\n')
def jsave(name,obj):save(name,json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def main():
    inv=j('MODEL_INVENTORY.json');cal=j('CALIBRATION_MANIFEST.json');e1=j('WIKITEXT_TEST_MANIFEST.json');e2=j('C4_EVAL_MANIFEST.json')
    smoke=j('BUDGET_SMOKE.json');tim=j('HISTORICAL_TIMING_EVIDENCE.json')
    bywidth={r['input_dim']:r for r in smoke['cases'] if r['rows']==32}
    core_seconds=24*10*(368*bywidth[896]['seconds']+28*bywidth[4864]['seconds'])
    # Width-equivalent proxies for k/v/gate were not measured historically; label them as proxies.
    histmean={k:statistics.mean(r['seconds_full10_3rows'] for r in v) for k,v in tim['v15_unique_timings'].items()}
    b3seconds=24*(math.ceil(896/3)*(histmean['q_proj']+histmean['o_proj']+histmean['down_proj'])+
        2*math.ceil(128/3)*histmean['q_proj']+2*math.ceil(4864/3)*histmean['up_proj'])
    from safetensors import safe_open
    snapshot=ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
    with safe_open(str(snapshot/'model.safetensors'),framework='pt',device='cpu') as sf:
        checkpoint_elements=sum(math.prod(sf.get_slice(k).get_shape()) for k in sf.keys())
        module_shapes={m['module']:sf.get_slice(m['module']+'.weight').get_shape() for m in csv.DictReader((OUT/'MODULE_INVENTORY.csv').open(encoding='utf-8'))}
    assert len(module_shapes)==168
    packed_bytes=inv['quantized_weight_elements']//2+inv['scale_zero_pairs']*2+inv['scale_zero_pairs']//2+245760*4+(checkpoint_elements-inv['quantized_weight_elements'])*2
    evidence=dict(official_commit=SHA,checkpoint_unique_elements=checkpoint_elements,checkpoint_fp16_mib=checkpoint_elements*2/2**20,
        shape_count_verified_from_safetensors=len(module_shapes),module_count=168,output_rows=304128,row_alpha_candidates=3041280,
        b32_batches_per_layer_small_width=368,b32_batches_per_layer_large_width=28,b32_scalar_alpha_calls=95040,
        b32_core_projection_seconds=core_seconds,b32_core_projection_hours=core_seconds/3600,
        v15_b3_projection_hours=b3seconds/3600,historical_mean_full10_3row_seconds=histmean,
        important='B32 is single alpha cold micro timing with historical H, excludes full calibration search loss, capture/H/load/save; projection not measured runtime',
        packed_layout_payload_estimate_mib=packed_bytes/2**20,lwc_total_parameters=2*inv['scale_zero_pairs'],lwc_parameters_active_per_block=2*inv['scale_zero_pairs']//24,
        omni_steps_per_block=2560,omni_total_optimizer_steps=61440,
        native_prefix_peak_from_v15_invalid_for_quantization=True)
    jsave('COMPUTE_BUDGET.json',evidence)
    protocol=dict(name='v16 Full-Model & OmniQuant Baseline Protocol Audit',status='FROZEN_DESIGN_NOT_EXECUTION',
        model='Qwen/Qwen2.5-0.5B-Instruct',revision='7ae557604adf67be50417f59c2c2f167def9a775',dtype='FP16',device='CUDA RTX4060 Laptop 8GiB',
        attention='sdpa for all M0-M3 using native Qwen2 forward; use_cache=False; TF32=False; eval mode',
        quantized_scope='all 168 decoder linears; embeddings/tied lm_head/norms/biases remain FP16',
        native=dict(bits=4,activation_bits=16,sym=True,group_size=128,block_size=128,damp_percent=.1,desc_act=False,static_groups=False,true_sequential=True,mse=0.,math='FP32',damp_auto_increment=0),
        search=dict(unit='per output row, one shared alpha over all G128 groups',grid=[k/10 for k in range(1,11)],
            prefix='self-method selected quantized weights, true sequential native group order',
            candidates='complete independent dynamic GPTQ from pristine current target W and common initial Hinv',
            loss='mean square of X_FP32 @ (Q_FP16.float()-W_FP16.float()).T, calibration only',
            loss_primary='direct projection in token chunks; Gram equivalent allowed only after real-X checks and ranking check',
            tie='lowest measured loss; exact ties prefer largest alpha',row_batch=32,alpha_batch=1,no_gate=True),
        omni=dict(name='OmniQuant-Qwen2.5 port, official W4A16 weight-only configuration',commit=SHA,
            bits=4,abits=16,group_size=128,symmetric=False,disable_zero_point=False,lwc=True,let=False,
            epochs=20,batch_size=1,lwc_lr=.01,let_lr=.005,weight_decay=0.,aug_loss=False,
            bound_init=4.,optimizer='AdamW + official AMP GradScaler, original fixed order/no shuffle/no LR schedule',
            objective='official MSELoss(block output of original FP prefix, quant block output of quant prefix)',
            original_weights_frozen=True,port_implemented=False,formal_training_executed=False),
        common_calibration=dict(manifest='CALIBRATION_MANIFEST.json',nsamples=128,seqlen=2048,token_budget=262144,seed=20261005),
        evaluation=dict(E1='WIKITEXT_TEST_MANIFEST.json',E2='C4_EVAL_MANIFEST.json',max_length=2048,stride=512,evaluator='benchmark.eval.ppl.evaluate_ppl',selection_or_tuning=False),
        primary_representation='FP16 dense dequantized weights for all methods, mathematical quality comparison, no compression/latency claim',
        packing='separate secondary GPTQ packed export with exact integer-code and layout verification; not assumed equal to dense FP16 candidate',
        formal_launch='only after next implementation/one-block smoke gates and empirical cost review; this audit stops',
        output_restriction='all v16 files in this directory; v8-v15 immutable')
    jsave('PROTOCOL.json',protocol)

    save('OMNIQUANT_AUDIT.md',f'''
# 官方 OmniQuant 源码审计

审计对象：[OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant)，固定 commit `{SHA}`。`official_source/` 是从官方 origin 取得的浅克隆；`OFFICIAL_SOURCE_SHA256.json` 固定读取源码的字节。没有安装依赖、修改算法或运行官方训练。Windows 大小写文件系统使 `imgs/OmniQuant.png` / `imgs/omniquant.png` 碰撞，工作树显示图片差异；Python、shell、toml 和许可证源码必须与 Git blob 一致，图片不参与实验。

## 来源、版本和许可证

`LICENCE` 是 MIT，copyright 2024 Mengzhao Chen；`pyproject.toml` 的 Apache classifier 与它不一致，按实际 MIT 许可证保留版权和许可文本，明确记录该元数据冲突。[固定许可证]({BASE}/LICENCE)。项目声明版本 0.1.0，Python>=3.8，README 安装示例 Python3.10。依赖没有官方 lock：torch>=2.0、transformers>=4.31、datasets>=2.0、tokenizers>=0.12.1、accelerate 未限定、peft>=0.2、sacrebleu==1.5.0 等，完整声明保存在 `official_source/pyproject.toml`。`utils.py` 还直接 import termcolor，不能只凭安装声明认为 import 一定成功。

本机 GPTQ 环境实际版本见 `ENVIRONMENT_AUDIT.json`：torch2.6.0+cu124、transformers4.49.0、tokenizers0.21.4、datasets5.0.1、accelerate1.14.0、GPTQModel2.2.0。不修改这个环境。后续使用单独 OmniQuant 环境，固定相同 torch/transformers/tokenizers 作为架构 API 和评测基准；先 import smoke，再生成完整 pip lock。完整传递依赖尚未安装或验证，不宣称官方支持该组合。packed real quant 的官方 AutoGPTQ-bugfix/Triton 依赖不能视为本机 Windows 已可用。

## CLI 的真实含义

官方 `scripts/llama/llama-7b/w4a16g128.sh` 是 `--wbits 4 --abits 16 --group_size 128 --lwc --epochs 20`，没有 `--let`，没有 `--symmetric`。因此主 M3 冻结官方 weight-only 配置：**LWC开启、LET关闭、asymmetric affine G128、20 epochs**。这本身是官方方法的 weight-only 设置，不是删去官方必需机制。[固定脚本]({BASE}/scripts/llama/llama-7b/w4a16g128.sh)。LET 是另一条可选路径，官方 W4A4 示例启用；本轮不加 LET/激活量化。

`main.py` CLI epochs 默认10而示例20，nsamples默认128，batch1，seed2，lwc_lr=.01，let_lr=.005，wd0，aug_loss默认false，symmetric默认false，disable_zero_point默认false。CLI `--alpha=.5` 是 LET 初始化指数，与 pure-search clipping alpha 无关。`lm.seqlen` 在 main 中强制2048。v16统一 seed20261005 和冻结数据清单是明确的数据协议变动，不能调用官方随机 sampler 悄悄换池。[CLI]({BASE}/main.py)。

## 量化与优化

`UniformAffineQuantizer` 使用 round STE、clamp 和 dequant；G128 reshape 到每行每组。LWC 每个 row-group 两个独立 bound factor，初始化4，经 sigmoid 缩放当前 weight min/max；默认 asym scale=(xmax-xmin)/15，clamp[1e-5,1e4]，zero=round(-xmin/scale)，zero另有[-1e4,1e4] clamp。官方不先将 extrema 扩到包含零。symmetric 可选时 scale=max(abs(bounds))/7、zero7，仍是[0,15]整数；**这不同于当前 GPTQ 的典型 scale=2max/15、zero8**。abits16 forward identity，不做低位激活或 KV 量化。没有 GPTQ Hessian compensation。[量化器]({BASE}/quantize/quantizer.py)。

`omniquant()` 先缓存首层输入，独立维护 FP 前缀和 quant 前缀。每层 FP teacher 输出来自原始 FP prefix；student 输入来自此前量化层。官方 `MSELoss` 比较整个 decoder block 输出，aug_loss=false 时只一项。原模型参数冻结；当前 qlayer.float()，FP32 LWC参数，经 AMP forward 和原 NativeScaler backward/AdamW 更新，学习率无 schedule，样本固定顺序，不按 loss early-stop。LET关闭时临时量化原权重，不改 attention/RMSNorm 参数。最终 qlayer.half() 后 official inplace fake quant，传播该层 quant outputs 到下一层，保存 LWC factors。[完整循环]({BASE}/quantize/omniquant.py)。

## fake、packed 和 PPL

默认是 dense fake-dequant FP16 权重，不是物理 int4 checkpoint；QuantLinear F.linear 执行。`--real_quant` 只允许2/3/4位且abits>=16，4位走 qlinear_triton，3位走qlinear_cuda，再 pack scales/zeros。官方保存 fake model 的 `save_pretrained` 路径并不能自动为未知 Qwen wrapper提供可靠 reload；必须采用原生Qwen dense导出，保留另存 factors/参数/校验和。不能把 fake model 的磁盘/VRAM当成4位压缩数据。

官方 `evaluate()` 只识别既有模型家族；WikiText用整流非重叠2048块、尾部丢弃、每块loss乘seqlen；C4在datautils抽256个验证窗口。`--limit` 提前break但分母仍用原nsamples，是额外陷阱。它与本仓库 token-weighted sliding evaluator 不同；M0-M3统一使用本仓库 frozen IDs + `evaluate_ppl`，**不混用官方报表**。官方 dataloader/test cache key缺 revision/seed/seqlen/tokenizer hash；新adapter必须使用 manifest hash 的缓存key。[数据采样]({BASE}/datautils.py)。

## 命名与公平性

M3：`OmniQuant-Qwen2.5 port (official W4A16 configuration; LWC on, LET off)`，仅当保留上述优化、目标、量化器、epochs及参数化才可使用此名。M4若将LWC改为逐Linear/row输出loss、换成GPTQ量化器、加GPTQ compensation或改训练参数化，命名 `LWC-controlled mechanism baseline`，不能冒充完整OmniQuant。M4本轮不执行、不进入必需矩阵。M3和M2共同数据/位宽/模型可做方法级PPL+校准成本比较，不能把差异仅归因于 enumeration versus gradients。
''')

    save('QWEN25_PORT_PLAN.md',f'''
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
''')

    save('FULLMODEL_SEARCH_PLAN.md','''
# Full-model Pure Search integration

固定 per-output-row shared-alpha，grid=.1,.2,...,1.0，无Gate、梯度、learnable parameter、proxy筛选或重要行子集。沿用v8 dynamic_core.py；新增full-model runner与processor写在新执行目录，不改历史core/run。Native G128/block128/damp.1/desc_act=false/static_groups=false/mse0；每个group入口基于此前compensation后的global W重新find_params，再scale乘当前row共享alpha，zero8保留native特殊fallback/all-positive语义。不是原始W预计算固定range。

## 原生顺序与自己的量化前缀

GPTQModel2.2.0实际调度是 `looper/module_looper.py::ModuleLooper.loop` 与 `GPTQProcessor`；BaseGPTQModel中旧循环不是新增实现的依据。每层groups严格 `[k,v,q] -> [o] -> [up,gate] -> [down]`。同组共享当前输入，先capture该组全部成员，再为每个成员独立build/复用该输入H和搜索；完成全部成员后才更新组状态。q/k/v和up/gate不能在组内错误地相互改变capture输入。

layer0输入从原模型embedding来。组0 capture时该层七Linear仍pristine；组1 capture时k/v/q已经安装M2自己选的Q；组2前o也已选定；组3前up/gate也已选定。最后把当前block的**M2最终量化输出**传给下一层。M1另一次运行传播自己的native alpha1 prefix。不要借用v15的native-D prefix、FP输入或另一个方法的H。当前target原始W是FP16 snapshot的该模块原权重（不受其他模块compensation），每个候选从同一W.clone和同一Hinv开始。

## 行可分与状态

当前配置下Hinv与row无关，列误差补偿 `E @ Hinv` 按输出行分别作用，per-channel find_params按row各自reduce，无跨row项。因此一行共享alpha可在共同H上独立选择，再拼装每行最终Q、scale、zero，等价于对相应行分别完整GPTQ；不联合搜索各组。B=32按row分块、alpha顺序执行；每候选完整遍历所有输入列，禁止只重跑一个group。alpha1在本模块**同一M2 prefix和H**下是搜索集合中的本地native点，不能直接复用另一个M1 prefix计算的alpha1。

候选结果最终Q_FP32 cast FP16，搜索loss使用实际Q_FP16.float-W_FP16.float；bias未变所以抵消。目标=mean square(Xdelta)，calibration唯一selection来源；exact loss tie选更大的alpha。每模块报告本地alpha1/search-selected loss和alpha分布，但不能由本地不劣声称最终PPL不劣。全模型PPL只在全部方法冻结后统一计算。

`native_hinv_from_documents`目前遇到dead columns会raise；full-model runner必须补齐原生dead处理：记录原始W_ref供输出目标；diag(H)==0的列对工作W置0、H diagonal置1，再原生damp/inverse，所有alpha共用相同dead掩码。不能把dead后的W作为改变了的teacher，不能增大damp隐藏失败。原生qcfg自动增加damp关，OOM inverse近似fallback也不许静默启用；不够显存就按原精度CPU/offload或停止。

## loss计算 / H缓存

默认直接token-chunk FP32 Xdelta，每篇样本2048tokens完整计数，FP32平方求和后host FP64累计，TF32关闭。可预计算独立 undamped G=X^TX/N，以delta^T G delta为等价loss；这是同一目标的计算变换，不是proxy。必须在真实cal X而非随机X上检查direct等价、全十候选排名/选择，近tie直接重算；不能使用damped H当loss。native H按sample数量缩放：128个等长样本时H_raw=2/128 ΣX_doc^TX_doc，理论G=H_raw/(2*2048)，但native累积缩放与直接Gram浮点并非bitwise相同；初版默认direct，Gram优化仅在检查通过后启用并记录。

H只可在同一method/prefix/input hash的同模块候选间复用；同组实际输入相同的q/k/v、up/gate可共享。不能跨模型方法、不同sequential组或不同prefix复用。v15提供H只用于本轮预算smoke，绝不成为正式full-model H。

## scale/zero与标准packing

每模块保存selected Q [out,in] FP16、每row alpha、scales/zeros [out,in/128] FP32原值、native g_idx [in] int32=floor(col/128)、pristine/input/H/config hashes。选择不同row-alpha不会增加packing自由度：GPTQ格式本就允许每row-group scale/zero。nativePackableQuantLinear.pack接受[out,groups]后转置为[groups,out]，qweight int32位打包shape[in/8,out]、qzeros[groups,out/8]、scales[groups,out]FP16；全部实际尺寸可整除8/128。embedding/head/norm/bias照旧。使用本机GPTQModel codec及正确v1/v2 zero格式metadata，不自行套用历史zero-minus-one公式。

**布局可表示，不代表数值已完成packed parity。** 官方pack先用FP32scale从dense Q恢复整数，但存FP16scale；dequant与逐元素FP16 Q可能存在差异。主quality协议因此M0/M1/M2/M3统一dense FP16 dequant权重：M1通过原生GPTQ量化结果、在pack前保留Q，M2保存selected Q，M3官方inplace fake Q。这是真实量化权重的self-prefix计算，磁盘仍FP16，不能宣称int4运行/压缩。packed导出是单独secondary artifact，要求codes正确、无range overflow、reload/checksum/scales布局和实际forward误差审核。不得先用dense candidates选alpha再把packed PPL当成同一误差目标的无差异结果。

如果后续主问题必须使用packed运行作为primary，需要预先注册“各candidate pack-roundtrip后计算loss并用该实际prefix传播”的representation-only协议；它仍使用原十alpha GPTQ，但与v15 Q_FP16候选不能按数值严格等同。本轮不切换也不暗中repack覆盖候选。

## 中断恢复与最小检查

按row batch保存十个candidate loss与best tensor/params，成功后原子commit模块，再commit完整native组。恢复key绑定model/revision/cal IDs/order/pristine W/current selected-prefix hash/Hinv hash/dtype/core hash。未完成组重新加载前一组committed state，不保留部分安装状态；不从候选mutable work继续。不必把全部十候选tensor永久存盘，选定权重及原始loss/scale信息保留，spot候选另存。每次记录logical row-alpha counts、actual calls、cache reuse、sync time、cache/H构建和I/O、phase peak allocated/reserved、失败/重试；这都是离线校准成本。

先测q/k/v/o/up/gate/down的alpha1 native parity、外部alpha使用、候选状态隔离、dead/allzero输入、direct/Gram、selected-scale ownership、dense reload、packed codec。小型self-prefix测试固定L0的完整四组，只跑前32行用于检查、其他行保持原权重；它只是smoke，不能称完整模型方法。数学行可分不替代GPU批量数值检查；本轮B3/B32两宽度alpha.9首三行FP16 exact只是有限spot证据。
''')

    save('COMPUTE_BUDGET.md',f'''
# 计算和存储预算：实测与外推分开

精确计数来自checkpoint config和168个实际safetensors weight shapes，详见MODEL_INVENTORY.json/MODULE_INVENTORY.csv。24blocks ×7=168 Linears；每block12,672 output rows；共304,128 rows，**3,041,280个行-alpha候选**。量化范围357,826,560 weights；2,795,520 row-group scale/zero pairs。lm_head tied embedding排除。不能把row-alpha逻辑次数与GPU调用数混报。

## 已有计时

v14的K4/8/16/32各480候选分别201.07/200.75/199.76/201.19秒；不同输入规模在该小row runner下没有比例增长，不能按token倍率推candidate core。v15三行批量、960 unique candidates、320 module-alpha calls，core+per-article search-loss133.842秒，native full-prefix133.076秒(16×128tokens)，FP capture2.640、Qfullforward3.674、target replay3.643、32H builds2.053秒。unique calls已对重复三行JSON去重；明细及来源hash见HISTORICAL_TIMING_EVIDENCE.json。

v15 `cost_summary.csv` 的native-prefix peak606,409,216bytes是在后续QPREFIX forward reset之后读取，**不能作为native量化峰值**。候选peak约256MiB也只覆盖旧小批量。v14约1.75–2.08GiB峰值属于旧完整capture/候选流程，也不等于新的全模型+Omni训练峰值。

## 本轮极小预算smoke

只用历史L2 q/down H和pristine前32行，各alpha.9，B3/B32，总70 slots，脚本内wall {smoke['wall_seconds']:.3f}秒（含cache读取与检查，不含Python/torch启动），四次同步candidate区间合计3.401秒。B32宽896 core={bywidth[896]['seconds']:.6f}s，宽4864={bywidth[4864]['seconds']:.6f}s，候选分阶段peak最大{max(r['peak_allocated_bytes'] for r in smoke['cases'])/2**20:.1f}MiB。无新calibration搜索/PPL/模型级实验。单冷调用不能声称加速比或代表steady-state。

## M2估算

B32、一次一个alpha：每block宽896 batches=28q+4k+4v+28o+152up+152gate=368，宽4864 down=28；全模型95,040次GPU完整candidate调用。

核心外推=24×10×(368×t896+28×t4864)={core_seconds:.1f}s≈**{core_seconds/3600:.2f}h**，假设其余同宽projection同吞吐，batch32端尾不更慢，使用该GPU/同core。它不包含262,144-token loss、capture、H、序列化，也不是观测到的full-model时间。三行旧runner按projection/行数推约{b3seconds/3600:.1f}h，同宽k/v/gate用q/up替代，是未经该projection测量的估算，不能直接作为正式执行路线。

直接loss乘法总约2×N×10×357,826,560≈1.876×10^15 FLOPs；若用同目标Gram约2×10×Σ_rows d_in²≈1.47×10^13 FLOPs，另计一次Gram构建。二者浮点排序必须验证，不能将理论FLOPs比写成实测加速。预留capture/H约20–60min（v15按token数、H数外推上界，cholesky不随N线性；不精确）；候选loss/导出/metadata/I/O和冷启动另加。**规划M2 12–24h**，不是可信置信区间；formal前在共同新cal数据的固定32行q/down做全部十alpha含真实loss测量，更新预算并验证不越48h执行上限。越界停止，禁止删grid或挑重要row。可校验Gram/增加row batch/候选vectorization，不改变算法；任何工程吞吐收益独立报告。

M1：已有native133秒是在2048 calibration tokens；新协议N为其128倍，capture/H增加而原生全row GPTQ列循环并不同比增加。规划20–60min，待同协议one-block计时核定。M0只加载和评测，无校准。不能把这些规划当作已运行成本。

## M3成本单独算

20 epochs ×128 samples/batch1=2,560 AdamW steps/block，总61,440steps。LWC总5,591,040个scalar参数，每次只训练当前block232,960个；每block梯度/FP32params/两个moments约3.56MiB，原权重冻结，但forward/backward激活和temporary quant weights还需显存。没有可核验的Omni-Qwen block training timing，不虚构wall。若one-step实测 t=.05/.1/.5/1s，则纯optimizer-loop分别约0.85/1.71/8.53/17.07h，**仅为敏感性情景，不是测量**；另加每层teacher生成、quant传播、初始化、export。下一轮one-block smoke先测forward/backward/GradScaler实际step及warmup，给正式预算；若OOM/非finite/超过预设24h上限，不启正式训练。不得为赶时删epochs或改变loss。

## VRAM和磁盘

本机8188MiB可见GPU总显存。checkpoint unique FP16 tensor载荷约{checkpoint_elements*2/2**20:.1f}MiB；模型数据header确认，未读取全tensor。128×2048×896 FP16 block-boundary cache=448MiB，Omni FP/quant两份约896MiB；不开aug_loss。最大H/Gram/Hinv单份4864²FP32=90.25MiB；Qsmall896²=3.06MiB。最大projection输入全CPU FP16约2.375GiB，GPU只流式2048-token块，约38MiB FP32；不把所有X移GPU。B32 serialalpha工作权重不到数MiB，但全模型/teacher/student/attention/AMP workspace另计。

设计M1/M2仅active block在GPU、embedding/norm按需移动，projection capture/H/loss流式；**规划2–4GiB**，不是完整流程实测峰值。M3 block训练+两个boundary cache+SDPA规划3–6GiB；8GiB能否满足必须实测，禁用会分配大[heads,L,L]tensor的旧自定义attention。VRAM阶段内reset，立即读取peak，记录allocated和reserved；不得在下一阶段reset后取前阶段peak。

标准GPTQ payload约{packed_bytes/2**20:.1f}MiB（int4+FP16scale+4bitzero+g_idx+未量化FP16权重，不计headers），是布局预算而非已导出checkpoint。primary dense每方法约{checkpoint_elements*2/2**20:.1f}MiB。十候选全部FP16 tensor约6.67GiB；本方案只永久保存selected tensor和十loss/params，避免全候选盘积。所有168模块X若全存约120GiB，禁止这样cache；每组消费后释放，boundary/prefix只保留当前和恢复所需。

保留96组Hinv和undampedGram约4.7GiB，可按恢复策略逐组删除/仅保留当前；当前down X约2.4GiB，加selected weights、两个dense prefix recovery checkpoints、三方法导出/metadata，规划**18GiB新磁盘空间上限**，cal/evaltoken JSON额外几MiB。所有预计算、H/cache、转移、load/save、optimizer和resume重试计入end-to-end成本，不把离线效率称为在线推理速度。
''')

    save('PPL_EVAL_PROTOCOL.md',f'''
# 冻结 PPL protocol

同一checkpoint revision、local fast tokenizer（tokenizers0.21.4）、无chat template、add_special_tokens=false、无BOS/EOS插入。所有M0-M3原生Qwen结构FP16推理、SDPA、eval、use_cache=false、TF32=false、batch1；primary dense dequant representation统一，量化校准数学另按各方法。

## E1 WikiText-2 test

Salesforce/wikitext，revision b08601e04326c79dfdd32d625aee71d232d685c3，config wikitext-2-raw-v1，完整test原序 `"\\n\\n".join(text)`（保留空行）。使用已落盘 WIKITEXT_TEST_MANIFEST.json 的IDs；输入 **{e1['num_input_tokens']:,}**，预期被评分causal targets **{e1['expected_evaluated_tokens']:,}**。source arrow/token hash已冻结。不是v8-v15 train-heldout；最终test从未用于alpha/hyperparam选择。本轮只tokenize，没有forward/PPL。

## E2 独立 C4 validation，固定token预算

allenai/c4 revision1588ec454efa1a09f29cd18ddd04fe05fc8653a2，en/c4-validation.00000-of-00008.json.gz。固定文档索引32..1055（1024篇，排除此前C4 smoke用过的前32篇），source order，双换行连接，无special tokens，取前 **131,072** IDs，预期评分 **131,071** targets。原始文本/文档hash/URL/timestamp和token清单保存在C4_EVAL_TEXT.txt / C4_EVAL_MANIFEST.json；池长{e2['num_pool_tokens']:,}tokens。只读取gzip前1056条后关闭流，没有下载完整C4数据；CPU准备时间见C4_PREPARATION_COST.json。

标签必须写 **C4 validation fixed token budget PPL**，不是standard/full C4 PPL，更不是官方OmniQuant随机256窗口protocol。选择这份池与任何候选PPL无关；不将校准改成C4来追求收益。WikiText-test和C4报告单独PPL与相对FP16/native delta，不能混算一个PPL。

## 所有方法同一计分器

复用 `benchmark/eval/ppl.py::evaluate_ppl`：max_length2048，stride512。第一window评分2047targets，以后只评分尚未覆盖的targets；重叠labels=-100，loss乘实际 `labels[:,1:]!=-100`数，累加total_nll/total_tokens再exp。最后不足窗口保留，首token无上下文故不计；每套验证 total_tokens=N−1。不得均匀平均window loss，不丢弃tail，不混官方非重叠计分表。

`prepare_ppl_input_ids`/dataset loader目前不显式传revision；本轮不改共享代码，未来wrapper直接load冻结IDs，检查tokenizer/data/source hashes，再调用共同evaluate_ppl。不重新在线加载浮动main。C4旧adapter只能从首文档前缀取数据且不能传revision，因此E2也用冻结IDs绕过这两个数据边界，不改PPL数学。

先tiny128-token smoke检查loss finite、计数127和reload；正式输出每window nll/valid_count、aggregate、model/config/token hashes、elapsed/peak，用独立工具核查。同一eval一旦完成，不根据PPL调alpha grid、calibration池、epochs、格式或方法定义。若发现实现错误，声明无效run，修复后全部受影响方法按同协议重跑；保留原始文件。

## 泄漏界限

cal是128篇train全文独立样本前2048tokens，全部排除v8-v15使用过的280篇文章；v8-v15的heldout从不作为E1/E2。没有token拆分作独立留出。跨数据集不能保证任何语义重复完全消失；完整文章的字符串hash与sample-token hash需检查并记录（VERIFY_AUDIT）。该检查也不能等同于训练预训练语料去重。模型预训练是否见过WT/C4不在本轮可辨识范围；所有方法相同checkpoint所以公开承认共同局限。

最终E1/E2只评一次冻结main matrix；成本与PPL分别报告，不启动HellaSwag/downstream/PrefixQuant。M3使用单独校准环境但primary quality统一回本机冻结eval环境，避免PPL跨库差异。旧benchmark PPL只作为工具历史，不直接搬入新表。
''')

    save('SMOKE_TEST_PLAN.md','''
# 下一阶段 compatibility smoke 与正式启动 gate

这是一份冻结计划，**不是PASS记录**。本轮只执行budget_smoke.py和CPU清单生成；Omni-Qwen port尚未实施。新任务目录使用独立v17（或用户后续指定目录），历史v8-v16不覆盖。

1. 环境：独立Omni环境，固定torch2.6.0+cu124/transformers4.49.0/tokenizers0.21.4，其他直接依赖从官方manifest解算并完整lock；先import QuantLinear/omniquant/Qwen，不改GPTQ环境。若本机full entry import缺termcolor/einops等，隔离补齐并记版本。real_quant/Triton不属于primary smoke。
2. architecture patch：只实施QWEN25_PORT_PLAN白名单。exact checkpoint在CPU加载，只L0在GPU。native/adapter关量化同token32/64 forward，比较hidden/logits；q/k/v bias和14/2头、RoPE、SDPA、mask/cache_position传递不变。附tinyCache generation一致性检查，正常cal/eval禁用cache。
3. 官方LWC kernel path：L0七Linear，W4A16 asym G128，官方bound初始化4、official optimizer/STE。2个共同cal样本的前64tokens，1个epoch/2steps，仅smoke-budget override；检查只有bound_factor有梯度、step更新、fpweights/norm/bias没有梯度、loss/weights/grad finite。明确这不是正式128×2048×20epoch run。临时量化/inplace输出有限，abits16 identity。
4. teacher/student prefix：L0 smoke使用同输入；再用32tokens的两block纯forward验证teacher来源FP、student来源本方法量化prefix，不需要第二block训练。保存trace和prefix/input hashes。不得把shared FP input误当全模型M3流程。
5. fake export/reload：L0已量化七Linear转native dense，其他blocks原FP，保存独立partial checkpoint与quantized module inventory。reload保持embedding-head tying；仅7个被量化模块，排除head/norm/bias。output allclose、no NaN/Inf。model不带未初始化的QuantLinear临时状态。tinyPPL128tokens/127targets，只做smoke不入正式表。
6. M2：固定L0组顺序，q/k/v/o/up/gate/down每模块先alpha1与native同W/H/init比较group entry、scale/zero/g_idx、Q_FP16、direct loss。dead-column/allzero smoke按native处理。外部alpha.9确实影响dynamic group scale，候选mutability不存在。
7. row batching/选择：共同cal新X上固定first32 q/down rows，全部十alpha，只做cost/selection正确性；candidate-state独立，FP16 selected Q对应其scale/zero；B3与B32 spot一致。direct delta loss与直接linear差分以及undampedGram allclose；near tie direct ranking复算。记录每阶段sync time和峰值，再预算全矩阵，不增加grid或用eval loss选择。
8. packing separate smoke：32输出行的一个projection导出标准GPTQ字段，检查整数code重建、group_idx、scale FP16 roundtrip和qzeros format，reload TorchQuantLinear eager post_init。记录dense/packed forward差异，不能预先要求bitwise相同或当作已经兼容。无Triton则走现有eager，不能为性能修复改变算法。

通过标准：各项finite、原生未量化架构比对rtol1e-5/atol1e-6，alpha1沿用v15 exact FP16/group参数规则；direct/Gram按1e-8+1e-5|reference|，ranking检查显式记录；export dense Q必须逐tensor相同，tiny127计数正确。若本机shape/quantizer差异致任一标准不成立，先输出mismatch，不跑formal。预算micro与正式training override分别序列化，不用smoke配置冒充官方baseline。

成本门槛：GPU峰值<7GiB，M2包含新cal actual loss的外推<=48h；M3依据实际one-block step+forward成本外推<=24h。超预算只报告STOP，可调整batch/offload/同目标缓存实现后重新测量，不能删候选/epochs/rows。正式启动需后续明确执行任务且smoke gates已通过，本轮不启动。
''')

    save('SMOKE_RESULTS.md',f'''
# 本轮实际 smoke：仅预算 core，非 OmniQuant compatibility

`BUDGET_SMOKE.json` 保存4个同步调用、70个行-alpha slots：L2 q/down前32 pristine行，v15 QPREFIX H，alpha=.9，B3/B32。脚本内wall总{smoke['wall_seconds']:.3f}s（含cache读取、检查和同步调用，不含Python/torch启动）；四次candidate同步区间合计3.401s；GPU RTX4060 Laptop。两宽度首三行最终FP16权重均exact相同，W/Hinv未修改，全部finite；随机64-token direct/Gram loss最大差1.863e-9，allclose PASS。它不验证新calibration数据选择、不验证全部projection/alpha1/native parity，不代表OmniQuant port通过。

OmniQuant Qwen forward/block-training/reload/tinyPPL **NOT RUN**。本轮没有任何完整模型量化、OmniQuant训练或PPL。CPU生成校准/test/C4 token清单不属于quality experiment。

Tokenizer对拼接test/C4长流发出超过model_max_length的警告；本轮只生成token清单，后续所有forward窗口最多2048tokens，不会将299,078或498,741个tokens一次送入模型。

准备中的两个软件错误：collect_audit最初假定v15嵌入historical_blacklist，KeyError后改为v14完整历史+v15新文章；budget_smoke最初误用weight key，实际cache是weight_fp16，KeyError后修复。这两次都发生在对应新实验计算之前，无候选/冻结结果被覆盖。C4首次网络读取遭sandbox WinError10013；经针对该bounded official-data命令的审批后成功。git默认Schannel凭据错误后改用git的OpenSSL后端，官方clone成功；大小写图片碰撞已披露，不涉及算法源码。
''')

    methods=[
        dict(id='M0',name='FP16 original',weight_bits=16,activation_bits=16,group_size='',format='FP16',optimizer='none',objective='none',epochs=0,grid='',prefix='original FP',representation='dense FP16',required=True,status='protocol only'),
        dict(id='M1',name='Native GPTQ',weight_bits=4,activation_bits=16,group_size=128,format='native symmetric zero8',optimizer='fixed GPTQ compensation',objective='native GPTQ',epochs=0,grid='alpha1',prefix='self native',representation='dense dequant FP16 primary; packed secondary',required=True,status='new common calibration not run'),
        dict(id='M2',name='GPTQ + Pure Range Search',weight_bits=4,activation_bits=16,group_size=128,format='same as M1',optimizer='enumeration + fixed GPTQ compensation',objective='local calibration output MSE per row',epochs=0,grid='0.1:0.1:1.0',prefix='self selected rows',representation='dense dequant FP16 primary; packed secondary',required=True,status='integration design; not implemented'),
        dict(id='M3',name='OmniQuant-Qwen2.5 port (official W4A16 config)',weight_bits=4,activation_bits=16,group_size=128,format='official asymmetric affine',optimizer='official LWC AdamW AMP; LET off',objective='official whole-block MSE',epochs=20,grid='',prefix='FP teacher + self quant student',representation='official fake/dequant FP16 primary',required=True,status='port design; not implemented'),
        dict(id='M4',name='LWC-controlled mechanism baseline',weight_bits=4,activation_bits=16,group_size=128,format='must match GPTQ for mechanism isolation',optimizer='not frozen / excluded',objective='not frozen / excluded',epochs='',grid='',prefix='must control',representation='not frozen',required=False,status='optional future study; not scheduled')]
    with (OUT/'BASELINE_MATRIX.csv').open('x',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=methods[0]);wr.writeheader();wr.writerows(methods)

    save('PROTOCOL_DRAFT.md',f'''
# v16 Full-Model & OmniQuant Baseline Protocol Audit

**Verdict: FULLMODEL_BASELINE_PROTOCOL_READY**。这是协议设计可进入实现/compatibility-smoke的判定，不是port实现、正式量化或PPL通过。证据清单与执行配置已冻结，smoke必须先通过，formal costs还需实测校正。完成本轮后 **STOP**。

## 已有证据与研究边界

当前读取v12 VALIDATION_REPORT、v13 REPORT、v14 REPORT、v15 REPORT/SUMMARY/correctness/RUN_REPORT和dynamic源码：Gate v1 independent validation FAIL（severe reduction42.03%、gain retention23.82%）；coverage hypothesis NOT SUPPORTED；v15 correctness PASS、TRANSFER_SIGNAL_PRESERVED。v15 FP/QPREFIX pooled local gain13.8722/15.8111%、median12.1283/12.9196%、alpha exact33/48、within-step47/48、35/37 positive保留、QPREFIX无<-5%row。v15 prefix来自native D，**不是M2 self-prefix**。这些局部输出重建证据没有证明full-model PPL收益。v13诊断只解释历史失败，未建立可用Gate。

主方法不用Gate，不更换模型/位宽/grid，不做coverage/robust objective/PrefixQuant或96-row扩展；v8-v15原始文件树由HISTORICAL_FINGERPRINTS固定并在交付核查。既有仓库其他未提交修改保留；本轮只新增本目录。未找到适用AGENTS.md文件，不推断不存在的仓库约定。

## 对15个必答问题的冻结答案

1. **下一主matrix**：M0 FP16 vs M1 Native GPTQ vs M2 GPTQ+Pure Search vs M3 OmniQuant-Qwen2.5 port；M4不纳入必需矩阵。完整配置见BASELINE_MATRIX.csv/PROTOCOL.json。
2. **Omni native Qwen支持**：没有；官方commit `{SHA}` 只有既有家族分派，AutoModel支持不能替代校准/quant支持。
3. **最小adapter**：新增native-decoder七Linear替换工厂；修改omniquant discovery/device及kwargs捕获/重放；新manifest入口和dense export/eval bridge。精确文件/类边界见QWEN25_PORT_PLAN。
4. **算法语义是否保持**：计划保持官方W4A16配置LWC-on/LET-off、量化器/20epochs/AdamW/AMP/block MSE/FP teacher和quant student不变。未实现，必须逐项smoke证明。若修改这些算法项，不能仍叫官方方法port。
5. **M2 sequential prefix**：原生 `[k,v,q]->[o]->[up,gate]->[down]`；组capture时此前组/层已装M2自身selected rows；每候选独立完整dynamic compensation，per-row shared alpha，经最终该block的quant output传播下一层。不是FP-input或v15 native-D prefix。
6. **candidate总数**：168 linears、304,128 rows、3,041,280 row-alpha candidates；B32 serialalpha约95,040 calls，所有行/所有十alpha保留。
7. **calibration wall预算**：极小timing外推M2 core约{core_seconds/3600:.2f}h；规划total12–24h，M1规划20–60min；它们不是正式测量。M3总61,440steps，没有one-block训练时间，不虚构点估计，按COMPUTE_BUDGET情景/next-step smoke核定。M2>48h/M3>24h停止正式启动，不能缩水协议。
8. **显存**：8GiB设备，M1/M2 active-block流式设计规划2–4GiB，M3规划3–6GiB；本轮只有budget core约192MiB峰值，完整训练峰值尚未验证，不能搬用v15错位reset后的prefix peak。
9. **单独环境**：需要隔离Omni校准环境，目标相同torch2.6/cu124、transformers4.49、tokenizer0.21.4；完整依赖lock待import smoke。primary质量统一在已冻结GPTQ eval环境，禁止修改现有环境或未经审计安装real_quant kernel。
10. **共同cal**：CALIBRATION_MANIFEST.json，128篇未用train文章×2048tokens=262,144，seed20261005，排除v8-v15共280篇IDs/hash，候选cal-only；same exact token_ids/order给M1/M2/M3。不使用最终test/PPL筛选。
11. **E1**：完整WT2 test，revision固定，{e1['num_input_tokens']:,} input/{e1['expected_evaluated_tokens']:,} scored tokens，max2048/stride512，token-weighted共享计分器，留尾，无special/chat。
12. **E2**：独立C4validation固定131,072 input/131,071 scored，官方固定revision第0shard docs32..1055；不是standard full C4。清单和文本已生成，无模型forward。
13. **方法级公平指标**：共同checkpoint/cal/eval/precision/目标scope下，PPL、deltaNLL/PPL、总离线calibrationwall、GPU allocated/reserved、磁盘representation、candidate/epochs/steps/trainableparams/失败重试可以比较。M3默认asym与M1/M2sym不同且无GPTQ，因此只能说整体方法结果。
14. **机制比较**：enumeration-vs-LWC、row共享-vs组独立、projectionloss-vsblockloss、symmetric-vsaffine、compensation差异需要另设控制对照。完整M3不能单独归因优化器；官方weight-only本来LET-off，不能把它称为人为删减。M4如控制同quantizer/同objective/同prefix需另命名，不在本轮执行。
15. **Luna Max下一任务**：在独立新执行目录，先固定环境并实现QWEN25_PORT_PLAN白名单的官方weight-only port、共同calibration loader和dense export；只做SMOKE_TEST_PLAN的L0 two-sample×64token two-step、关量化native/port forward、7模块/梯度/LWC/finite/reload/tiny127tokens检查；再实现M2原生sequential接口并做first32row十alpha/new-cal cost checks。输出SMOKE_RESULTS、patch manifest、pip lock、actual step/candidate time、成本复算，**先停在实现+smoke gate，不跑正式四模型PPL表**。不自动升级为full-model训练。

## READY 判定与可验证边界

Qwen mapping由config+safetensors shapes及本机native forward代码明确；Omni port范围明确；M2 own-prefix/row independence/dead处理/scale/zero/表示策略明确；候选数量精确且存在B32计时与有边界预算；cal/E1/E2 exact IDs/hash冻结；disjoint和历史完整性由VERIFY_AUDIT.json检查。`READY`仅满足这些**设计门槛**，并不宣称环境安装、Qwen port forward、packed精度或full-model质量已通过。

主要未验证项：Omni兼容环境/训练step成本、port teacher/student接线、M2整个168模块self-prefix的实现、全grid真实X批量等价、packed与dense差异、最终PPL。primary统一dense-dequant避免将不同packing误差混入当前range研究；standard GPTQ layout能表达per-row scales，但真实packed导出仍是smoke门槛。

## 论文与工程方向

论文方向值得进入受控模型级验证，当前只支持“局部信号跨native量化prefix仍存在”，不支持PPL收益、算法优于OmniQuant或新颖性/发表承诺。下一证据是同cal/eval的完整M0-M3质量与成本，以及失败case；不能追正结果换目标。

工程亮点现阶段是动态GPTQ原生parity、候选状态隔离、版本/数据hash冻结、原始结果复算、负结果门槛和预算/恢复设计。full-model integration、标准packed导出、Omni port尚属于设计项，不能写成已完成能力；搜索工程耗时改进属于离线校准，不是在线推理加速。
''')

    save('RUN_REPORT.md',r'''
# v16 audit run record

实际执行：读取用户附件/历史报告与源代码；官方固定commit浅克隆；CPU读取本地Arrow及tokenizer冻结cal/E1；bounded官方C4读取并tokenize；4次极小GPTQ core调用共70row-alpha slots；生成协议/计划/预算与检查。没有formal full-model quantization、Omni training、PPL、PrefixQuant或Gate实验。

命令（repo根目录，已有文件不可覆盖；completed结果应复用）：

```powershell
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\collect_audit.py
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\freeze_c4.py
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\budget_smoke.py
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\build_documents.py
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\verify_audit.py
```

文档生成是本轮交付过程，前四个命令不是formal实验入口；clean reproduction需空的v16目录及同commit official_source。已冻结目录只运行verify，避免重复smoke/更换cal。C4脚本第一次sandbox网络失败后经限定审批成功；这次审批只针对固定1056doc数据读取，不等于formal实验授权。软件准备错误详情见SMOKE_RESULTS。

历史文件前后SHA由verify核查；官方code blob核查排除已披露case-colliding images。原repo存在的README/benchmark修改均保留，本轮不新增对它们的修改。最终verdict以PROTOCOL_DRAFT/VERIFY_AUDIT为准，STOP后不启动大跑。
''')
    print('documents written; core projection hours',core_seconds/3600)
if __name__=='__main__':main()
