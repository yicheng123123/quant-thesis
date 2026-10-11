# v34 Source Audit — AWQ scaling 与 M2 GPTQ

## 范围

静态审计 v18、v32、v33、AWQ recipe/checkpoint metadata/raw benchmark、v17.3 GPTQ wrapper，以及 MIT 官方固定 commit 源码。本轮没有 GPU、前向、GPTQ、候选生成或 PPL。根目录未找到 AGENTS.md；历史文件只读。

## MIT AWQ 固定源码版本

源码基准：**mit-han-lab/llm-awq commit d6e797a42b9ef7778de8ee2352116e0f48a78d61**，不是浮动 main/master。

- [auto_scale.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py)
- [pre_quant.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/pre_quant.py)
- [quantizer.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py)
- [entry.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/entry.py)
- [auto_clip.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_clip.py)

### 核对结果

1. **激活统计**：captured input 按最后一维摊平样本/token 维，逐输入通道求 mean(abs(x))；scale 运算跟随 X 的 tensor dtype。v34 的目标输入是 FP16，需保留来源 dtype，不要悄悄升成 FP32。
2. **ratio 与归一化**：n_grid=20，ratio 为0.00、0.05、…、0.95，不含1.00；s=act_mean^ratio，逐通道 clamp 下限1e-4，再除以 sqrt(max(s)*min(s))。ratio=0 给 identity。
3. **候选目标/恢复**：保存 block 原 state；逐 ratio 将 mapping 中 Linear 权重乘 s、假量化再除回 s，运行目标 module/block，以原输出与候选输出的 FP32 MSE 选 D。每个候选后恢复原 state；严格小于才更新，故精确并列保留先到的较小 ratio。没有梯度。
4. **候选格式**：pseudo_quantize_tensor 根据 q_config 分组、round/clamp/dequant。entry 默认 zero_point=true、q_group_size=-1（可显式配置）；v34 的 G128 是显式值，非 CLI 默认。默认 asymmetric affine W4 范围0..15，scale=clamp(max-min,1e-5)/15，zero=clip(-round(min/scale),0,15)，再 round、加 zero、clamp、减 zero 和反量化。zero_point=false 是 absmax/7 对称分支，整数-8..7、zero=0。v34 将 D selector 定义为 asymmetric W4/G128 RTN proxy；最终 GPTQ 保留 M2 symmetric W4/G128。
5. **Qwen mapping**：input_layernorm→q/k/v，评价 self-attention；post_attention_layernorm→共享 gate_proj、up_proj，评价整个 MLP；up_proj→down_proj 使用 down_proj 输入并默认评价 down Linear。v_proj→o_proj 只有两个 weight shape 相同才采用；本模型 GQA 维度不匹配，故跳过。gate/up 与 up/down 是两种不同合法坐标变换，不能合并成同一个 D。
6. **折叠边界**：norm→Linear 时 norm weight 和 bias（若存在）除以 s，目标 Linear 输入列乘 s。Linear→Linear 时前 Linear 对应输出行和 bias 除以 s，下一 Linear 输入列乘 s。Qwen MLP 的 up→down mapping 可将 up 分支按通道缩放折入 down 输入列，因为 SiLU(gate)*up 对 up 支路的正对角缩放仍线性；存在 bias 时也必须处理。缓存输入如供后续 clip 使用也会变换。必须依据实际图、共享分支、残差和消费者折叠。
7. **Scaling 与 clipping 分离**：pre_quant.py 先 auto_scale/apply_scale，再由独立 mse_range 分支执行 auto_clip/apply_clip。auto_clip 是对称 weight shrink/clamp 搜索，不是 scaling ratio。v34 不运行 auto_clip，实验只能叫 AWQ-auto-scale-selected-D + GPTQ，不是 full AWQ。

## 本仓库 GPTQ 冻结定义

v18 模型 Qwen/Qwen2.5-0.5B-Instruct，revision 7ae557604adf67be50417f59c2c2f167def9a775，FP16；原正式设备 RTX 4060 Laptop GPU。M2 覆盖24层、168个 decoder Linear；W4A16、symmetric、G128、block128、desc_act=false、static_groups=false、true_sequential=true、damp_percent=0.1、damp_auto_increment=0、mse=0。GPTQ统计/补偿 FP32，dense 权重保存 FP16。alpha为0.1到1.0，每输出行独立选一个 alpha、跨该行动态 G128 groups 共享；目标是相同实际输入下 direct local reconstruction MSE；精确并列取较大 alpha。

v17.3 optimized_native_m2.py 每候选从 pristine W clone；输入/H/Hinv 可在完全同配置候选间共享，但候选不能改写共享状态。H 准备先将零对角 dead columns 设为1，再加 0.1*mean(diag(H)) damping；native GPTQ 内仍做动态组参数估计、列级 error compensation 和最终因子步骤。缩放坐标必须用真实 X′重建 H′并重新 dead-column/damp/factor；不能未经验证地变换旧阻尼 H。alpha=1 是同坐标 native GPTQ 锚点，不代表两个坐标的 GPTQ 结果应相同。

## v18 可核验来源及成本基数

- experiments/range_search_w4/results/range_w4_fullmodel_v18/PROTOCOL.json：SHA256 136c9c363c609aa10fe98f2d9f4e5a94961293da02f06b2b5f2090ecb09a78a2
- 同目录 REPORT.md：SHA256 6fc304e09bc3436a6ab05dec468249f451a8b239b5a618d08565529bedabdb76
- M2/RUN_REPORT.json：SHA256 7557d450ca05fc913a6b71b7e9dbaef5fb77b261066bf59a0695957ad2854d980
- FORMAL_TIMING.csv：SHA256 a54214354e122d3800877d053df22cc072dd321622142ef338d6a358a0cd5e7b
- wrapper：experiments/range_search_w4/results/range_w4_m2_runtime_v17_3/optimized_native_m2.py

M2 已完成1,680次 full-Linear 候选、3,041,280个 row-alpha score slots；这是不同计数。正式 calibration 是 v16 manifest 中128篇完整 WT2-train文章×2048 tokens（262,144 tokens），seed 20261005。M2 校准2,115.94秒。这里只复用配置和成本，不重跑。

FORMAL_TIMING.csv 的完整矩阵、128×2048输入均值：gate GPTQ 0.3420106s/score 0.6599522s；up GPTQ 0.3455335s/score 0.6594002s；down GPTQ 1.8631694s/score 0.9235557s。up_gate→down refresh 均值2.37228s。AWQ 的 full-MLP D 搜索没有相同记录阶段，不能声称实测加速。

## 仓库已有 AWQ 产物

recipe 位于：

- artifacts/qwen2.5-0.5b-instruct-awq-w4a16-g64-sym-calib64-wttrain512/
- artifacts/qwen2.5-0.5b-instruct-awq-w4a16-g64-sym-duo-calib64-wttrain512/

两者均为 AWQModifier→QuantizationModifier，n_grid=20、W4A16、G64、symmetric，64篇 WikiText-2 train calibration×512 tokens、来自128篇 pool（hash 2bd6cacca4c7894b00d8779d76fc9b1d5776345bd69bc0d981054b2dbf579a76）；duo_scaling false/true。raw environment 为 llmcompressor 0.13.0、compressed-tensors 0.18.0。false 的 full WT2 test PPL 14.489811，true 为14.549754。AWQ-family scaling-before-quantize 的整模型流程做过，但没有 GPTQ；G64/校准/实现与 M2 不同，不是 MIT upstream 代码运行。recipe 未声明独立 clipping stage。phase13 baseline_awq.json 的16.6189是 smoke PPL，不和 full WT2 混用。

## v32/v33 审计

- v32 frozen protocol SHA256 fa6b61785cd2ce633f402b0b93be7761389ffc9e48e4bdb10f151ecd62e1b12e；report SHA256 59d14b07347b8563a37a3e47bb8da991156a1e300f414ecc950dcf86695dbfa8。
- v33 frozen protocol SHA256 540d8ee9a7ada015eab26dafea017bcefc2b56f4b044886e456ea3bd34481e03；report SHA256 f98a0be2c3af176d34333d77d58c9511b58db875401af457c41640543989f83b。

v32 对每 target/split 用单个预定 activation-derived D：mean(abs(X_j))、1e-8 floor 后开平方、端点几何均值归一化；不是 AWQ ratio 搜索/full MLP target。只测六个 target、四个固定行、两个 split；变换后输入重建 H，原生动态 G128/compensation 保留。v33 重用 v24/v32 固定候选，每个 target/split 的四行在 identity 与固定 D 间选择；固定 D 选中3/12单位。S0/S1 pooled gain +10.542%/+8.510%，但净节省主要来自 L23 down_proj；移除后仅+0.196%。最差行退化38.326%，3/48行超过历史10%退化口径。文章已被看过，属回顾性诊断；v34 不用 held 选层或调参。

结论：
- 仓库 AWQ-family 整模型缩放量化：**做过，llmcompressor 实现**；
- 固定 activation scale + GPTQ：**做过，v32 局部实验**；
- MIT upstream auto_scale-D + GPTQ alpha=1：**无已运行证据**；
- 同一 MIT auto_scale-D + scalar10：**无已运行证据**；
- 完整 MIT AWQ + GPTQ：**未做，v34 不把混合流程称完整 AWQ**。

## 坐标推导与待测合法变换

令 X∈R^(N×d)，W∈R^(o×d)，D为正对角矩阵：

- W′=WD，X′=XD⁻¹
- X′W′ᵀ=XWᵀ（实数精度）
- 若 H∝XᵀX，则 H′=D⁻¹HD⁻¹。它不是复用已阻尼、已分解 H 的依据。

GPTQ 不假定坐标变换不变。M2从真实输入重估 H，处理 dead columns，按平均对角加 damping 并重新 factor；damping 随坐标改变。必须重新构建和补偿。

待测 post_attention_layernorm→{gate_proj,up_proj}：norm weight与存在的norm bias除以D；两个Linear输入列乘D。Linear输出bias不受输入坐标缩放影响。D是共享归一化输入坐标；residual/attention其他路径不改。量化gate/up提交后，实际SiLU(gate)*up产生臂特定down input，应重捕获并重建H_down。

另一合法 mapping up_proj→down_proj：up对应输出行和bias除以D，down输入列乘D。v34最小实验只选norm→gate/up，不叠加两种缩放。

## 未来运行前 correctness 清单

1. 未量化变换输出等价；分开报告解析等价、FP64/FP32误差和源码兼容FP16 in-place折叠误差。
2. ratio=0、D=I回归原checkpoint/input hashes和冻结标准。
3. 原坐标与缩放坐标alpha=1各自对三种完整Linear匹配独立native dynamic GPTQ：group-entry compensated weights、scale、zero、g_idx、最终权重、全行直接输出；不要求跨坐标量化张量相同。
4. 每个alpha从pristine W/fresh GPTQ state开始；pristine W/H hashes不变；逐split、逐坐标反序复跑完整down_proj。
5. D、逐行alpha、候选绑定输入/H/W/D hashes和完整矩阵shape；selection lock后才读held。
6. stitch全部W/scale/zero/g_idx；held重算完整MLP、decoder residual/block输出和真实L13输入边界。整块提交后刷新一次，不按行批刷新。
7. 记录norm/linear bias、scale dtype、cache和数据来源；held不参与activation stats、D/alpha选择或候选顺序。

这些只是协议要求，不表示本轮检查已运行或通过。

