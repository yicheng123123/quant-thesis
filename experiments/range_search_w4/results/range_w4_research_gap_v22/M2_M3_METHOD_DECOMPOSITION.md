# M2 / M3 方法拆解

本轮只读审计。比较对象是同一 Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775` 的冻结结果；不是新实验。

## 已核实实现

| 维度 | M2：v18 | M3：v21.4 / v21.5 |
|---|---|---|
| 权重/激活 | W4A16，G128，dense-dequant FP16 保存 | W4A16，G128，dense-dequant FP16 保存 |
| 格式 | symmetric，整数 0..15，zero=8 | asymmetric affine，整数 0..15，独立上下边界，rounded zero |
| 范围时机 | 原生 GPTQ 每组入口，在已补偿的当前 W 上估计范围；scale 乘 α | pristine group min/max 乘 sigmoid(logit)，每次 fake quant 重新计算 scale/zero；没有 GPTQ trajectory |
| 自由度 | 304,128 个行 α；同一行所有组共享 | 2,795,520 个 row/group，两个端点各一个参数，共 5,591,040 标量 |
| 优化 | 0.1..1.0 十候选穷举，独立行 argmin；精确 ties 选较大 α | AdamW，lr=.01，wd=0，20 epochs ×128 samples/block，batch=1，round STE，AMP |
| 目标 | 当前量化前缀输入 X 上 `mean((X(Q-W)^T)^2)` | block MSE：FP prefix 的 FP block 输出，对量化 prefix 的 quantized block 输出 |
| 更新范围 | 单个 Linear；行之间可分，row stitch | 七个 projection 的 LWC 在同一 block loss 下联合更新 |
| 误差补偿 | 原生 GPTQ Hessian/Cholesky/Hinv、damping=.1、blocksize=128；desc_act/static_groups=false | 没有 GPTQ compensation；通过 block reconstruction 调整 LWC 参数 |
| 前缀 | 各方法自己的已提交量化前缀，依赖边界 refresh | FP teacher trajectory 与量化 student trajectory 分别推进；`aug_loss=False` |
| 不启用项 | 无梯度/旋转/混合精度/激活或 KV 量化 | LET=false，A16；只优化 LWC，不能称作 LET+LWC 或权重微调 |

M2 每候选均从 pristine target W 和新 GPTQ mutable state 开始。α=1 没有范围 hook，严格对应 native dynamic GPTQ。原始全局 `find_params` 不裁剪，后续动态 G128 估计才缩 scale。真实 symmetric affine 舍入/截断见安装的 GPTQModel quantizer，不能自行替换为 `absmax/7` 的另一实现。

M3 `UniformAffineQuantizer` 将 W reshape 为 `[-1,128]`，使用 group 的 `amin/amax`，**不在代码中强制 min/max 包含零**；上下端点分别乘 sigmoid。scale=`(xmax-xmin)/15` 并 clamp 到 `[1e-5,1e4]`；zero=`round(clamp(-xmin/scale,-1e4,1e4))`；量化整数 clamp 0..15。LWC logit 初值为 4，即 coefficient≈.982014，不是严格 MinMax=1。无 LET 时 temporary weight 直接取原始 module weight，未学习 W 本身。

## 为什么可能影响 PPL

最值得检验的是 H1 的目标/teacher alignment 与 H2 的容量差异。Block loss 经过 attention、SwiGLU、残差等计算，会对局部误差方向给出不同权重，并能处理其他量化模块已有误差。M3 还以 FP trajectory 为目标，包含前缀误差恢复意图；这与“在同一 quantized X 上重建 Linear”是另一差异。联合参数变化和连续优化可能帮助达到更好的 block 解。

这些都是机制解释，不是已识别的贡献比例。**asymmetric 格式、上下端点自由度、row/group 粒度、teacher 输入、联合更新、连续梯度、GPTQ 有无补偿同时变化**。M3 PPL 更好不能证明其中任一单项导致收益。

## 共同控制与限制

128×2048 WT2 train 校准文档及 hash 相同；WT2 test 与 C4 fixed-budget token 流、FP16 SDPA、TF32 off、2048/512 evaluator 相同。M3 61,440 scheduled /61,438 successful /2 native scaler skips，dense reload exact；历史 prefix 的成功更新计数有推断性 caveat。M3 exact calibration wall 未保存，3052s 是日志时间跨度，不能计算与 M2 的可信 wall ratio。

依据：v18 `PROTOCOL.json`、v19 `REPORT.md`；v17.3 `optimized_native_m2.py`；v21.4 `FORMAL_PROTOCOL_LOCK.json`、`run_formal_m3.py:485`、official_port `quantizer.py` / `omniquant.py` / `utils.py`；v21.5 `REPORT.md`。对应 SHA256 见本目录 `INPUT_PROVENANCE.json`。
