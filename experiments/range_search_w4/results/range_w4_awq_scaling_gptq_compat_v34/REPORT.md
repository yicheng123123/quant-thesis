# v34 — 官方 AWQ scaling × GPTQ 兼容性审计

状态：**静态审计完成，后续实验协议已冻结；本轮未运行模型、GPU、前向、候选搜索、GPTQ、PPL 或整模型量化。**

## 结论

仓库**做过 AWQ 风格的整模型缩放量化**：两份 W4A16、G64、对称 AWQModifier 产物（duo_scaling=false/true），由 llmcompressor 0.13.0 和 compressed-tensors 0.18.0 运行；false 版本有完整 WikiText-2 test PPL 14.48981。它不是 MIT 官方仓库直接运行出的 checkpoint，也没有使用 v18 GPTQ。

v32/v33 **没有测试 MIT 官方 auto_scale D 选择接入 v18 动态 GPTQ**。v32 是固定 activation-derived D，在少数 Linear 的四个预定输出行上诊断；v33 只在两个已有坐标之间按四行 search SSE 选择共享坐标。两者均非官方 AWQ scaling ratio 搜索，也非完整 MLP 输出目标。v32 中 scalar10+固定 D 相对原坐标 scalar10 的局部 held pooled 改善为 3.688%，但仍只代表固定行；v33 的净节省高度集中于 L23 down_proj，移除该 target 后仅 +0.196%。这些回顾性结果不支持扩大固定 D 路线，也不用于选择本协议 target 或调参。

有一个定义明确但尚未测试的兼容性问题，值得做一次局部、有停止门槛的验证：

> 使用 MIT AWQ auto_scale 的激活统计、ratio 网格和完整 Qwen2 MLP 目标，按其 RTN 假量化误差选择一个 post-attention norm 到 gate/up 的共享 D；再比较同一 D 下原生 alpha=1 与 scalar10 GPTQ。检验 D 能否在完整 L12 MLP 上迁移，不预设胜出。

这个组合不是“完整 AWQ”：不运行 MIT AWQ 的最终 AWQ 量化流程，不做 auto_clip，不改变最终 M2 对称 W4 G128 动态 GPTQ。D 选择器明确使用 MIT quantizer 可配置的 asymmetric W4 G128 RTN proxy；这是 AWQ scaling selector 与 GPTQ 的跨量化器组合，不能称为相同量化格式或完整 AWQ。

## 实验分类

| 已有路径 | 实际执行内容 | 与待测路径的差异 |
|---|---|---|
| 仓库 AWQ 整模型量化 | llmcompressor AWQModifier 搜索/折叠 scaling，随后 QuantizationModifier；W4A16、G64、对称；duo=false 和 true | 不是 GPTQ；非 MIT pinned source；G64 不同于 M2 G128 |
| MIT 官方 auto_scale | 固定源码版本；20 个 ratio，伪量化候选，按目标 module/block FP32 输出 MSE 选 D | 仅源码审计，仓库未按此源码运行 |
| v32 固定 activation scaling + GPTQ | 预定统计公式产生固定 D；动态 GPTQ alpha=1 或 scalar10 | 无 D ratio 搜索；六个 target、每个四行、历史文章 |
| v33 D/identity 共享选择 | 四行 search SSE 选择 v32 固定 D 或 identity，held 事后评价 | 不搜索 AWQ ratio；不覆盖完整模块和全部输出 |
| MIT auto_scale-D + native GPTQ | 未运行 | 待测 arm C |
| MIT auto_scale-D + scalar10 GPTQ | 未运行 | 待测 arm D；和 C 共用同一个 D |

## 后续实验与成本

协议选中中间层 L12（24 层模型的 index 12），完整 MLP 的 gate_proj、up_proj、down_proj 和全部输出行。不是根据 v33 held 结果选层，也不只看历史四行。两个 split 各有16篇 search 和16篇新 held WT2-train 完整文章。每个 split 独立按 search 选 D；split 内 C/D 完全共享该 D。须先按全文 hash 排除 v18、v24/v31/v32/v33 等全部历史文章及校准文章；不足64篇就停止，不拆文章、不替换为旧文章。

每 split 计划20个 AWQ RTN D 候选、62次主要 full-Linear GPTQ 候选、214,272个 row-alpha score slots、6份候选无关 Hessian；两个 split 合计40个 D 候选、124次主要 GPTQ、428,544 score slots、12份 H。另有16次 alpha=1 parity（包含四种实际 down input 状态）和4次完整 down_proj 候选隔离复跑，须分开计数。A/B 共享原坐标 gate/up bank，C/D 共享缩放坐标 gate/up bank；四臂 down input 因 gate/up 的提交结果不同而分别重算。

以 v18 相同完整矩阵 shape、较大128×2048校准输入的实际阶段时间作保守参考：124个主要候选 GPTQ 约1.83分钟；对应 all-row direct scoring 约1.56分钟；parity/replay 检查约0.7分钟。另计 D 搜索完整 MLP forwards、四份 search/held prefix 捕获、八次 search gate/up→down refresh、八次 held MLP arm 前向、缓存写入和环境开销。规划区间约 **6–10 GPU 分钟**，不是 benchmark。AWQ D 搜索没有完全相同的 v18 阶段计时，首批完整 MLP 候选必须实测复核；禁止按代表行时间外推。累计30 GPU分钟 hard cap 包含启动、预热、重试与写盘；预测超25分钟则停止，不继续第二 split。

## 推进门槛

仅当 correctness 全过、文章拆分经 hash 验证、且实测预测在 cap 内才运行局部实验。search selection lock 写盘前不读取 held。D 和 alpha 仅依 search 选择。D arm 相对 B 的 held full-MLP 输出 MSE 须在 S0、S1 都至少改善1%，并超过对应 FP16 未量化折叠误差的五倍。任一 gate/up/down projection pooled held MSE 相对 B 退化超过5%、超过5%的投影输出行退化超过10%、或单行退化超过50%，就停止该方向并保留负结果。通过也不自动触发整模型量化或 PPL。

D selector 用 upstream ratio grid 严格小于规则（精确并列保留更小 ratio）；M2 alpha 精确并列仍取较大值。分解 C−A（scale 本身，alpha=1）、D−C（缩放下范围搜索）、D−B（组合相对现有 scalar10），并报告所有投影行和 full MLP 输出分布。

## 本轮交付

- [SOURCE_AUDIT.md](SOURCE_AUDIT.md)：仓库 AWQ / v18 / v32 / v33 源证、固定 upstream commit、格式/边界与数学审计。
- [COMPARISON_TABLE.csv](COMPARISON_TABLE.csv)：实验路径对照。
- [NEXT_EXPERIMENT_PROTOCOL.json](NEXT_EXPERIMENT_PROTOCOL.json)：可执行实验的冻结定义、预算和停止门槛。
- 本轮未创建运行脚本；尚无 v34 新量化结果，未来运行需实现独立 harness。

## 静态复核命令

从仓库根目录运行以下只读检查，核对本报告引用的主要来源文件：

~~~powershell
$files = @(
  'experiments/range_search_w4/results/range_w4_fullmodel_v18/PROTOCOL.json',
  'experiments/range_search_w4/results/range_w4_fullmodel_v18/FORMAL_TIMING.csv',
  'experiments/range_search_w4/results/range_w4_fixed_channel_scale_scalar_v32/FROZEN_PROTOCOL.json',
  'experiments/range_search_w4/results/range_w4_shared_coordinate_selection_v33/FROZEN_PROTOCOL.json',
  'artifacts/qwen2.5-0.5b-instruct-awq-w4a16-g64-sym-calib64-wttrain512/recipe.yaml',
  'benchmark/results/raw/awq_g64.json',
  'benchmark/results/raw/awq_g64_duo_true.json'
)
$files | ForEach-Object { Get-FileHash -Algorithm SHA256 -LiteralPath $_ }
~~~

