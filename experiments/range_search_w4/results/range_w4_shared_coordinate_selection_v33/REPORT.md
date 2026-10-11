# v33 — 已有两种坐标下的共享缩放选择诊断

> 只复用 v24/v32 已冻结的数据与候选；所有文章已被查看，本轮是探索性诊断，不是新质量验证。四行只是固定代表行，不能代表完整 Linear。

## 冻结选择规则

选择单位为 target×split。对 D=I 和 v32 已有固定 D，分别逐行按 search MSE 从原 10 点网格选 alpha，精确并列取较大 alpha，再将四行 MSE×32768 相加。总分较低的 D 获选；D 总分精确相同时取 D=I，不用容差。每个单位四行共用同一个 D，行可各自选 alpha。search 记录和候选提交审计先保存并锁定，随后才读取 held。

search 候选集合包含关系检查：True；D=I 时重现 G01 alpha/score：True；固定 D 时重现 v32 G11 alpha/score：True。共选择固定 D 3/12 个 target×split 单位，D=I 9/12 个。

## Held 对 G01

改善为 pooled SSE 下降；W/T/L 按冻结的 v24 per-row epsilon 分类，严重退化按相对 G01 严格低于 −10% 统计。

| Split | G01 SSE | 共享 D 选择 SSE | 改善 | W/T/L | 中位逐行改善 | 最差 | >10% 退化 | Top-3 正节省占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 2023.691860 | 1810.347689 | +10.542% | 3/20/1 | +0.000% | -32.549% (L23_down_proj_R672_S0) | 1 | 100.00% |
| S1 | 1820.882374 | 1665.919013 | +8.510% | 6/16/2 | +0.000% | -38.326% (L23_down_proj_R672_S1) | 2 | 95.60% |

Overall pooled gain +9.580%; W/T/L 9/36/3; median +0.000%; worst -38.326% (L23_down_proj_R672_S1); >10% regressions 3/48; top-3 share of positive savings 72.37%.

### Target 汇总（两个 split 配对）

| Target | Pooled gain | W/T/L | Median | Worst | >10%退化 |
|---|---:|---:|---:|---:|---:|
| L0 down_proj | +0.000% | 0/8/0 | +0.000% | +0.000% (L0_down_proj_R0_S0) | 0 |
| L0 up_proj | +2.121% | 3/4/1 | +0.000% | -28.256% (L0_up_proj_R0_S1) | 1 |
| L12 down_proj | +0.000% | 0/8/0 | +0.000% | +0.000% (L12_down_proj_R0_S0) | 0 |
| L12 up_proj | +0.000% | 0/8/0 | +0.000% | +0.000% (L12_up_proj_R0_S0) | 0 |
| L23 down_proj | +21.241% | 6/0/2 | +25.399% | -38.326% (L23_down_proj_R672_S1) | 2 |
| L23 up_proj | +0.000% | 0/8/0 | +0.000% | +0.000% (L23_up_proj_R0_S0) | 0 |

### 固定 D 被选中的 held 结果与尾部

在被 search 选中固定 D 的 3 个单位中，held pooled SSE 改善/退化/精确持平为 3/0/0。行级 W/T/L 与每个单位细节见 `SHARED_UNIT_HELD.csv`、`HELD_ROW_RESULTS.csv`。
v32 G11 相对 G01 有 19 个 >10% 严重退化行；共享坐标选择避开 16 个，仍有 3 个；新出现的 >10% 严重退化 0 个，新增 epsilon 判定 loss 0 个。

最大净节省 target 为 L23 down_proj，净节省 364.132 SSE。移除该 target 后，其余目标相对 G01 的 pooled 变化为 +0.196%。正收益目标 2/6。

## 计算成本

CPU 阶段预算上限 300 秒；`BUDGET.json` 中记录的累计阶段 wall 为 16.44 秒，包含 4 次 CPU-only 实现检查失败及修复时间。没有 GPU 调用、模型前向、GPTQ、PPL 或整模型量化。

## 结论

仅当 S0 和 S1 都改善且尾部/集中度未明显恶化时才提出新文章冻结验证。若 split 增益方向不一致、目标贡献集中或尾部退化仍明显，则停止当前两种坐标的联合选择扩展。该诊断不构成新文章盲测，不是完整 Linear 的全行最优选择，也不支持 AWQ 新方法、正 interaction 或 PPL 改善结论。

本轮结果：S0/S1 pooled gain 分别为 +10.542% / +8.510%；最大目标贡献占净节省 98.87%（净节省可能因其他目标退化而被抵消）。据此停止当前两坐标联合选择扩展，不增加第三种 D 或更多缩放强度。

GPU、模型前向、GPTQ、PPL 与整模型量化均未运行。完整逐行负例、搜索选择、alpha 和候选张量审计见 CSV/JSON；复现命令见 `REPRODUCE.md`。
