# v42 REPORT

## 结果与推进判断

**D：正确性或预算限制，质量未验证。** 状态：PILOT_COMPLETE_BUDGET_STOP。完成后停止，无自动后续实验。

正式修复覆盖 0/168 Linear、0/304,128 行。完整 block0 pilot 覆盖 7/7 Linear、12,672/12,672 行。

未生成经部署核对的完整 R，未运行 R 的正式 PPL。不能判断局部信号是否迁移。

已核查模型文件哈希、冻结 token manifest 和原始窗口重新汇总后复用的历史值：

- wikitext：M0=12.46340950，M1=13.88436957，M2=13.76055605；R 及其相对变化未测量。
- c4_fixed_budget：M0=19.24855582，M1=22.23392445，M2=22.03288316；R 及其相对变化未测量。

完整 block0 实测 226.107 秒；按全部真实相同矩阵形状，24 block ×1.35 +30 秒，正式修复保守预测 7355.856 秒，另预留保存/reload/PPL 360 秒；门控 预算不支持，pilot 后停止。这是成本预测，不是全模型实测。

## 修复与正确性

本报告统计范围为 已完成 pilot：12,672 行，接受 378,993 次单坐标修改，实际候选增量评价 902,037,314 次。同一冻结本组输入下的局部重构 MSE 合计 8.92532269562 → 7.76991472887（下降 12.9453%）。各组输入来自当前 R 前缀；跨模块合计仅是局部诊断，不是端到端损失。

v41 原始代码/协议一致：坐标升序、−1 再+1、首次最小值；预测下降阈值 max(1e−14,1e−8×初始直接 MSE)，每次接受必须直接重算严格下降且增量误差满足原容差。每行最多32次接受、128×输入维度候选评价上限；边界过滤。没有更换算法或放宽容差。

96 条 v41 A 轨迹的终态、实际评价次数、接受次数及5个预算节点状态逐一一致；4 条原始逐行参考与批量路径一致。模型搜索主循环跨行批量，非逐行 Python 搜索。每次接受后 p 按实际 FP16 网格增量更新。

正式 M2 本来是 dense FP16 部署，不存在历史 packed kernel。整数码来自可信保存的 FP32 scale/zero/g_idx，枚举全部16个网格值唯一逐位匹配，不是近似除法恢复。辅助4位打包仅用于存储与 roundtrip；部署结构和 kernel 沿用正式 dense FP16/SDPA。全部168模块零修改参数权重逐位一致，64-token 输出回归逐位一致。C 为独立未 damping 的完整 FP64 Gram，保留跨组/非对角项。

每次接受直接输出重算、固定1/8/16/24/32更新点 p 漂移和初始/最终核对见 CORRECTNESS.jsonl；原始移动、整数码、固定网格参数、输入/C哈希、实际预算及提交状态见各依赖组 checkpoint。CPU 落盘审计重新加载全部本阶段 checkpoint，逐步重放修改，核对整数码、网格、packing、实际候选评价与更新上限，并确认历史来源哈希及冻结实现/协议哈希均未改变（OUTPUT_AUDIT.json）。完整模型的最终 pack/reload 与 PPL 迁移未验证。

## 成本

GPU 硬预算整体实测墙时 252.257 秒 /5400秒，包含预热、v41复现、pilot及所有已执行诊断；本轮无失败重试。同步分阶段合计 240.642 秒；未分项墙时 11.614 秒单列为调度/CPU/记录等开销，不能冒称 GPU kernel 时间。峰值 allocated 1.613 GiB、reserved 1.930 GiB。

pilot 主要阶段实测：C 构建 71.167 秒、候选评分 2.353 秒、接受更新 0.307 秒、直接重算 133.578 秒。所有成本预测单独保存在 COST_PROJECTION.json。

CPU 来源审计实测 23.439 秒，在 GPU 总预算开始前。历史正式 M2 校准/量化成本 2115.935 秒单列复用；本轮新增成本见 STAGE_TIMINGS.csv、COST_BILL.csv。候选评价次数不是完整 Linear/GPTQ 执行节省，也不等于运行时间。校准修复和 PPL 评测分阶段记账。

## 边界

局部重构下降不保证 PPL 下降；本轮不是 ReQuant 官方复现，也不证明论文原创贡献。一次评测不代表跨 seed 稳定。不读取 P3、不重跑 GPTQ、不再搜 alpha、不改 scale/zero/g_idx、无配对/残差目标/新局部 held gate。历史 v41 是旧文章探索，本轮正式评测未用于调参。没有把局部 MSE 转换成 PPL 预测。

Sol high 为用户请求；当前接口不能确认或切换模型设置，故不声明已确认。

## 交付

PROTOCOL.json、IMPLEMENTATION_LOCK.json、SOURCE_AUDIT.json、BASELINE_PROVENANCE_AUDIT.json、run_v42.py、V41_A_REPRODUCTION.json、STAGE_TIMINGS.csv、COST_BILL.csv、MODULE_RESULTS.csv、LAYER_RESULTS.csv、各阶段 ROW_RESULTS.csv、CORRECTNESS.jsonl、group checkpoints、COST_PROJECTION.json、DECISION.json；完整时另有 CHECKPOINT_LOCK.json、R checkpoint、R 两语料原始窗口和 PPL_COMPARISON.csv。复现命令见 REPRODUCE.md。
