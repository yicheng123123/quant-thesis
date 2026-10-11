# 毕设定位与章节建议

## 建议标题

**动态 GPTQ 的输出行级范围搜索：量化质量、离线校准成本与迁移边界评估**

可加副标题：以 Qwen2.5-0.5B-Instruct 的 W4A16 对称 G128 为例。标题描述实证评估，不声称提出新算法。

## 研究问题

1. 保留 GPTQ 动态 G128 范围估计与原误差补偿时，alpha10 候选和逐输出行选择是否改善模型级质量？
2. 质量变化对应多少额外离线校准成本？
3. 局部 MSE、网格密度、选择数据变化的收益能否稳定迁移？
4. 理论等价 scorer 在当前浮点路径下能否保持选择与提交张量一致？

## 章节结构

### 第 1 章：问题、范围和研究问题
低比特权重量化背景；固定 W4A16 对称 G128、动态 GPTQ 和原 compensation 语义；区分离线校准与在线推理。

### 第 2 章：GPTQ 与范围搜索的实现定义
量化器参数、动态 group range 时机、候选执行和 row-alpha score slots 的区别；完整 Linear 候选生成、逐行 argmin、whole-Linear stitch。

### 第 3 章：冻结实验协议
模型 revision/dtype/device、128 篇 WT2 train 校准、文章/token hash、M0/M1/M2、候选隔离、alpha=1 parity、评测窗口和复现方式。

### 第 4 章：M0/M1/M2 正式整模型对照
24 层/168 Linear 覆盖；WT2 和 C4 fixed-budget PPL/NLL；M2 对 M1 小幅改善、仍逊于 FP16、校准 wall 增加。

### 第 5 章：候选密度和选择数据迁移
dense46 对 scalar10 的 held SSE、S0/S1、逐单位胜平负、尾部、集中度与 oracle gap；v31 仅作已查看数据上的回顾性诊断。

### 第 6 章：评分等价性与离线工程成本
二次型公式、v27 局部验证、v28 row 884 浮点选择差异、v29 direct-dependent trigger；分开 scorer-stage、full calibration 和 inference latency。

### 第 7 章：扩展路线负结果
weight-error penalty、固定 D、shared coordinate、official AWQ selector+GPTQ；说明 v37 修复后 S0 负结果、v36 旧 reference 无效及 S1/P3/PPL 未运行。

### 第 8 章：结论和限制
单模型 M2 结果、复现工程价值、局部迁移边界；不声称新算法、普遍泛化、PPL 加速或推理加速。

## 写作限制

- M2 整模型 PPL 是已完成结果，不写“没有整模型结果”。
- C4 必须标为 fixed-budget validation。
- v27/v28 scorer-stage 省时不得写成完整校准加速；v29 尚无独立 direct-free hybrid run。
- v30-v33 和 v37 S0 是回顾性/已查看数据，不是新盲测。
- M3 是未完成训练尝试，不纳入质量排名。

