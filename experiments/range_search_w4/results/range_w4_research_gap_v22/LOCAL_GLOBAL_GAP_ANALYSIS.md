# 局部收益与最终likelihood不是同一种量

## 已验证

v19从原CSV复算：304,128行中283,818行search MSE严格改善，20,310精确持平；不加权的行相对收益均值12.826748%。这是search calibration上的指标；“候选包含native α=1”的保障只适用于同X/H/目标的search，不保障留出或PPL。

v18/v19冻结WT2 M2相对M1 PPL改善.891748%，C4 fixed-budget .904210%。v21.5 M3相对M2分别改善1.564610%/2.666266%。这支持本次M2有效、M3更好的方法级事实；不支持统计显著或跨模型结论。

**12.8%与.89%不能相除成收益转化率。** 一个是不加权row-relative search MSE平均，一个是全token likelihood的指数函数变化。行norm、absolute误差、projection大小与下游敏感性不同；search与test不同。当前没有量化逐项损失归因或统计转换模型。

## 理论上为什么幅度不一致

- Residual：局部误差与主信号、其他量化误差方向叠加；减小自己的norm可能破坏原先抵消。加法自身不是非线性，但已有误差产生交叉项。
- RMSNorm：径向变化可能被归一化弱化，方向变化可能保留或放大；W MSE/local输出MSE不包含这个敏感度。
- Attention：Q/K误差影响相似度与softmax权重，token之间耦合；同norm的不同误差方向可能有不同block影响。
- SwiGLU：gate与up的乘法及SiLU使某些通道/符号/样本更敏感，down重新混合通道。
- Sequential propagation：M2各依赖边界使用自己的量化prefix，已经处理部分输入漂移；但当前Linear目标仍是同X下的FP输出，不重建原FP trajectory或末端likelihood。
- NLL：最终lm_head/log-softmax对错误token方向敏感，并非隐藏态各方向等权。Block MSE也不是NLL，升级block目标仍可能无法迁移到PPL。

上述是结构性推断，**本轮未逐项测量其贡献**。不将其中任何一项写成v18 gap的已证实原因。

## 历史证据的边界

v13同一X的不同row可有−412%到+73%不同留出gain，支持error direction×geometry的重要性；它发生在Linear层，不能证明attention或RMSNorm是灾难来源。v15量化prefix局部signal保留，反驳“全部来自FP-input artifact”；没有证明prefix全局问题消失。v11长尾和集中度说明mean/pooled收益不能代替稳健性；v14更多coverage帮助native但未解决clipping风险。

M3不仅block目标不同，也有FP teacher prefix对齐、格式、容量、joint gradient差异；目前最合理的**工作假设**是目标敏感度与结构化容量共同作用，而不是精确分配因果比例。E1先隔离目标；E2独立隔离组容量；E3作为廉价反例测试粗网格解释。

模型级验证尚缺：受控方法的held-out block改善、全模型same-format ablation、WT2/C4原protocol评测、跨模型重复；任何新PPL评测都需下一轮授权。本轮STOP。
