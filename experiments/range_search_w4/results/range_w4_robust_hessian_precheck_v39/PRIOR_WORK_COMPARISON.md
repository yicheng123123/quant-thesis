# RQC / SARQC 查重与区别

查阅主来源：[Saliency-Aware Regularized Quantization Calibration for Large Language Models，arXiv:2605.05693v2](https://arxiv.org/html/2605.05693v2)。版本页记录v1于2026-05-07、v2于2026-05-08；本轮访问日期2026-10-05。这是作者预印本，不添加未核验发表信息。

## 已有内容（主来源定位）

- Eq.7：RQC 的 reconstruction SSE＋λ||Q−W||F²；Eq.8：SARQC 使用λ||(Q−W)S||F²，S=I恢复RQC。
- Theorem3.1：固定有限量化类、iid输入向量、有界输入范数下，给出与权重偏差半径相关的统一风险界。Corollary3.2 的受约束/惩罚最优解对应需要 supportedness，不是任意有限候选都自动成立。
- Eq.9及AppendixE.2：scaling候选的 regularized选择，GS实现使用两项min-max normalization，λ由校准训练/验证数据选择。
- Eq.10 / Algorithm2：GBS改用XXᵀ＋λSSᵀ，改变量化补偿矩阵，不能等同于只选择固定GPTQ候选。

## 本仓库对照（以下是本轮推导/判断）

v30/拟议v39的未加权标量目标在S=I、统一SSE/MSE尺度后与RQC代数一致；λ=Nρ。把它解释为谱球worst-case，是同一惩罚的支持函数解释，不构成新的候选、新的目标或新的算法证据。

本仓库搜索变量是动态GPTQ的共享clipping alpha，不是论文GS的通道scaling alpha。候选事先生成后固定，ρ只改变selector；GBS中的regularization会参与候选生成。两者不能因为目标类似就被声称有相同数值结果。本轮不复现论文或引入saliency penalty。

论文GS两项min-max若范围非零，与 b+ρm 的等效ρ=λ·range(b)/range(m)（单行的未归一化SSE中再转换N）；其数据驱动相对尺度不是谱不确定半径覆盖证明。范围为零时必须另行定义，不能忽略退化。该操作不是本轮提出的规则，也不实施。

Theorem3.1不能直接把当前n写成token总数；本文的输入向量假设与文章内相关、候选由同数据GPTQ H产生、固定M2前缀的统计关系不同。有限候选bank生成后大小10，不能自动替代事前假设类的大小；除非独立生成/统一覆盖或稳定性分析另有证明。

可能构成实质新增的内容：在明确文章级总体风险下，为此固定动态GPTQ候选selector构造有依据、可校验的search-only矩阵半径，并验证其事前预测。仅“ρ因矩阵而异”、经验分位数、改名Robust Hessian或重复线性惩罚都不足以构成该新增。本轮未完成这些条件，也未进行穷尽全领域的新颖性检索；不能声称谱球推导或方法原创。
