# 共享α与有限结构化group α

M2：每行1个α，实际group range仍动态。输入维896的q/k/v/o/gate/up有7组；down输入4864有38组。各组weight distribution、X_g几何和补偿后权重可不同，同一α不是同一绝对range。

M3最终参数有2,795,520个row/group×上下端点=5,591,040个标量；M2有304,128个行变量，标量数约18.38倍。这里是可调系数数目，不代表有效统计维数、模型新增权重、同等参数语义或优化调用数。

## 新只读证据

M3组间端点平均系数跨度（`mean(lower,upper)`再对group取max-min）全行median=.06415，p95=.37016；54.63%行>.05。down median=.23585，v=.25721，gate=.02791。L0 median=.00728，L12=.02889，L23=.22241。显然存在row/group和层结构；但跨度随组数增长有天然机会效应，down38组不能直接与7组按max-min公平比较。标准差同样列入JSON：down median=.04894，v=.08681，gate=.00917。

端点不对称平均绝对系数差=.01376，仅7.15%组差>.05，不能推论全部需要二维搜索。更少自由度可能捕获部分模式，但上下端点不对称能力与组粒度必须分开评价。

## 推荐容量阶梯

1. 保留symmetric native dynamic G128 GPTQ，只给每行 **2个连续region α**。7组按前三/后四；38组按19/19；不能按观察结果挑分区。
2. 从共享α解开始，逐region搜索原10点，包含当前解，最多2个固定sweep。原始候选之间都从pristine该行W完整重跑GPTQ。
3. 仅当2-region独立held-out通过才考虑4-region；本轮不运行也不授权扩展。4-region增加容量与候选调用，应另行冻结。

完整任意group α组合为10^7或10^38，禁止联合穷举。2-region全枚举也有100组合；先用有限坐标预算诊断，若效果没有超过46点评价预算的共享α对照，不将差异归因于容量。

## GPTQ交互与缓存语义

当前组α改变补偿残差，后续组的当前W、MinMax、Q都可能变化。更新是 **修改range后从pristine W重跑完整GPTQ**，不是仅换当前组Q。H/Hinv可复用；前一候选working W或error state不能复用。不能用仅更新当前group残差的缓存公式评价这种更新。每个已接受的配置保存α向量、actual scale/zero/g_idx、全行输出loss和实际候选次数。

局部 `sum_g ||e_g||²` 与 `||sum_g e_g||²`排序可能不同，容量试验使用全行output MSE，保存cross term但不将它等同于非线性block interaction。E2与共享α保持相同objective/format，避免混入H1。

更大容量可适应search特有H及error方向，v11–v14已提示泛化风险。数据不按token拆分，固定完整文档留出，报告负尾/层投影覆盖/预算；M3存在参数差异是H2的动机，未验证纯搜索2-region的效果。
