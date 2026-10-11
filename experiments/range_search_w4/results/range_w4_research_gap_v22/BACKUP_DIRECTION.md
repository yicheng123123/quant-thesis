# BACKUP_DIRECTION（冻结）

**Limited structured group-range search：每行2-region α，保留symmetric dynamic GPTQ和local output objective。**

触发：E1按冻结范围未支持目标方向，或通过correctness但成本gate不允许block评分。只能在下一明确授权任务按E2运行，本轮不自动继续。

动机：M3最终参数确有组间非均匀性，M2共享α缺少该容量；但这是跨格式/目标的描述证据，不能声称2-region已经更好或能替代OmniQuant。2-region以较少参数检验H2，比自由group α更容易控制预算与泛化。

固定分区、原10点、两sweep、从shared解初始化、实际完整GPTQ重跑；保存所有接受变化。与原10点及46-score共享α对照比较，报告search/held/负尾、score/真实调用/wall/memory；只有held-out和预算对照均通过才考虑4-region。group交叉项可以作为诊断，不作为未经正确重算的更新缓存。

它与PRIMARY保持正交：E2不升级block loss、不切asymmetric、不新增gradient/rotation/激活量化。若两方向都不支持，STOP并报告证据，不临时改目标或引入其他模块。密网格E3仍是便宜的判别项，不能未经结果自动升为新主方向。
