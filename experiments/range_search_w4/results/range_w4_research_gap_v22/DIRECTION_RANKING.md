# 十个历史方向的重新排序

依据是当前质量证据与下一步可判别性，不按概念新颖度评分。

| 历史方向 | 当前定位 | 原因与下一步 |
|---|---|---|
| 6 downstream/block-aware objective | **PRIMARY** | 可在同candidate/格式/行粒度下直接检验H1；触及local-to-model敏感度，先staged单行诊断，不能一步联合全block |
| 7 structured group-wise α | **BACKUP** | 新M3参数确有组内变化，2-region保留pure search和GPTQ；须独立held+匹配score预算，防容量过拟合 |
| 3 adaptive coarse-to-fine | 排第三，先固定dense诊断 | 成本低、可能反驳主假设；hist不足以证明主瓶颈。暂不实现adaptive，以固定网格避免额外适应性混杂 |
| 2 quantization-state candidate generation | 后续容量/工程选项 | 状态断点可能优于均匀α，但dynamic compensation使断点随轨迹变化；现无same-budget held优势，先E3 |
| 10 compensation-aware cheap proxy | 以后block成本确实高时 | M2已实际完整GPTQ评分，当前不是RTN误排造成的主缺陷；proxy需先匹配真实block ranking/heldout |
| 4 staged calibration scoring | 工程支持项 | 35min已可运行；若blockforward成为瓶颈再profiling，不能拿少文档proxy改变已冻选择语义 |
| 8 informative calibration selection | 机制辅助，低优先级 | v13指出方向×geometry，但heldout方向不能作为search-time oracle；v14否定简单增加coverage，选择新规则需独立validation |
| 9 transfer/hot-start | 工程支持项 | v15说明信号保留但α仅68.8% exact，一步内97.9%；不能保证跨层/模型复用，warm start不能省掉必要候选验证 |
| 1 GPTQ trajectory reuse | 暂不主攻 | 已缓存α无关H逆，mutable trajectory依赖α，复用会改变动态MinMax和补偿；缺少质量研究增量且正确性风险高 |
| 5 robust multi-article objective | 暂不主攻 | v13灾难case search文章高度一致，投票不能识别未见方向；robust目标不一定无效，但没有主要故障机制支持 |

## 为什么不回退到旧路线

Gate v1独立validation FAIL，不能用原gain/margin重新调阈值当新研究。更多calibration改善nativeGPTQ却未系统修复clipping风险；文章一致性不是v13灾难的主要可见预警。M2全模型仅35.27min且工作单位已纠正，纯加速不足以回答M3质量gap。若后续block loss成本升级，profiling/缓存才成为从属工程问题。

实际runtime瓶颈并非完全不存在：v18 candidate938.43s、scoring718.77s占校准主要部分；工程可优化评分/数据生命周期，但不能为了加速偷换目标或补偿语义。当前主要科研问题是质量与解释，而非不存在的“304万GPTQ调用”。

联合projection搜索暂列H1后续而非第三个独立大方向：qkv和gate/up互作有结构理由，没有受控收益证据，先objective ranking诊断，再有限2×2 interaction，避免把目标和容量一起改。
