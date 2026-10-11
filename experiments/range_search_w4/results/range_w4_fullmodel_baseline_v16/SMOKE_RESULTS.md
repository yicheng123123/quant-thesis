# 本轮实际 smoke：仅预算 core，非 OmniQuant compatibility

`BUDGET_SMOKE.json` 保存4个同步调用、70个行-alpha slots：L2 q/down前32 pristine行，v15 QPREFIX H，alpha=.9，B3/B32。脚本内wall总3.857s（含cache读取、检查和同步调用，不含Python/torch启动）；四次candidate同步区间合计3.401s；GPU RTX4060 Laptop。两宽度首三行最终FP16权重均exact相同，W/Hinv未修改，全部finite；随机64-token direct/Gram loss最大差1.863e-9，allclose PASS。它不验证新calibration数据选择、不验证全部projection/alpha1/native parity，不代表OmniQuant port通过。

OmniQuant Qwen forward/block-training/reload/tinyPPL **NOT RUN**。本轮没有任何完整模型量化、OmniQuant训练或PPL。CPU生成校准/test/C4 token清单不属于quality experiment。

Tokenizer对拼接test/C4长流发出超过model_max_length的警告；本轮只生成token清单，后续所有forward窗口最多2048tokens，不会将299,078或498,741个tokens一次送入模型。

准备中的两个软件错误：collect_audit最初假定v15嵌入historical_blacklist，KeyError后改为v14完整历史+v15新文章；budget_smoke最初误用weight key，实际cache是weight_fp16，KeyError后修复。这两次都发生在对应新实验计算之前，无候选/冻结结果被覆盖。C4首次网络读取遭sandbox WinError10013；经针对该bounded official-data命令的审批后成功。git默认Schannel凭据错误后改用git的OpenSSL后端，官方clone成功；大小写图片碰撞已披露，不涉及算法源码。
