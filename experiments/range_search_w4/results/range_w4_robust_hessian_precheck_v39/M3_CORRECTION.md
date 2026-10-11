# v39：v38 / v25 的 M3 记录纠正

## 结论：已解决，属于汇总遗漏

v38 的 EV02 引用了 v21.1 停止快照，遗漏 v21.4 续跑和 v21.5 最终化。不是两种不同定义的模型都被命名为 M3，也不是只有缺乏有效证据的 PPL 数字。v21.1 的停止记录当时有效，但不能作为后续项目的最终 M3 状态。旧报告保留，只在本目录纠正。

### 来源关系与运行身份

- 初始协议：`range_w4_omniquant_full_v21_1_M3`；保留前缀的 attempt ID 为 `20261001_183359_6c00bf87`。完成 block 0–20，53760 个已提交迭代；L21 epoch0 sample0 的有限梯度检查停止。v21.1 没有最终导出或 M3 PPL。
- 续跑协议：`range_w4_omniquant_resume_v21_4_M3`；正式完成 attempt ID 为 `20261001_234305_7e9e2ddd`。从上述 block20 的状态和 RNG 恢复，在完整 block 边界继续；最终 block23 提交时间为 2026-10-01 23:59:08+0800。
- v21.5 是已有训练、导出、评测的证据最终化，不是新的训练或评测 run。v25 读取其 `WT2_M3_FINAL.json` / `C4_M3_FINAL.json`，这两份文件又明确指向 v21.4 原始评测 JSON 和窗口 CSV。
- v38 EV02 停在初始阶段。EV01 中若将 M0–M2 当成整个项目唯一完整模型质量证据，亦不完整；M2 的原生 GPTQ 对照结果本身不因此失效。

### 模型与配置

Qwen/Qwen2.5-0.5B-Instruct，revision `7ae557604adf67be50417f59c2c2f167def9a775`，分析/运行 FP16。M3 是 OmniQuant LWC、W4A16、asymmetric affine G128，LET 关闭；不是 M2 symmetric 动态 GPTQ scalar10。

初始和续跑的模型、校准 manifest、ordered token hash、训练 config、量化定义一致。128×2048，20 epochs，batch1，LWC learning rate0.01。训练 config SHA256 为 `c9d40132698c1dda4fcaf8f0502f1eb148bb608c80fb25792dc3fcb0b509d09a`。

两阶段 source-audit/runner hash 不相同，不能冒充完全相同的运行文件。实际 `OMNIQUANT_PATCH.diff` 增加 block-boundary 恢复、前缀 replay、诊断 hook，并把检查放在 native GradScaler 处理之后；未改变量化器、损失、AdamW 更新数学或原生 AMP 操作次序。续跑有两次被 native GradScaler 跳过的迭代。61440 scheduled / 61438 successful 的历史前缀部分计数属于旧证据推断，不是完整原始 optimizer-hook 实测。

### 本轮独立核验

`M3_SOURCE_RECONCILIATION.json`、`M3_EVALUATION_RECOMPUTE.csv` 和 `CORRECTNESS.json` 保存检查结果；`SOURCE_MANIFEST.csv` 保存逐文件来源。

- 初始 block20 manifest：`950421d4b4954fc8512300cbb5683665e7455afedc457fbf89f937fc42847411`。核对历史与隔离复制的状态/RNG 文件哈希一致。
- 最终 block23 manifest：`3641627bf11979d159b769123eeaf5dabdea2ff35475fb77f30b66900fcf9366`；状态、RNG 与 latest pointer 一致。
- 168 个 Linear 的 dense-dequant FP16 export 已存在；逐一核验 export manifest 所列文件。`model.safetensors` 文件 SHA256：`168ddbfb526cd87d458af7648d12768f1bba099d270e41ed9aa2bd68d143fbf6`，988097536 bytes。其文件哈希不同于原评测保存的多文件聚合标识 `bd9c11ba...`，不混淆这两种标识。
- 原始窗口 CSV 与 JSON 的每个窗口边界、loss/NLL 一致，index 连续。用 `sum(window_nll)/sum(valid_tokens)` 重算，再指数化；不得平均窗口 PPL。
- WT2：582 windows，299077 scored targets，2048 context / stride512；PPL **13.545256974943815**。评测时间 UTC 2026-10-01 16:09:23.906638 至 16:13:05.119049。
- C4：253 windows，131071 scored targets，131072 input tokens 的冻结预算；PPL **21.44542796230099**。具体原运行时间见复算 CSV。不是完整标准 C4 评测。
- token manifest、revision、special-token 关闭、chat-template 关闭与 v18 来源一致；v25 已引用的数值与本次复算一致。没有新 PPL 运行。

### 纠正后的使用边界

可以引用已完成 M3 的上述正式指标。它低于同窗口 M2 的 WT2 13.760556051038424 和 C4 22.032883164065005；这是方法整体比较，不能把差异归因于单个范围变量：M3 的格式、可学习 LWC 与训练目标均不同。

M3 完整校准 perf-counter wall 未持久化；旧报告的前缀/续跑时间跨度不能用于精确 M2/M3 成本比。本轮没有重新验证训练张量数值或前向传播，只复核既有训练/导出证据、文件身份和原始评测汇总。M3 冲突已解决，不标为 UNRESOLVED；历史逐步计数和严格成本的证据缺口仍保留。
