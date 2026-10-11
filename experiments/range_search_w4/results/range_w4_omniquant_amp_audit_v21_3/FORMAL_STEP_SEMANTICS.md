# Formal iteration 与 successful optimizer-step 语义

## 官方20 epochs按样本循环定义

upstream quantize/omniquant.py:251–254是`for epochs in range(args.epochs)`和`for j in range(args.nsamples//args.batch_size)`。没有成功step计数条件、重试队列或skip后补跑。同一block optimizer/scaler跨sample/epoch保留，下一block重新创建。

当前配置24 blocks×20 epochs×128 samples/batch1 = **61440 scheduled training iterations**。successful optimizer steps = iterations − scaler skips；不保证61440成功更新，不可为满足旧计数标签额外补跑sample。

checkpoint persistence是另一含义：只有完成整block并通过参数/量化检查才commit。诊断中的成功step不等于持久checkpoint更新。

## 历史block0–20的53760

主attempt `20261001_183359_6c00bf87`，逐行核对21×20×128=53760个prefix iterations，layer/epoch/sample顺序唯一且完整。每行loss有限，14个owned LWC grads在native scaler之后全有限；LET off，optimizer没有其他参数。按exact native scaler控制流，可推断这些iterations都成功step，因此本已完成prefix下iterations与successful steps均为53760。

证据限制：历史没有保存step hook/found_inf/scaler snapshots，**53760 successful steps是基于完整gradient日志与源码语义的推断，不是新拿到的直接step计数**。没有证据支持prefix曾发生owned-gradient overflow，也不把它扩展为所有未优化intermediate均无Inf。

training_epochs主attempt420行，loss/norm/参数有限。主attempt第53761行是L21/sample0失败，未计入block20。最初checkpoint IO失败attempt的2560行也不属于当前持久state，不能相加。

## 原manifest标签问题

runner save_block_checkpoint:307直接写`completed_layers * EPOCHS * SAMPLES`，不是optimizer.step计数器；FORMAL_PROTOCOL_LOCK的total_optimizer_updates=61440也是schedule乘积。历史prefix没有skip时标签数值恰好相等；允许dynamic skip之后标签的普遍含义必须修正。

未来新版本同时记录：attempted_iterations、successful_optimizer_steps、scaler_skips、checkpoint_committed_blocks，并保持每sample只消费一次。历史manifest/协议锁不修改，在新审计和新resume metadata中解释其旧标签。源文件、CSV SHA见SOURCE_SHA256和PROTECTED_HASHES；逐行核验结果见HISTORICAL_UPDATE_COUNT_AUDIT.json。
