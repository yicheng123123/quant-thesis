# Formal resume plan — 下一轮执行方案，尚未实施

## 依据与边界

此次修改应是runner correctness fix：官方GradScaler原本允许skip/backoff，额外runner gate和update计数应准确观察该控制流。保留官方NativeScaler/AdamW、autocast、loss、LWC参数化、LR、epochs20、样本/顺序、量化器、teacher和prefix。禁止手动指定scale、retry或补跑skip样本。

当前实际gate为post-scaler，未来要调整它；不能声称删除了不存在的pre-scaler fatal gate。原v21.1源码/锁/checkpoint/结果保留，用新目录、新runner与新审计锁记录最小差异。

## 必要runner修正

1. 在optimizer_init边界新建官方AdamW/GradScaler，如原生每block语义。observer读取unscale之后、update清空tracking之前的found_inf，并用实际optimizer.step hook统计成功调用，不依赖AdamW返回值（成功step通常也返回None）。observer不得更改梯度、返回值、scale或调用次数。
2. step callback仅在官方wrapper返回后检查：loss有限、scaler状态有限、所有实际参数有限、optimizer state有限。known found_inf>0时要求step跳过、参数/状态未变、官方backoff已发生；将非有限grad/norm作为overflow诊断。found_inf=0但owned gradient非有限/实际step状态不一致，应STOP。
3. 对实际成功step要求owned unscaled grads及更新后参数/state有限；不得直接optimizer.step或额外unscale。missing LWC gradients等可保留一致性gate；LET仍off。
4. epoch_cb不能再要求所有norm_mean有限：合法skip可能使raw norm_mean为NaN。保留loss/参数安全gate，另记skip数、finite norm统计和raw norm状态；JSON用null+finite flag表达非有限统计，不偷偷丢弃events。upstreamnorm是诊断量，不影响optimizer。
5. 移除最终finite_gradients==steps的绝对要求，改为iterations=successful_steps+scaler_skips，并核对61440 schedule。scaler state reset/growth均交给官方，不为保证固定successful step数增补sample。
6. 新checkpoint metadata分别记录completed_blocks、scheduled/attempted iterations、successful steps、skips、计数provenance。继承block20的53760成功step以历史推断标记，不冒充hook实测。若未来支持层内恢复，必须同时保存optimizer/scaler/iteration/RNG；当前只需block边界恢复，不能继续失败进程内的半成品状态。

## 恢复步骤

先验证原block20 model/omni/RNG/config/source/token哈希。按已有native resume逻辑重构已完成0..20的FP teacher与learned quant prefix；从checkpoint RNG恢复，fresh L21 LWC初始化、fresh AdamW、fresh dynamic GradScaler。不要加载本轮一次性diagnostic的三个更新。

从L21 epoch0/sample0开始完整官方20×128循环；sample0应消费一次，允许scaler skip，并在sample1使用backoff后的scale。结束L21后按原算法量化/传播/整block检查再commit。L22/23尚未运行，仍按原生fresh per-block scaler处理，不预先承诺不会overflow。

## 下一轮执行前的门禁

先以同一冻结fixture核对新runner与本轮four-sample trace的loss/scale/found_inf/step/参数状态，及正常样本路径不变；review所有新增observer和gate差异。冻结新runner hash与iteration计数定义，保留当前cost/memory边界并监测剩余正式校准成本。该计划只允许以后继续calibration，export/WT2/C4须按后续明确授权和既有验收顺序。

## 本轮状态

没有实现或执行上述runner修正。当前M3不完整，正式STOP保持。审计verdict中的RESUME_ALLOWED指“可作为下一轮的官方语义一致方案”，不是已恢复/已完成。
