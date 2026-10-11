# v26 因子缓存作用域

## 已存在的 M2 缓存

v17.3 `prepare_native_state(H, nsamples=128)` 对一个完整 Linear 的同一份 FP32 Hessian 预先计算 dead-column 处理、0.1 damping、lower Cholesky、`GPTQ.block_cholesky_inverse` 的中间矩阵。每个候选均从 pristine FP16 W、独立 `GPTQ` 实例和 `core.H = prepared.hessian.clone()` 开始。原生 `GPTQ.quantize()` 随后仍执行自己的 dead/damping、lower Cholesky、最终 upper Cholesky、动态 G128 `find_params` 和逐列误差补偿。中间 inverse 在每个候选调用中被缓存返回，未重算。共享中间矩阵不应被候选修改；本轮同一候选重复运行结果完全一致，且原始 W/H 哈希未变。

既有缓存的正确复用条件是同一完整 Linear、同一原始 H 的字节内容与 shape、FP32 dtype、同一 CUDA device、128 个校准文档/`nsamples`、相同 dead-column 规则、damping 0.1、相同列排序（`desc_act=false`）、block size 128、G128 分组和 GPTQ 实现版本。任何输入/H/设置变化都应重新构建。若要跨进程持久化，必须把这些信息及源码哈希写入键和 manifest；现有 M2 只在当前 Linear 的候选期间内存复用。

## 计划中的 upper factor 缓存：未实施

冻结的 `PROTOCOL.json` 记录了成本门槛通过时才会启用的设计。该门槛未通过，因此 v26 **没有新增 upper factor 缓存、开关或缓存版量化路径**，也没有进行缓存版/原版 parity、命中率、增量显存或速度对照。不能把原有中间 inverse 的命中称为新 upper-factor 命中。

若未来另行验证，upper factor 还须绑定 damped H、dead mask、dtype/device、列顺序、block Cholesky 实现和所有上述条件，保持只读，并从 pristine W 创建每个候选。对 `[896,896]` 和 `[4864,4864]` 的 FP32 upper factor，单张量理论占用分别约 3.06 MiB 与 90.25 MiB；这是大小计算，**并非实测新增峰值显存**。
