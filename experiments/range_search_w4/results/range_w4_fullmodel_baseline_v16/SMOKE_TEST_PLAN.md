# 下一阶段 compatibility smoke 与正式启动 gate

这是一份冻结计划，**不是PASS记录**。本轮只执行budget_smoke.py和CPU清单生成；Omni-Qwen port尚未实施。新任务目录使用独立v17（或用户后续指定目录），历史v8-v16不覆盖。

1. 环境：独立Omni环境，固定torch2.6.0+cu124/transformers4.49.0/tokenizers0.21.4，其他直接依赖从官方manifest解算并完整lock；先import QuantLinear/omniquant/Qwen，不改GPTQ环境。若本机full entry import缺termcolor/einops等，隔离补齐并记版本。real_quant/Triton不属于primary smoke。
2. architecture patch：只实施QWEN25_PORT_PLAN白名单。exact checkpoint在CPU加载，只L0在GPU。native/adapter关量化同token32/64 forward，比较hidden/logits；q/k/v bias和14/2头、RoPE、SDPA、mask/cache_position传递不变。附tinyCache generation一致性检查，正常cal/eval禁用cache。
3. 官方LWC kernel path：L0七Linear，W4A16 asym G128，官方bound初始化4、official optimizer/STE。2个共同cal样本的前64tokens，1个epoch/2steps，仅smoke-budget override；检查只有bound_factor有梯度、step更新、fpweights/norm/bias没有梯度、loss/weights/grad finite。明确这不是正式128×2048×20epoch run。临时量化/inplace输出有限，abits16 identity。
4. teacher/student prefix：L0 smoke使用同输入；再用32tokens的两block纯forward验证teacher来源FP、student来源本方法量化prefix，不需要第二block训练。保存trace和prefix/input hashes。不得把shared FP input误当全模型M3流程。
5. fake export/reload：L0已量化七Linear转native dense，其他blocks原FP，保存独立partial checkpoint与quantized module inventory。reload保持embedding-head tying；仅7个被量化模块，排除head/norm/bias。output allclose、no NaN/Inf。model不带未初始化的QuantLinear临时状态。tinyPPL128tokens/127targets，只做smoke不入正式表。
6. M2：固定L0组顺序，q/k/v/o/up/gate/down每模块先alpha1与native同W/H/init比较group entry、scale/zero/g_idx、Q_FP16、direct loss。dead-column/allzero smoke按native处理。外部alpha.9确实影响dynamic group scale，候选mutability不存在。
7. row batching/选择：共同cal新X上固定first32 q/down rows，全部十alpha，只做cost/selection正确性；candidate-state独立，FP16 selected Q对应其scale/zero；B3与B32 spot一致。direct delta loss与直接linear差分以及undampedGram allclose；near tie direct ranking复算。记录每阶段sync time和峰值，再预算全矩阵，不增加grid或用eval loss选择。
8. packing separate smoke：32输出行的一个projection导出标准GPTQ字段，检查整数code重建、group_idx、scale FP16 roundtrip和qzeros format，reload TorchQuantLinear eager post_init。记录dense/packed forward差异，不能预先要求bitwise相同或当作已经兼容。无Triton则走现有eager，不能为性能修复改变算法。

通过标准：各项finite、原生未量化架构比对rtol1e-5/atol1e-6，alpha1沿用v15 exact FP16/group参数规则；direct/Gram按1e-8+1e-5|reference|，ranking检查显式记录；export dense Q必须逐tensor相同，tiny127计数正确。若本机shape/quantizer差异致任一标准不成立，先输出mismatch，不跑formal。预算micro与正式training override分别序列化，不用smoke配置冒充官方baseline。

成本门槛：GPU峰值<7GiB，M2包含新cal actual loss的外推<=48h；M3依据实际one-block step+forward成本外推<=24h。超预算只报告STOP，可调整batch/offload/同目标缓存实现后重新测量，不能删候选/epochs/rows。正式启动需后续明确执行任务且smoke gates已通过，本轮不启动。
