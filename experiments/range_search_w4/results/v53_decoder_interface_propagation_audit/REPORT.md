# v53 REPORT

**接口修复与独立传播验证通过。block0可限定复用；v52 layer1以后必须重建。**

## 修复与验证

新实现为独立`runtime_v53.py`及`decoder_interface.py`。仅支持冻结flags下的Tensor[B,T,H]和合法单元素tuple；未知类型明确报错。提取返回原Tensor，不改内容/dtype；所有embedding、qkv、hook和block刷新缓存先验证完整shape/dtype严格相等，再写入，禁止广播。

最小diff仅改变embed/capture/advance_block。其他函数/类AST保持一致；C/p、固定整数网格、候选顺序、32更新上限和评价预算、v52初始化FP64复核与接受规则未改变。相关源码与验证协议已在GPU前冻结，之后hash未变。

21个确定性用例通过，含可区分token的Tensor/tuple、首token误取、shape/dtype错误、未知返回、非有限值和拒绝后缓存不变。

真实参考由模型原生`model.forward`和输入prehook独立取得，参考路径不调用新提取/缓存函数。每次保留完整2048 token和全部24 Transformer；仅lm_head输出选择最后一个位置，减少不影响层输入的验证开销，不计算loss/PPL。

- 正式128篇，按原M2→qkv→o→up/gate→down重建block0状态：4组全部输入、sibling共享输入和历史source hash逐字节一致。
- 原M2＋4个已提交block0组，正式128篇block0→layer1：完整hidden及qkv输入与原生forward逐字节一致；旧广播边界被拒绝。
- 预冻结文章0/64/127，各完整2048 token，24层hidden和qkv输入对照全部逐字节一致。该24层逐输入比较覆盖3篇；不能改写成128篇×24层全部逐输入对照。
- 原code/grid/packed和实际FP16权重与v51/v52继承状态一致。仅在内存重建锁定参数；block0以外参数版本未变。没有新模型checkpoint、C构建、修复、held或PPL。

## 历史影响

`CALLSITE_INVENTORY.csv`列出30个项目危险取法（含分步索引和经norm别名的等价路径），archive角色单列；无调用关系的vendor及hook tuple/batch取法另表，不混为decoder错误。

v51/v52广播有历史输入/边界内容和环境证据，layer1后传播无效。v18/v28/v29/v40记录4.49，支持tuple接口，未发现同类Tensor误用证据。v45/v46/v49保存元数据支持4.49，但历史直接返回类型缺少日志，全链校准有效性仍为UNKNOWN，不能仅凭当前重放认定有效/无效。v36/v37和v48 L3 replay相应不确定性列入影响表。

5个历史模型的10份旧PPL核验通过：实际checkpoint SHA、result/window SHA、连续窗口、manifest/token与valid-token加权NLL一致。可继续作为对应真实模型的测量值使用，不自动证明其校准传播符合算法协议。不要求仅因本次发现重评同一旧checkpoint；也不使用旧PPL替代未来新模型评测。

## 可执行恢复边界

从**原正式M2**重建完整模型，仅装入`VALID_RESUME_BOUNDARY.json`列出的v51 block0四组，跳过重复修复；用v53新接口重放本次完整prefix。期望正确hidden raw hash：`9488e56e397608a5d19291595f43645135ad6b6ed477cd6d97eeba7398bfeed6`；正确layer1 qkv raw hash：`70cd41104db3769a3d434c27d4612a41becf4be98439c6328c6010e5b6eae7ac`。

随后从**layer1/qkv，原M2 code/scale/zero/g_idx**开始。不得载入v52 layer1–23的已修复组、诊断最终模型、旧广播边界或失败tentative状态。若来源/网格/新边界不匹配，停止，后续从原M2全部重新开始。v53验证缓存只是对照证据，不自动成为新生产activation缓存；未来仍从自身模型重放。旧boundary预期必须由独立正确记录替换，并在下一任务冻结，不改历史报告。

本轮未启动下一正式运行。剩余时间不等于正确输入下完整修复一定能容纳；必须依据正确传播另行作保守成本检查，不用v52错误输入上的执行时间承诺速度。

## 成本与交付

- 原共同账本精确携带：5278.453026s（用户显示5278.453）。
- v53新增验证活跃作业：**398.975s / 600s**，包含预热、载入、IO、来源、全部比较；实际643次原生完整forward，不作PPL。
- 共同累计：**5677.428s / 10800s**；剩余**5122.572s**。未重置旧657.305s或任何失败。应用等待不计入。
- CPU首次扫描到停止请求220.547s，进程CPU145.922s；收窄后完成审计147.354s，确定性测试0.043s，另列。前两者与验证wall可能重叠，不相加冒充GPU时间。只停止本轮CPU审计任务，未结束自动重启的辅助进程。
- 阶段实测`GPU_VALIDATION_TIMINGS.csv`，内部嵌套打点`STAGE_TIMINGS.csv`；不要把嵌套表相加冒充exclusive total。预测552.25s与实测分开。

交付：最小diff/接口契约、冻结lock、确定性和独立传播记录、30处callsite与历史影响/PPL来源表、纠正说明、恢复清单、输入hash/shape/dtype锁、原始阶段与预算、复现命令。历史来源前后hash均核验，报告和checkpoint保持只读。

沿用已授权可用执行入口；本轮未重测普通沙盒或扩大环境修复。Sol high是请求设置，接口不能确认实际设置。完成后停止，不自动新版本/整模型/PPL，也不声明质量提升或理论保证。
