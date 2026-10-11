# v25 稠密标量搜索收益、成本与迁移审计

**COMPLETE；跨 split 迁移 gate FAIL，STOP新增GPU试运行与整模型量化。**

## 直接比较

dense46相对scalar10 pooled held SSE改善 1.384531%；S0 3.006073%，S1 -0.417619%。S1失败，不能以总体正值替代用户要求的跨split一致性。
scalar10/dense46绝对pooled SSE 3844.57423449/3791.34492523，下降 53.2293092584。48单位W/T/L=20/12/16；median 0.000000%；worst -12.446746% (L23_up_proj_R224_S1)。
S0 W/T/L=11/6/7；S1=9/6/9。epsilon沿用v24，不重新调阈值。每单位16篇×2048元素，SSE=MSE×32768；此处绝对SSE不混称MSE总和。

- L0/up_proj：pooled 3.936049%；S0/S1 -0.594871%/8.658452%；W/T/L 4/2/2；median 2.150109%；worst -9.282963%。
- L0/down_proj：pooled -0.262864%；S0/S1 -1.253510%/0.800758%；W/T/L 3/1/4；median -0.363605%；worst -7.860984%。
- L12/up_proj：pooled 0.438047%；S0/S1 2.400267%/-1.573226%；W/T/L 2/3/3；median 0.000000%；worst -4.928758%。
- L12/down_proj：pooled 1.069923%；S0/S1 0.701731%/1.435940%；W/T/L 4/1/3；median 0.797134%；worst -2.955407%。
- L23/up_proj：pooled -0.487340%；S0/S1 1.576531%/-2.789183%；W/T/L 2/4/2；median 0.000000%；worst -12.446746%。
- L23/down_proj：pooled 3.011475%；S0/S1 4.922511%/0.842849%；W/T/L 5/1/2；median 1.425720%；worst -3.393108%。

## 集中度与oracle

- top1正收益单位解释正收益 28.0255%、净收益 59.8730%；移除后pooled gain 0.584732%。
- top3正收益单位解释正收益 66.5906%、净收益 142.2625%；移除后pooled gain -0.721250%。
- top5正收益单位解释正收益 79.3112%、净收益 169.4385%；移除后pooled gain -1.327434%。
净收益主要由L23/down驱动：去掉整个L23/down后仅约0.0753%；前三个正贡献是L23_down_R0_S0、L23_down_R224_S0和L23_up_R224_S0，全部来自S0。L12/down与L23/down的两split方向都正，其他四个target方向跨split改变；不据此挑层推广。
跨split配对outcome：`{"win/loss": 6, "tie/win": 3, "loss/loss": 2, "loss/win": 3, "win/win": 3, "loss/tie": 2, "tie/loss": 1, "tie/tie": 2, "win/tie": 2}`。top名单、逐target移除敏感度和全部负单位见DIRECT_SCALAR_AUDIT及DIRECT_UNIT_RESULTS。
scalar10/dense46 held oracle SSE 3791.744603/3647.18996474；oracle capacity改善 3.812352%。search选择相对各自oracle regret 1.393280%/3.952494%。
- S0：scalar10/dense46 selection regret 2.673094%/5.396013%；oracle capacity 5.511923%。
- S1：scalar10/dense46 selection regret 0.007843%/2.446263%；oracle capacity 1.972515%。

Oracle仅回溯诊断，不设计选择规则或挑层。更多候选的search最优不劣是集合包含关系；不能保证held/PPL更好。oracle存在潜力与search选择不能稳定泛化是两个观察，不构成因果机制证明。

## 实际协议、实现和正确性

Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`，FP16/SDPA/RTX4060 Laptop，TF32 off；W4A16 symmetric G128，block128、damping.1、desc_act=False、static_groups=False、true_sequential=True、mse=0，H/GPTQ FP32。
原生code[0,15]、zero8，round后clamp并反量化。group入口按当前compensated W估计range，再乘scale alpha；alpha1无hook。保留原补偿。10点=.1..1.0；46点=原10 union {.1+.9*j/37,j=1..36}，不是均匀.02网格。
两种scalar共用同一完整Linear候选、输入、pristine W、H及选择目标，唯一差异是网格。直接读取552个scalar候选/held sidecar，重算2208个行分数的选择与oracle；独立核对CSV及W/scale/zero/g_idx tensor hash，不通过structured反推。
代表层0/12/23，up/down都固定rows [0,224,448,672]（up不是四分位抽样，不能外推全行分布）。正式calibration为v18冻结128篇WT2 train×2048、seed20261005；本局部诊断沿用v22/v23/v24的64完整文章，每篇2048tokens；S0 search0..15/held16..31，S1 search32..47/held48..63。与正式128篇calibration hash无交集，不拆token、不重抽样。
输入是固定v18 M2 checkpoint前缀：v23捕获up输入，v24 replay block捕获down输入。目标W pristine FP16。不是dense46自身prefix，也不是正式dense模型迁移。
v18已经每alpha一次完整Linear产生所有行、全行评分、流式选择及W/scale/zero拼接，g_idx一致检查、一次完整提交；candidate独立clone。未发现逐行重复GPTQ或每32行full-prefix刷新。因此没有编造加速修复。
共享X/H和逆运算中间结果，但完整逆Cholesky因子尚未全缓存：core每candidate仍做damped-H lower Cholesky与inverse upper Cholesky，中间inverse已缓存。这是实际重复计算，但本轮未单独profile；不能将整个GPTQ时间当作可节省时间。gate失败后不增加新runtime优化。
run_block_capture当前运行完整block，所以qkv->o和o->up/gate捕获后的未使用suffix forward确实存在；依赖边界次数已合理，仍有partial-forward工程潜力。没有独立profile，不估造可节省比例。
qkv、up/gate各共享一个输入/H，原生处理order不产生sibling forward dependency。每block四input/H组；完整group提交后qkv->o、o->up/gate、up/gate->down；完整down提交后仅为下一block刷新。24blocks：96次H准备、168次Linear提交、72次block内capture refresh、23次down->next，128×95=12160次block forward。初始embedding/一次qkv norm另计，next qkv norm融合在down->next，最后block无需下游input，零row-batch refresh。
本轮correctness PASS仅指CPU原始数据/参数/协议审计；v18/v24历史runtime PASS可复用。新dense46 own-prefix alpha1/order/commit/propagation试运行是NOT_RUN，不冒充PASS。没有优化前后测速/张量对照，因为未改实现。

## 真实成本

正式10点1680完整Linear executions、3,041,280 row-alpha scores；46点7728 executions、13,989,888 scores。增加6048 calls/10,948,608 scores，候选执行数量4.6倍；35次历史native parity参考额外列账。
- v24 original10：120完整GPTQ / 480行search评分；GPTQ 162.807s，search 13.660s，held 14.901s，candidate IO 0.305s。
- v24 additional36：432完整GPTQ / 1728行search评分；GPTQ 574.341s，search 45.406s，held 45.092s，candidate IO 1.116s。
- v24 dense46：552完整GPTQ / 2208行search评分；GPTQ 737.148s，search 59.066s，held 59.993s，candidate IO 1.420s。

这是共享bank实测阶段，非两个方法独立总wall；v24只评分4行×16篇，不能用其评分时间外推全行×128篇。v24失败重试与预处理保留原RUN_REPORT，不计入scalar专属candidate时间。
正式M2校准wall 2115.935s：GPTQ+全行评分占 78.32%，dependency refresh占 6.19%。分阶段实测和未归因overhead详见COST_AUDIT，全部168个矩阵/形状详见COST_BY_REAL_LINEAR与COST_BY_SHAPE。
按已完成全部真实矩阵计时，仅候选/评分/选择放大46/10、固定成本保留：历史dense投影 2.246h；1.5倍保守 3.370h。较新v24匹配up/down完整形状实测GPTQ替换对应均值的混合1.5倍估计 3.613h。均非本轮实测或runtime cost gate PASS；own-prefix H、环境时变需新pilot核验。
预定block0 pilot历史估计 425.1s，1.5倍 637.7s（上限1800s）；正式新量化上限21600s不变。不是代表行时间乘全模型行数。
历史M2 peak allocated/reserved 2.187/2.500GiB；本轮无CUDA，新GPU峰值NOT_RUN。

## 门槛与正式PPL

PROTOCOL先冻结复算规则、跨split gate与条件pilot（完整block0覆盖896/4864输入维度和实际依赖）。S1失败，按用户“收益不稳定则停止重计算”停止：pilot、新held、own-prefix提交传播、dense整模型量化/PPL均NOT_RUN。成本不是停止原因；正确性审计PASS也不能覆盖质量gate FAIL。
以下已有正式基线从原始windows按valid_tokens加权复算，并核查revision、token hash和checkpoint SHA，未重复评测：

- M0 WT2 PPL 12.46340950；C4 fixed-budget PPL 19.24855582。
- M1 WT2 PPL 13.88436957；C4 fixed-budget PPL 22.23392445。
- M2 WT2 PPL 13.76055605；C4 fixed-budget PPL 22.03288316。
- M3 WT2 PPL 13.54525697；C4 fixed-budget PPL 21.44542796。

C4是固定131071计分tokens validation流，不是标准/full C4。dense46 PPL和相对M1/M2/M3变化未测，不推断。M3 asymmetric LWC梯度block目标无GPTQ，与M2不只网格不同，非纯搜索因果对照。校准wall和evaluation wall分账；M3精确校准timer缺失不作速度比。
E1 raw复算pooled held block gain -0.00012097%，verdict E1_BLOCK_OBJECTIVE_SIGNAL_NOT_SUPPORTED；E2仍保留不支持结果，两路线不扩展。

## 下一实验与研究/工程结论

建议结束当前扩大scalar网格的质量扩展。现有小幅pooled局部收益跨split不稳定，不能支撑新方法优势或PPL迁移；也不意味着所有scalar网格都无效。
一个有依据的下一实验：冻结10点M2，在完整up/down矩阵与正式128篇输入上做<10分钟的阶段profile，分别量两次Cholesky、逐列补偿、全行直接评分与IO。只有factorization确是主要成本，再验证完整upper-factor缓存parity及同协议计时。无需新held/PPL、不扩大grid、不用oracle制定选择规则；本轮未执行。
工程亮点是执行单位/完整矩阵成本审计、native dynamic候选隔离、row stitch/真实dependency、hash恢复、跨split负结果与oracle regret追踪。离线校准优化不叫在线推理加速。

## 复现

```powershell
& 'E:/quant-thesis/.venv-gptq/Scripts/python.exe' -B 'E:/quant-thesis/experiments/range_search_w4/results/range_w4_dense_scalar_audit_v25/audit_dense_v25.py'
```

仅CPU解析和hash验证，所有新增产物限v25；本次CPU分析wall 61.414s，不混入历史GPU实验cost。原始CSV/JSON、correctness、输入哈希、冻结协议与cache manifest齐全。CPU审计增加哈希检查时有一次编码定义错误，见AUDIT_ATTEMPT_FAILURE_01.json，已按历史JSON编码修正，数据未改；该失败pass计时未记录，因此不把本次成功pass时间声称为所有开发重试总wall。
