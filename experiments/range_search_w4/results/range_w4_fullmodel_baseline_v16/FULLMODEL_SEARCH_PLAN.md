# Full-model Pure Search integration

固定 per-output-row shared-alpha，grid=.1,.2,...,1.0，无Gate、梯度、learnable parameter、proxy筛选或重要行子集。沿用v8 dynamic_core.py；新增full-model runner与processor写在新执行目录，不改历史core/run。Native G128/block128/damp.1/desc_act=false/static_groups=false/mse0；每个group入口基于此前compensation后的global W重新find_params，再scale乘当前row共享alpha，zero8保留native特殊fallback/all-positive语义。不是原始W预计算固定range。

## 原生顺序与自己的量化前缀

GPTQModel2.2.0实际调度是 `looper/module_looper.py::ModuleLooper.loop` 与 `GPTQProcessor`；BaseGPTQModel中旧循环不是新增实现的依据。每层groups严格 `[k,v,q] -> [o] -> [up,gate] -> [down]`。同组共享当前输入，先capture该组全部成员，再为每个成员独立build/复用该输入H和搜索；完成全部成员后才更新组状态。q/k/v和up/gate不能在组内错误地相互改变capture输入。

layer0输入从原模型embedding来。组0 capture时该层七Linear仍pristine；组1 capture时k/v/q已经安装M2自己选的Q；组2前o也已选定；组3前up/gate也已选定。最后把当前block的**M2最终量化输出**传给下一层。M1另一次运行传播自己的native alpha1 prefix。不要借用v15的native-D prefix、FP输入或另一个方法的H。当前target原始W是FP16 snapshot的该模块原权重（不受其他模块compensation），每个候选从同一W.clone和同一Hinv开始。

## 行可分与状态

当前配置下Hinv与row无关，列误差补偿 `E @ Hinv` 按输出行分别作用，per-channel find_params按row各自reduce，无跨row项。因此一行共享alpha可在共同H上独立选择，再拼装每行最终Q、scale、zero，等价于对相应行分别完整GPTQ；不联合搜索各组。B=32按row分块、alpha顺序执行；每候选完整遍历所有输入列，禁止只重跑一个group。alpha1在本模块**同一M2 prefix和H**下是搜索集合中的本地native点，不能直接复用另一个M1 prefix计算的alpha1。

候选结果最终Q_FP32 cast FP16，搜索loss使用实际Q_FP16.float-W_FP16.float；bias未变所以抵消。目标=mean square(Xdelta)，calibration唯一selection来源；exact loss tie选更大的alpha。每模块报告本地alpha1/search-selected loss和alpha分布，但不能由本地不劣声称最终PPL不劣。全模型PPL只在全部方法冻结后统一计算。

`native_hinv_from_documents`目前遇到dead columns会raise；full-model runner必须补齐原生dead处理：记录原始W_ref供输出目标；diag(H)==0的列对工作W置0、H diagonal置1，再原生damp/inverse，所有alpha共用相同dead掩码。不能把dead后的W作为改变了的teacher，不能增大damp隐藏失败。原生qcfg自动增加damp关，OOM inverse近似fallback也不许静默启用；不够显存就按原精度CPU/offload或停止。

## loss计算 / H缓存

默认直接token-chunk FP32 Xdelta，每篇样本2048tokens完整计数，FP32平方求和后host FP64累计，TF32关闭。可预计算独立 undamped G=X^TX/N，以delta^T G delta为等价loss；这是同一目标的计算变换，不是proxy。必须在真实cal X而非随机X上检查direct等价、全十候选排名/选择，近tie直接重算；不能使用damped H当loss。native H按sample数量缩放：128个等长样本时H_raw=2/128 ΣX_doc^TX_doc，理论G=H_raw/(2*2048)，但native累积缩放与直接Gram浮点并非bitwise相同；初版默认direct，Gram优化仅在检查通过后启用并记录。

H只可在同一method/prefix/input hash的同模块候选间复用；同组实际输入相同的q/k/v、up/gate可共享。不能跨模型方法、不同sequential组或不同prefix复用。v15提供H只用于本轮预算smoke，绝不成为正式full-model H。

## scale/zero与标准packing

每模块保存selected Q [out,in] FP16、每row alpha、scales/zeros [out,in/128] FP32原值、native g_idx [in] int32=floor(col/128)、pristine/input/H/config hashes。选择不同row-alpha不会增加packing自由度：GPTQ格式本就允许每row-group scale/zero。nativePackableQuantLinear.pack接受[out,groups]后转置为[groups,out]，qweight int32位打包shape[in/8,out]、qzeros[groups,out/8]、scales[groups,out]FP16；全部实际尺寸可整除8/128。embedding/head/norm/bias照旧。使用本机GPTQModel codec及正确v1/v2 zero格式metadata，不自行套用历史zero-minus-one公式。

**布局可表示，不代表数值已完成packed parity。** 官方pack先用FP32scale从dense Q恢复整数，但存FP16scale；dequant与逐元素FP16 Q可能存在差异。主quality协议因此M0/M1/M2/M3统一dense FP16 dequant权重：M1通过原生GPTQ量化结果、在pack前保留Q，M2保存selected Q，M3官方inplace fake Q。这是真实量化权重的self-prefix计算，磁盘仍FP16，不能宣称int4运行/压缩。packed导出是单独secondary artifact，要求codes正确、无range overflow、reload/checksum/scales布局和实际forward误差审核。不得先用dense candidates选alpha再把packed PPL当成同一误差目标的无差异结果。

如果后续主问题必须使用packed运行作为primary，需要预先注册“各candidate pack-roundtrip后计算loss并用该实际prefix传播”的representation-only协议；它仍使用原十alpha GPTQ，但与v15 Q_FP16候选不能按数值严格等同。本轮不切换也不暗中repack覆盖候选。

## 中断恢复与最小检查

按row batch保存十个candidate loss与best tensor/params，成功后原子commit模块，再commit完整native组。恢复key绑定model/revision/cal IDs/order/pristine W/current selected-prefix hash/Hinv hash/dtype/core hash。未完成组重新加载前一组committed state，不保留部分安装状态；不从候选mutable work继续。不必把全部十候选tensor永久存盘，选定权重及原始loss/scale信息保留，spot候选另存。每次记录logical row-alpha counts、actual calls、cache reuse、sync time、cache/H构建和I/O、phase peak allocated/reserved、失败/重试；这都是离线校准成本。

先测q/k/v/o/up/gate/down的alpha1 native parity、外部alpha使用、候选状态隔离、dead/allzero输入、direct/Gram、selected-scale ownership、dense reload、packed codec。小型self-prefix测试固定L0的完整四组，只跑前32行用于检查、其他行保持原权重；它只是smoke，不能称完整模型方法。数学行可分不替代GPU批量数值检查；本轮B3/B32两宽度alpha.9首三行FP16 exact只是有限spot证据。
