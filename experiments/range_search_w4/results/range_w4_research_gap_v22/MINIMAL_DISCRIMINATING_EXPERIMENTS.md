# 三个最小判别实验：设计冻结，尚未执行

目的：区分H1目标、H2容量、H3网格。不是下一轮运行授权，也不是已完成实验。先各做小型机制诊断，不启动全模型/PPL/其他模型。

## 共同冻结协议

- 模型/revision/FP16/SDPA/TF32 off与v18相同；symmetric W4A16、G128、dynamic native GPTQ，damping=.1/blocksize128/desc_act=false/static_groups=false；α=1无hook。Bias/norm保留FP16。
- Blocks固定0/12/23；按projection输出维n固定行 `[0,floor(n/4),floor(n/2),floor(3n/4)]`。q为[0,224,448,672]；up为[0,1216,2432,3648]；down为[0,224,448,672]。不得根据结果选层/行。
- 使用已有WT2 train语料/工具，**新建独立完整文章manifest**：排除v11–v18所有已使用文章ID/full hash、最终WT2/C4评测文章hash；合格文章可提供2048 tokens。确定性排序键`SHA256('v22-next-diagnostic-20261022:' + full_article_sha256)`，取前64篇；每篇前2048token为一个完整校准样本，同一文章只归一个集合。A split：排序位置0..15 search、16..31 held；B：32..47 search、48..63 held。不足64则STOP，不改变排序/预算。记录文章索引、全文/token hash及mask/position，禁止token拆分留出。
- 这是**未来选择规则**，本轮未抽样、没有假称已有新manifest。两个split预先固定，held仅最终一次评分，不能看held再调配置或阈值。已训练M3权重参数仅作当前只读动机，不参与新候选选择。
- Quantized prefix来自冻结v18 M2 dense checkpoint。新文章单次forward捕获三个block入口，所有实验共用并记录hash；每block context重置为该v18 block，target恢复pristine W用于candidate generation。每target独立context，不把一项实验改动commit到另一项。各split重新构建其search H，不用held建H。
- 每Linear的10 full-shape候选从相同pristine W/new mutable GPTQ state生成，共享immutable H/Hinv。所有row选出local解后形成该target Linear作为基准，其余模块固定v18。下游固定Q但使用候选实际输入重新forward；**不**在每候选重新量化下游。teacher为同一个quantized-prefix block输入下的pristine FP block。
- 固定候选/数据双重复评分估计数值噪声；loss FP32差值、FP64 SSE累加。判别epsilon=max(1e-12,5×重复MSE最大差,1e-6×baseline MSE)。通过finite、α=1 native parity、状态隔离、stitch metadata/selected-row output检查再运行。失败STOP并定位。
- 校准/scoring/capture/refresh/IO分别同步计时、峰值allocated/reserved，记录score slots与full-Linear/per-row GPTQ执行数。不把候选分数数目当调用次数。

## E1 — H1：同候选的单行local vs block选择

Targets：三个block的q_proj/up_proj，共6Linear×4行×2split=48比较单位。每target生成10 full candidates；其他行固定local-M2 stitch，仅当前一行替换。对完全相同的10个单行配置，分别以local row output MSE、实际whole-block output MSE选α，ties取较大α。候选bank/输入/H/范围自由度/最终GPTQ/非目标模块完全一致，唯一选择目标不同。

需要120次full-Linear候选、480个block配置×32篇=15,360单文档block forward（可缓存teacher但不能复用错误的后续激活）。保存search/held各候选loss、排名Spearman、选中α、损失差和每文档分布，actualscale/zero/g_idx。单行干预只用于判别目标；**不**把各行block最优同时stitch成正式方法。

推进gate（研究路线门槛，非显著性检验）：两个split pooled held block SSE都改善超过epsilon，至少4/6 targets跨split合计改善，非平局changed单位≥60% held优于local，没有单个case block MSE相对local恶化>1%。Search排名/选择差异必须超过epsilon；若同排名或无held收益，H1在此预选范围不获支持，保留负结果并切换已冻结BACKUP设计，不能换teacher追正结果。微小单行收益通过只允许下一小阶段联合/完整Linear诊断，**不授权PPL，也不证明可弥合M3 gap**。

held最佳候选仅作retrospective regret诊断，不能拿来部署或重选search α。单行效应可能小，需报告绝对SSE而非只报相对百分比，任何低于数值地板的收益当tie。

## E2 — H2：2-region与共享α，local目标保持不变

Targets：三个block的up/down，共6Linear×4行×2split=48单位。原10点shared α基准；2-region从此解初始化，7组按[0..2]/[3..6]，38组按[0..18]/[19..37]。按输入顺序region0→1，**恰好2个预设sweep**；每region评价9个不同于当前解的α，当前解作为已知候选包含。只按search整行output MSE接受严格改进，每次记录接受变化。

每次修改α_g后从pristine row W完整重跑GPTQ，后续组也随compensation变化，不能仅替换当前group量化权重。先验证shared α逐行reference与full-shape row输出/scale/zero/g_idx在冻结tolerance内一致，FAIL则停止。

每单位最多10基准+36新配置=46 candidate score positions。Matched-score-budget scalar对照：原10点加固定36点`.1+.9*j/37, j=1..36`，同46点评价预算；报告原10 vs structured、46 scalar vs structured。预算相同不能完全消除不同搜索几何，也**不等于runtime相同**：scalar可full-Linear批行执行，structured少量行是per-row GPTQ。

独立运行基准120次full-Linear；scalar扩展432次full-Linear；structured最多1,728次新per-row execution，基准480 row scores取自120 full runs。两方法各2,208 score positions，须分清调用与分数。重复配置即使缓存也记录去重前后次数；预算对照按实际score/执行/时间分别报告，不虚构公平wall。

推进gate：structured相对原10点与46 scalar均在两个split pooled held row SSE改善超过epsilon，至少4/6 targets合计正向；相比shared基准不得新增gain<−10% nativeD的失败case。通过后才能考虑4-region设计，未通过则容量方向不扩大。候选loss保证只对search适用，不保证held。

## E3 — H3：coarse vs dense，额外预算与分辨率分开

Targets与E1相同48单位，仍只用local objective。原网格G10=.1..1；G16=G10并集{.825,.85,.875,.925,.95,.975}；匹配10点评价预算G10ref={.1,.7,.8,.825,.85,.875,.9,.925,.95,1}。参数仍为整行共享α，其他GPTQ不变。

G16全部192次full-Linear执行，768 row-alpha scores，原G10与G10ref均是其中子集，无需重复已存候选。G16−G10混合“更多候选+局部细化”；G10ref−G10保持候选数但改变覆盖，不能将其视为完美因果隔离。共享同search/held输入和最终candidate真实Q。只能以search选α，held oracle仅作明确标注的上限诊断。

推进gate：G16与G10ref对G10均在两个split pooled held row SSE改善超过epsilon，至少4/6 targets合计正向、不新增<−10% nativeD失败。若只有G16通过，证据偏向额外预算，不能宣称分辨率是主因；都不过则停止密网格扩展。即使通过，未运行模型级NLL就不能声称解释1.76×/2.98×gap。

## 一天内可行性与成本闸门

三项设计可共用新文档prefix caches、q/up的coarse candidate bank；每split/target/H身份一致才允许复用。不同objective只共享量化候选，不能共享它们不等价的loss。每实验先测最深层、最大input的2候选+2文档warmup/同步计时，纳入预计算与IO；估算全量×1.5。单项上限6h，三项加准备总预算24h；不通过STOP，不临时缩规模。当前只有历史cost锚点，没有新完整forward timing，因此“≤一天”是执行gate，不是未经测量的完成承诺。

此设计不改变历史full-model 48h hard gate；未来任何全模型实验仍需独立估算与授权。当前无新增实验运行。
