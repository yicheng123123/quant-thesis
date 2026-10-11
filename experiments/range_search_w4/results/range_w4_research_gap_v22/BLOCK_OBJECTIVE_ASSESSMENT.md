# Pure-search block objective：可行性与边界

冻结研究方向：在原有symmetric W4A16/G128/native dynamic GPTQ候选生成下，评价 **实际量化block输出**，先少量行干预做目标消融，再考虑staged有限配置搜索。无梯度/反向传播/学习参数，不实现新算法。

定义当前量化前缀在block入口的固定输入X_b，teacher=`F_b^FP(X_b)`；候选输出=`F_b^Q(X_b;theta)`；loss=`mean((candidate-teacher)^2)`。先同X隔离块内效应。M3实际teacher是FP prefix下的FP block输出，包含输入漂移，不能声称这个同X目标完全复刻M3。以后需独立teacher-alignment消融。

## 为什么行可分GPTQ不意味着block可分

GPTQ在冻结H上补偿不跨output row，full candidates可以row stitch；但attention、SwiGLU与其他量化模块已有残差使 **block loss** 不能按行argmin后无条件拼接。多个单行最优一起提交可能相互干扰。E1每次只改一个预选行，context完整复位；正式方法以后若通过，应串行固定预算坐标/配置搜索，每次重算实际block loss后才接受，不能假装一次独立行搜索已优化block。

线性化仅作解释：block perturbation≈Jδ，平方误差包含 `δ^T J^T Jδ` 与已有误差e_0的交叉项；J不同于local的单位权重。没有在本轮计算J，没有声称二阶代理替代真实forward。

## staged依赖

| 部分 | 实际计算依赖 | 可能的有限搜索 | 哪些缓存必须刷新 |
|---|---|---|---|
| q/k/v | 共享norm输入；QK影响softmax，V影响attention输出 | 先单row目标检验；后预设q/k/v配置坐标；2×2只用于interaction诊断 | 共同输入/H可缓存；任何候选都重算attention→o→MLP→block输出 |
| o | 输入来自量化attention，输出进入residual/RMSNorm | 在已提交qkv context下搜索 | qkv修改后o输入/H失效；o变后gate/up输入与down输入刷新 |
| gate/up | 共享post-attention norm输入；SiLU(gate)×up | 先up单row；以后gate/up配对有限配置 | 输入/H同context可共享；每候选重算乘积、down与block输出 |
| down | 输入来自gate/up乘积 | 固定上游后末阶段校正 | gate/up变则输入/H刷新；down变后下一block入口失效 |

“刷新activation/H并重新量化下游”与“固定下游Q，只forward新输入”是两种实验。E1选择第二种隔离单row downstream效应；未来staged正式语义必须另行明确，不能混合汇报。每Linear所有rows共用同一个X，不能每32行refresh prefix。

## 防止无效消融

如果只改down_proj，其他部分FP且teacher共享相同down输入，那么residual相消后block输出MSE就是down local输出MSE，无法检验H1。E1选q_proj/up_proj，传播经过attention或SwiGLU/down。即使使用down也必须说明已有量化context/teacher残差，否则把等价目标作为新目标是错误的。

## 成本

H1设计共有120次full-Linear候选，48个单行比较单位，480个block配置；每配置32篇2048-token文档，合计15,360次单文档block forward，另计teacher、prefix capture、评分/IO。不是480次full-prefix forward，更不是所有行候选各自重跑全模型。

历史M2分解：GPTQ candidates938.43s、all-row scoring718.77s、H/Hinv90.01s；对应全1680候选，平均GPTQ .559s/完整Linear，不是单row调用。它仅说明小型候选生成规模有限，不能直接预测block评分时间。

下一实验前必须同步测量最深block的单文档teacher/student耗时，并估算：`T_capture + sum_modules(T_H + 10*T_GPTQ + T_local_score) + 15360*T_block_doc + T_teacher + T_IO`。保守1.5×、每实验6h gate，三实验+准备≤24h；若不过则停止报成本，不临时缩样本或更换冻结对象追结果。candidate、capture、dependency refresh、block scoring、IO分别记录。当前无新runtime测量或吞吐承诺。
