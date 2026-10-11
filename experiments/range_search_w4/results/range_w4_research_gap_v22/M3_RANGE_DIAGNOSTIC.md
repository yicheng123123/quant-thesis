# 最终 M3 LWC 的只读诊断

本轮实际运行CPU artifact analysis，未加载模型、forward、训练、GPTQ或PPL。入口 `analyze_artifacts_v22.py`，输出JSON/CSV可重现；所有读取输入执行前后SHA256一致。正确性仅覆盖artifact解析和配对，不等于新方法correctness。

## 来源与参数含义

读取v21.4最终 `M3/checkpoints/block_23/omni_parameters.pth`，包含24层、每层14个FP16 endpoint-logit tensor；按源码reshape为[out_row,n_input_groups]。与v18 `M2/selected_alpha.csv` 按完整module名和row号配对，168模块/304,128行全覆盖，无缺失。group数为7或38，累计2,795,520组、5,591,040标量；5,565,915标量最终logit不同于初值4。

源码定义 `c_lower=sigmoid(lowbound_factor)`，`c_upper=sigmoid(upbound_factor)`，分别乘**pristine当前group的min/max**。本分析对已保存FP16 logit转FP32再sigmoid，提供稳定分布统计，不复现训练AMP下每一步的参数舍入，也不计算实际scale/rounded zero或effective range width。

`(c_lower+c_upper)/2`只是统计摘要。实际width需要xmin/xmax加权并考虑scale clamp/zero rounding；本轮不加载原始大权重。没有把它称为alpha-equivalence。

## 实测分布

全组lower/upper均值=.949117/.949122，中位数均≈.975762，p05≈.803945；endpoint系数11.772%低于.9、2.307%高于.99。全组端点系数绝对差均值=.013758、median=.001205、p95=.071686；7.148%组差>.05。

| projection | M2 mean α | M3 lower/upper mean | 行内group端点平均系数跨度median | 跨度>.05的行 |
|---|---:|---:|---:|---:|
| q | .869615 | .955284/.955332 | .040080 | 45.92% |
| k | .901400 | .924380/.924613 | .179605 | 86.00% |
| v | .886458 | .868465/.868053 | .257205 | 99.41% |
| o | .867108 | .922947/.923150 | .178511 | 85.37% |
| gate | .893195 | .961577/.961596 | .027911 | 39.49% |
| up | .885993 | .951599/.951636 | .070206 | 57.28% |
| down | .883980 | .940635/.940555 | .235853 | 89.41% |

同一行group跨度整体median=.064151、p95=.370159；54.625%行>.05。down有38组，其他7组，max-min存在组数机会效应，不能单凭down跨度排名判断敏感性；JSON同时提供population std。

层级（group加权endpoint mean / row加权跨度median）：L0 .9767/.9768 / .0073，L12 .9654/.9653 / .0289，L23 .8843/.8842 / .2224。总体上深层更广、更aggressive，但不是每层单调，不能推出前缀漂移的因果贡献。

配对M2 α与M3行端点平均系数的descriptive Pearson=.14227；七projection为.0498–.4536。不作显著性检验或因果解读：样本强依赖且格式、objective、compensation不同。M3总体≈.95而M2≈.886不表示M3“应该更少裁剪”以获得收益；二者基准范围和实际quantizer不相同。

## 能说与不能说

已验证：M3学出的参数确有projection/层/组模式；并非全模型统一.9；组间不均匀与共享α的容量差异值得受控验证。v的平均系数更小、gate更接近初值、深层跨度更大都属当前checkpoint描述。

未验证：这些模式是否必要、是否泛化、是否解释M3 gap、是否可由2-region复现。极小系数的极值不能用作“应重点优化这些组”的选择依据。参数不同于初值不证明每个参数贡献了收益。没有真实scale/zero分析，不报告quantizer等价或clipping幅度的跨方法比较。

原始派生统计：`M3_RANGE_STATISTICS.json`；168模块CSV：`M2_M3_MODULE_RANGE_SUMMARY.csv`；按预先确定行号0/四分之一/二分之一/四分之三/末行的示例：`M3_PRESELECTED_ROW_EXAMPLES.json`，未按结果挑行。
