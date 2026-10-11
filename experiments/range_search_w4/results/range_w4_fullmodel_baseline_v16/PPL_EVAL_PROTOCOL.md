# 冻结 PPL protocol

同一checkpoint revision、local fast tokenizer（tokenizers0.21.4）、无chat template、add_special_tokens=false、无BOS/EOS插入。所有M0-M3原生Qwen结构FP16推理、SDPA、eval、use_cache=false、TF32=false、batch1；primary dense dequant representation统一，量化校准数学另按各方法。

## E1 WikiText-2 test

Salesforce/wikitext，revision b08601e04326c79dfdd32d625aee71d232d685c3，config wikitext-2-raw-v1，完整test原序 `"\n\n".join(text)`（保留空行）。使用已落盘 WIKITEXT_TEST_MANIFEST.json 的IDs；输入 **299,078**，预期被评分causal targets **299,077**。source arrow/token hash已冻结。不是v8-v15 train-heldout；最终test从未用于alpha/hyperparam选择。本轮只tokenize，没有forward/PPL。

## E2 独立 C4 validation，固定token预算

allenai/c4 revision1588ec454efa1a09f29cd18ddd04fe05fc8653a2，en/c4-validation.00000-of-00008.json.gz。固定文档索引32..1055（1024篇，排除此前C4 smoke用过的前32篇），source order，双换行连接，无special tokens，取前 **131,072** IDs，预期评分 **131,071** targets。原始文本/文档hash/URL/timestamp和token清单保存在C4_EVAL_TEXT.txt / C4_EVAL_MANIFEST.json；池长498,741tokens。只读取gzip前1056条后关闭流，没有下载完整C4数据；CPU准备时间见C4_PREPARATION_COST.json。

标签必须写 **C4 validation fixed token budget PPL**，不是standard/full C4 PPL，更不是官方OmniQuant随机256窗口protocol。选择这份池与任何候选PPL无关；不将校准改成C4来追求收益。WikiText-test和C4报告单独PPL与相对FP16/native delta，不能混算一个PPL。

## 所有方法同一计分器

复用 `benchmark/eval/ppl.py::evaluate_ppl`：max_length2048，stride512。第一window评分2047targets，以后只评分尚未覆盖的targets；重叠labels=-100，loss乘实际 `labels[:,1:]!=-100`数，累加total_nll/total_tokens再exp。最后不足窗口保留，首token无上下文故不计；每套验证 total_tokens=N−1。不得均匀平均window loss，不丢弃tail，不混官方非重叠计分表。

`prepare_ppl_input_ids`/dataset loader目前不显式传revision；本轮不改共享代码，未来wrapper直接load冻结IDs，检查tokenizer/data/source hashes，再调用共同evaluate_ppl。不重新在线加载浮动main。C4旧adapter只能从首文档前缀取数据且不能传revision，因此E2也用冻结IDs绕过这两个数据边界，不改PPL数学。

先tiny128-token smoke检查loss finite、计数127和reload；正式输出每window nll/valid_count、aggregate、model/config/token hashes、elapsed/peak，用独立工具核查。同一eval一旦完成，不根据PPL调alpha grid、calibration池、epochs、格式或方法定义。若发现实现错误，声明无效run，修复后全部受影响方法按同协议重跑；保留原始文件。

## 泄漏界限

cal是128篇train全文独立样本前2048tokens，全部排除v8-v15使用过的280篇文章；v8-v15的heldout从不作为E1/E2。没有token拆分作独立留出。跨数据集不能保证任何语义重复完全消失；完整文章的字符串hash与sample-token hash需检查并记录（VERIFY_AUDIT）。该检查也不能等同于训练预训练语料去重。模型预训练是否见过WT/C4不在本轮可辨识范围；所有方法相同checkpoint所以公开承认共同局限。

最终E1/E2只评一次冻结main matrix；成本与PPL分别报告，不启动HellaSwag/downstream/PrefixQuant。M3使用单独校准环境但primary quality统一回本机冻结eval环境，避免PPL跨库差异。旧benchmark PPL只作为工具历史，不直接搬入新表。
