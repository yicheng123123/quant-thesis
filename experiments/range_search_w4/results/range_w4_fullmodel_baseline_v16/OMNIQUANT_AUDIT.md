# 官方 OmniQuant 源码审计

审计对象：[OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant)，固定 commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`。`official_source/` 是从官方 origin 取得的浅克隆；`OFFICIAL_SOURCE_SHA256.json` 固定读取源码的字节。没有安装依赖、修改算法或运行官方训练。Windows 大小写文件系统使 `imgs/OmniQuant.png` / `imgs/omniquant.png` 碰撞，工作树显示图片差异；Python、shell、toml 和许可证源码必须与 Git blob 一致，图片不参与实验。

## 来源、版本和许可证

`LICENCE` 是 MIT，copyright 2024 Mengzhao Chen；`pyproject.toml` 的 Apache classifier 与它不一致，按实际 MIT 许可证保留版权和许可文本，明确记录该元数据冲突。[固定许可证](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/LICENCE)。项目声明版本 0.1.0，Python>=3.8，README 安装示例 Python3.10。依赖没有官方 lock：torch>=2.0、transformers>=4.31、datasets>=2.0、tokenizers>=0.12.1、accelerate 未限定、peft>=0.2、sacrebleu==1.5.0 等，完整声明保存在 `official_source/pyproject.toml`。`utils.py` 还直接 import termcolor，不能只凭安装声明认为 import 一定成功。

本机 GPTQ 环境实际版本见 `ENVIRONMENT_AUDIT.json`：torch2.6.0+cu124、transformers4.49.0、tokenizers0.21.4、datasets5.0.1、accelerate1.14.0、GPTQModel2.2.0。不修改这个环境。后续使用单独 OmniQuant 环境，固定相同 torch/transformers/tokenizers 作为架构 API 和评测基准；先 import smoke，再生成完整 pip lock。完整传递依赖尚未安装或验证，不宣称官方支持该组合。packed real quant 的官方 AutoGPTQ-bugfix/Triton 依赖不能视为本机 Windows 已可用。

## CLI 的真实含义

官方 `scripts/llama/llama-7b/w4a16g128.sh` 是 `--wbits 4 --abits 16 --group_size 128 --lwc --epochs 20`，没有 `--let`，没有 `--symmetric`。因此主 M3 冻结官方 weight-only 配置：**LWC开启、LET关闭、asymmetric affine G128、20 epochs**。这本身是官方方法的 weight-only 设置，不是删去官方必需机制。[固定脚本](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/scripts/llama/llama-7b/w4a16g128.sh)。LET 是另一条可选路径，官方 W4A4 示例启用；本轮不加 LET/激活量化。

`main.py` CLI epochs 默认10而示例20，nsamples默认128，batch1，seed2，lwc_lr=.01，let_lr=.005，wd0，aug_loss默认false，symmetric默认false，disable_zero_point默认false。CLI `--alpha=.5` 是 LET 初始化指数，与 pure-search clipping alpha 无关。`lm.seqlen` 在 main 中强制2048。v16统一 seed20261005 和冻结数据清单是明确的数据协议变动，不能调用官方随机 sampler 悄悄换池。[CLI](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/main.py)。

## 量化与优化

`UniformAffineQuantizer` 使用 round STE、clamp 和 dequant；G128 reshape 到每行每组。LWC 每个 row-group 两个独立 bound factor，初始化4，经 sigmoid 缩放当前 weight min/max；默认 asym scale=(xmax-xmin)/15，clamp[1e-5,1e4]，zero=round(-xmin/scale)，zero另有[-1e4,1e4] clamp。官方不先将 extrema 扩到包含零。symmetric 可选时 scale=max(abs(bounds))/7、zero7，仍是[0,15]整数；**这不同于当前 GPTQ 的典型 scale=2max/15、zero8**。abits16 forward identity，不做低位激活或 KV 量化。没有 GPTQ Hessian compensation。[量化器](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/quantizer.py)。

`omniquant()` 先缓存首层输入，独立维护 FP 前缀和 quant 前缀。每层 FP teacher 输出来自原始 FP prefix；student 输入来自此前量化层。官方 `MSELoss` 比较整个 decoder block 输出，aug_loss=false 时只一项。原模型参数冻结；当前 qlayer.float()，FP32 LWC参数，经 AMP forward 和原 NativeScaler backward/AdamW 更新，学习率无 schedule，样本固定顺序，不按 loss early-stop。LET关闭时临时量化原权重，不改 attention/RMSNorm 参数。最终 qlayer.half() 后 official inplace fake quant，传播该层 quant outputs 到下一层，保存 LWC factors。[完整循环](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/quantize/omniquant.py)。

## fake、packed 和 PPL

默认是 dense fake-dequant FP16 权重，不是物理 int4 checkpoint；QuantLinear F.linear 执行。`--real_quant` 只允许2/3/4位且abits>=16，4位走 qlinear_triton，3位走qlinear_cuda，再 pack scales/zeros。官方保存 fake model 的 `save_pretrained` 路径并不能自动为未知 Qwen wrapper提供可靠 reload；必须采用原生Qwen dense导出，保留另存 factors/参数/校验和。不能把 fake model 的磁盘/VRAM当成4位压缩数据。

官方 `evaluate()` 只识别既有模型家族；WikiText用整流非重叠2048块、尾部丢弃、每块loss乘seqlen；C4在datautils抽256个验证窗口。`--limit` 提前break但分母仍用原nsamples，是额外陷阱。它与本仓库 token-weighted sliding evaluator 不同；M0-M3统一使用本仓库 frozen IDs + `evaluate_ppl`，**不混用官方报表**。官方 dataloader/test cache key缺 revision/seed/seqlen/tokenizer hash；新adapter必须使用 manifest hash 的缓存key。[数据采样](https://github.com/OpenGVLab/OmniQuant/blob/feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14/datautils.py)。

## 命名与公平性

M3：`OmniQuant-Qwen2.5 port (official W4A16 configuration; LWC on, LET off)`，仅当保留上述优化、目标、量化器、epochs及参数化才可使用此名。M4若将LWC改为逐Linear/row输出loss、换成GPTQ量化器、加GPTQ compensation或改训练参数化，命名 `LWC-controlled mechanism baseline`，不能冒充完整OmniQuant。M4本轮不执行、不进入必需矩阵。M3和M2共同数据/位宽/模型可做方法级PPL+校准成本比较，不能把差异仅归因于 enumeration versus gradients。
