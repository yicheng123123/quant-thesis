# v51 M2 scalar10 + v49 fast integer repair — stopped (D)

## 结果

v51 未完成全模型修复，未保存完整模型，也未运行 WT2/C4。质量迁移结论为 D（运行/评测未完成）。block0 是正式流程的一部分，已完整修复并提交一次；layer1 的第一个 qkv Linear 在进入候选搜索前触发 v48 的冻结初始一致性检查，随后停止。没有绕过检查、改容差或切换评分路径。

M2 的 code/网格接入预检通过。正式 M2 保存的是 dense FP16，但其逐层参数含可信 FP16 权重和原 scale/zero/g_idx；v42/v45 的历史审计对 16 个合法 code 做逐值位模式枚举并证明每个坐标映射唯一。v51 再次逐层核对 168 个 Linear、304128 行、pack/unpack 和 M2 权重位模式，没有套用 M1 网格或近似舍入反推。

停止点诊断重建了原 M2 + v51 已提交 block0 的确定性前缀。layer1 `self_attn.k_proj` 第 122 行的 FP32 direct loss 为 `9.85896058681e-08`，FP64 quadratic 为 `9.85847303563e-08`，差 `4.87551e-12`，超过冻结误差界 `2.97169e-12`。独立 raw-X FP64 为 `9.85847303563e-08`，与 quadratic 只差 `3.3749e-21`；其余 127/128 行通过。这个近零行的 FP32 direct 舍入误差导致原 guard 拒绝开始修复。它没有显示 code/grid、C 或 p 的状态偏差，但仍不能绕过冻结 guard，所以没有全模型结果。

历史 PPL 只用于上下文，不能代替 v51：

| 数据集 | M1 | M2 scalar10 | 原 R2（M2+v45 修复） | v49 R1（M1+v49 快修） | v51 |
|---|---:|---:|---:|---:|---:|
| WT2 test | 13.884370 | 13.760556 | 13.553100 | 13.628060 | 未评测 (D) |
| C4 fixed validation | 22.233924 | 22.032883 | 21.941981 | 22.077789 | 未评测 (D) |

基线 NLL 是按 valid tokens 加权的历史值，完整结果及哈希见 `HISTORICAL_QUALITY_COMPARISON.csv`。v49 使用 M1 初始化；R2 使用旧 v45 修复，比较均为描述性历史结果，不能推断 v51 相对质量。v51 没有 PPL/NLL，也没有锁定 checkpoint。

## 覆盖与正确性

- 正式 block0：7/168 Linear、12,672/304,128 行；四个组 checkpoint 均提交，码/网格/终态检查通过。接受更新共 379,055。恢复重放核对了四个 block0 输入哈希并跳过修复，没有重复提交。
- layer1：`self_attn.k_proj` 首次初始化 direct/quadratic guard 失败，尚未评分任何候选，也没有提交 layer1 组。code 范围、M2 scale/zero/g_idx、FP16 实际网格没有发现不一致。
- 未产生完整模型、reload/output regression、PPL 或新的质量结论；没有读取 P3。block0 局部重构记录不作为质量替代。

## 成本

M2 scalar10 正式初始化历史实测 2115.935s，单列且不计入本轮 GPU cap。v51 从 GPU_START 到最后一项诊断累计 657.305s（含失败、恢复和诊断），低于 10,800s 硬上限；所有已打点 CUDA 同步阶段合计 264.619s，按 mode 的分解在 `PARTIAL_STAGE_COSTS.csv`。两种口径不同：同步阶段计时不含模型加载外的所有间隔，生命周期 wall 会计入等待/脚本间隔。
block0 一次完整正式修复实测 187.085s；恢复时只读重放 block0 3.455s，没有再次修复。v51 保存/reload/PPL 均未运行。历史 v49 M1+快修 formal repair 为 4543.075s，完整 job wall 5126.394s。v51 只完成 block0，不能据此声称完整作业加速或标量范围质量收益。
M2 初始化 + v51 已发生不完整 job 的账面合计为 2773.240s；这不是完成的初始化+全模型修复流水线。原 winner-union 重复执行成本没有计作首次候选发现成本。

## 冻结与复现

协议 SHA256 `478171ef4807d646162c6ee4e45ce2915dba68169bd766607d8f15443ae6620b`；runner SHA256 `6d3f8c1cfd7d999eec3ac50ed0b6ab4440e1dea363d348c3357a1e1ddeb3157c`；v48 `fast_accept.py` SHA256 `c3a760c0f48723b3784774263af71bc9e469115c2ab5d0c35ba3145671e079a2`；规则 SHA256 `7c6dd06068f46b0b9efdead69055c711356588f4700d751cfc744018505631d6`。v49 runner 的唯一运行后 amendment 修正只读 v46 成本预测文件的路径；没有修改算法或协议，协议 hash 前后相同。

原始失败状态与日志保持在 `RUN_STATUS.json`、`FAILED_ATTEMPTS.jsonl`、`run.log` 和 `run_resume.log`。数值诊断为重建状态，不覆盖原失败记录。复现及证据路径见 `REPRODUCE.md`。

Luna Max 是用户请求的设置，但本接口无法确认或切换模型设置。结果没有被解释为 ReQuant 官方复现，也没有原创贡献主张。
