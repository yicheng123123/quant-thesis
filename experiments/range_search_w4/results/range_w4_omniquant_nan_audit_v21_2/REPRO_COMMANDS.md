# v21.2 可复现命令

工作目录固定为 `E:\quant-thesis`，使用已有 `.venv-gptq`。不安装依赖、不下载模型。父配置、block20和冻结tokens必须可读取；fixtures的哈希见FIXTURE_MANIFEST。

## 已运行的诊断链（历史记录，不需重复）

顺序为 capture_replay.py → replay_basic.py → run_diagnostics.py → audit_gemm_boundary.py → audit_state_data.py → run_full_fp32.py → finalize_audit.py。具体原始命令输出保存在本目录CONSOLE logs。capture_replay.py已生成固定fixture，检测到existing实际L21 fixture会拒绝覆盖。诊断脚本有固定输出名；不要为了验收重复运行或覆盖它们。

## 在新文件中独立重放失败

该命令仅重放单层/单样本3次，D1使用原生GradScaler，但失败步骤实际AdamW不会执行。output必须为本目录尚不存在的文件；脚本拒绝已有文件和目录外路径。

```powershell
Set-Location -LiteralPath 'E:\quant-thesis'
& '.venv-gptq\Scripts\python.exe' -B -X utf8 'experiments\range_search_w4\results\range_w4_omniquant_nan_audit_v21_2\minimal_repro.py' --mode D1 --repeats 3 --output 'experiments\range_search_w4\results\range_w4_omniquant_nan_audit_v21_2\REEXECUTE_D1_new.json'
```

全FP32数值对照：同一命令改 `--mode D2_FULL_FP32 --repeats 1`，并使用另一个新的output名。D3为相同AMP forward/单位 backward scale。minimal_repro只生成基础梯度统计；首算子详细trace和操作数证据使用已保存的BACKWARD_NODE_TRACE及GEMM_OVERFLOW_VERIFICATION。不要把单层重放称正式M3。

## 仅重建汇总，不执行 GPU forward/backward

```powershell
& '.venv-gptq\Scripts\python.exe' -B -X utf8 'experiments\range_search_w4\results\range_w4_omniquant_nan_audit_v21_2\finalize_audit.py'
```

此命令只读取原始JSON/fixture、检查protected哈希并刷新本目录派生摘要。无需也不会继续正式训练。历史正式源码、checkpoint与结果不覆盖。

验证原始JSON/CSV、脚本语法、code/fixture/parent SHA256（CPU-only）：

```powershell
& '.venv-gptq\Scripts\python.exe' -B -X utf8 'experiments\range_search_w4\results\range_w4_omniquant_nan_audit_v21_2\verify_artifacts.py'
```
