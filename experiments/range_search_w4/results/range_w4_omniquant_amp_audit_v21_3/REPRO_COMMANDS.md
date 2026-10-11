# Reproducibility

工作目录E:\quant-thesis，已有.venv-gptq，torch2.6.0+cu124/RTX4060 Laptop。单步状态来源为v21.2验证过的checkpoint20恢复fixture；checkpoint、fixture、source哈希在PROTOCOL/SOURCE_SHA256/PROTECTED_HASHES。

已执行命令（cached raw结果存在时run_amp_audit.py拒绝重跑和覆盖）：

```powershell
& '.venv-gptq\Scripts\python.exe' -B -X utf8 'experiments\range_search_w4\results\range_w4_omniquant_amp_audit_v21_3\run_amp_audit.py'
```

CPU-only汇总与验证，不执行forward/backward：

```powershell
& '.venv-gptq\Scripts\python.exe' -B -X utf8 'experiments\range_search_w4\results\range_w4_omniquant_amp_audit_v21_3\finalize_audit.py'
```

本轮没有安装依赖、下载模型、调用正式run_calibration、写新正式checkpoint。保留CONSOLE.log。不得删除历史raw以强制重跑；独立重执行应另行授权并使用新的隔离目录及同一协议。
