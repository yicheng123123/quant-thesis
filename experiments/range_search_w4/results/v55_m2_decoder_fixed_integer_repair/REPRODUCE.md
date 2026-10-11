# v55复现记录

修复使用本目录冻结PROTOCOL.json、run_v55.py、fast_accept_v52.py、runtime_v54.py、decoder_interface.py及源锁。原正式runner命令为：

    python run_v55.py --resume

正式修复已完整结束并保存checkpoint。该runner在最后的原始config JSON序列化比较处留下FAILED_STOP；原失败保留在FAILED_ATTEMPTS.jsonl。之后使用同一当前Transformers loader核验有效配置，并运行一次冻结评测续接脚本：

    python continue_v55_eval.py

该脚本只适用于未写入v55评测结果的新隔离副本；本目录已完成一次WT2及一次C4评测，不能在本目录重复执行。评测协议与manifest SHA记录在QUALITY_EVALUATION_SOURCE.json中。
