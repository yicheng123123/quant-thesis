# v21.1 runner 与 frozen upstream 的 AMP 对照

结论：AMP/backward/unscale/step/update相同；额外梯度fatal gate不属于官方算法。**实际STOP发生在scaler处理之后，必须纠正“pre-scaler STOP”的假设。**

## 按执行顺序对比

1. upstream omniquant.py:257–262在traincast内量化、forward、loss；port:360–370同样操作，Qwen调用原生block_forward。未换训练目标。
2. upstream:263–265仅对非有限loss进入pdb；port:373–374改为raise。当前loss有限，未触发。
3. upstream:268 optimizer.zero_grad；port:383同样zero_grad。
4. upstream:269调用NativeScaler；port:399同样调用。utils.py的SHA完全相同（`b209ca43c737e583eb4e23c8bdcd9292c592e7378c6f5c5aac87a8700600b868`）。两者都是scale(loss).backward → unscale_ → norm → scaler.step → scaler.update。
5. 在owned gradient非有限时，官方scaler已经skip/backoff。然后port:400–401才调用v21_step_check_callback。
6. runner install_source_callbacks.step_check:424–442检查**此时仍非有限的unscaled LWC gradients**，日志写入后raise，阻止下一sample。scaler不会把Inf/NaN修复成有限梯度；安全性来自跳过更新。这种post-scaler finite-gradient绝对要求同样会拒绝合法AMP event。
7. upstream:270–274保存/汇总norm并写日志；runner epoch_cb:414–422额外要求norm_mean有限。一旦合法skip产生NaN norm，该gate仍会在epoch末STOP。
8. runner:575额外要求finite_gradients==steps；:307/:579/:587等将iteration乘积命名为effective_optimizer_updates。这些都要在未来runner修正方案中一起处理，只改一处callback不足。

## 关键时序代码

文件：`E:\quant-thesis\experiments\range_search_w4\results\range_w4_omniquant_full_v21_1\M3\official_port\quantize\omniquant.py`；SHA256：`7319c46bf73595b95796c6cf894f823ae862b12718249cda020c9af7ac66db60`；lines 397–403。

```python
397:                     optimizer_step_times.clear()
398:                     backward_optimizer_started = __import__("time").perf_counter()
399:                     norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
400:                     if hasattr(args, "v21_step_check_callback"):
401:                         args.v21_step_check_callback(qlayer, i, epochs, j, loss)
402:                     norm_list.append(norm.data)
403:                     completed_steps += 1
```

文件：`E:\quant-thesis\experiments\range_search_w4\results\range_w4_omniquant_full_v21_1\M3\run_formal_m3.py`；SHA256：`d69a93b55090d40b75011ed37115c563fa367db7e39b37d4824847f58fbd81f6`；lines 424–442。

```python
424:     def step_check(module, layer: int, epoch: int, sample: int, loss: torch.Tensor):
425:         params = list(lwc_parameters(module))
426:         grads = [param.grad.detach().reshape(-1) for param in params if param.grad is not None]
427:         if len(grads) != len(params) or not grads:
428:             raise FloatingPointError(f"missing LWC gradients at layer={layer}, epoch={epoch}, sample={sample}")
429:         flat = torch.cat([grad.float() for grad in grads])
430:         flags = torch.stack((torch.isfinite(flat).all(), torch.any(flat != 0))).detach().cpu().tolist()
431:         finite_grad, nonzero_grad = bool(flags[0]), bool(flags[1])
432:         loss_value = float(loss.detach().float().item())
433:         step_csv.write({"attempt_id": attempt_id, "layer": layer, "epoch_index": epoch,
434:                         "sample_index": sample, "loss": loss_value,
435:                         "lwc_grad_param_count": len(params), "lwc_grad_finite": finite_grad,
436:                         "lwc_nonzero": nonzero_grad}, flush=(sample == SAMPLES - 1))
437:         counts["steps"] += 1
438:         counts["finite_losses"] += int(np.isfinite(loss_value))
439:         counts["finite_gradients"] += int(finite_grad)
440:         counts["nonzero_gradients"] += int(nonzero_grad)
441:         if not np.isfinite(loss_value) or not finite_grad:
442:             raise FloatingPointError(f"LWC loss/gradient check failed at layer={layer}, epoch={epoch}, sample={sample}")
```


原始formal runner SHA：`d69a93b55090d40b75011ed37115c563fa367db7e39b37d4824847f58fbd81f6`；port driver SHA：`7319c46bf73595b95796c6cf894f823ae862b12718249cda020c9af7ac66db60`。

v21.2正确定位了scaled HALF gradient overflow。本轮补充了严重度判断：该首次event在官方control flow下可恢复，不能等同fatal参数损坏。v21.1/v21.2历史文档没有覆盖，新结论保存在v21.3。
