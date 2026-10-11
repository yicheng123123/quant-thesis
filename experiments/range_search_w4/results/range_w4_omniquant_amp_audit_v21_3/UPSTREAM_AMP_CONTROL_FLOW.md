# Frozen upstream AMP control flow

官方仓库 OpenGVLab/OmniQuant，commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`。直接使用保存的该commit源码，不用最新README、其他commit或runner封装代替官方证据。全部257个源码manifest条目哈希验证通过；精确文件哈希见SOURCE_SHA256.json。

## Case A：官方使用 dynamic GradScaler

weight-only LWC 路径位于 quantize/omniquant.py 的 `omniquant`。LET off 时优化器LET组为空，另一组为LWC；activation bits16不做激活量化。W4默认 deactive_amp=False，student训练使用 CUDA autocast；qlayer.float()保存FP32参数。每个block新建AdamW和NativeScaler，每个epoch/sample不重建scaler。

文件：`E:\quant-thesis\experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\official_source\main.py`；SHA256：`509769f0b35ba5f1b073a250690b58d01142dc5fb4fe0348e8a752741a1ae8f9`；lines 225–225。

```python
225:     parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
```

文件：`E:\quant-thesis\experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\official_source\main.py`；SHA256：`509769f0b35ba5f1b073a250690b58d01142dc5fb4fe0348e8a752741a1ae8f9`；lines 246–247。

```python
246:     if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
247:         args.deactive_amp = True
```

文件：`E:\quant-thesis\experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\official_source\quantize\omniquant.py`；SHA256：`70dd574209ac7d726eb5d347f685d4dfd804c09d774e9434ee32d36bc02800c0`；lines 103–110。

```python
103:     if args.deactive_amp and args.epochs>0:
104:         dtype = torch.float
105:         traincast = nullcontext
106:     else:
107:         dtype = torch.float16
108:         traincast = torch.cuda.amp.autocast
109:     inps = torch.zeros(
110:         (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
```


## 精确 training loop

函数：`omniquant`。先autocast forward/loss；非有限loss进入官方pdb诊断；finite loss时zero_grad并调用NativeScaler。20 epochs按for循环迭代samples，不根据successful steps重试。

文件：`E:\quant-thesis\experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\official_source\quantize\omniquant.py`；SHA256：`70dd574209ac7d726eb5d347f685d4dfd804c09d774e9434ee32d36bc02800c0`；lines 243–274。

```python
243:         if args.epochs > 0:
244:             with torch.no_grad():
245:                 qlayer.float()      # required for AMP training
246:             # create optimizer
247:             optimizer = torch.optim.AdamW(
248:                 [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
249:             loss_scaler = utils.NativeScalerWithGradNormCount()
250:             
251:             for epochs in range(args.epochs):
252:                 loss_list = []
253:                 norm_list = []
254:                 for j in range(args.nsamples//args.batch_size):    
255:                     index = j * args.batch_size
256:                     # obtain output of quantization model
257:                     with traincast():
258:                         smooth_and_quant_temporary(qlayer, args, is_llama)
259:                         quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
260:                         loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
261:                         if args.aug_loss:
262:                             loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
263:                     if not math.isfinite(loss.item()):
264:                         logger.info("Loss is NAN, stopping training")
265:                         pdb.set_trace()
266:                         
267:                     loss_list.append(loss.detach().cpu())
268:                     optimizer.zero_grad()
269:                     norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
270:                     norm_list.append(norm.data)
271: 
272:                 loss_mean = torch.stack(loss_list).mean()
273:                 norm_mean = torch.stack(norm_list).mean()
274:                 logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
```


## 精确 backward / unscale / step / update

类：`NativeScalerWithGradNormCount`，方法`__init__`、`__call__`。不传init_scale参数；调用当前torch默认dynamic scaler。正式调用没有clip_grad，使用else分支：unscale后计算norm（可为NaN），不因norm非有限中止，然后执行scaler.step及update。没有独立的裸optimizer.step调用。

文件：`E:\quant-thesis\experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\official_source\utils.py`；SHA256：`b209ca43c737e583eb4e23c8bdcd9292c592e7378c6f5c5aac87a8700600b868`；lines 27–47。

```python
27: class NativeScalerWithGradNormCount:
28:     state_dict_key = "amp_scaler"
29: 
30:     def __init__(self):
31:         self._scaler = torch.cuda.amp.GradScaler()
32: 
33:     def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,retain_graph=False):
34:         self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
35:         if update_grad:
36:             if clip_grad is not None:
37:                 assert parameters is not None
38:                 self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
39:                 norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
40:             else:
41:                 self._scaler.unscale_(optimizer)
42:                 norm = ampscaler_get_grad_norm(parameters)
43:             self._scaler.step(optimizer)
44:             self._scaler.update()
45:         else:
46:             norm = None
47:         return norm
```


## 当前安装torch实现与实际检查

torch2.6.0+cu124。默认scale65536、growth_factor2、backoff_factor0.5、growth_interval2000，见本地torch/amp/grad_scaler.py:119–125。CUDA兼容包装继承torch.amp.GradScaler，没有覆盖这些默认参数。

unscale_只检查optimizer-owned grads，通过`_amp_foreach_non_finite_check_and_unscale_`记录found_inf；_maybe_opt_step仅在found_inf总和为0时调用optimizer.step；update使用found_inf和backoff/growth参数更新scale、清空本iteration的optimizer tracking。

文件：`E:\quant-thesis\.venv-gptq\Lib\site-packages\torch\amp\grad_scaler.py`；SHA256：`36938ff3866c898438a761a397a171be8a2095f9417db087bb84344dbfef9938`；lines 277–283。

```python
277:             for device, per_dtype_grads in per_device_and_dtype_grads.items():
278:                 for grads in per_dtype_grads.values():
279:                     torch._amp_foreach_non_finite_check_and_unscale_(
280:                         grads,
281:                         per_device_found_inf.get(device),
282:                         per_device_inv_scale.get(device),
283:                     )
```

文件：`E:\quant-thesis\.venv-gptq\Lib\site-packages\torch\amp\grad_scaler.py`；SHA256：`36938ff3866c898438a761a397a171be8a2095f9417db087bb84344dbfef9938`；lines 343–353。

```python
343:     def _maybe_opt_step(
344:         self,
345:         optimizer: torch.optim.Optimizer,
346:         optimizer_state: Dict[str, Any],
347:         *args: Any,
348:         **kwargs: Any,
349:     ) -> Optional[float]:
350:         retval: Optional[float] = None
351:         if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
352:             retval = optimizer.step(*args, **kwargs)
353:         return retval
```

文件：`E:\quant-thesis\.venv-gptq\Lib\site-packages\torch\amp\grad_scaler.py`；SHA256：`36938ff3866c898438a761a397a171be8a2095f9417db087bb84344dbfef9938`；lines 521–531。

```python
521:             torch._amp_update_scale_(
522:                 _scale,
523:                 _growth_tracker,
524:                 found_inf_combined,
525:                 self._growth_factor,
526:                 self._backoff_factor,
527:                 self._growth_interval,
528:             )
529: 
530:         # To prepare for next iteration, clear the data collected from optimizers this iteration.
531:         self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
```

本地实现路径和SHA256见SOURCE_SHA256；关键可执行代码区段：torch/amp/grad_scaler.py:338–341（检测状态）、343–353（有条件step）、521–531（dynamic update与下一iteration状态清理）。不是仅引用文档：GRADSCALER_BEHAVIOR_CHECK.json在当前GPU/torch运行了overflow和后续finite两次iteration，观察到了skip/backoff和后续成功更新。

官方loop不会因一次scaled-gradient overflow退出；梯度norm非有限只进入日志。下一iteration的optimizer.zero_grad清除优化器参数的旧梯度，低scale继续参与backward。正常的skip并不保证之后任意sample/epoch都稳定。
