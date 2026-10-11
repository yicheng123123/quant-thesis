import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from models.int_qwen2_layer import QuantQwen2DecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import hashlib
import utils
import os
import pdb
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    is_qwen2 = getattr(model.config, "model_type", None) == "qwen2"
    if is_qwen2:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
        DecoderLayer = QuantQwen2DecoderLayer
        layer_name_prefix = "model.layers"
    elif "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            self.is_qwen2 = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            if self.is_qwen2:
                cache.setdefault("layer_kwargs", []).append(dict(kwargs))
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    layers[0].is_qwen2 = is_qwen2

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if is_qwen2:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    def block_forward(block, hidden, sample_index):
        if is_qwen2:
            if args.batch_size != 1:
                raise ValueError("Qwen2.5 adapter currently preserves official batch_size=1 replay semantics")
            return block(hidden, **cache["layer_kwargs"][sample_index])[0]
        return block(hidden, attention_mask=attention_mask, position_ids=position_ids)[0]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    
    
    smoke_layer_limit = getattr(args, "smoke_layer_limit", len(layers))
    if not 1 <= smoke_layer_limit <= len(layers):
        raise ValueError(f"invalid smoke_layer_limit={smoke_layer_limit}")
    v21_resume_completed_layers = int(getattr(args, "v21_resume_completed_layers", 0))
    if not 0 <= v21_resume_completed_layers <= smoke_layer_limit:
        raise ValueError(f"invalid v21_resume_completed_layers={v21_resume_completed_layers}")
    for i in range(smoke_layer_limit):
        if args.resume and i == v21_resume_completed_layers and hasattr(args, "v21_restore_rng"):
            args.v21_restore_rng()
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    teacher_started = __import__("time").perf_counter()
                    for j in range(args.nsamples):
                        fp_inps[j] = block_forward(qlayer, fp_inps[j].unsqueeze(0), j)
                        if args.aug_loss:
                            fp_inps_2[j] = block_forward(qlayer, quant_inps[j].unsqueeze(0), j)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev)
                    if hasattr(args, "v20_timing_callback"):
                        args.v20_timing_callback("teacher_forward", __import__("time").perf_counter() - teacher_started)
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume and i in omni_parameters:
            qlayer.load_state_dict(omni_parameters[i], strict=False)

        # Resume only at a completed-block boundary. Reconstruct each saved
        # quantized prefix block with the official path so quant_inps and
        # fp_inps have the same dependency semantics as an uninterrupted run.
        if args.resume and i < v21_resume_completed_layers:
            if i not in omni_parameters:
                raise RuntimeError(f"resume checkpoint lacks completed block {i}")
            qlayer.half()
            smooth_and_quant_inplace(qlayer, args, is_llama)
            if args.epochs > 0:
                with torch.no_grad():
                    with traincast():
                        for j in range(args.nsamples):
                            quant_inps[j] = block_forward(qlayer, quant_inps[j].unsqueeze(0), j)
                register_scales_and_zeros(qlayer)
                layers[i] = qlayer.to("cpu")
            else:
                register_scales_and_zeros(qlayer)
                layers[i] = qlayer.to("cpu")
            if hasattr(args, "v21_layer_callback"):
                args.v21_layer_callback("replayed_prefix", i, qlayer)
            del layer
            torch.cuda.empty_cache()
            continue
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            if hasattr(args, "v21_layer_callback"):
                args.v21_layer_callback("optimizer_init", i, qlayer)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            optimizer_step_times = []
            if hasattr(optimizer, "register_step_pre_hook"):
                def _step_pre(*_):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev)
                    optimizer_step_times.append(__import__("time").perf_counter())
                def _step_post(*_):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev)
                    optimizer_step_times.append(__import__("time").perf_counter())
                optimizer.register_step_pre_hook(_step_pre)
                optimizer.register_step_post_hook(_step_post)

            def _measure_reconstruction():
                measured = []
                with torch.no_grad():
                    for sample_index in range(args.nsamples):
                        with traincast():
                            smooth_and_quant_temporary(qlayer, args, is_llama)
                            out = block_forward(qlayer, quant_inps[sample_index:sample_index+1], sample_index)
                        measured.append(float(loss_func(fp_inps[sample_index:sample_index+1], out).float().item()))
                    clear_temp_variable(qlayer)
                return sum(measured) / len(measured)

            if hasattr(args, "v20_loss_callback"):
                args.v20_loss_callback("before_optimization", _measure_reconstruction())
            
            smoke_step_limit = getattr(args, "smoke_step_limit", None)
            completed_steps = 0
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    if smoke_step_limit is not None and completed_steps >= smoke_step_limit:
                        break
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev)
                    step_started = __import__("time").perf_counter()
                    index = j * args.batch_size
                    # obtain output of quantization model
                    observe_step = hasattr(args, "v20_step_callback")
                    if observe_step:
                        quant_forward_start = torch.cuda.Event(enable_timing=True)
                        quant_forward_end = torch.cuda.Event(enable_timing=True)
                        loss_end = torch.cuda.Event(enable_timing=True)
                        quant_forward_start.record()
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        if is_qwen2:
                            quant_out = block_forward(qlayer, quant_inps[index:index+1], index)
                        else:
                            quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        if observe_step:
                            quant_forward_end.record()
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                        if observe_step:
                            loss_end.record()
                    if not math.isfinite(loss.item()):
                        raise FloatingPointError(f"nonfinite reconstruction loss at layer={i}, epoch={epochs}, sample={j}")
                        
                    loss_list.append(loss.detach().cpu())
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev)
                    student_loss_seconds = __import__("time").perf_counter() - step_started
                    if observe_step:
                        student_quant_forward_seconds = quant_forward_start.elapsed_time(quant_forward_end) / 1000.0
                        loss_seconds = quant_forward_end.elapsed_time(loss_end) / 1000.0
                    optimizer.zero_grad()
                    lwc_params = list(lwc_parameters(qlayer))
                    lwc_before = [p.detach().clone() for p in lwc_params] if hasattr(args, "v20_step_callback") else None
                    lwc_before_names = [(name, param) for name, param in qlayer.named_parameters() if "bound_factor" in name]
                    if observe_step:
                        lwc_before_state = []
                        for name, param in lwc_before_names:
                            value = param.detach().cpu().contiguous()
                            value_f = value.float()
                            lwc_before_state.append({
                                "name": name, "dtype": str(value.dtype), "shape": list(value.shape),
                                "min": float(value_f.min()), "mean": float(value_f.mean()), "max": float(value_f.max()),
                                "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                            })
                    optimizer_step_times.clear()
                    backward_optimizer_started = __import__("time").perf_counter()
                    norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
                    if hasattr(args, "v21_step_check_callback"):
                        args.v21_step_check_callback(qlayer, i, epochs, j, loss)
                    norm_list.append(norm.data)
                    completed_steps += 1
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(dev)
                    step_wall = __import__("time").perf_counter() - step_started
                    if hasattr(args, "v20_step_callback"):
                        grads = [p.grad.detach() for p in lwc_params if p.grad is not None]
                        grad_sq = sum(float(torch.sum(g.float() ** 2).item()) for g in grads)
                        grad_norm = math.sqrt(grad_sq)
                        update_l1 = sum(float(torch.sum((after.detach().float() - before.float()).abs()).item())
                                        for before, after in zip(lwc_before, lwc_params)) if lwc_before is not None else 0.0
                        changed_names = [name for (name, after), before in zip(
                            [(name, param) for name, param in qlayer.named_parameters() if "bound_factor" in name], lwc_before or []
                        ) if not torch.equal(before, after.detach())]
                        changed = bool(changed_names)
                        lwc_after_state = []
                        for name, param in [(name, param) for name, param in qlayer.named_parameters() if "bound_factor" in name]:
                            value = param.detach().cpu().contiguous()
                            value_f = value.float()
                            lwc_after_state.append({
                                "name": name, "dtype": str(value.dtype), "shape": list(value.shape),
                                "min": float(value_f.min()), "mean": float(value_f.mean()), "max": float(value_f.max()),
                                "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                            })
                        opt_wall = (optimizer_step_times[-1] - optimizer_step_times[-2]
                                    if len(optimizer_step_times) == 2 else float("nan"))
                        backward_optimizer_seconds = __import__("time").perf_counter() - backward_optimizer_started
                        post_loss = _measure_reconstruction() if hasattr(args, "v20_loss_callback") else None
                        if post_loss is not None:
                            args.v20_loss_callback(f"after_step_{completed_steps}", post_loss)
                        named_lwc = [(name, param) for name, param in qlayer.named_parameters() if "bound_factor" in name]
                        non_lwc_grads = [name for name, param in qlayer.named_parameters()
                                         if "bound_factor" not in name and param.grad is not None]
                        args.v20_step_callback({
                            "layer": i, "epoch_index": epochs, "sample_index": index,
                            "loss_before_update": float(loss.detach().float().item()),
                            "grad_norm": grad_norm, "grad_tensor_count": len(grads),
                            "gradients_finite": all(bool(torch.isfinite(g).all().item()) for g in grads),
                            "nonzero_grad_tensor_count": sum(bool(torch.any(g != 0).item()) for g in grads),
                            "lwc_parameter_names": [name for name, _ in named_lwc],
                            "lwc_parameter_count": len(named_lwc),
                            "lwc_before_state": lwc_before_state,
                            "lwc_after_state": lwc_after_state,
                            "lwc_update_l1": update_l1,
                            "lwc_changed_parameter_names": changed_names,
                            "non_lwc_parameter_grad_names": non_lwc_grads,
                            "lwc_changed": changed, "student_loss_seconds": student_loss_seconds,
                            "student_quant_forward_seconds": student_quant_forward_seconds,
                            "loss_seconds": loss_seconds,
                            "backward_optimizer_seconds": backward_optimizer_seconds,
                            "backward_seconds_excluding_optimizer": max(0.0, backward_optimizer_seconds - opt_wall),
                            "optimizer_step_seconds": opt_wall,
                            "step_total_seconds": step_wall,
                        })

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
                if hasattr(args, "v21_epoch_callback"):
                    args.v21_epoch_callback(i, epochs, loss_mean, norm_mean, qlayer)
                if smoke_step_limit is not None and completed_steps >= smoke_step_limit:
                    break
            if hasattr(args, "v21_layer_callback"):
                args.v21_layer_callback("optimizer_complete", i, qlayer)
            clear_temp_variable(qlayer)
            del optimizer
        qlayer.half() 
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = block_forward(qlayer, quant_inps[j].unsqueeze(0), j)
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            if hasattr(args, "v21_layer_callback"):
                args.v21_layer_callback("quantized_block", i, qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

