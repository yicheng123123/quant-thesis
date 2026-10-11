"""Bounded, reproducible OmniQuant Qwen2.5 compatibility smoke. No formal M3 run."""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import types
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V16 = RESULTS / "range_w4_fullmodel_baseline_v16"
V18 = RESULTS / "range_w4_fullmodel_v18"
UPSTREAM = HERE / "upstream"
MODEL_DIR = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
COMMIT = "feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def move_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_tree(v, device) for v in value)
    if isinstance(value, list):
        return [move_tree(v, device) for v in value]
    if isinstance(value, dict):
        return {k: move_tree(v, device) for k, v in value.items()}
    return value


def tensor_tree_summary(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
                "finite": bool(torch.isfinite(value).all())}
    if isinstance(value, (tuple, list)):
        return [tensor_tree_summary(v) for v in value]
    if isinstance(value, dict):
        return {k: tensor_tree_summary(v) for k, v in value.items()}
    return value


def maxdiff(a: torch.Tensor, b: torch.Tensor):
    d = (a.float() - b.float()).abs()
    return {"shape": list(a.shape), "dtype_a": str(a.dtype), "dtype_b": str(b.dtype),
            "max_abs_diff": float(d.max().item()), "mean_abs_diff": float(d.mean().item()),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "allclose_rtol_1e-5_atol_1e-6": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-6)),
            "bitwise_equal": bool(torch.equal(a, b))}


def create_patch_and_upstream_manifests():
    source = V16 / "official_source"
    expected = json.loads((V16 / "OFFICIAL_SOURCE_SHA256.json").read_text(encoding="utf-8"))
    checks = []
    tree = hashlib.sha256()
    for rel, expected_hash in sorted(expected.items()):
        src = source / rel
        actual = sha_file(src) if src.is_file() else None
        checks.append({"path": rel, "expected_sha256": expected_hash, "actual_sha256": actual,
                       "pass": actual == expected_hash})
        if actual == expected_hash:
            tree.update(rel.replace("\\", "/").encode("utf-8") + b"\0")
            tree.update(bytes.fromhex(actual))
    upstream_hash = tree.hexdigest()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    patch_rows = [
        ["upstream/models/int_qwen2_layer.py", "QuantQwen2DecoderLayer", "not present", "deep-copy native Qwen2 layer and replace exactly seven nn.Linear with unmodified QuantLinear", "Qwen2 architecture mapping", "yes", "no"],
        ["upstream/quantize/omniquant.py", "omniquant Qwen2 dispatch", "supports llama/opt/falcon/mixtral branches", "add config.model_type=qwen2 layer/device discovery and Qwen-specific native decoder adapter", "architecture plumbing", "yes", "no"],
        ["upstream/quantize/omniquant.py", "Catcher.forward/block_forward Qwen2 branch", "captures shared attention_mask and limited position_ids; replays fixed kwargs", "capture and replay each sample's full native Qwen decoder kwargs including position_embeddings/cache_position", "preserve Qwen forward semantics", "yes", "no"],
        ["upstream/quantize/omniquant.py", "smoke_layer_limit/smoke_step_limit", "full upstream layer/epoch loop", "optional explicit 1-layer/2-update smoke budget; default formal loop remains full", "bounded compatibility smoke only", "no", "no"],
        ["upstream/quantize/omniquant.py", "v20 loss/gradient/update/timing observers", "no v20 observers", "optional callbacks and synchronized timing events; observers do not feed model state or optimizer inputs", "smoke diagnostics", "no", "no"],
        ["run_smoke.py", "quant-off parity/discovery/calibration/export/evaluator harness", "not present", "fixed local checkpoint/manifests; no random sampler; invoke official block optimizer path for 2 updates", "experiment harness", "no", "no"],
    ]
    with (HERE / "OMNIQUANT_PATCH_MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["file", "function_class", "upstream_behavior", "patched_behavior", "reason", "architecture_only", "algorithm_semantic_change"]); writer.writerows(patch_rows)
    protected = ["quantize/quantizer.py", "quantize/int_linear.py", "quantize/utils.py", "utils.py"]
    protected_hashes = {p: {"official": sha_file(source/p), "v20_copy": sha_file(UPSTREAM/p), "identical": sha_file(source/p) == sha_file(UPSTREAM/p)} for p in protected}
    patch_files = ["quantize/omniquant.py", "models/int_qwen2_layer.py"]
    write_json("UPSTREAM_MANIFEST.json", {
        "repository": "https://github.com/OpenGVLab/OmniQuant", "commit": commit, "expected_commit": COMMIT,
        "commit_pass": commit == COMMIT, "frozen_source_manifest": str(V16/"OFFICIAL_SOURCE_SHA256.json"),
        "frozen_source_manifest_sha256": sha_file(V16/"OFFICIAL_SOURCE_SHA256.json"),
        "source_tree_hash_algorithm": "sha256 over sorted (relative POSIX path + NUL + raw file SHA256 bytes) from the 257-file frozen source manifest",
        "source_tree_sha256": upstream_hash, "source_files_checked": len(checks),
        "source_file_mismatches": [x for x in checks if not x["pass"]],
        "patched_upstream_files": patch_files, "protected_algorithm_files": protected_hashes,
        "algorithm_semantic_change": any(r[-1] == "yes" for r in patch_rows),
        "patch_manifest": "OMNIQUANT_PATCH_MANIFEST.csv"})
    return commit == COMMIT and all(x["pass"] for x in checks) and all(x["identical"] for x in protected_hashes.values()), upstream_hash


def environment_outputs(device):
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT)
    (HERE / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    props = torch.cuda.get_device_properties(device)
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "datasets", "accelerate", "timm", "termcolor", "peft", "sentencepiece", "sacrebleu", "einops"):
        try:
            spec = importlib.util.find_spec(name)
            packages[name] = {"available": spec is not None, "origin": spec.origin if spec else None}
        except (ImportError, ValueError):
            packages[name] = {"available": False, "origin": None}
    report = {"python": sys.version, "executable": sys.executable, "platform": sys.platform,
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "transformers": __import__("transformers").__version__, "tokenizers": __import__("tokenizers").__version__,
              "cuda_available": True, "device": torch.cuda.get_device_name(device),
              "vram_bytes": int(props.total_memory), "vram_gib": props.total_memory/(1024**3),
              "compute_capability": [props.major, props.minor], "optional_upstream_dependencies": packages,
              "environment_isolation": "read-only reuse of the frozen .venv-gptq runtime; no packages installed or upgraded. Missing termcolor is shimmed to identity only for optional logger coloring. This smoke does not establish a complete OmniQuant dependency lock or upstream-wide compatibility.",
              "pip_freeze": "pip_freeze.txt"}
    (HERE / "ENVIRONMENT_REPORT.md").write_text(
        "# v20 runtime environment\n\n" + "\n".join(f"- **{k}:** `{v}`" for k,v in report.items() if k != "optional_upstream_dependencies") +
        "\n\nOptional/import dependencies:\n\n" + "\n".join(f"- `{k}`: {'available' if v['available'] else 'not installed'}" for k,v in packages.items()) + "\n", encoding="utf-8")
    return report


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(UPSTREAM))
    # OmniQuant imports termcolor solely to color logger prefixes; do not add an environment dependency for that optional feature.
    if importlib.util.find_spec("termcolor") is None:
        termcolor = types.ModuleType("termcolor")
        termcolor.colored = lambda text, *args, **kwargs: text
        sys.modules["termcolor"] = termcolor

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from models.int_qwen2_layer import QuantQwen2DecoderLayer
    from quantize.int_linear import QuantLinear
    from quantize.omniquant import omniquant
    from quantize.utils import get_omni_parameters, lwc_parameters, smooth_and_quant_temporary

    torch.manual_seed(20261005); np.random.seed(20261005)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    up_ok, upstream_hash = create_patch_and_upstream_manifests()
    env = environment_outputs(device)
    (HERE / "smoke_state").mkdir(exist_ok=True)
    correctness = {"official_commit_and_tree": up_ok, "algorithm_semantic_change": False}
    if not up_ok:
        write_json("correctness_checks.json", {"status":"FAIL","checks":correctness})
        raise RuntimeError("official upstream freeze verification failed; STOP")

    cal_path = V18 / "CALIBRATION_MANIFEST.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    frozen_hash = "664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3"
    if sha_file(cal_path) != frozen_hash or len(cal["samples"]) != 128 or cal["split"] != "train":
        write_json("CALIBRATION_LOADER_CHECK.json", {"status":"FAIL","reason":"frozen calibration identity mismatch"})
        raise RuntimeError("frozen calibration identity mismatch; STOP")
    v16_cal = V16 / "CALIBRATION_MANIFEST.json"
    if sha_file(v16_cal) != sha_file(cal_path):
        raise RuntimeError("v16 and v18 calibration manifest differ; STOP")
    ids2 = [s["token_ids"][:64] for s in cal["samples"][:2]]
    cal_loader_rows = []
    for i, sample in enumerate(cal["samples"]):
        token_hash = sha_bytes(json.dumps(sample["token_ids"], separators=(",", ":")).encode())
        # The frozen manifest hash protects serialized bytes; reproduce its token hash convention too.
        token_hash_default = sha_bytes(json.dumps(sample["token_ids"]).encode())
        if token_hash_default != sample["token_sha256"]:
            raise RuntimeError(f"sample token hash mismatch at manifest index {i}")
        cal_loader_rows.append({"sample_index": i, "source_article_index": sample["source_article_index"],
                                "article_sha256": sample["article_sha256"], "token_count": len(sample["token_ids"]),
                                "token_sha256": sample["token_sha256"]})
    write_json("CALIBRATION_LOADER_CHECK.json", {"status":"PASS", "manifest_sha256":sha_file(cal_path),
        "dataset":cal["dataset"],"split":cal["split"],"full_samples":len(cal["samples"]),
        "full_tokens_per_sample":sorted({len(x["token_ids"]) for x in cal["samples"]}),
        "ordered_sample_metadata_sha256":sha_bytes(json.dumps(cal_loader_rows,separators=(",",":"),sort_keys=True).encode()),
        "smoke_samples":[{"manifest_index":i,"source_article_index":cal_loader_rows[i]["source_article_index"],
                           "article_sha256":cal_loader_rows[i]["article_sha256"],"tokens":64,
                           "token_ids_sha256":sha_bytes(np.asarray(ids2[i],dtype="<i8").tobytes())} for i in range(2)],
        "resampling":False,"tokenizer_invoked_for_smoke":False,"full_manifest_hashes_valid":True})

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"frozen model snapshot unavailable: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float16,
        low_cpu_mem_usage=True, local_files_only=True, attn_implementation="sdpa").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    config = model.config
    if config.model_type != "qwen2" or config._commit_hash not in (None, REVISION):
        raise RuntimeError(f"unexpected frozen model config/hash: {config.model_type}/{config._commit_hash}")

    from safetensors import safe_open
    checkpoint_file=MODEL_DIR/"model.safetensors"
    safetensor_shapes={}
    with safe_open(checkpoint_file, framework="pt", device="cpu") as checkpoint:
        for li in range(len(model.model.layers)):
            for target in TARGETS:
                base=f"model.layers.{li}.{target}"
                for suffix in ("weight","bias"):
                    key=f"{base}.{suffix}"
                    if key in checkpoint.keys():
                        safetensor_shapes[key]=list(checkpoint.get_slice(key).get_shape())

    module_map = []
    for li, block in enumerate(model.model.layers):
        found = []
        for name in TARGETS:
            mod = block.get_submodule(name)
            if not isinstance(mod, nn.Linear):
                raise TypeError(f"L{li} {name} is {type(mod).__name__}, expected nn.Linear")
            base=f"model.layers.{li}.{name}"
            weight_key=f"{base}.weight"
            bias_key=f"{base}.bias"
            checkpoint_weight_shape=safetensor_shapes.get(weight_key)
            checkpoint_bias_shape=safetensor_shapes.get(bias_key)
            short=name.split(".")[-1]
            head_dim=config.hidden_size//config.num_attention_heads
            expected_shapes={
                "q_proj":[config.num_attention_heads*head_dim,config.hidden_size],
                "k_proj":[config.num_key_value_heads*head_dim,config.hidden_size],
                "v_proj":[config.num_key_value_heads*head_dim,config.hidden_size],
                "o_proj":[config.hidden_size,config.num_attention_heads*head_dim],
                "gate_proj":[config.intermediate_size,config.hidden_size],
                "up_proj":[config.intermediate_size,config.hidden_size],
                "down_proj":[config.hidden_size,config.intermediate_size],
            }
            weight_shape_matches=(checkpoint_weight_shape==list(mod.weight.shape)==expected_shapes[short])
            bias_shape_matches=(checkpoint_bias_shape==list(mod.bias.shape)) if mod.bias is not None else checkpoint_bias_shape is None
            found.append({"name":f"model.layers.{li}.{name}","in_features":mod.in_features,
                          "out_features":mod.out_features,"weight_shape":list(mod.weight.shape),
                          "bias_shape":list(mod.bias.shape) if mod.bias is not None else None,
                          "bias_present":mod.bias is not None,"weight_dtype":str(mod.weight.dtype),
                          "safetensors_weight_shape":checkpoint_weight_shape,"safetensors_bias_shape":checkpoint_bias_shape,
                          "config_expected_weight_shape":expected_shapes[short],
                          "checkpoint_and_config_shape_match":weight_shape_matches and bias_shape_matches})
        module_map.extend(found)
    shape_ok = (len(model.model.layers)==24 and len(module_map)==168
                and all(x["checkpoint_and_config_shape_match"] for x in module_map))
    write_json("QWEN25_MODULE_MAP.json", {"status":"PASS" if shape_ok else "FAIL","model_revision":REVISION,
        "model_type":config.model_type,"layers":len(model.model.layers),"target_projection_count":len(module_map),
        "safetensors_file":"model.safetensors","all_projection_shapes_match_checkpoint_and_config":all(x["checkpoint_and_config_shape_match"] for x in module_map),
        "expected_target_projection_count":168,"hidden_size":config.hidden_size,"intermediate_size":config.intermediate_size,
        "num_attention_heads":config.num_attention_heads,"num_key_value_heads":config.num_key_value_heads,
        "head_dim":config.hidden_size//config.num_attention_heads,"gqa_repeat_factor":config.num_attention_heads//config.num_key_value_heads,
        "modules":module_map})
    correctness["module_discovery_24x7"] = shape_ok
    if not shape_ok:
        write_json("correctness_checks.json", {"status":"FAIL","checks":correctness})
        raise RuntimeError("module discovery failed; STOP")

    adapter_args = SimpleNamespace(weight_quant_params={"n_bits":4,"per_channel_axes":[0],"symmetric":False,
        "dynamic_method":"per_channel","group_size":128,"lwc":True,"disable_zero_point":False},
        act_quant_params={"n_bits":16,"per_channel_axes":[],"symmetric":False,"dynamic_method":"per_token"})
    device_model = model.to(device).eval()
    fixed_ids = torch.tensor(ids2[0],dtype=torch.long,device=device)[None]
    selected_layers=(0,12,23)
    native_layer_out={}; native_attention_out={}; native_qkv={}; layer_in={}; layer_kwargs={}; layer_kwargs_used_summary={}
    handles=[]
    for li in selected_layers:
        block=device_model.model.layers[li]
        def out_hook(module, inputs, output, idx=li):
            native_layer_out[idx]=output[0].detach().cpu()
        handles.append(block.register_forward_hook(out_hook))
        def attention_out_hook(module, inputs, output, idx=li):
            native_attention_out[idx]=(output[0] if isinstance(output, (tuple, list)) else output).detach().cpu()
        handles.append(block.self_attn.register_forward_hook(attention_out_hook))
    def pre0(module, args, kwargs):
        layer_in[0]=args[0].detach().cpu()
        layer_kwargs_used_summary[0]=tensor_tree_summary(kwargs)
        layer_kwargs[0]=move_tree(kwargs,"cpu")
    handles.append(device_model.model.layers[0].register_forward_pre_hook(pre0,with_kwargs=True))
    for li in selected_layers:
        native_qkv[li]={}
        for proj in ("q_proj","k_proj","v_proj"):
            def qkv_hook(module, inputs, output, idx=li, name=proj):
                native_qkv[idx][name]=output.detach().cpu()
            handles.append(device_model.model.layers[li].get_submodule("self_attn."+proj).register_forward_hook(qkv_hook))
    with torch.inference_mode():
        native_output=device_model(input_ids=fixed_ids,use_cache=False).logits.detach().cpu()
    for h in handles: h.remove()
    baseline_outputs={"logits":native_output}
    baseline_outputs.update({f"layer_{i}":native_layer_out[i] for i in selected_layers})
    parity_rows=[]
    parity_logger=SimpleNamespace(info=lambda *a,**k:None)
    for li in selected_layers:
        original=device_model.model.layers[li]
        port=QuantQwen2DecoderLayer(config,original,adapter_args).to(device).eval()
        device_model.model.layers[li]=port
        capture={}
        def capture_port_output(module, inputs, output, c=capture):
            c["out"] = output[0].detach().cpu()
        def capture_port_attention(module, inputs, output, c=capture):
            c["attention"] = (output[0] if isinstance(output, (tuple, list)) else output).detach().cpu()
        port_handles=[port.register_forward_hook(capture_port_output),
                      port.native_layer.self_attn.register_forward_hook(capture_port_attention)]
        capture["qkv"]={}
        for proj in ("q_proj","k_proj","v_proj"):
            def capture_port_qkv(module, inputs, output, c=capture, name=proj):
                c["qkv"][name]=output.detach().cpu()
            port_handles.append(port.native_layer.get_submodule("self_attn."+proj).register_forward_hook(capture_port_qkv))
        with torch.inference_mode():
            port_logits=device_model(input_ids=fixed_ids,use_cache=False).logits.detach().cpu()
        for handle in port_handles: handle.remove()
        device_model.model.layers[li]=original
        vals={"logits":maxdiff(baseline_outputs["logits"],port_logits),
              f"layer_{li}":maxdiff(baseline_outputs[f"layer_{li}"],capture["out"]),
              "attention_block_output":maxdiff(native_attention_out[li],capture["attention"])}
        for proj in ("q_proj","k_proj","v_proj"):
            vals[f"{proj}_output"]=maxdiff(native_qkv[li][proj],capture["qkv"][proj])
        parity_rows.append({"layer":li,"comparisons":vals})
        del port,port_logits
        torch.cuda.empty_cache()
    parity_pass=all(v["allclose_rtol_1e-5_atol_1e-6"] and v["finite"] for row in parity_rows for v in row["comparisons"].values())
    write_json("QUANT_OFF_PARITY.json", {"status":"PASS" if parity_pass else "FAIL","input_tokens":64,
        "token_ids_sha256":sha_bytes(np.asarray(ids2[0],dtype="<i8").tobytes()),"dtype":"FP16","device":str(device),
        "rtol":1e-5,"atol":1e-6,"native_vs_port":parity_rows})
    kwargs_summary=layer_kwargs_used_summary[0]
    qkv_shapes={str(li):{name:{"shape":list(value.shape),"dtype":str(value.dtype),"finite":bool(torch.isfinite(value).all())}
                         for name,value in projections.items()} for li,projections in native_qkv.items()}
    attention_ok=(qkv_shapes["0"].get("q_proj",{}).get("shape")==[1,64,896]
                  and qkv_shapes["0"].get("k_proj",{}).get("shape")==[1,64,128]
                  and qkv_shapes["0"].get("v_proj",{}).get("shape")==[1,64,128]
                  and all(x.get("finite") for layer in qkv_shapes.values() for x in layer.values())
                  and all(torch.isfinite(native_attention_out[li]).all() for li in selected_layers)
                  and "position_embeddings" in layer_kwargs[0]
                  and "cache_position" in layer_kwargs[0]
                  and layer_kwargs[0].get("use_cache") is False)
    write_json("ATTENTION_SEMANTICS_CHECK.json", {"status":"PASS" if attention_ok else "FAIL",
        "num_attention_heads":config.num_attention_heads,"num_key_value_heads":config.num_key_value_heads,
        "head_dim":config.hidden_size//config.num_attention_heads,"gqa_repeat_factor":config.num_attention_heads//config.num_key_value_heads,
        "native_qkv_projection_outputs":qkv_shapes,
        "attention_block_outputs":{str(li):{"shape":list(native_attention_out[li].shape),"dtype":str(native_attention_out[li].dtype),"finite":bool(torch.isfinite(native_attention_out[li]).all())} for li in selected_layers},
        "native_port_qkv_attention_comparisons":[row["comparisons"] for row in parity_rows],
        "first_layer_forward_kwargs":kwargs_summary,
        "native_block_forward_reused":True,"attention_implementation":config._attn_implementation,
        "position_embeddings_and_cache_position_preserved":("position_embeddings" in layer_kwargs[0] and "cache_position" in layer_kwargs[0]),
        "past_key_value":str(layer_kwargs[0].get("past_key_value")),"use_cache":layer_kwargs[0].get("use_cache")})
    correctness["quant_off_native_port_parity"] = parity_pass
    correctness["gqa_rope_kwargs_semantics"] = attention_ok
    if not parity_pass or not attention_ok:
        write_json("correctness_checks.json", {"status":"FAIL","checks":correctness})
        raise RuntimeError("quant-off parity or attention semantics failed; STOP")

    # Save a pristine native L0 reference for a post-training dense export roundtrip.
    pristine_l0=deepcopy(device_model.model.layers[0]).cpu()
    del device_model
    model=model.cpu()
    torch.cuda.empty_cache(); gc.collect()
    model.config.use_cache=True
    cal_loader=[(torch.tensor(row,dtype=torch.long)[None],) for row in ids2]

    # Parameter-map probe uses the adapter itself and is discarded before official training.
    probe=QuantQwen2DecoderLayer(config,pristine_l0,adapter_args)
    param_map=[]; total_lwc=0
    for n,m in probe.named_modules():
        if isinstance(m,QuantLinear):
            q=m.weight_quantizer
            expected_groups=math.ceil(m.in_features/128)
            for pname,p in (("upbound_factor",q.upbound_factor),("lowbound_factor",q.lowbound_factor)):
                param_map.append({"module":n,"parameter":pname,"shape":list(p.shape),"dtype":str(p.dtype),
                                  "initial_min":float(p.min()),"initial_mean":float(p.float().mean()),"initial_max":float(p.max()),
                                  "expected_shape":[m.out_features*expected_groups,1],"shape_matches":list(p.shape)==[m.out_features*expected_groups,1]})
                total_lwc+=p.numel()
    target_quantlinears=sum(isinstance(m,QuantLinear) for m in probe.modules())
    param_pass=(target_quantlinears==7 and len(param_map)==14 and total_lwc==232960 and all(p["shape_matches"] and p["initial_min"]==4 and p["initial_max"]==4 for p in param_map))
    write_json("LWC_PARAMETER_MAP.json", {"status":"PASS" if param_pass else "FAIL","projection_count":target_quantlinears,
        "bound_factor_tensor_count":len(param_map),"bound_factor_scalar_count":total_lwc,"expected_scalar_count":232960,
        "initial_value":4.0,"parameters":param_map})
    del probe; gc.collect()
    correctness["lwc_parameter_creation"] = param_pass
    if not param_pass:
        write_json("correctness_checks.json", {"status":"FAIL","checks":correctness})
        raise RuntimeError("LWC parameter map failed; STOP")

    from quantize import omniquant as omni_module
    from quantize.omniquant import omniquant as run_omniquant

    # The import above returns the submodule via package attribute in this upstream layout.
    if hasattr(omni_module,"__file__"):
        core=omni_module
    else:
        core=importlib.import_module("quantize.omniquant")

    metric_rows=[]; step_rows=[]; timing_rows=[]
    class Logger:
        def info(self,*args,**kwargs):
            with (HERE/"smoke.log").open("a",encoding="utf-8") as f: f.write(" ".join(map(str,args))+"\n")
    def loss_callback(phase,loss): metric_rows.append({"phase":phase,"loss":float(loss),"finite":math.isfinite(float(loss))})
    def step_callback(row): step_rows.append(row)
    def timing_callback(stage,seconds): timing_rows.append({"stage":stage,"seconds":float(seconds)})
    args=SimpleNamespace(net="qwen2",nsamples=2,seqlen=64,batch_size=1,deactive_amp=False,
        epochs=20,smoke_step_limit=2,smoke_layer_limit=1,aug_loss=False,let=False,lwc=True,
        wbits=4,abits=16,group_size=128,symmetric=False,disable_zero_point=False,
        w_dynamic_method="per_channel",a_dynamic_method="per_token",alpha=0.5,
        let_lr=5e-3,lwc_lr=1e-2,wd=0.0,resume=None,output_dir=str(HERE/"smoke_state"),real_quant=False,
        weight_quant_params={"n_bits":4,"per_channel_axes":[0],"symmetric":False,"dynamic_method":"per_channel",
                             "group_size":128,"lwc":True,"disable_zero_point":False},
        act_quant_params={"n_bits":16,"per_channel_axes":[],"symmetric":False,"dynamic_method":"per_token"},
        v20_loss_callback=loss_callback,v20_step_callback=step_callback,v20_timing_callback=timing_callback)
    class LM:
        pass
    lm=LM(); lm.model=model; lm.device=device; lm._device=device; lm.seqlen=64
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats(device)
    train_start=time.perf_counter()
    quantized_model=run_omniquant(lm,args,cal_loader,{}, {}, logger=Logger())
    torch.cuda.synchronize(device)
    train_wall=time.perf_counter()-train_start
    smoke_peak_alloc=int(torch.cuda.max_memory_allocated(device)); smoke_peak_reserved=int(torch.cuda.max_memory_reserved(device))
    steps_exactly_two=len(step_rows)==2 and [r["sample_index"] for r in step_rows]==[0,1]
    finite_losses=all(r["finite"] for r in metric_rows) and all(math.isfinite(r["loss_before_update"]) for r in step_rows)
    gradients_ok=steps_exactly_two and all(r["gradients_finite"] and r["grad_tensor_count"]==14 and r["nonzero_grad_tensor_count"]>0 and r["grad_norm"]>0 and math.isfinite(r["grad_norm"]) for r in step_rows)
    no_frozen_grads=all(not r["non_lwc_parameter_grad_names"] for r in step_rows)
    updates_ok=steps_exactly_two and all(r["lwc_changed"] and r["lwc_update_l1"]>0 and r["lwc_changed_parameter_names"] for r in step_rows)
    write_json("LWC_GRADIENT_CHECK.json", {"status":"PASS" if gradients_ok and no_frozen_grads else "FAIL",
        "optimizer_steps":len(step_rows),"steps":step_rows,"all_gradients_finite":all(r["gradients_finite"] for r in step_rows),
        "lwc_nonzero_gradient_each_step":all(r["nonzero_grad_tensor_count"]==14 for r in step_rows),
        "non_lwc_parameters_with_grad":sorted({n for r in step_rows for n in r["non_lwc_parameter_grad_names"]})})
    write_json("LWC_UPDATE_CHECK.json", {"status":"PASS" if updates_ok else "FAIL","optimizer_steps":len(step_rows),
        "steps":[{"sample_index":r["sample_index"],"lwc_changed":r["lwc_changed"],"lwc_update_l1":r["lwc_update_l1"],
                   "changed_parameter_names":r["lwc_changed_parameter_names"]} for r in step_rows],
        "lwc_value_checkpoints":{
            "step_0_before_first_update":step_rows[0]["lwc_before_state"] if step_rows else [],
            "step_1_after_first_update":step_rows[0]["lwc_after_state"] if step_rows else [],
            "step_2_after_second_update":step_rows[-1]["lwc_after_state"] if step_rows else [],
        },
        "step_1_matches_second_update_start":bool(step_rows and step_rows[-1]["lwc_before_state"]==step_rows[0]["lwc_after_state"])})
    write_json("RECONSTRUCTION_SMOKE.json", {"status":"PASS" if finite_losses else "FAIL","objective":"official torch.nn.MSELoss between FP teacher block output and quantized student block output",
        "teacher_source":"same frozen FP Qwen2.5 L0 input; qlayer weight and activation quant initially disabled",
        "student_source":"official QuantLinear LWC/STE path; quantized weights evaluated on the same frozen block inputs",
        "formal_epoch_config":20,"smoke_optimizer_steps":len(step_rows),"losses":metric_rows,
        "step_losses":[{"step":i+1,"pre_update_loss":r["loss_before_update"]} for i,r in enumerate(step_rows)],
        "finite":finite_losses,"monotonic_decrease_required":False})
    with (HERE/"OMNI_STEP_TIMING.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["kind","repeat","warmup_excluded","teacher_forward_seconds","student_quant_forward_seconds","loss_seconds","student_loss_seconds","backward_seconds","optimizer_step_seconds","total_step_seconds","loss"]
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in timing_rows:
            if r["stage"]=="teacher_forward": writer.writerow({"kind":"official_2_sample_teacher_capture","repeat":"combined","warmup_excluded":"not_applicable","teacher_forward_seconds":r["seconds"]})
        for i,r in enumerate(step_rows,1):
            writer.writerow({"kind":"official_compatibility_smoke_update","repeat":i,"warmup_excluded":False,
                "student_quant_forward_seconds":r["student_quant_forward_seconds"],"loss_seconds":r["loss_seconds"],
                "student_loss_seconds":r["student_loss_seconds"],"backward_seconds":r["backward_seconds_excluding_optimizer"],
                "optimizer_step_seconds":r["optimizer_step_seconds"],"total_step_seconds":r["step_total_seconds"],
                "loss":r["loss_before_update"]})
    # Preserve the trained Qwen decoder result as a dense native nn.Linear state dict.
    trained_wrapper=quantized_model.model.layers[0]
    if not hasattr(trained_wrapper,"native_layer"):
        raise RuntimeError("expected Qwen adapter wrapper after OmniQuant; STOP")
    dense_layer=deepcopy(trained_wrapper.native_layer).cpu()
    quant_aux={}
    from models.int_qwen2_layer import _set_submodule
    for name,module in list(dense_layer.named_modules()):
        if isinstance(module,QuantLinear):
            q=module.weight_quantizer
            quant_aux[name]={"scales":q.scales.detach().cpu(),"zeros":q.zeros.detach().cpu(),
                "upbound_factor":q.upbound_factor.detach().cpu(),"lowbound_factor":q.lowbound_factor.detach().cpu(),
                "scales_sha256":tensor_sha(q.scales),"zeros_sha256":tensor_sha(q.zeros)}
            native=nn.Linear(module.in_features,module.out_features,bias=module.bias is not None,dtype=module.weight.dtype)
            with torch.no_grad():
                native.weight.copy_(module.weight.detach().cpu())
                if native.bias is not None: native.bias.copy_(module.bias.detach().cpu())
            _set_submodule(dense_layer,name,native)
    if any(isinstance(m,QuantLinear) for m in dense_layer.modules()):
        raise RuntimeError("dense export retained QuantLinear; STOP")
    artifact=HERE/"L0_DENSE_EXPORT.pt"
    torch.save({"model_revision":REVISION,"layer_index":0,"representation":"dense-dequant FP16 native Qwen2DecoderLayer",
                "state_dict":dense_layer.state_dict(),"quantizer_aux":quant_aux},artifact)
    pristine_reload=deepcopy(pristine_l0)
    payload=torch.load(artifact,map_location="cpu",weights_only=False)
    load_result=pristine_reload.load_state_dict(payload["state_dict"],strict=True)
    pristine_reload=pristine_reload.to(device).eval()
    ref_input=layer_in[0].to(device); ref_kwargs=move_tree(layer_kwargs[0],device)
    dense_layer=dense_layer.to(device).eval()
    with torch.inference_mode():
        dense_pre=dense_layer(ref_input,**ref_kwargs)[0].detach().cpu()
        dense_post=pristine_reload(ref_input,**ref_kwargs)[0].detach().cpu()
    block_reload=maxdiff(dense_pre,dense_post)
    dense_pass=(block_reload["allclose_rtol_1e-5_atol_1e-6"] and block_reload["finite"] and not load_result.missing_keys and not load_result.unexpected_keys)
    write_json("DENSE_EXPORT_RELOAD.json", {"status":"PASS" if dense_pass else "FAIL","artifact":"L0_DENSE_EXPORT.pt",
        "artifact_sha256":sha_file(artifact),"scope":"L0 seven projection weights only; all other L0 decoder modules unchanged FP16",
        "quantlinears_after_export":sum(isinstance(m,QuantLinear) for m in dense_layer.modules()),
        "native_linears_after_export":sum(isinstance(m,nn.Linear) for m in dense_layer.modules()),
        "strict_reload_missing_keys":load_result.missing_keys,"strict_reload_unexpected_keys":load_result.unexpected_keys,
        "block_output_before_vs_after_reload":block_reload,
        "quantizer_auxiliary":{k:{x:{"shape":list(v[x].shape),"sha256":v[x+"_sha256"]} for x in ("scales","zeros")}
                                for k,v in quant_aux.items()}})

    # Replace L0 in the partial smoke model with the reloaded native dense block and run the frozen shared evaluator.
    quantized_model.model.layers[0]=pristine_reload.to("cpu")
    quantized_model=quantized_model.to(device).eval()
    test_manifest=json.loads((V18/"evaluation/manifests/WIKITEXT_TEST_MANIFEST.json").read_text(encoding="utf-8"))
    tiny_ids=torch.tensor(test_manifest["token_ids"][:128],dtype=torch.long)[None]
    from benchmark.eval.ppl import evaluate_ppl
    with torch.inference_mode():
        tiny_ppl,tiny_nll,tiny_tokens=evaluate_ppl(quantized_model,tiny_ids,max_length=128,stride=128,progress_every=0)
    tiny_pass=(tiny_tokens==127 and math.isfinite(tiny_ppl) and math.isfinite(tiny_nll))
    write_json("TINY_EVAL_SMOKE.json", {"status":"PASS" if tiny_pass else "FAIL","method":"shared benchmark.eval.ppl.evaluate_ppl",
        "model_scope":"L0 dense-dequant OmniQuant-smoke weights + remaining 23 native FP16 blocks; not a full M3 checkpoint",
        "evaluation_stream":"first 128 fixed WT2 test IDs; evaluator wiring smoke only",
        "input_tokens":128,"scored_targets":tiny_tokens,"expected_scored_targets":127,"max_length":128,"stride":128,
        "mean_nll":tiny_nll,"ppl":tiny_ppl,"quality_claim":False,
        "token_ids_sha256":sha_bytes(np.asarray(test_manifest["token_ids"][:128],dtype="<i8").tobytes())})

    # Profile five repeated one-step trials plus a discarded warmup on a disposable L0 adapter.
    # Each trial resets all LWC parameters and AdamW state; only the two actual OmniQuant smoke updates are retained.
    profile=QuantQwen2DecoderLayer(config,pristine_l0,adapter_args).to(device).float().train()
    for name,p in profile.named_parameters():
        p.requires_grad_("bound_factor" in name)
    profile_params=list(lwc_parameters(profile))
    optimizer=torch.optim.AdamW([{"params":[],"lr":5e-3},{"params":profile_params,"lr":1e-2}],weight_decay=0.0)
    import utils as omni_utils
    scaler=omni_utils.NativeScalerWithGradNormCount()
    opt_times=[]
    def opt_pre(*_):
        torch.cuda.synchronize(device); opt_times.append(time.perf_counter())
    def opt_post(*_):
        torch.cuda.synchronize(device); opt_times.append(time.perf_counter())
    optimizer.register_step_pre_hook(opt_pre); optimizer.register_step_post_hook(opt_post)
    profile_input=ref_input.detach(); profile_target=native_layer_out[0].to(device)
    profile_kwargs=ref_kwargs
    profile_rows=[]
    for rep in range(6):
        with torch.no_grad():
            for p in profile_params: p.fill_(4.0)
        optimizer.state.clear(); optimizer.zero_grad(set_to_none=True); opt_times.clear()
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device); start=time.perf_counter()
        with torch.cuda.amp.autocast():
            forward_start_event=torch.cuda.Event(enable_timing=True)
            forward_end_event=torch.cuda.Event(enable_timing=True)
            loss_end_event=torch.cuda.Event(enable_timing=True)
            forward_start_event.record()
            smooth_and_quant_temporary(profile,args,isllama=False)
            student=profile(profile_input,**profile_kwargs)[0]
            forward_end_event.record()
            loss=torch.nn.functional.mse_loss(profile_target,student)
            loss_end_event.record()
        torch.cuda.synchronize(device); student_s=time.perf_counter()-start
        student_forward_s=forward_start_event.elapsed_time(forward_end_event)/1000.0
        loss_s=forward_end_event.elapsed_time(loss_end_event)/1000.0
        back_start=time.perf_counter()
        grad_norm=scaler(loss,optimizer,parameters=get_omni_parameters(profile,use_shift=False))
        torch.cuda.synchronize(device); back_all=time.perf_counter()-back_start
        opt_s=opt_times[-1]-opt_times[-2] if len(opt_times)==2 else float("nan")
        total=time.perf_counter()-start
        profile_rows.append({"repeat":rep,"warmup_excluded":rep==0,"student_quant_forward_seconds":student_forward_s,
            "loss_seconds":loss_s,"student_loss_seconds":student_s,
            "backward_seconds":max(0.0,back_all-opt_s),"optimizer_step_seconds":opt_s,"total_step_seconds":total,
            "loss":float(loss.detach().float().item()),"finite":bool(torch.isfinite(loss)),
            "peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device))})
        smooth_clear=getattr(omni_module,"clear_temp_variable",None)
        if smooth_clear is not None: smooth_clear(profile)
    with (HERE/"OMNI_STEP_TIMING.csv").open("a",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["kind","repeat","warmup_excluded","teacher_forward_seconds","student_quant_forward_seconds","loss_seconds","student_loss_seconds","backward_seconds","optimizer_step_seconds","total_step_seconds","loss"])
        for r in profile_rows:
            writer.writerow({"kind":"disposable_reset_one_step_profile","repeat":r["repeat"],"warmup_excluded":r["warmup_excluded"],
                "student_quant_forward_seconds":r["student_quant_forward_seconds"],"loss_seconds":r["loss_seconds"],
                "student_loss_seconds":r["student_loss_seconds"],"backward_seconds":r["backward_seconds"],
                "optimizer_step_seconds":r["optimizer_step_seconds"],"total_step_seconds":r["total_step_seconds"],"loss":r["loss"]})
    measured=[r for r in profile_rows if not r["warmup_excluded"]]
    timing_stats={}
    for field in ("student_quant_forward_seconds","loss_seconds","student_loss_seconds","backward_seconds","optimizer_step_seconds","total_step_seconds"):
        vals=np.asarray([r[field] for r in measured],dtype=np.float64)
        timing_stats[field]={"mean":float(vals.mean()),"median":float(np.median(vals)),"p95":float(np.percentile(vals,95))}
    write_json("VRAM_REPORT.json", {"device":torch.cuda.get_device_name(device),"total_bytes":int(torch.cuda.get_device_properties(device).total_memory),
        "formal_capacity_gib":8,"functional_smoke_peak_allocated_bytes":smoke_peak_alloc,"functional_smoke_peak_reserved_bytes":smoke_peak_reserved,
        "disposable_profile_peak_allocated_bytes":max(r["peak_allocated_bytes"] for r in profile_rows),
        "disposable_profile_peak_reserved_bytes":max(r["peak_reserved_bytes"] for r in profile_rows),
        "smoke_peak_below_8gib":max(smoke_peak_reserved,max(r["peak_reserved_bytes"] for r in profile_rows))<=8*(1024**3),
        "full_2048_token_training_vram_measured":False,"formal_2048_token_vram_feasibility_proven":False,
        "full_run_vram_conclusion":"not proven by the 64-token smoke"})
    write_json("TIMING_SUMMARY.json", {"functional_updates":step_rows,"profile_repeats":profile_rows,
        "profile_repeat_count_excluding_warmup":len(measured),"timing_stats":timing_stats,
        "teacher_forward_capture_seconds":next((r["seconds"] for r in timing_rows if r["stage"]=="teacher_forward"),None),
        "timing_scope":"64-token L0 smoke shape. Disposable profile clones are reset to LWC=4/empty AdamW state each repeat; they are not retained as method results."})

    # Simple transparent shape-scaling envelope; the 64-token measurements cannot validate long-context scaling.
    t=float(timing_stats["total_step_seconds"]["median"])
    total_steps=24*20*128
    factor={"optimistic_linear_sequence":2048/64,"central_effective_exponent_1_5":(2048/64)**1.5,
            "conservative_quadratic_attention":(2048/64)**2}
    hours={k:total_steps*t*v/3600 for k,v in factor.items()}
    estimate_ok=hours["conservative_quadratic_attention"]<=24
    (HERE/"FORMAL_M3_COST_ESTIMATE.md").write_text(
        "# Formal M3 cost estimate (low confidence)\n\n"
        f"Frozen official semantics require 24 blocks × 20 epochs × 128 samples / batch 1 = **{total_steps:,} optimizer updates**. The only step measurements are L0, batch 1, 64 tokens. The median disposable-profile step is **{t:.6f} s**. No 2048-token optimization step was run.\n\n"
        "Scaling scenarios apply the measured 64-token median to sequence factors 32 (all work linear), 32^1.5 (central interpolation), and 32^2 (attention-dominant upper scenario). They omit model/block handoff, initial/final calibration capture, save/export and any additional host overhead, so they are not performance guarantees.\n\n"
        + "\n".join(f"- {k}: factor {factor[k]:.3f}; compute-only estimate **{hours[k]:.2f} h**" for k in factor)
        + f"\n\nThe conservative estimate is {'within' if estimate_ok else 'above'} the frozen 24-hour gate. Because the smoke/full sequence ratio is 32× and the conservative estimate is an unvalidated extrapolation, report it as low confidence. Formal 20-epoch training was not run. The measured 64-token smoke reserved VRAM is reported separately; 2048-token formal-training VRAM feasibility was not measured and does not pass its readiness gate.\n",encoding="utf-8")

    semantics_ok=all(r["algorithm_semantic_change"]=="no" for r in csv.DictReader((HERE/"OMNIQUANT_PATCH_MANIFEST.csv").open(encoding="utf-8")))
    status_checks={"official_commit_and_tree":up_ok,"architecture_only_patch":semantics_ok,
        "quant_off_parity":parity_pass,"seven_projection_discovery":shape_ok,
        "gqa_rope_kwargs":attention_ok,"frozen_calibration_loader":True,
        "lwc_parameters":param_pass,"lwc_gradients":gradients_ok and no_frozen_grads,
        "optimizer_updates_lwc":updates_ok,"reconstruction_finite":finite_losses,
        "dense_export_reload":dense_pass,"shared_evaluator_tiny_smoke":tiny_pass,
        "conservative_runtime_le_24h":estimate_ok,
        "smoke_reserved_vram_le_8gib":max(smoke_peak_reserved,max(r["peak_reserved_bytes"] for r in profile_rows))<=8*(1024**3),
        "formal_2048_token_vram_feasibility_proven":False}
    all_pass=all(status_checks.values())
    readiness="M3_FORMAL_EXECUTION_READY" if all_pass else "M3_FORMAL_EXECUTION_NOT_READY"
    functional_pass=all(status_checks[k] for k in list(status_checks)[:12])
    write_json("correctness_checks.json", {"status":"PASS" if functional_pass else "FAIL",
        "status_scope":"functional correctness gates A-L only; runtime and formal-shape VRAM readiness gates are reported separately",
        "readiness_gates_status":"PASS" if all_pass else "NOT_READY","checks":status_checks,
        "formal_execution_readiness":readiness,"formal_training_executed":False,"formal_ppl_executed":False})
    report=("# v20 OmniQuant Qwen2.5 compatibility smoke\n\n"
        f"**Verdict: `{readiness}`.** This means {'the frozen M3 full calibration may enter a later explicitly requested run' if all_pass else 'the formal full calibration gate is not met'}; no formal training or PPL was run.\n\n"
        f"- Upstream commit/hash gate: {'PASS' if up_ok else 'FAIL'}; quantizer/LWC/optimizer math sources are byte-identical to upstream; the driver has only declared Qwen plumbing and smoke-only controls/observers.\n"
        f"- Quant-off native/port parity on L0/mid/last and logits: {'PASS' if parity_pass else 'FAIL'}, 64 frozen tokens.\n"
        f"- Qwen target modules: {len(module_map)} across {len(model.model.layers)} layers; mapping {'PASS' if shape_ok else 'FAIL'}.\n"
        f"- Attention semantics and native kwargs: {'PASS' if attention_ok else 'FAIL'}; LWC map: {'PASS' if param_pass else 'FAIL'} ({total_lwc:,} scalars).\n"
        f"- Functional LWC smoke: {len(step_rows)} optimizer steps; gradients {'PASS' if gradients_ok and no_frozen_grads else 'FAIL'}, updates {'PASS' if updates_ok else 'FAIL'}, reconstruction {'PASS' if finite_losses else 'FAIL'}.\n"
        f"- Dense L0 export/reload: {'PASS' if dense_pass else 'FAIL'}; shared evaluator: {'PASS' if tiny_pass else 'FAIL'} ({tiny_tokens} scored targets; smoke only).\n"
        f"- Calibration wall for one L0/two-sample smoke: {train_wall:.2f}s. Disposable timing repeats: {len(measured)} after one excluded warmup; L0 64-token median forward {timing_stats['student_quant_forward_seconds']['median']:.4f}s, loss {timing_stats['loss_seconds']['median']:.4f}s, backward {timing_stats['backward_seconds']['median']:.4f}s, optimizer {timing_stats['optimizer_step_seconds']['median']:.4f}s, total {timing_stats['total_step_seconds']['median']:.4f}s.\n"
        f"- Runtime extrapolation: optimistic {hours['optimistic_linear_sequence']:.2f}h, central {hours['central_effective_exponent_1_5']:.2f}h, conservative {hours['conservative_quadratic_attention']:.2f}h; 64-to-2048 scaling is low-confidence.\n"
        f"- Actual smoke peak reserved VRAM: {max(smoke_peak_reserved,max(r['peak_reserved_bytes'] for r in profile_rows))/(1024**3):.2f} GiB (<8 GiB); full 2048-token formal-training feasibility is unproven and its readiness gate is not passed.\n\n"
        "M3 remains a method-level comparator to M1/M2 because OmniQuant uses asymmetric affine quantization, LWC block reconstruction and no GPTQ compensation. Smoke losses/PPL are not model-quality results. STOP.\n")
    (HERE/"REPORT.md").write_text(report,encoding="utf-8")
    print(json.dumps({"readiness":readiness,"checks":status_checks,"runtime_hours":hours,"step_timing":timing_stats,
        "functional_steps":len(step_rows),"tiny_eval_tokens":tiny_tokens},indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
