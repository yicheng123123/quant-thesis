"""Derive a narrowly M2-bound runner from the frozen v49 runner template."""
import hashlib, json, time
from pathlib import Path

ROOT = Path(r"E:\quant-thesis")
BASE = ROOT / "experiments/range_search_w4/results"
OUT = BASE / "v51_m2_fast_integer_repair_fullmodel"
V49 = BASE / "v49_fullmodel_fast_integer_repair"
V45 = BASE / "v45_m2_single_integer_repair_fullmodel"
V18 = BASE / "range_w4_fullmodel_v18"
V46 = BASE / "v46_m1_single_integer_repair_fullmodel"
V48 = BASE / "v48_fast_numerical_acceptance"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def replace_once(text, old, new, tag):
    n = text.count(old)
    assert n == 1, f"{tag}: expected one match, got {n}"
    return text.replace(old, new, 1)

def main():
    protocol_path = OUT / "PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_BEFORE_GPU"
    if (OUT / "GPU_START.json").exists():
        started = json.loads((OUT / "GPU_START.json").read_text(encoding="utf-8"))
        assert started["protocol_sha256"] == sha(protocol_path), "never change the protocol after GPU start"

    # Record and freeze the same Torch build used by the completed v49 execution.
    v49_status = json.loads((V49 / "RUN_STATUS.json").read_text(encoding="utf-8"))
    assert v49_status["torch"] == "2.6.0+cu124"
    protocol["execution_runtime"] = {
        "python_executable": str(ROOT / ".venv/Scripts/python.exe"),
        "torch_version_required": "2.6.0+cu124",
        "cuda_build_required": "12.4",
        "source": "v49 completed RUN_STATUS.json; exact matching repository venv selected",
        "device_settings": "TF32 off; FP32 matmul highest; unchanged v49 runtime",
    }
    protocol["initialization"]["alpha_grid"] = [0.1 * i for i in range(1, 11)]
    protocol["initialization"]["alpha_selection_source"] = "already completed formal v18 M2; no v51 alpha search or regeneration"
    protocol["pristine_fp_weights"] = json.loads((OUT / "SOURCE_AUDIT.json").read_text(encoding="utf-8"))["pristine_fp_weight_files"]
    protocol.setdefault("freeze_epoch", time.time())
    tmp = protocol_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(protocol_path)
    amendment = {
        "amendment": "001", "before_gpu": True,
        "reason": "Bind the exact Torch/CUDA build used by v49 and record the frozen M2 alpha-grid source and pristine FP hash list; no algorithm or acceptance rule change.",
        "torch": "2.6.0+cu124", "cuda_build": "12.4",
        "evidence": str(V49 / "RUN_STATUS.json"), "evidence_sha256": sha(V49 / "RUN_STATUS.json"),
        "protocol_sha256_after_amendment": sha(protocol_path), "timestamp_epoch": time.time(),
    }
    (OUT / "PROTOCOL_AMENDMENT_001.json").write_text(json.dumps(amendment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    template_path = V49 / "run_v49.py"
    text = template_path.read_text(encoding="utf-8")
    text = replace_once(text,
        "OUT=BASE/'v49_fullmodel_fast_integer_repair';V46=BASE/'v46_m1_single_integer_repair_fullmodel';V48=BASE/'v48_fast_numerical_acceptance';V18=BASE/'range_w4_fullmodel_v18'",
        "OUT=BASE/'v51_m2_fast_integer_repair_fullmodel';V46=BASE/'v46_m1_single_integer_repair_fullmodel';V45=BASE/'v45_m2_single_integer_repair_fullmodel';V49=BASE/'v49_fullmodel_fast_integer_repair';V48=BASE/'v48_fast_numerical_acceptance';V18=BASE/'range_w4_fullmodel_v18'",
        "result root and historical source bindings")
    text = replace_once(text,
        "model=AutoModelForCausalLM.from_pretrained(str(V18/'M1/checkpoint/final'),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()",
        "model=AutoModelForCausalLM.from_pretrained(str(V18/'M2/checkpoint/final'),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()",
        "M2 initialization load")
    text = replace_once(text,
        "source_m1=V18/'M1/checkpoint/final/model.safetensors';assert sha(source_m1)==read('PROTOCOL.json')['initialization']['sha256']",
        "source_m2=V18/'M2/checkpoint/final/model.safetensors';assert sha(source_m2)==read('PROTOCOL.json')['initialization']['model_sha256']",
        "M2 source model hash")
    text = replace_once(text,
        "rt=loadmod('run_v42',V46/'runtime_v46.py');rt.OUT=OUT;rt.WALL_START=None;rt.START=None;rt.deadline=deadline",
        "rt=loadmod('run_v42',V46/'runtime_v46.py');rt.OUT=OUT;rt.WALL_START=None;rt.START=None;rt.deadline=deadline\nassert Path(rt.FP).resolve()==(ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775').resolve(),'pristine FP target source mismatch'",
        "pristine FP source binding")
    # Assert the frozen code is the imported v48 module, and this runner itself uses the matched runtime.
    text = replace_once(text,
        "assert read('PREFLIGHT.json')['pass'] and read('PROTOCOL.json')['status']=='FROZEN_BEFORE_GPU'",
        "assert read('PREPARATION.json')['status']=='CPU_SOURCE_AUDIT_PASS' and read('PROTOCOL.json')['status']=='FROZEN_BEFORE_GPU'\n assert torch.__version__ == '2.6.0+cu124', f'v49 runtime mismatch: {torch.__version__}'",
        "v51 preflight and exact runtime gate")
    text = replace_once(text,
        "assert sha(V48/'fast_accept.py')==read('PROTOCOL.json')['algorithm_source']['sha256']",
        "assert sha(V48/'fast_accept.py')==read('PROTOCOL.json')['algorithm_source']['fast_accept_sha256']",
        "v51 fast acceptance source hash key")
    text = replace_once(text,
        "return sum(vals),max(vals),read('COST_PROJECTION.json')",
        "return sum(vals),max(vals),json.loads((V46/'COST_PROJECTION.json').read_text(encoding='utf8'))",
        "read-only v46 cost projection source")

    start = text.index("  mcfg=V18/")
    end = text.index("  if model is not None:del model", start)
    replacement = r'''  mcfg=V18/'M2/checkpoint/final/config.json';outcfg=OUT/'R1/checkpoint/final/config.json'
  cfg=json.loads(mcfg.read_text(encoding='utf8'));cfg2=json.loads(outcfg.read_text(encoding='utf8'));ignore={'_name_or_path','transformers_version'}
  r2cfg=json.loads((V45/'R/checkpoint/final/config.json').read_text(encoding='utf8'))
  config_equal=({k:v for k,v in cfg.items() if k not in ignore}=={k:v for k,v in cfg2.items() if k not in ignore}=={k:v for k,v in r2cfg.items() if k not in ignore});assert config_equal
  modelpath=OUT/'R1/checkpoint/final/model.safetensors';refpath=V45/'R/checkpoint/final/model.safetensors';m2path=V18/'M2/checkpoint/final/model.safetensors';diffs=[];diff_m2=[];comp_start=time.perf_counter()
  with safe_open(str(modelpath),framework='pt',device='cpu') as ours,safe_open(str(refpath),framework='pt',device='cpu') as r2,safe_open(str(m2path),framework='pt',device='cpu') as m2:
   assert set(ours.keys())==set(r2.keys())==set(m2.keys())
   for key in ours.keys():
    a=ours.get_tensor(key);b=r2.get_tensor(key);c=m2.get_tensor(key)
    if not torch.equal(a.view(torch.uint8),b.view(torch.uint8)):
     diffs.append({'key':key,'v51_tensor_sha256':tensor_hash(a),'original_R2_tensor_sha256':tensor_hash(b),'max_abs_v51_minus_R2':float((a.float()-b.float()).abs().max()),'unequal_bytes':int((a.view(torch.uint8)!=b.view(torch.uint8)).sum())})
    if not torch.equal(a.view(torch.uint8),c.view(torch.uint8)):
     diff_m2.append(key)
  tensor_key_count=0
  with safe_open(str(modelpath),framework='pt',device='cpu') as f:tensor_key_count=len(f.keys())
  rt.TIMINGS.append({'mode':'deployment','layer':-1,'module':'all_model_tensors','stage':'named_tensor_comparison_vs_original_R2','seconds':time.perf_counter()-comp_start,'elapsed_seconds':elapsed(),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'type':'measured CPU wall'})
  quant_rows=[]
  for li in range(24):
   src=ORIG.get(li) or torch.load(OUT/f'original_codes/layer_{li:02d}.pt',map_location='cpu',weights_only=True)
   for g in GROUP_ORDER:
    ck=torch.load(OUT/'formal/groups'/f'layer_{li:02d}_{g}.pt',map_location='cpu',weights_only=False);assert_group(ck,li,g,src)
    refcpgroup=torch.load(V45/'formal/groups'/f'layer_{li:02d}_{g}.pt',map_location='cpu',weights_only=False)
    for name,res in ck['modules'].items():
     refcp=refcpgroup['modules'][name]
     quant_rows.append({'layer':li,'group':g,'module':name,'codes_equal_original_R2':torch.equal(res['codes'],refcp['codes']),'scales_equal_M2_R2':torch.equal(res['scales'],refcp['scales']),'zeros_equal_M2_R2':torch.equal(res['zeros'],refcp['zeros']),'g_idx_equal_M2_R2':torch.equal(res['g_idx'],refcp['g_idx']),'FP16_grid_weights_equal_R2':torch.equal(res['weight'],refcp['weight']),'v51_input_sha256':ck['input_sha256'],'original_R2_input_sha256':refcpgroup['input_sha256'],'input_matches_original_R2':ck['input_sha256']==refcpgroup['input_sha256'],'v51_C_sha256':ck['C_sha256'],'original_R2_C_sha256':refcpgroup['C_sha256']})
  rt.csvwrite('reports/MODULE_VS_ORIGINAL_R2.csv',quant_rows);rt.csvwrite('reports/TENSOR_DIFFERENCES_VS_ORIGINAL_R2.csv',diffs)
  equal=(not diffs) and config_equal
  lock={'final_model_sha256':sha(modelpath),'config_sha256':sha(outcfg),'protocol_sha256':sha(OUT/'PROTOCOL.json'),'group_checkpoints':[{'path':str(p),'sha256':sha(p)} for p in sorted((OUT/'formal/groups').glob('*.pt'))],'modules_compared_with_original_R2':len(quant_rows),'named_model_tensor_differences_vs_original_R2':len(diffs),'inference_config_equal_original_R2':config_equal,'equivalent_to_original_R2':equal,'locked_before_evaluation_or_reuse_unix':time.time()}
  assert len(lock['group_checkpoints'])==96;dump('CHECKPOINT_LOCK.json',lock)
  deployment={'pass':True,'all_168_grid_pack_reload_and_sample_output_checks':True,'model_sha256':sha(modelpath),'config_sha256':sha(outcfg),'named_model_tensors_equal_original_R2':not diffs,'M2_quantized_target_tensors_different':len(diff_m2),'original_R2_model_sha256':sha(refpath),'M2_checkpoint_sha256':sha(m2path),'config_equal_original_R2':config_equal,'max_abs_diff_summary':max((d['max_abs_v51_minus_R2'] for d in diffs),default=0.0),'tensor_keys_compared':tensor_key_count,'completed_unix':time.time()};dump('DEPLOYMENT_COMPARISON.json',deployment)
  audit=read(V46/'PPL_SOURCE_AUDIT.json');copies=[]
  if equal:
   # The only PPL reuse path is exact named-tensor and normalized-config equality to the audited old R2.
   for ds,key in [('wikitext','wikitext'),('c4_fixed_budget','c4_fixed_budget')]:
    record=audit['historical_source_records'][f'R2_{key}'];srcres=Path(record['result_path']);srcwin=Path(record['windows_path'])
    assert sha(srcres)==record['result_sha256'] and sha(srcwin)==record['windows_sha256']
    dstres=OUT/f'evaluation/M2_FAST_{ds}.json';dstwin=OUT/f'evaluation/M2_FAST_{ds}_WINDOWS.csv'
    shutil.copy2(srcres,dstres);shutil.copy2(srcwin,dstwin)
    copies.append({'dataset':ds,'result_source_sha256':sha(srcres),'result_copy_sha256':sha(dstres),'windows_source_sha256':sha(srcwin),'windows_copy_sha256':sha(dstwin),'PPL':json.loads(srcres.read_text())['ppl'],'NLL':json.loads(srcres.read_text())['nll'],'valid_tokens':record['valid_tokens'],'windows':record['windows'],'reused_because_all_named_tensors_and_config_equal_original_R2':True})
   rt.dump('QUALITY_REUSE_SOURCE.json',{'pass':True,'new_PPL_evaluations':0,'reason':'All named dense model tensors and normalized inference config equal v45 original R2 bitwise/config-normalized.','P3_read':False,'source_audit_sha256':sha(V46/'PPL_SOURCE_AUDIT.json'),'reused_results':copies})
  else:
   rt.MODE='evaluation';assert rt.P['v18']['source_protocol']=={'v16_wikitext_manifest_sha256':EVAL_KEY_MAP['wikitext'],'v16_c4_manifest_sha256':EVAL_KEY_MAP['c4_fixed_budget']}
   rt.evaluate(model)
   for ds in ('wikitext','c4_fixed_budget'):
    srcres=OUT/f'R1_{ds}.json';srcwin=OUT/f'R1_{ds}_WINDOWS.csv';dstres=OUT/f'evaluation/M2_FAST_{ds}.json';dstwin=OUT/f'evaluation/M2_FAST_{ds}_WINDOWS.csv'
    assert srcres.exists() and srcwin.exists();srcres.replace(dstres);srcwin.replace(dstwin)
    res=json.loads(dstres.read_text(encoding='utf8'))
    assert res['model_sha256']==sha(modelpath) and res['manifest_sha256']==EVAL_KEY_MAP['wikitext' if ds=='wikitext' else 'c4_fixed_budget']
    copies.append({'dataset':ds,'result_sha256':sha(dstres),'windows_sha256':sha(dstwin),'PPL':res['ppl'],'NLL':res['nll'],'valid_tokens':res['scored_tokens'],'windows':res['windows'],'fresh_frozen_protocol_evaluation':True})
   rt.dump('QUALITY_REUSE_SOURCE.json',{'pass':True,'new_PPL_evaluations':2,'reason':'At least one named model tensor differed from original R2; one frozen evaluation per corpus after lock.','P3_read':False,'evaluator_code_sha256':sha(V46/'runtime_v46.py'),'source_manifests':hashes,'new_results':copies})
'''
    text = text[:start] + replacement + text[end:]
    text = text.replace("'equivalent_to_R1':equal,'PPL_run':not equal,'PPL_reused':equal", "'equivalent_to_original_R2':equal,'PPL_run':not equal,'PPL_reused':equal")
    text = text.replace("'equivalent_to_R1':equal,'model_ready_unix'", "'equivalent_to_original_R2':equal,'model_ready_unix'")
    text = text.replace("'v49 180-minute cumulative GPU budget exhausted", "'v51 180-minute cumulative GPU budget exhausted")
    text = text.replace("'source_code_not_modified':True", "'source_code_not_modified':True")
    out = OUT / "run_v51.py"
    out.write_text(text, encoding="utf-8", newline="")
    # Compile before any GPU use; this is a CPU syntax check only.
    compile(text, str(out), "exec")
    report = {
        "pass": True, "source_template": str(template_path), "source_template_sha256": sha(template_path),
        "derived_runner": str(out), "derived_runner_sha256": sha(out),
        "frozen_core_sha256": protocol["algorithm_source"]["fast_accept_sha256"],
        "changes": ["new v51 output root", "load official M2 checkpoint and validate its hash", "compare deployment tensors/config with original v45 R2", "reuse R2 PPL only after full equality, otherwise run frozen evaluator exactly once per corpus", "read-only v46 cost-projection source for budget gates", "v51 output metadata"],
        "search_core_modified": False, "search_budget_neighborhood_order_and_acceptance_changed": False,
        "v50_mixed_gram_imported": False, "compiled_cpu_only": True,
    }
    (OUT / "RUNNER_DERIVATION.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("V51_RUNNER_DERIVED_AND_COMPILED", sha(out), flush=True)

if __name__ == "__main__":
    main()
