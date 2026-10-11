from pathlib import Path
import hashlib, json, shutil, difflib

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
RESULTS = REPO / 'experiments/range_search_w4/results'
HIST = RESULTS / 'range_w4_omniquant_resume_v21_4'
SRC = HIST / 'M3'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def put(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def replace_once(s, a, b):
    if s.count(a) != 1: raise RuntimeError(('replacement count', a[:100], s.count(a)))
    return s.replace(a,b)

def main():
    dst = OUT / 'M3'
    if dst.exists() and {p.name for p in dst.iterdir()} != {'measurement.py'}:
        raise RuntimeError('measurement source already exists; refusing rebuild')
    for p in [OUT, dst, dst/'logs', dst/'checkpoints', OUT/'memory', OUT/'timings']:
        p.mkdir(parents=True, exist_ok=True)
        probe = p/'WRITE_PROBE.txt'
        with probe.open('x') as f: f.write('writable before calibration\n')
    shutil.copytree(SRC/'official_port', dst/'official_port', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['LOCAL_MODEL_MANIFEST.json','TOKENIZER_CHECK.json']:
        shutil.copy2(HIST/name, OUT/name)
    protocol = json.loads((HIST/'PROTOCOL.json').read_text())
    protocol.update(protocol_id='v61_repository_M3_LWC_cost_memory', outputs_root=str(OUT),
        method_name='repository M3/LWC implementation', historical_source=str(SRC),
        measurement_only=True, start_from='pristine FP; no historical M3 parameters loaded',
        workload='24 blocks x 20 epochs x 128 samples; batch=1, seqlen=2048',
        no_new_ppl=True, official_full_reproduction_claim=False,
        timing_rule='monotonic; GPU sync only at newly added stage boundaries; pre-existing step sync/checks retained',
        sampler='external supervisor; target period 0.2 s; device total and each accessible process RSS; flush each tick',
        allocator_rule='save each stage peak before reset; full peak is max of all stages, including failed stage',
        diagnostic_scope='v21.4 existing finite/AMP/checkpoint diagnostics retained for all blocks; historical prefix diagnostics were less detailed',
        cost_comparison='new measured run, not reconstruction of historical elapsed time; no M1/M2 speed ratio',
        permitted_changes=['paths/output labels', 'phase markers/measurement', 'manifest return metadata repair only'])
    put(OUT/'PROTOCOL.json', protocol)
    runner0 = (SRC/'run_formal_m3.py').read_text(encoding='utf-8')
    runner = runner0.replace('PREFLIGHT = ROOT / "LOCAL_PREFLIGHT"', f'PREFLIGHT = Path({str(HIST/"LOCAL_PREFLIGHT")!r})')
    runner = runner.replace('BUNDLE = ROOT.parent / "range_w4_omniquant_kaggle_v20_1" / "kaggle_bundle"',
        f'BUNDLE = Path({str(RESULTS/"range_w4_omniquant_kaggle_v20_1/kaggle_bundle")!r})')
    runner = replace_once(runner, '    return checkpoint_dir / manifest["state_file"], completed, {"rng_state": rng_state, "state": state}',
        '    return checkpoint_dir / manifest["state_file"], completed, {"rng_state": rng_state, "state": state, "manifest": manifest}')
    runner = replace_once(runner, '    config, calibration, token_ids, lock = frozen_context()',
        '    METER.switch("source_validation")\n    config, calibration, token_ids, lock = frozen_context()')
    runner = replace_once(runner, '    model, identity = load_exact_model(', '    METER.switch("model_load")\n    model, identity = load_exact_model(')
    runner = replace_once(runner, '    model, identity = load_exact_model(MODEL_ID, MODEL_REVISION, str(FROZEN_MODEL), DEVICE, HERE)',
        '    model, identity = load_exact_model(MODEL_ID, MODEL_REVISION, str(FROZEN_MODEL), DEVICE, HERE)\n    METER.record_runtime(model, args)')
    runner = replace_once(runner, '    tokenizer = AutoTokenizer.from_pretrained(str(FROZEN_MODEL), local_files_only=True, use_fast=True)',
        '    METER.switch("data_prepare")\n    tokenizer = AutoTokenizer.from_pretrained(str(FROZEN_MODEL), local_files_only=True, use_fast=True)')
    runner = replace_once(runner, '    counts, csv_logs = install_source_callbacks(args, lock, resume_state, attempt_id)',
        '    counts, csv_logs = install_source_callbacks(args, lock, resume_state, attempt_id)\n    args.v61_phase_callback = METER.switch')
    runner = replace_once(runner, '            record = save_block_checkpoint(value, lock, original_torch_save, args.v21_count_state)',
        '            METER.switch("checkpoint_save", len(value)-1)\n            record = save_block_checkpoint(value, lock, original_torch_save, args.v21_count_state)\n            METER.switch("block_cleanup", len(value)-1)')
    runner = runner.replace('    torch.cuda.reset_peak_memory_stats(DEVICE)', '    # stage/global peak bookkeeping is owned by METER')
    runner = replace_once(runner, '    export_total_started = time.perf_counter()',
        '    METER.switch("export_finalize")\n    export_total_started = time.perf_counter()')
    runner = replace_once(runner, '    export_started = time.perf_counter()',
        '    METER.switch("model_save")\n    export_started = time.perf_counter()')
    runner = replace_once(runner, '    reload_started = time.perf_counter()',
        '    METER.switch("reload_check")\n    reload_started = time.perf_counter()')
    runner = replace_once(runner, 'if __name__ == "__main__":\n    raise SystemExit(main())',
        'if __name__ == "__main__":\n    from measurement import StageMeter\n    METER = StageMeter(ROOT)\n    try:\n        METER.switch("runtime_setup")\n        result = main()\n    except BaseException as exc:\n        METER.finish("FAILED", repr(exc))\n        raise\n    else:\n        METER.finish("COMPLETE")\n        raise SystemExit(result)')
    (dst/'run_formal_m3.py').write_text(runner, encoding='utf-8')
    srcp = SRC/'official_port/quantize/omniquant.py'
    port0 = srcp.read_text(encoding='utf-8')
    port = replace_once(port0, '    logger.info("Starting ...")',
        '    def phase(name, block=-1):\n        args.v61_phase_callback(name, block)\n    phase("initial_activation_capture")\n    logger.info("Starting ...")')
    port = replace_once(port, '        logger.info(f"=== Start quantize layer {i} ===")',
        '        phase("block_setup", i)\n        logger.info(f"=== Start quantize layer {i} ===")')
    port = replace_once(port, '                    teacher_started = __import__("time").perf_counter()',
        '                    phase("teacher_capture", i)\n                    teacher_started = __import__("time").perf_counter()')
    port = replace_once(port, '        # init smooth parameters', '        phase("block_setup", i)\n        # init smooth parameters')
    port = replace_once(port, '        if args.epochs > 0:\n            with torch.no_grad():\n                qlayer.float()',
        '        if args.epochs > 0:\n            phase("optimization", i)\n            with torch.no_grad():\n                qlayer.float()')
    port = replace_once(port, '        qlayer.half() \n        # real smooth and quantization',
        '        phase("quantization_finalize", i)\n        qlayer.half() \n        # real smooth and quantization')
    port = replace_once(port, '            # update input of quantization model',
        '            phase("student_activation_refresh", i)\n            # update input of quantization model')
    port = replace_once(port, '            register_scales_and_zeros(qlayer)\n            layers[i] = qlayer.to("cpu")\n            omni_parameters[i]',
        '            phase("quantization_finalize", i)\n            register_scales_and_zeros(qlayer)\n            layers[i] = qlayer.to("cpu")\n            omni_parameters[i]')
    port = replace_once(port, '    del inps', '    phase("calibration_cleanup")\n    del inps')
    (dst/'official_port/quantize/omniquant.py').write_text(port, encoding='utf-8')
    audit=json.loads((SRC/'OMNIQUANT_SOURCE_AUDIT.json').read_text())
    audit['measurement_parent_audit_sha256']=sha(SRC/'OMNIQUANT_SOURCE_AUDIT.json')
    for name, rec in audit['files'].items():
        rec['historical_effective_sha256']=sha(SRC/'official_port'/name)
        rec['patched_sha256']=sha(dst/'official_port'/name)
    put(dst/'OMNIQUANT_SOURCE_AUDIT.json',audit)
    diff=''.join(difflib.unified_diff(runner0.splitlines(True),runner.splitlines(True),
        fromfile='historical/run_formal_m3.py',tofile='v61/run_formal_m3.py'))
    diff+=''.join(difflib.unified_diff(port0.splitlines(True),port.splitlines(True),
        fromfile='historical/official_port/quantize/omniquant.py',tofile='v61/official_port/quantize/omniquant.py'))
    (OUT/'IMPLEMENTATION.diff').write_text(diff,encoding='utf-8')
    paths=[SRC/'run_formal_m3.py', SRC/'OMNIQUANT_SOURCE_AUDIT.json',HIST/'PROTOCOL.json',
        RESULTS/'range_w4_omniquant_full_v21_1/M3/FORMAL_PROTOCOL_LOCK.json',
        RESULTS/'range_w4_omniquant_finalize_v21_5/FINAL_M3_ARTIFACT_AUDIT.json',
        RESULTS/'range_w4_omniquant_finalize_v21_5/M3_FORMAL_RUNTIME_FINAL.json',
        RESULTS/'range_w4_omniquant_kaggle_v20_1/kaggle_bundle/configs/m3_preflight.json',
        RESULTS/'range_w4_omniquant_kaggle_v20_1/kaggle_bundle/calibration/CALIBRATION_MANIFEST.json']
    paths += [SRC/'official_port'/x for x in audit['files']]
    paths += [SRC/'checkpoints/block_23/omni_parameters.pth',SRC/'dense_export/model.safetensors']
    put(OUT/'SOURCE_AUDIT.json', {'status':'FROZEN','sources':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths],
        'algorithm_change':False,'original_finalization_error':'verify_latest_checkpoint metadata did not return manifest; added only return metadata',
        'history_missing':['exact original full internal wall','prefix 0-20 direct optimizer hook/found_inf snapshots'],
        'fresh_training_not_bitwise_historical_claim':True})
    print(OUT)

if __name__=='__main__': main()
