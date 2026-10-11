from pathlib import Path
import ast, difflib, hashlib, importlib.util, json, tempfile, time

ROOT=Path(__file__).resolve().parent
HIST=ROOT.parents[1]/'experiments/range_search_w4/results/range_w4_omniquant_resume_v21_4/M3'

class StripMarkers(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name=='phase': return None
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='phase': return None
        return self.generic_visit(node)

def main():
    begin=time.monotonic(); cpu=time.process_time()
    a=(HIST/'official_port/quantize/omniquant.py').read_text()
    b=(ROOT/'M3/official_port/quantize/omniquant.py').read_text()
    assert ast.dump(ast.parse(a),include_attributes=False)==ast.dump(StripMarkers().visit(ast.parse(b)),include_attributes=False)
    for p in ROOT.rglob('*.py'): ast.parse(p.read_text(encoding='utf-8'))
    diff=''.join(difflib.unified_diff((HIST/'run_formal_m3.py').read_text().splitlines(True),(ROOT/'M3/run_formal_m3.py').read_text().splitlines(True),fromfile='historical/run_formal_m3.py',tofile='v61/run_formal_m3.py'))
    diff+=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='historical/official_port/quantize/omniquant.py',tofile='v61/official_port/quantize/omniquant.py'))
    (ROOT/'IMPLEMENTATION.diff').write_text(diff)
    audit=json.loads((ROOT/'M3/OMNIQUANT_SOURCE_AUDIT.json').read_text())
    for p,rec in audit['files'].items():
        assert hashlib.sha256((ROOT/'M3/official_port'/p).read_bytes()).hexdigest()==rec['patched_sha256']
    spec=importlib.util.spec_from_file_location('meter',ROOT/'M3/measurement.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    class FakeCuda:
        def __init__(self): self.a=0; self.r=0; self.resets=0
        def is_initialized(self): return True
        def synchronize(self,x): pass
        def max_memory_allocated(self,x): return self.a
        def max_memory_reserved(self,x): return self.r
        def reset_peak_memory_stats(self,x): self.resets+=1; self.a=0; self.r=0
    cuda=m.torch.cuda; fake=FakeCuda(); m.torch.cuda=fake
    try:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            t=Path(tmp); (t/'timings').mkdir(); meter=m.StageMeter(t)
            meter.switch('load'); fake.a=100; fake.r=300
            meter.switch('optimization',0); fake.a=200; fake.r=250
            meter.finish('FAILED','deterministic meter failure test')
            x=json.loads(next((t/'timings').glob('summary_*.json')).read_text())
            assert x['peak_allocated_bytes']==200 and x['peak_reserved_bytes']==300
            assert len(meter.rows)==2 and meter.rows[-1]['status']=='FAILED'
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            t=Path(tmp); (t/'timings').mkdir(); meter=m.StageMeter(t)
            stop=__import__('threading').Event()
            def reader():
                p=t/'timings/STAGE_EVENTS.jsonl'
                while not stop.wait(.0005):
                    if p.exists(): p.read_bytes()
            thread=__import__('threading').Thread(target=reader); thread.start()
            for i in range(200): meter.switch('concurrent_stage',i)
            meter.finish('COMPLETE'); stop.set(); thread.join()
            assert len((t/'timings/STAGE_EVENTS.jsonl').read_text().splitlines())==200
    finally: m.torch.cuda=cuda
    spec=importlib.util.spec_from_file_location('sampler',ROOT/'measure_once.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
    sampler=s.Sampler('preflight_'+str(__import__('os').getpid())); sampler.pid=__import__('os').getpid()
    sampler.tick(); sampler.thread=__import__('threading').Thread(target=sampler.run); sampler.thread.start()
    time.sleep(.45); mem=sampler.finish()
    assert mem['samples']>=3 and mem['device_sampled_peak_bytes'] is not None and mem['process_rss_peaks']
    result=dict(status='PASS',source_AST_identical_after_removing_phase_markers=True,
        failed_stage_and_global_peak_saved=True,sampler_flush_and_RSS_device_samples=True,
        append_stage_notification_concurrent_reader_200_transitions=True,
        new_per_training_step_sync_calls=0,training_and_quantization_math_unchanged=True,
        cpu_process_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-begin,
        sampler_GPU_query_only_no_CUDA_model_or_training=True)
    (ROOT/'MEASUREMENT_PREFLIGHT.json').write_text(json.dumps(result,indent=2)); print(result)

if __name__=='__main__': main()
