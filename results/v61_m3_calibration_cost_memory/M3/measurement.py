from pathlib import Path
import csv, json, os, time, sys, hashlib, platform
import torch

class StageMeter:
    def __init__(self, root):
        self.root=Path(root)
        self.attempt=time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
        self.file=(self.root/'timings'/('stages_'+self.attempt+'.csv')).open('x',newline='')
        self.fields=['attempt','phase','block','start_monotonic_ns','end_monotonic_ns','wall_seconds','cpu_process_seconds',
            'allocated_peak_bytes','allocated_peak_GiB','reserved_peak_bytes','reserved_peak_GiB','status']
        self.writer=csv.DictWriter(self.file,fieldnames=self.fields); self.writer.writeheader(); self.file.flush()
        self.phase=None; self.rows=[]; self.global_a=0; self.global_r=0
        self.start=time.monotonic_ns(); self.cpu_start=time.process_time_ns()
        self.statefile=self.root/'timings'/'STAGE_EVENTS.jsonl'

    def sync(self):
        if torch.cuda.is_initialized(): torch.cuda.synchronize(0)

    def switch(self, phase, block=-1):
        self.sync()
        now=time.monotonic_ns(); cpu=time.process_time_ns()
        self.close_phase(now,cpu,'COMPLETE')
        self.phase=phase; self.block=block
        if torch.cuda.is_initialized(): torch.cuda.reset_peak_memory_stats(0)
        self.phase_start=now; self.phase_cpu=cpu
        with self.statefile.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'phase':phase,'block':block,'attempt':self.attempt,'workload_pid':os.getpid(),'monotonic_ns':now})+'\n')
            stream.flush()

    def record_runtime(self, model, args):
        import transformers, tokenizers, inspect
        source=Path(inspect.getfile(type(model.model.layers[0])))
        runtime=dict(python=sys.version,executable=sys.executable,pid=os.getpid(),platform=platform.platform(),
            torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,
            cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
            gpu_capability=list(torch.cuda.get_device_capability(0)),device='cuda:0',
            model_code=str(source),model_code_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            effective_model_config=model.config.to_dict(),attention=model.config._attn_implementation,
            attention_module=type(model.model.layers[0].self_attn).__name__,model_training=model.training,
            parameter_dtype=str(next(model.parameters()).dtype),use_cache_before_calibration=model.config.use_cache,
            calibration_temporarily_disables_use_cache=True,
            args={k:v for k,v in vars(args).items() if isinstance(v,(str,int,float,bool,type(None),dict,list))},
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
            flash_sdp=torch.backends.cuda.flash_sdp_enabled(),memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math_sdp=torch.backends.cuda.math_sdp_enabled(),torch_num_threads=torch.get_num_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
            optimizer='AdamW: LWC-only owned parameters; LET group empty',optimizer_default_betas=[.9,.999],optimizer_default_eps=1e-8,
            epochs=20,batch_size=1,samples=128,sequence_length=2048,
            precision='FP16 model/input/teacher, qlayer.float for optimization under native CUDA FP16 autocast + dynamic GradScaler, qlayer.half finalization',
            sampler_thread_is_outside_workload=True)
        (self.root/'ACTUAL_RUNTIME.json').write_text(json.dumps(runtime,indent=2,default=str))

    def close_phase(self, now, cpu, status):
        if self.phase is None: return
        a=int(torch.cuda.max_memory_allocated(0)) if torch.cuda.is_initialized() else 0
        r=int(torch.cuda.max_memory_reserved(0)) if torch.cuda.is_initialized() else 0
        self.global_a=max(self.global_a,a); self.global_r=max(self.global_r,r)
        row=dict(attempt=self.attempt,phase=self.phase,block=self.block,
            start_monotonic_ns=self.phase_start,end_monotonic_ns=now,
            wall_seconds=(now-self.phase_start)/1e9,cpu_process_seconds=(cpu-self.phase_cpu)/1e9,
            allocated_peak_bytes=a,allocated_peak_GiB=a/2**30,reserved_peak_bytes=r,reserved_peak_GiB=r/2**30,status=status)
        self.writer.writerow(row); self.file.flush(); self.rows.append(row); self.phase=None

    def finish(self,status,error=None):
        sync_error=None
        try: self.sync()
        except BaseException as e: sync_error=repr(e)
        now=time.monotonic_ns(); cpu=time.process_time_ns()
        self.close_phase(now,cpu,status)
        self.file.close()
        summary=dict(status=status,error=error,sync_error=sync_error,attempt=self.attempt,
            wall_seconds=(now-self.start)/1e9,cpu_process_seconds=(cpu-self.cpu_start)/1e9,
            peak_allocated_bytes=self.global_a,peak_allocated_GiB=self.global_a/2**30,
            peak_reserved_bytes=self.global_r,peak_reserved_GiB=self.global_r/2**30,
            peak_accounting='maximum of saved stage peaks; all reset boundaries covered; failed phase retained',
            stages=len(self.rows),stage_csv=str(Path(self.file.name)))
        (self.root/'timings'/('summary_'+self.attempt+'.json')).write_text(json.dumps(summary,indent=2))
