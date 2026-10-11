"""v26 bounded full-shape M2 factor profile and conditional native-equivalent factor cache.
All writes stay under this directory. The installed GPTQ quantize() body is untouched.
"""
import csv,gc,hashlib,json,os,statistics,sys,time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM
O=Path(__file__).resolve().parent;R=O.parents[3];V=O.parent
V18=V/'range_w4_fullmodel_v18';V173=V/'range_w4_m2_runtime_v17_3';V171=V/'range_w4_m2_parity_v17_1'
sys.path[:0]=[str(V18),str(V173),str(V171)]
import run_precheck as rp
import optimized_native_m2 as om
from native_m2 import native_dynamic_gptq
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.quantizer import Quantizer
from gptqmodel.quantization import QuantizeConfig
P=json.loads((O/'PROTOCOL.json').read_text('utf-8'))
CAL=json.loads((V18/'CALIBRATION_MANIFEST.json').read_text('utf-8'))
CV=json.loads((V18/'CALIBRATION_VALIDATION.json').read_text('utf-8'))
D=torch.device('cuda:0'); N=128; SEQ=2048
BASE=R/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors'
FINAL=V18/'M2/checkpoint/final'; WNAMES=['up_proj','down_proj']
ACT=O/'cache'; ACT.mkdir(exist_ok=True)
RAW=O/'RAW_EVENTS.jsonl'
def write(name,obj):
 p=O/name;temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf-8');os.replace(temp,p)
def event(kind,**payload):
 with RAW.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(kind=kind,**payload),ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def sync():torch.cuda.synchronize()
def timed(fn):
 sync();s=time.perf_counter();r=fn();sync();return r,time.perf_counter()-s
class Budget:
 def __init__(self):
  old=json.loads((O/'RUN_STATUS.json').read_text('utf-8')) if (O/'RUN_STATUS.json').exists() else {}
  self.previous=float(old.get('gpu_wall_consumed_seconds',0));self.start=time.perf_counter();self.limit=600
 def used(self):return self.previous+time.perf_counter()-self.start
 def check(self,label,reserve=20):
  if self.used()+reserve>self.limit:
   event('budget_stop',label=label,seconds=self.used());write('RUN_STATUS.json',dict(status='BUDGET_STOP',gpu_wall_consumed_seconds=self.used(),label=label));raise RuntimeError('v26 600s GPU budget reached before '+label)
 def save(self,status,**extra):write('RUN_STATUS.json',dict(status=status,gpu_wall_consumed_seconds=self.used(),**extra))
def verify_sources():
 for rel,expected in P['input_sha256'].items():
  if sha(R/rel)!=expected:raise RuntimeError('Frozen source changed: '+rel)
 assert sha(V18/'CALIBRATION_MANIFEST.json')==CV['manifest_sha256']
 assert len(CAL['samples'])==128 and all(len(x['token_ids'])==2048 for x in CAL['samples'])
 packed=np.asarray([t for x in CAL['samples'] for t in x['token_ids']],dtype='<i8')
 assert hashlib.sha256(packed.tobytes()).hexdigest()==CV['ordered_token_ids_sha256_i64le']
def load_model_and_weights(budget):
 budget.check('model_load',90)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 t=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(str(FINAL),torch_dtype=torch.float16,low_cpu_mem_usage=True,local_files_only=True,attn_implementation='sdpa').eval().to(D);sync()
 event('model_loaded',seconds=time.perf_counter()-t,checkpoint_sha256=P['input_sha256'][str((FINAL/'model.safetensors').relative_to(R))]);budget.save('MODEL_LOADED')
 with safe_open(str(BASE),framework='pt',device='cpu') as f:
  pristine={name:f.get_tensor('model.layers.0.mlp.'+name+'.weight').to(device=D,dtype=torch.float16).contiguous() for name in WNAMES}
 return model,pristine
def capture_inputs(model,budget):
 shape={'up_proj':896,'down_proj':4864};names={n:ACT/f'L0_{n}_128x2048.f16' for n in WNAMES}
 manifest=O/'ACTIVATION_CACHE.json'
 if manifest.exists():
  m=json.loads(manifest.read_text('utf-8'))
  assert m['calibration_sha256']==sha(V18/'CALIBRATION_MANIFEST.json') and m['m2_checkpoint_sha256']==P['input_sha256'][str((FINAL/'model.safetensors').relative_to(R))]
  assert all(names[n].stat().st_size==N*SEQ*shape[n]*2 and sha(names[n])==m['input_sha256'][n] for n in WNAMES)
  event('capture_cache_hit',manifest_sha256=sha(manifest));return {n:np.memmap(names[n],dtype=np.float16,mode='r',shape=(N,SEQ,shape[n])) for n in WNAMES}
 budget.check('full_128_document_capture',150)
 tmp={n:names[n].with_suffix('.tmp') for n in WNAMES};mm={n:np.memmap(tmp[n],dtype=np.float16,mode='w+',shape=(N,SEQ,shape[n])) for n in WNAMES}
 seen=({n:0 for n in WNAMES});i=-1
 def hook(name):
  def h(_mod,args):
   seen[name]+=1;mm[name][i]=args[0][0].detach().to('cpu').numpy()
  return h
 layer=model.model.layers[0];handles=[layer.mlp.up_proj.register_forward_pre_hook(hook('up_proj')),layer.mlp.down_proj.register_forward_pre_hook(hook('down_proj'))]
 pos=torch.arange(SEQ,device=D)[None];started=time.perf_counter()
 try:
  for i,sample in enumerate(CAL['samples']):
   budget.check('capture_doc_'+str(i),20)
   ids=torch.tensor(sample['token_ids'],dtype=torch.long,device=D)[None]
   with torch.inference_mode():
    hidden=model.model.embed_tokens(ids)
    _=layer(hidden,attention_mask=None,position_ids=pos,cache_position=pos[0],position_embeddings=model.model.rotary_emb(hidden,pos),use_cache=False)
   if i%32==31:event('capture_progress',documents=i+1,seconds=budget.used())
  sync()
 finally:
  for h in handles:h.remove()
 assert seen=={'up_proj':128,'down_proj':128}
 for m in mm.values():m.flush();m._mmap.close()
 for n in WNAMES:os.replace(tmp[n],names[n])
 hashes={n:sha(names[n]) for n in WNAMES};elapsed=time.perf_counter()-started
 write('ACTIVATION_CACHE.json',dict(calibration_sha256=sha(V18/'CALIBRATION_MANIFEST.json'),m2_checkpoint_sha256=P['input_sha256'][str((FINAL/'model.safetensors').relative_to(R))],input_sha256=hashes,shape={n:[N,SEQ,shape[n]] for n in WNAMES},capture_seconds=elapsed,documents=128,token_count=262144))
 event('capture_complete',seconds=elapsed,input_sha256=hashes);budget.save('CAPTURE_COMPLETE')
 return {n:np.memmap(names[n],dtype=np.float16,mode='r',shape=(N,SEQ,shape[n])) for n in WNAMES}
class OpProfile:
 """CUDA events around coarse native stages; one synchronization after whole call."""
 def __init__(self):self.records=[];self._old={}
 def measure(self,kind,func,*args,**kw):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
  a.record();t=time.perf_counter();out=func(*args,**kw);cpu=time.perf_counter()-t;b.record()
  self.records.append((kind,a,b,cpu));return out
 @contextmanager
 def patch(self):
  self._old=dict(chol=torch.linalg.cholesky,diag=torch.diag,mean=torch.mean,clone=GPTQ._clone_module,find=Quantizer.find_params,inverse=GPTQ.block_cholesky_inverse,methodtype=om.MethodType)
  original=self._old
  def chol(x,*args,**kw):return self.measure('upper_cholesky' if kw.get('upper',False) else 'lower_cholesky',original['chol'],x,*args,**kw)
  def diag(*a,**kw):return self.measure('H_diag',original['diag'],*a,**kw)
  def mean(*a,**kw):return self.measure('H_mean_or_other',original['mean'],*a,**kw)
  def clone(core,*a,**kw):return self.measure('W_clone_FP32',original['clone'],core,*a,**kw)
  def find(quant,*a,**kw):return self.measure('find_params',original['find'],quant,*a,**kw)
  def inverse(core,*a,**kw):return self.measure('real_inverse',original['inverse'],core,*a,**kw)
  def metatype(fn,core):
   profiler=self
   def wrapped(core_self,*a,**kw):return profiler.measure('cached_inverse_hit',fn,core_self,*a,**kw)
   return original['methodtype'](wrapped,core)
  torch.linalg.cholesky=chol;torch.diag=diag;torch.mean=mean
  GPTQ._clone_module=clone;Quantizer.find_params=find;GPTQ.block_cholesky_inverse=inverse;om.MethodType=metatype
  try:yield self
  finally:
   torch.linalg.cholesky=original['chol'];torch.diag=original['diag'];torch.mean=original['mean']
   GPTQ._clone_module=original['clone'];Quantizer.find_params=original['find'];GPTQ.block_cholesky_inverse=original['inverse'];om.MethodType=original['methodtype']
 def summary(self):
  sync();out={};counts={};cpu={}
  for name,a,b,c in self.records:
   out[name]=out.get(name,0)+a.elapsed_time(b)/1000;counts[name]=counts.get(name,0)+1;cpu[name]=cpu.get(name,0)+c
  return dict(gpu_seconds=out,counts=counts,cpu_dispatch_envelopes_seconds=cpu,
   warning='CPU envelopes and GPU event time are different clocks; do not sum CPU envelopes or nested totals')

def profile_candidate(weight,prepared,alpha):
 prof=OpProfile();sync();torch.cuda.reset_peak_memory_stats()
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
 a.record();t=time.perf_counter()
 with prof.patch():
  result=om.optimized_dynamic_gptq(weight,prepared,alpha=alpha,check_cholesky=(alpha==1))
 b.record();sync();wall=time.perf_counter()-t
 details=prof.summary();gpu=a.elapsed_time(b)/1000
 covered=sum(details['gpu_seconds'].values());details.update(dict(mode='original',alpha=alpha,candidate_wall_seconds=wall,candidate_gpu_stream_elapsed_seconds=gpu,
  nonoverlap_profiled_gpu_seconds=covered,residual_gpu_seconds=gpu-covered,
  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
 return result,details

def profile_prepared(H):
 prof=OpProfile();sync();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
 a.record();t=time.perf_counter()
 with prof.patch():state=om.prepare_native_state(H,nsamples=N)
 b.record();sync();d=prof.summary();d.update(wall_seconds=time.perf_counter()-t,gpu_stream_seconds=a.elapsed_time(b)/1000)
 return state,d

def measure_score(x,w,result):
 sync();torch.cuda.reset_peak_memory_stats();s=time.perf_counter();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 start.record();loss=rp.score_rows(x,w,result.weight_fp16);end.record();sync()
 return loss,dict(wall_seconds=time.perf_counter()-s,gpu_stream_elapsed_seconds=start.elapsed_time(end)/1000,rows=int(w.shape[0]),documents=128,tokens_per_document=2048,
  peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())

def stitch_two(one,eight,s1,s8):
 sync();t=time.perf_counter();choose=s8<s1;alpha=torch.where(choose,torch.full_like(s8,.8),torch.ones_like(s1))
 q=one.weight_fp16.clone();q[choose]=eight.weight_fp16[choose]
 scale=one.scales.clone();scale[choose]=eight.scales[choose]
 zero=one.zeros.clone();zero[choose]=eight.zeros[choose]
 assert torch.equal(one.g_idx,eight.g_idx)
 committed=q.clone();sync()
 return dict(alpha=alpha,weight=committed,scale=scale,zero=zero,g_idx=one.g_idx.clone()),time.perf_counter()-t

def profile_original(model,pristine,inputs,budget):
 outcomes={}
 historical=torch.load(V18/'checkpoints/M2/layer_00.pt',map_location='cpu',weights_only=False)['modules']
 for name in WNAMES:
  budget.check('H_'+name,45)
  W=pristine[name];x=inputs[name];module=model.model.layers[0].mlp.get_submodule(name)
  H,ns,hmeta=rp.build_hessian(module,x,[],f'v26 L0 {name}')
  assert ns==N and H.shape==(W.shape[1],W.shape[1])
  h0=thash(H);w0=thash(W)
  reference=historical['mlp.'+name]
  if sha(ACT/f'L0_{name}_128x2048.f16')!=reference['input_sha256']:raise RuntimeError('Formal M2 input provenance mismatch '+name)
  if h0!=reference['hessian_sha256']:raise RuntimeError('Formal M2 H mismatch '+name)
  prepared,preptime=profile_prepared(H)
  assert prepared.h_sha256==h0
  budget.check('warmup_'+name,30)
  _,warm=timed(lambda:om.optimized_dynamic_gptq(W,prepared,alpha=.8))
  records=[];first={};first_losses={}
  for alpha in [1.0,.8]:
   for rep in [1,2]:
    budget.check(f'profile_{name}_{alpha}_{rep}',25)
    result,detail=profile_candidate(W,prepared,alpha)
    loss,scoring=measure_score(x,W,result)
    detail.update(projection=name,replicate=rep,scoring=scoring)
    records.append(detail);event('original_candidate',projection=name,alpha=alpha,replicate=rep,candidate_wall=detail['candidate_wall_seconds'],score_wall=scoring['wall_seconds'])
    if rep==1:first[alpha]=result;first_losses[alpha]=loss
    else:
     assert all(torch.equal(getattr(first[alpha],f),getattr(result,f)) for f in ['weight_fp16','scales','zeros','g_idx'])
     assert torch.equal(first_losses[alpha],loss)
    budget.save('ORIGINAL_PROFILE',projection=name,last_alpha=alpha,last_replicate=rep)
  stitched,stitch_wall=stitch_two(first[1.0],first[.8],first_losses[1.0],first_losses[.8])
  assert thash(W)==w0 and thash(H)==h0
  outcomes[name]=dict(shape=list(W.shape),input_shape=list(x.shape),input_sha256=sha(ACT/f'L0_{name}_128x2048.f16'),H_sha256=h0,pristine_weight_sha256=w0,
   H_build_seconds=hmeta['seconds'],H_build_peak_bytes=hmeta['peak_allocated'],H_preparation=preptime,warmup_seconds=warm,records=records,
   score_and_two_alpha_stitch_seconds=stitch_wall,selected_08_rows=int((stitched['alpha']==.8).sum()),
   selected_weight_sha256=thash(stitched['weight']),selected_scale_sha256=thash(stitched['scale']),selected_zero_sha256=thash(stitched['zero']),selected_g_idx_sha256=thash(stitched['g_idx']))
  write('PROFILE_ORIGINAL.json',dict(status='IN_PROGRESS',targets=outcomes,gpu_wall_seconds=budget.used()))
  event('original_projection_complete',projection=name,seconds=budget.used())
 return outcomes
