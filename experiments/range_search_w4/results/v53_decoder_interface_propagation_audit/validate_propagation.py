"""Bounded native-full-forward reference. No quantization, repair, Gram, loss or PPL."""
import csv,gc,hashlib,importlib.util,json,math,os,sys,threading,time,traceback
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch,transformers
from transformers import AutoModelForCausalLM
from decoder_interface import extract_hidden
import budget_v53 as budget
O=Path(__file__).resolve().parent;B=O.parent;ROOT=Path('E:/quant-thesis');V51=B/'v51_m2_fast_integer_repair_fullmodel';V52=B/'v52_m2_fast_integer_repair_fp64_initial_review';V46=B/'v46_m1_single_integer_repair_fullmodel';V18=B/'range_w4_fullmodel_v18'
sys.path.insert(0,str(V46));sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('v53_runtime',O/'runtime_v53.py');rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt);rt.OUT=O
ROWS=[];REF_CALLS=0;TIMING=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def ah(a):
 h=hashlib.sha256()
 for i in range(len(a)):h.update(np.ascontiguousarray(a[i]).tobytes())
 return h.hexdigest()
def deadline(reserve=0):
 total,local=budget.seconds()
 if local>600-reserve or total>10800-reserve:raise TimeoutError('v53 local/shared hard budget exhausted')
def elapsed():return budget.seconds()[0]
rt.elapsed=elapsed;rt.deadline=deadline
@contextmanager
def timer(stage):
 deadline();torch.cuda.synchronize();t=time.perf_counter()
 try:yield
 finally:
  torch.cuda.synchronize();r={'stage':stage,'synchronized_seconds':time.perf_counter()-t,'shared_charged_seconds':elapsed(),'v53_active_seconds':budget.seconds()[1],'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved()};TIMING.append(r)
  with (O/'GPU_VALIDATION_TIMINGS.csv').open('a',newline='',encoding='utf8') as f:
   w=csv.DictWriter(f,fieldnames=list(r));
   if f.tell()==0:w.writeheader()
   w.writerow(r)
def compare(a,b,label):
 assert a.shape==b.shape and a.dtype==b.dtype,(label,a.shape,b.shape,a.dtype,b.dtype)
 maxabs=0.;maxratio=0.;changed=0;tokens=0
 for i in range(len(a)):
  aa=np.asarray(a[i]);bb=np.asarray(b[i]);diff=np.abs(aa.astype(np.float32)-bb.astype(np.float32));tol=1e-6+1e-5*np.abs(bb.astype(np.float32));assert np.isfinite(diff).all() and np.isfinite(tol).all()
  maxabs=max(maxabs,float(diff.max()));maxratio=max(maxratio,float((diff/tol).max()));changed+=int(np.count_nonzero(aa.view(np.uint16)!=bb.view(np.uint16)));tokens+=int(np.count_nonzero(np.any(aa!=bb,axis=-1)));assert bool((diff<=tol).all()),f'{label} exceeds frozen output regression tolerance, maxratio={maxratio}'
 r={'label':label,'shape':list(a.shape),'dtype':str(a.dtype),'max_abs_error':maxabs,'max_error_ratio':maxratio,'bitwise_equal':changed==0,'unequal_elements':changed,'unequal_token_positions':tokens,'full_token_positions_checked':int(np.prod(a.shape[:-1])),'rtol':1e-5,'atol':1e-6};ROWS.append(r);dump('PROPAGATION_COMPARISONS.json',{'comparisons':ROWS});return r

def mmap(tag,dim):return np.lib.format.open_memmap(O/'validation_cache'/f'{tag}.npy',mode='w+',dtype=np.float16,shape=(128,2048,dim))

def native_group(model,group,tag):
 """Reference uses native model.forward and raw hook args; NEVER interface extractor/cache writer."""
 global REF_CALLS
 names=rt.GROUPS[group];dim=model.model.layers[0].get_submodule(names[0]).in_features;dest=mmap(tag,dim);active={};handles=[]
 def hook(module,args):
  value=args[0];assert torch.is_tensor(value) and tuple(value.shape)==(1,2048,dim) and value.dtype==torch.float16
  if active['seen']==0:
   arr=value.detach().cpu().numpy();assert arr.shape==(1,2048,dim);assert dest[active['doc']].shape==arr[0].shape and dest.dtype==arr.dtype
   np.copyto(dest[active['doc']],arr[0],casting='no');active['first']=value.detach().clone()
  else:assert torch.equal(active['first'],value),'native sibling hook input mismatch'
  active['seen']+=1
 try:
  handles=[model.model.layers[0].get_submodule(n).register_forward_pre_hook(hook) for n in names]
  with torch.inference_mode():
   for i,s in enumerate(rt.CAL['samples']):
    deadline();active.clear();active.update(doc=i,seen=0)
    ids=torch.tensor(s['token_ids'],device='cuda')[None]
    output=model(input_ids=ids,use_cache=False,output_attentions=False,logits_to_keep=1)
    REF_CALLS+=1;assert active['seen']==len(names);del output
 finally:
  for h in handles:h.remove()
 dest.flush();return dest

def native_boundary(model):
 global REF_CALLS
 dest=mmap('native_block0_to_layer1_hidden',896);qdest=mmap('native_layer1_qkv_X',896);active={};handles=[]
 def raw_hook(kind,module,args):
  val=args[0];assert torch.is_tensor(val) and tuple(val.shape)==(1,2048,896) and val.dtype==torch.float16
  arr=val.detach().cpu().numpy();target=(dest if kind=='hidden' else qdest)[active['doc']]
  assert arr[0].shape==target.shape and arr.dtype==target.dtype;np.copyto(target,arr[0],casting='no');active[kind]+=1
 try:
  handles=[model.model.layers[1].register_forward_pre_hook(lambda m,a:raw_hook('hidden',m,a)),model.model.layers[1].self_attn.q_proj.register_forward_pre_hook(lambda m,a:raw_hook('qkv',m,a))]
  with torch.inference_mode():
   for i,s in enumerate(rt.CAL['samples']):
    deadline();active.update(doc=i,hidden=0,qkv=0);out=model(input_ids=torch.tensor(s['token_ids'],device='cuda')[None],use_cache=False,output_attentions=False,logits_to_keep=1);REF_CALLS+=1;assert active['hidden']==active['qkv']==1;del out
 finally:
  for h in handles:h.remove()
 dest.flush();qdest.flush();return dest,qdest

def full24_articles(model):
 global REF_CALLS
 traces=[]
 for doc in read(O/'PROTOCOL.json')['fixed_articles']:
  refs={};hooks=[]
  def observe(key,module,args):
   val=args[0];assert val.shape==(1,2048,896) and val.dtype==torch.float16;refs[key]=val.detach().cpu().numpy().copy()
  try:
   for li,layer in enumerate(model.model.layers):
    hooks += [layer.register_forward_pre_hook(lambda m,a,k=(li,'hidden'):observe(k,m,a)),layer.self_attn.q_proj.register_forward_pre_hook(lambda m,a,k=(li,'qkv'):observe(k,m,a))]
   ids=torch.tensor(rt.CAL['samples'][doc]['token_ids'],device='cuda')[None]
   with torch.inference_mode():out=model(input_ids=ids,use_cache=False,output_attentions=False,logits_to_keep=1);REF_CALLS+=1;del out
  finally:
   for h in hooks:h.remove()
  assert len(refs)==48
  positions=torch.arange(2048,device='cuda')[None]
  with torch.inference_mode():
   states=model.model.embed_tokens(ids)
   for li,layer in enumerate(model.model.layers):
    deadline();raw=states.cpu().numpy();q=layer.input_layernorm(states).cpu().numpy()
    a=compare(raw,refs[(li,'hidden')],f'fixed_article{doc}_layer{li}_hidden');b=compare(q,refs[(li,'qkv')],f'fixed_article{doc}_layer{li}_qkv')
    result=layer(states,attention_mask=None,position_ids=positions,cache_position=positions[0],position_embeddings=model.model.rotary_emb(states,positions),use_cache=False)
    original=result if torch.is_tensor(result) else result[0]
    states=extract_hidden(result,expected_shape=(1,2048,896));assert states is original
    traces.append({'article':doc,'layer':li,'runtime_return_type':type(result).__name__,'runtime_return_shape':list(original.shape),'extracted_full_shape':list(states.shape),'hidden_bitwise_equal':a['bitwise_equal'],'qkv_bitwise_equal':b['bitwise_equal'],'complete_token_axis':2048})
  del refs;gc.collect()
 dump('FIXED_ARTICLE_24_LAYER_CHECK.json',{'pass':True,'articles':[0,64,127],'layers':24,'comparisons':traces,'native_reference_does_not_use_new_extractor':True,'model_state':'originalM2 + only independently audited v51 block0 repaired weights'})

def main():
 assert read(O/'DETERMINISTIC_TESTS.json')['pass']
 for n,h in read(O/'FROZEN_VALIDATION_LOCK.json')['files'].items():assert sha(O/n)==h,n
 assert not (O/'GPU_VALIDATION_RESULT.json').exists(),'single validation run only'
 budget.start('v53 independent native-forward propagation and block0 source verification')
 total,local=budget.seconds()
 def hardstop():
  dump('GPU_VALIDATION_STOP.json',{'status':'HARD_CAP_STOP','shared_seconds':budget.seconds()[0],'v53_seconds':budget.seconds()[1]});budget.finish();os._exit(124)
 watcher=threading.Timer(min(600-local,10800-total),hardstop);watcher.daemon=True;watcher.start()
 rt.WALL_START=time.time();rt.START=time.perf_counter();rt.MODE='validation';rt.LAYER=0;rt.MODULE='propagation'
 try:
  with timer('warmup_model_load_and_source_verification'):
   z=torch.zeros((64,64),device='cuda');z@z
   p=read(V52/'PROTOCOL.json');mp=V18/'M2/checkpoint/final';assert sha(mp/'model.safetensors')==p['initialization']['model_sha256'];assert sha(V18/'CALIBRATION_MANIFEST.json')==p['calibration']['manifest_sha256']
   assert len(rt.CAL['samples'])==128 and all(len(s['token_ids'])==2048 for s in rt.CAL['samples'])
   model=AutoModelForCausalLM.from_pretrained(str(mp),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()
   original=torch.load(V52/'original_codes/layer_00.pt',map_location='cpu',weights_only=True)
   initial_versions={n:x._version for n,x in model.named_parameters()};modified=set();source_rows=[]
  hidden=rt.embed(model);group_checks=[];group_names=['qkv','o','up_gate','down']
  for group in group_names:
   rt.MODULE=group
   with timer('native_full_forward_reference_'+group):ref=native_group(model,group,'native_block0_'+group+'_X')
   with timer('patched_capture_'+group):actual=rt.capture(model,0,hidden,group)
   with timer('full_input_comparison_and_history_source_'+group):
    comp=compare(actual.reshape(ref.shape),ref,'formal128_block0_'+group)
    cp=torch.load(V51/'formal/groups'/f'layer_00_{group}.pt',map_location='cpu',weights_only=False);inherited=torch.load(V52/'formal/groups'/f'layer_00_{group}.pt',map_location='cpu',weights_only=False)
    actualhash=ah(actual.reshape(ref.shape));refhash=ah(ref);assert actualhash==refhash
    source_match=refhash==cp['input_sha256'];assert cp['committed'] and set(cp['modules'])==set(rt.GROUPS[group])
    for name,s in cp['modules'].items():
     orig=original[name];sib=inherited['modules'][name]
     for key in ['codes','scales','zeros','g_idx','weight','packed_u4']:assert torch.equal(s[key],sib[key]),f'inherited{key} changed'
     for key in ['scales','zeros','g_idx']:assert torch.equal(s[key],orig[key]),f'M2 grid changed{key}'
     assert bool((s['codes']>=0).all() and (s['codes']<=15).all() and (s['updates']<=32).all() and (s['evaluations']<=128*s['codes'].shape[1]).all())
     assert torch.equal(s['zeros'],torch.full_like(s['zeros'],8)) and torch.equal(s['g_idx'].long(),torch.arange(s['codes'].shape[1])//128)
     codes=rt.unpack(s['packed_u4']);assert torch.equal(codes,s['codes'])
     grid=(s['scales'][:,s['g_idx'].long()]*(codes.float()-8)).half();assert torch.equal(grid.view(torch.int16),s['weight'].view(torch.int16))
     raworig=(orig['scales'][:,orig['g_idx'].long()]*(orig['codes'].float()-8)).half();mod=model.model.layers[0].get_submodule(name);assert torch.equal(raworig.view(torch.int16),mod.weight.detach().cpu().view(torch.int16))
     assert 'terminal_initial_raw64' in s and bool((s['terminal_final_raw64']<=s['terminal_initial_raw64']).all())
     source_rows.append({'group':group,'module':name,'v51_checkpoint_sha256':sha(V51/'formal/groups'/f'layer_00_{group}.pt'),'v52_container_sha256':sha(V52/'formal/groups'/f'layer_00_{group}.pt'),'grid_code_weight_equal_inherited':True,'all_rows_terminal_check_in_source':True,'native_current_reference_matches_historical_input_hash':source_match})
    group_checks.append({'group':group,'source_states_before_capture':list(modified),'reference_model':'originalM2 with preceding block0 groups committed in original order','all128_fulltokens':True,'reference_and_patched_bitwise_equal':comp['bitwise_equal'],'historical_source_input_sha256':cp['input_sha256'],'native_reference_input_sha256':refhash,'historical_source_input_matches_native':source_match})
    dump('BLOCK0_GROUP_SOURCE_CHECKS.json',{'groups':group_checks,'modules':source_rows})
   with timer('source_group_reconstruction_'+group):
    with torch.no_grad():
     for name,s in cp['modules'].items():model.model.layers[0].get_submodule(name).weight.copy_(s['weight'].cuda());modified.add('model.layers.0.'+name+'.weight')
   del ref,actual,cp,inherited;gc.collect();torch.cuda.empty_cache()
   print('V53_GROUP_VALIDATED',group,'v53_seconds',round(budget.seconds()[1],3),'historical_hash_match',source_match,flush=True)
  with timer('native_full_forward_boundary_reference'):refH,refQ=native_boundary(model)
  with timer('patched_block0_to_layer1_full128'):newH=rt.advance_block(model,0,hidden);newQ=rt.capture(model,1,newH,'qkv')
  with timer('full128_boundary_comparison'):
   compH=compare(newH,refH,'formal128_block0_to_layer1_hidden');compQ=compare(newQ.reshape(refQ.shape),refQ,'formal128_true_layer1_qkv')
   trueH=ah(newH);trueQ=ah(newQ.reshape(refQ.shape));oldH=read(V51/'formal/PREFIX_STATUS_layer_00.json')['hidden_sha256'];assert trueH!=oldH
   before=read(V52/'PREFIX_COMPATIBILITY_DIAGNOSIS.json');assert trueH==before['true_full_token_prefix_sha256'] and trueQ==before['true_layer1_X_sha256']
   np.save(O/'validation_cache/corrected_block0_boundary_hidden.npy',newH)
   boundary={'pass':True,'all128_articles_all2048tokens':True,'hidden_bitwise_equal_native_forward':compH['bitwise_equal'],'qkv_bitwise_equal_native_forward':compQ['bitwise_equal'],'correct_hidden_sha256':trueH,'correct_layer1_qkv_X_sha256':trueQ,'old_broadcast_hidden_sha256':oldH,'old_boundary_rejected':True,'no_layer1plus_repaired_parameters_loaded':True,'source_model_checkpoint_sha256':p['initialization']['model_sha256']}
   dump('BLOCK0_BOUNDARY_INDEPENDENT_CHECK.json',boundary)
  del newQ,refH,refQ;gc.collect()
  with timer('fixed_complete_articles_all24_layers'):full24_articles(model)
  for n,x in model.named_parameters():
   if n not in modified:assert x._version==initial_versions[n],f'unplanned parameter mutation{n}'
  allowed=all(g['historical_source_input_matches_native'] for g in group_checks)
  result={'pass':True,'new_interface_and_independent_propagation':True,'runtime_transformers':transformers.__version__,'runtime_torch':torch.__version__,'native_full_forward_calls':REF_CALLS,'native_reference_uses_extractor':False,'full128_block0_group_sources_match_historical':allowed,'v51_v52_block0_reuse_allowed':allowed,'v52_layer1plus_reuse_allowed':False,'corrected_boundary':boundary,'fixed_article_full24_layer_check':True,'model_structure_and_all_outside_block0_parameter_versions_unchanged':True,'no_C_p_grid_search_acceptance_or_budget_modified':True,'no_repair_PPL_or_held':True,'shared_charged_seconds':budget.seconds()[0],'v53_active_seconds':budget.seconds()[1],'checkpoint_creation':False}
  dump('GPU_VALIDATION_RESULT.json',result);print('V53_GPU_VALIDATION_COMPLETE',json.dumps(result),flush=True)
 except BaseException as ex:
  dump('GPU_VALIDATION_STOP.json',{'error':repr(ex),'traceback':traceback.format_exc(),'new_rules_or_thresholds_added':False,'shared_seconds':budget.seconds()[0],'v53_seconds':budget.seconds()[1]});raise
 finally:watcher.cancel();budget.finish()
if __name__=='__main__':main()
