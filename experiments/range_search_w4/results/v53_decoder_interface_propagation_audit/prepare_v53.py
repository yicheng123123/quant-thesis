import ast,csv,difflib,hashlib,json,time
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parent;R=Path('E:/quant-thesis');t=time.perf_counter()
for n in ['diagnostics','validation','reports','validation_cache']: (O/n).mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
interface=r'''"""Explicit Qwen2 decoder contract under use_cache=False/output_attentions=False."""
import numpy as np
import torch

def extract_hidden(result, *, expected_shape):
    # Supported native APIs: new Tensor, legacy one-element tuple.
    if torch.is_tensor(result):
        hidden=result
    elif type(result) is tuple and len(result)==1 and torch.is_tensor(result[0]):
        hidden=result[0]
    else:
        raise TypeError(f'unsupported decoder result type={type(result).__name__}; expected Tensor or (Tensor,)')
    expected=tuple(expected_shape)
    if hidden.ndim!=3 or tuple(hidden.shape)!=expected:
        raise ValueError(f'decoder hidden shape {tuple(hidden.shape)} != full [batch,tokens,hidden] {expected}')
    if not hidden.is_floating_point() or not bool(torch.isfinite(hidden).all()):
        raise ValueError('decoder hidden is not finite floating-point')
    return hidden  # Identity and bytes retained; no cast, clone, slicing or reshape.

def write_document(cache, document, hidden):
    if not isinstance(cache,np.ndarray) or cache.ndim!=3:
        raise ValueError('cache must have [documents,tokens,features] shape')
    if not (0<=document<len(cache)):raise IndexError('document outside cache')
    expected=tuple(cache[document:document+1].shape)
    if not torch.is_tensor(hidden) or hidden.ndim!=3 or tuple(hidden.shape)!=expected:
        raise ValueError(f'cache source shape {getattr(hidden,"shape",None)} != target {expected}; broadcasting forbidden')
    source=hidden.detach().cpu().numpy()
    target=cache[document:document+1]
    if source.shape!=target.shape or source.dtype!=target.dtype:
        raise ValueError(f'cache source shape/dtype {source.shape}/{source.dtype} != target {target.shape}/{target.dtype}')
    np.copyto(target,source,casting='no')  # Shape equality already established.
'''
(O/'decoder_interface.py').write_text(interface,encoding='utf8')
p=B/'v46_m1_single_integer_repair_fullmodel/runtime_v46.py';src=p.read_text(encoding='utf8');new=src
new=new.replace('import numpy as np\n','import numpy as np\nfrom decoder_interface import extract_hidden, write_document\n',1)
old='for i,s in enumerate(CAL[\'samples\']):hidden[i]=model.model.embed_tokens(torch.tensor(s[\'token_ids\'],device=\'cuda\')[None])[0].cpu().numpy()'
assert old in new;new=new.replace(old,"for i,s in enumerate(CAL['samples']):write_document(hidden,i,model.model.embed_tokens(torch.tensor(s['token_ids'],device='cuda')[None]))")
old="if 'x' not in active:active['x']=args[0].detach();X[active['doc']]=args[0][0].detach().cpu().numpy()"
assert old in new;new=new.replace(old,"if 'x' not in active:active['x']=args[0].detach();write_document(X,active['doc'],args[0])")
old="if group=='qkv':X[i]=layer.input_layernorm(h)[0].cpu().numpy()";assert old in new;new=new.replace(old,"if group=='qkv':write_document(X,i,layer.input_layernorm(h))")
old="out[i]=layer(h,attention_mask=None,position_ids=positions,cache_position=positions[0],position_embeddings=model.model.rotary_emb(h,positions),use_cache=False)[0][0].cpu().numpy()"
assert old in new;new=new.replace(old,"result=layer(h,attention_mask=None,position_ids=positions,cache_position=positions[0],position_embeddings=model.model.rotary_emb(h,positions),use_cache=False)\n            states=extract_hidden(result,expected_shape=(1,hidden.shape[1],hidden.shape[2]))\n            write_document(out,i,states)")
(O/'runtime_v53.py').write_text(new,encoding='utf8');compile(new,'runtime_v53.py','exec')
(O/'MINIMAL_REPAIR.diff').write_text(''.join(difflib.unified_diff(src.splitlines(True),new.splitlines(True),fromfile=str(p),tofile=str(O/'runtime_v53.py'))),encoding='utf8')
a={x.name:ast.dump(x,include_attributes=False) for x in ast.parse(src).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))};b={x.name:ast.dump(x,include_attributes=False) for x in ast.parse(new).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
changed=[n for n in a if a[n]!=b[n]];assert changed==['embed','capture','advance_block']
dump('UNCHANGED_ALGORITHM_AUDIT.json',{'pass':True,'only_changed_definitions':changed,'all_other_function_and_class_AST_identical':True,'original_runtime_sha256':sha(p),'local_runtime_sha256':sha(O/'runtime_v53.py'),'C_p_grid_search_acceptance_and_evaluator_unchanged':True,'v52_fast_core_source_sha256':sha(B/'v52_m2_fast_integer_repair_fp64_initial_review/fast_accept_v52.py'),'v52_review_source_sha256':sha(B/'v52_m2_fast_integer_repair_fp64_initial_review/initialization_review.py')})
ledger=json.loads((B/'v52_m2_fast_integer_repair_fp64_initial_review/ACTIVE_JOB_LEDGER.json').read_text());assert all('charged_seconds' in s for s in ledger['segments']);prior=ledger['v51_charged_seconds']+sum(s['charged_seconds'] for s in ledger['segments']);assert abs(prior-5278.453)<.001
ledger.update(v53_start_shared_charged_seconds=prior,v53_new_gpu_cap_seconds=600,prior_ledger_read_only_path=str(B/'v52_m2_fast_integer_repair_fp64_initial_review/ACTIVE_JOB_LEDGER.json'),prior_ledger_sha256=sha(B/'v52_m2_fast_integer_repair_fp64_initial_review/ACTIVE_JOB_LEDGER.json'),user_reported_prior_seconds=5278.453)
dump('ACTIVE_JOB_LEDGER.json',ledger)
# Snapshot only relevant history, all version files remain read-only.
history=[]
for dirname in ['v42_m2_single_integer_repair_fullmodel','v43_m2_single_integer_repair_fullmodel','v44_m2_single_integer_repair_fullmodel','v45_m2_single_integer_repair_fullmodel','v46_m1_single_integer_repair_fullmodel','v49_fullmodel_fast_integer_repair','v51_m2_fast_integer_repair_fullmodel','v52_m2_fast_integer_repair_fp64_initial_review','range_w4_fullmodel_v18']:
 D=B/dirname
 for fn in ['PROTOCOL.json','REPORT.md','RUN_STATUS.json','GPU_START.json','CHECKPOINT_LOCK.json','RUNTIME_PREFLIGHT.json','FORMAL_RUNTIME.json','run_v42.py','runtime_v46.py','run_v45.py','run_v49.py','run_v51.py','run_v52.py','ACTIVE_JOB_LEDGER.json']:
  p=D/fn
  if p.exists():history.append(dict(path=str(p),sha256=sha(p)))
for p in (B/'v51_m2_fast_integer_repair_fullmodel/formal/groups').glob('layer_00_*.pt'):history.append(dict(path=str(p),sha256=sha(p)))
for p in (B/'v52_m2_fast_integer_repair_fp64_initial_review/formal/groups').glob('layer_00_*.pt'):history.append(dict(path=str(p),sha256=sha(p)))
dump('HISTORY_BEFORE.json',history)
protocol={'version':'v53','purpose':'decoder interface correction and independent propagation audit only','requested_model_setting':'Sol high requested; API cannot verify or switch','algorithm_changes':False,'supported_returns':['Tensor[batch,tokens,hidden]','plain one-element tuple(Tensor[batch,tokens,hidden],) under frozen cache/attention flags'],'unknown_returns':'hard TypeError','cache_policy':'strict rank/shape/dtype equality before np.copyto; no broadcasting/casting','fixed_articles':[0,64,127],'article_tokens':2048,'corpus':'formal128x2048 v18 calibration, no P3/held/PPL','reference':'native complete causal-LM forward, independent prehooks, logits_to_keep=1 only skips unnecessary lm_head positions; all24Transformer blocks execute','reference_does_not_call_decoder_interface':True,'comparison':'bitwise first; fixed original output tolerance rtol1e-5 atol1e-6; no later relaxation','block0_sources':'original formal M2 plus v51 four committed block0 groups; verify v52 inherited tensors; no layer1+ repaired parameters used','group_states':['original M2 qkv input','qkv committed -> o input','qkv+o committed -> up_gate input','qkv+o+up_gate committed -> down input','allblock0 committed -> true layer1 boundary'],'new_gpu_cap_seconds':600,'shared_cap_seconds':10800,'prior_shared_charged_seconds':prior,'no_new_fullmodel_repair_or_PPL':True,'freeze_unix':time.time()}
dump('PROTOCOL.json',protocol)
(O/'INTERFACE_CONTRACT.md').write_text('''# Decoder/cache contract

For the actual frozen use_cache=False/output_attentions=False calls: support Tensor[B,T,H] and legacy plain one-element tuple containing Tensor[B,T,H]. Unknown result/container types, extra tuple items, rank/shape mismatch, nonfloating/nonfinite hidden states fail explicitly. No content changes or dtype conversion in extraction; returned Tensor identity is retained.

Cache writes use a B=1 document slice [1,T,H]. Source and target rank, every dimension, and dtype must exactly agree before copy; [H] and [T,H] cannot enter this path. Embedding, qkv, sibling-hook and block-refresh cache writes use the same strict write policy. Reference native forward does not use this extraction/write implementation.

Only embed/capture/advance_block change in isolated runtime_v53.py. Remaining function/class ASTs, C builder, candidate scoring, p updates, integer grid, tie order, budgets, acceptance and evaluator are unchanged. v52 initialization review and fast repair sources are read-only, not exercised as a new repair.

Future integration must explicitly load runtime_v53.py as the shared runtime and retain v52 initialization_review and fast_accept_v52 unchanged. Do not run historical entry points which import the old runtime. This audit does not authorize a new full repair or PPL.
''',encoding='utf8')
dump('PREPARATION.json',{'cpu_seconds':time.perf_counter()-t,'output_write_probe':True,'history_snapshot_files':len(history),'prior_ledger_not_reset':True})
print('V53_PREPARED',prior,changed)
