from __future__ import annotations
import hashlib, importlib.util, json, sys, types
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(r'E:/quant-thesis/experiments/range_search_w4/results/range_w4_omniquant_full_v21_1')
MODEL = Path(r'E:/quant-thesis/artifacts/models/qwen2.5-0.5b-instruct-7ae5576/7ae557604adf67be50417f59c2c2f167def9a775')
BUNDLE = Path(r'E:/quant-thesis/experiments/range_search_w4/results/range_w4_omniquant_kaggle_v20_1/kaggle_bundle')
CAL = ROOT/'LOCAL_PREFLIGHT/CALIBRATION_MANIFEST.json'
V20_PARITY = Path(r'E:/quant-thesis/experiments/range_search_w4/results/range_w4_omniquant_smoke_v20/QUANT_OFF_PARITY.json')
LOCK = json.loads((BUNDLE/'manifests/MODEL_FILE_HASHES.json').read_text(encoding='utf-8'))
MANIFEST = json.loads((ROOT/'LOCAL_MODEL_MANIFEST.json').read_text(encoding='utf-8'))

def sha_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def write(path: Path, value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def compare(a: torch.Tensor, b: torch.Tensor):
    diff=(a.float()-b.float()).abs()
    return {'shape':list(a.shape),'dtype_a':str(a.dtype),'dtype_b':str(b.dtype),'max_abs_diff':float(diff.max().item()),'mean_abs_diff':float(diff.mean().item()),'finite':bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),'allclose_rtol_1e-5_atol_1e-6':bool(torch.allclose(a,b,rtol=1e-5,atol=1e-6)),'bitwise_equal':bool(torch.equal(a,b))}

if MANIFEST.get('status')!='PASS': raise RuntimeError('local model manifest did not pass')
assert all((MODEL/x['relative_path']).is_file() and sha_file(MODEL/x['relative_path'])==x['expected_sha256'] and (MODEL/x['relative_path']).stat().st_size==x['expected_bytes'] for x in MANIFEST['files'])

# Load the exact isolated snapshot; no Hub/network fallback is permitted.
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.manual_seed(20261005)
tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True)
cal=json.loads(CAL.read_text(encoding='utf-8'))
cal_hash=sha_file(CAL)
expected_cal_sha='664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3'
assert cal_hash==expected_cal_sha and cal.get('tokenizer_revision')==LOCK['revision']
ids=cal['samples'][0]['token_ids'][:64]
ids_hash=sha_bytes(np.asarray(ids,dtype='<i8').tobytes())
v20=json.loads(V20_PARITY.read_text(encoding='utf-8'))
assert ids_hash==v20['token_ids_sha256'] and len(ids)==64

tokenizer_files=[]
for rel in ('tokenizer.json','tokenizer_config.json','merges.txt','vocab.json'):
    row=next(x for x in MANIFEST['files'] if x['relative_path']==rel)
    tokenizer_files.append({'path':rel,'bytes':row['bytes'],'sha256':row['sha256'],'matches_v20_1_lock':row['matches_v20_1_lock']})
tokcheck={'status':'PASS','model_revision':LOCK['revision'],'v18_calibration_tokenizer_revision':cal.get('tokenizer_revision'),'v20_formal_tokenizer_library':'tokenizers 0.21.4','local_tokenizers_version':__import__('tokenizers').__version__,'local_transformers_version':__import__('transformers').__version__,'tokenizer_class':type(tokenizer).__name__,'is_fast':bool(tokenizer.is_fast),'vocab_size':int(tokenizer.vocab_size),'len_tokenizer':int(len(tokenizer)),'tokenizer_files':tokenizer_files,'all_tokenizer_file_hashes_match_frozen_lock':all(x['matches_v20_1_lock'] for x in tokenizer_files),'calibration_manifest_sha256':cal_hash,'frozen_sample_index':0,'restored_token_count':len(ids),'restored_token_ids_sha256_little_endian_int64':ids_hash,'matches_v20_64_token_input_hash':ids_hash==v20['token_ids_sha256'],'reencoded_from_text':False}
write(ROOT/'TOKENIZER_CHECK.json',tokcheck)

model=AutoModelForCausalLM.from_pretrained(str(MODEL),torch_dtype=torch.float16,low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True).eval()
param_count=sum(p.numel() for p in model.parameters())
dtypes=sorted({str(p.dtype) for p in model.parameters()})
nonfinite=[]
for name,p in model.named_parameters():
    if p.is_floating_point() and not bool(torch.isfinite(p).all().item()): nonfinite.append(name)
config=model.config
loadcheck={'status':'PASS' if config.model_type=='qwen2' and len(model.model.layers)==24 and dtypes==['torch.float16'] and not nonfinite and param_count>0 else 'FAIL','model_id':LOCK['model_id'],'revision':LOCK['revision'],'local_snapshot_path':str(MODEL),'local_files_only':True,'config_commit_hash':getattr(config,'_commit_hash',None),'dtype':dtypes,'parameter_count':param_count,'model_type':config.model_type,'decoder_layers':len(model.model.layers),'hidden_size':config.hidden_size,'intermediate_size':config.intermediate_size,'vocab_size':config.vocab_size,'num_attention_heads':config.num_attention_heads,'num_key_value_heads':config.num_key_value_heads,'all_floating_weights_finite':not nonfinite,'nonfinite_parameter_names':nonfinite[:20],'model_manifest_sha256':sha_file(ROOT/'LOCAL_MODEL_MANIFEST.json'),'loaded_from_isolated_directory':True}
write(ROOT/'LOCAL_MODEL_LOAD_CHECK.json',loadcheck)
if loadcheck['status']!='PASS': raise RuntimeError('frozen model load/config/finite check failed; stop before parity')

if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU parity substitute')
device=torch.device('cuda:0')
model=model.to(device).eval()
torch.cuda.synchronize(device)
sys.path.insert(0,str(BUNDLE/'omniquant_port/upstream'))
if importlib.util.find_spec('termcolor') is None:
    shim=types.ModuleType('termcolor');shim.colored=lambda text,*a,**k:text;sys.modules['termcolor']=shim
from models.int_qwen2_layer import QuantQwen2DecoderLayer
from quantize.utils import set_quant_state
adapter_args=types.SimpleNamespace(weight_quant_params={'n_bits':4,'per_channel_axes':[0],'symmetric':False,'dynamic_method':'per_channel','group_size':128,'lwc':True,'disable_zero_point':False},act_quant_params={'n_bits':16,'per_channel_axes':[],'symmetric':False,'dynamic_method':'per_token'})
input_ids=torch.tensor(ids,dtype=torch.long,device=device)[None]
selected=(0,12,23)
native_layer={};native_attn={};native_qkv={}
handles=[]
for li in selected:
    block=model.model.layers[li]
    handles.append(block.register_forward_hook(lambda m,a,o,idx=li:native_layer.__setitem__(idx,o[0].detach().cpu())))
    handles.append(block.self_attn.register_forward_hook(lambda m,a,o,idx=li:native_attn.__setitem__(idx,(o[0] if isinstance(o,(tuple,list)) else o).detach().cpu())))
    native_qkv[li]={}
    for proj in ('q_proj','k_proj','v_proj'):
        handles.append(block.get_submodule('self_attn.'+proj).register_forward_hook(lambda m,a,o,idx=li,key=proj:native_qkv[idx].__setitem__(key,o.detach().cpu())))
with torch.inference_mode(): native_logits=model(input_ids=input_ids,use_cache=False).logits.detach().cpu()
for h in handles:h.remove()
parity=[]
for li in selected:
    original=model.model.layers[li]
    port=QuantQwen2DecoderLayer(config,original,adapter_args).to(device).eval()
    set_quant_state(port,weight_quant=False,act_quant=False)
    captured={};captures=[]
    captures.append(port.register_forward_hook(lambda m,a,o:captured.__setitem__('layer',o[0].detach().cpu())))
    captures.append(port.native_layer.self_attn.register_forward_hook(lambda m,a,o:captured.__setitem__('attention',(o[0] if isinstance(o,(tuple,list)) else o).detach().cpu())))
    captured['qkv']={}
    for proj in ('q_proj','k_proj','v_proj'):
        captures.append(port.native_layer.get_submodule('self_attn.'+proj).register_forward_hook(lambda m,a,o,key=proj:captured['qkv'].__setitem__(key,o.detach().cpu())))
    model.model.layers[li]=port
    with torch.inference_mode(): candidate_logits=model(input_ids=input_ids,use_cache=False).logits.detach().cpu()
    model.model.layers[li]=original
    for h in captures:h.remove()
    metrics={'logits':compare(native_logits,candidate_logits),f'layer_{li}':compare(native_layer[li],captured['layer']),'attention_block_output':compare(native_attn[li],captured['attention'])}
    for proj in ('q_proj','k_proj','v_proj'):metrics[f'{proj}_output']=compare(native_qkv[li][proj],captured['qkv'][proj])
    parity.append({'layer':li,'comparisons':metrics})
    del port,candidate_logits,captured
    torch.cuda.empty_cache()

all_comparisons=[m for row in parity for m in row['comparisons'].values()]
parity_result={'status':'PASS' if all(x['finite'] and x['allclose_rtol_1e-5_atol_1e-6'] for x in all_comparisons) else 'FAIL','input_tokens':64,'token_ids_sha256':ids_hash,'dtype':'FP16','device':str(device),'rtol':1e-5,'atol':1e-6,'layers':list(selected),'native_vs_port':parity,'overall_bitwise_equal':all(x['bitwise_equal'] for x in all_comparisons),'peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device)),'peak_reserved_bytes':int(torch.cuda.max_memory_reserved(device)),'formal_optimizer_updates_executed':0}
write(ROOT/'QUANT_OFF_PARITY.json',parity_result)
print(json.dumps({'model_load':loadcheck,'tokenizer_check':tokcheck,'parity':{k:v for k,v in parity_result.items() if k!='native_vs_port'},'layer_logit_diffs':[{'layer':x['layer'],'max_abs_diff':x['comparisons']['logits']['max_abs_diff'],'bitwise_equal':x['comparisons']['logits']['bitwise_equal']} for x in parity]},indent=2))
if parity_result['status']!='PASS': raise SystemExit('quant-off parity failed; stop before 2048-token preflight')
