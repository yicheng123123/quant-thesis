"""v17.2 source certification and native-input capture preflight."""
from pathlib import Path
import copy, hashlib, importlib.metadata, inspect, json, sys
import torch
from safetensors import safe_open
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[3]
V16=OUT.parent/'range_w4_fullmodel_baseline_v16'; V17=OUT.parent/'range_w4_fullmodel_smoke_v17'
V171=OUT.parent/'range_w4_m2_parity_v17_1'
sys.path.insert(0,str(V171))
REV='7ae557604adf67be50417f59c2c2f167def9a775'
MODEL=ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/REV
GRID=[i/10 for i in range(1,11)]
TARGETS=['self_attn.k_proj','self_attn.v_proj','self_attn.q_proj','self_attn.o_proj','mlp.up_proj','mlp.gate_proj','mlp.down_proj']

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):return sha_bytes(Path(p).read_bytes())
def sha(t):return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def meta(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'sha256':sha(t)}
def cfg():return QuantizeConfig(bits=4,group_size=128,sym=True,desc_act=False,static_groups=False,true_sequential=True,damp_percent=.1,damp_auto_increment=0,mse=0)
def make_core(module,H,ns):
    g=GPTQ(copy.deepcopy(module),qcfg=cfg());g.quantizer.configure(perchannel=True,bits=4,sym=True);g.H=H.clone();g.nsamples=ns;return g
def exact(a,b):return bool(torch.equal(a.detach().cpu(),b.detach().cpu()))

def main():
    torch.manual_seed(20261005);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    v171map=json.loads((V171/'SOURCE_MAP.json').read_text(encoding='utf-8'))
    certified=V171/'native_m2.py';native_fn=inspect.unwrap(GPTQ.quantize);native_src=inspect.getsourcefile(native_fn)
    pathcheck={'certified_path':str(certified.resolve()),'certified_source_sha256':sha_file(certified),
        'certified_expected_sha256':v171map['restored_interface']['file_sha256'],'function':'native_m2.native_dynamic_gptq',
        'native_gptq_source':str(Path(native_src).resolve()),'native_gptq_source_sha256':sha_file(native_src),
        'native_gptq_expected_sha256':v171map['native']['file_sha256'],'gptqmodel_version':importlib.metadata.version('gptqmodel'),
        'import_origin':str(Path(__import__('native_m2').__file__).resolve()),'model_revision':REV}
    pathcheck['pass']=pathcheck['certified_source_sha256']==pathcheck['certified_expected_sha256'] and pathcheck['native_gptq_source_sha256']==pathcheck['native_gptq_expected_sha256'] and pathcheck['import_origin']==str(certified.resolve())
    dump('CERTIFIED_PATH_CHECK.json',pathcheck)
    if not pathcheck['pass']:raise RuntimeError('Certified implementation source/path mismatch; STOP')

    manifest_path=V16/'CALIBRATION_MANIFEST.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert len(manifest['samples'])==128 and all(len(s['token_ids'])==2048 for s in manifest['samples'])
    ids=[torch.tensor(s['token_ids'][:64],device='cuda',dtype=torch.long)[None] for s in manifest['samples'][:2]]
    with safe_open(str(MODEL/'model.safetensors'),framework='pt',device='cpu') as f:
        emb=f.get_tensor('model.embed_tokens.weight').half().to('cuda')
    with torch.inference_mode():docs=[torch.nn.functional.embedding(i,emb) for i in ids]
    del emb
    config=AutoConfig.from_pretrained(str(MODEL),local_files_only=True);config._attn_implementation='sdpa'
    layer=Qwen2DecoderLayer(config,layer_idx=0).half().to('cuda').eval()
    with safe_open(str(MODEL/'model.safetensors'),framework='pt',device='cpu') as f:
        weights={key[len('model.layers.0.'):]:f.get_tensor(key).half() for key in f.keys() if key.startswith('model.layers.0.')}
    layer.load_state_dict(weights,strict=True);del weights
    rope=Qwen2RotaryEmbedding(config=config).to('cuda')
    captured={name:[] for name in TARGETS};handles=[]
    for name in TARGETS:
        module=layer.get_submodule(name)
        def hook(m,args,key=name):captured[key].append(args[0].detach().clone())
        handles.append(module.register_forward_pre_hook(hook))
    with torch.inference_mode():
        for doc in docs:
            pos=torch.arange(doc.shape[1],device='cuda')[None]
            layer(doc,attention_mask=None,position_ids=pos,cache_position=pos[0],position_embeddings=rope(doc,pos),use_cache=False)
    for h in handles:h.remove()
    assert all(len(captured[n])==2 for n in TARGETS)

    # Verify input equality from actual forward hooks. These are distinct
    # module invocations, each measuring the Linear's real first argument.
    q=torch.cat([z[0] for z in captured['self_attn.q_proj']]);k=torch.cat([z[0] for z in captured['self_attn.k_proj']]);v=torch.cat([z[0] for z in captured['self_attn.v_proj']])
    shared_qkv=exact(q,k) and exact(q,v)
    up=torch.cat(captured['mlp.up_proj']);gate=torch.cat(captured['mlp.gate_proj'])
    sibling_mlp=exact(up,gate)
    qkv={'shape':list(q.shape),'q':meta(q),'k':meta(k),'v':meta(v),'q_equals_k':exact(q,k),'q_equals_v':exact(q,v),
        'max_abs_q_k':float((q.float()-k.float()).abs().max()),'max_abs_q_v':float((q.float()-v.float()).abs().max()),'pass':shared_qkv}
    dump('QKV_INPUT_CHECK.json',qkv)

    names={name:name for name in TARGETS}
    descriptions={
      'self_attn.q_proj':('decoder layer input_layernorm output','RMSNorm(hidden_states); q projection then rotary embedding','shares pre-projection tensor with k_proj and v_proj',[128,896]),
      'self_attn.k_proj':('decoder layer input_layernorm output','RMSNorm(hidden_states); k projection then rotary embedding','shares pre-projection tensor with q_proj and v_proj',[128,896]),
      'self_attn.v_proj':('decoder layer input_layernorm output','RMSNorm(hidden_states); value projection','shares pre-projection tensor with q_proj and k_proj',[128,896]),
      'self_attn.o_proj':('self-attention context after Q/K/V, RoPE, causal attention and head reshape','attention weighted values are merged across heads','not a Q/K/V sibling',[128,896]),
      'mlp.up_proj':('post-attention residual after post_attention_layernorm','downstream of self_attn.o_proj and residual addition; MLP pre_feedforward norm','shares normalized MLP input with gate_proj',[128,896]),
      'mlp.gate_proj':('post-attention residual after post_attention_layernorm','same MLP input; gate branch before SiLU and multiplication','shares normalized MLP input with up_proj',[128,896]),
      'mlp.down_proj':('Silu(gate_proj(x)) * up_proj(x)','MLP down projection after gated activation product','no sibling projection input',[128,4864]),
    }
    sem=[]
    for name in TARGETS:
        tensors=captured[name];x=torch.cat([z[0] for z in tensors],dim=0)
        source,preceding,sibling,shape=descriptions[name]
        sem.append({'target_module':name,'actual_forward_input_tensor_source':source,'preceding_computation':preceding,
            'sibling_projection_input_relation':sibling,'capture_hook_location':'forward_pre_hook on exact target Linear; args[0]',
            'expected_concatenated_shape':shape,'actual_shape':list(x.shape),'per_sample_shapes':[list(y.shape) for y in tensors],
            'input_dtype':str(x.dtype),'input_hashes':[sha(y) for y in tensors],'concatenated_hash':sha(x),
            'finite':bool(torch.isfinite(x).all())})
    dump('CAPTURE_SEMANTICS.json',{'model_revision':REV,'calibration_manifest_sha256':sha_file(manifest_path),'samples':[0,1],
        'tokens_per_sample':64,'modules':sem,'qkv_sibling_inputs_exact':shared_qkv,'up_gate_sibling_inputs_exact':sibling_mlp,
        'pass':shared_qkv and sibling_mlp and all(m['finite'] and m['actual_shape']==m['expected_concatenated_shape'] for m in sem)})
    md=['# L0 activation capture semantics','','Each input is captured at the target Linear forward pre-hook, from `args[0]`. The two frozen v16 calibration samples use their first 64 token IDs. Shapes below concatenate the two actual invocations.','', '| Target | Actual input source | Preceding computation | Sibling relation | Shape |', '|---|---|---|---|---:|']
    for m in sem:md.append(f"| `{m['target_module']}` | {m['actual_forward_input_tensor_source']} | {m['preceding_computation']} | {m['sibling_projection_input_relation']} | `{m['actual_shape']}` |")
    md+=['','q/k/v receive bitwise identical normalized hidden-state inputs in this forward. up/gate likewise share their normalized MLP input. GPTQ grouping/order is not a forward dependency; see `SEQUENTIAL_QUANT_ORDER.json` and `FORWARD_DEPENDENCY.json`.','']
    (OUT/'CAPTURE_SEMANTICS.md').write_text('\n'.join(md),encoding='utf-8')
    if not json.loads((OUT/'CAPTURE_SEMANTICS.json').read_text())['pass'] or not shared_qkv:raise RuntimeError('Projection input capture semantics failed; STOP before candidates')

    from gptqmodel.models.definitions.qwen2 import Qwen2GPTQ
    quant_order=[list(g) for g in Qwen2GPTQ.layer_modules]
    dump('SEQUENTIAL_QUANT_ORDER.json',{'source':'gptqmodel.models.definitions.qwen2.Qwen2GPTQ.layer_modules',
        'source_sha256':sha_file(inspect.getsourcefile(Qwen2GPTQ)),'groups':quant_order,'true_sequential':True,
        'semantics':'GPTQ captures each group member input at group entry, quantizes the listed modules, then forwards the updated group before capturing the next group.',
        'actual_forward_hook_calls':names})
    forward={'decoder_forward_graph':'input_layernorm -> independent Q/K/V projections -> rotary Q/K -> causal attention/value mix -> o_proj -> residual -> post_attention_layernorm -> independent up/gate projections -> SiLU(gate)*up -> down_proj -> residual',
        'dependencies':{'q_proj':['input_layernorm'],'k_proj':['input_layernorm'],'v_proj':['input_layernorm'],
            'o_proj':['q_proj','k_proj','v_proj','rotary','causal_attention'],'up_proj':['o_proj','attention_residual','post_attention_layernorm'],
            'gate_proj':['o_proj','attention_residual','post_attention_layernorm'],'down_proj':['up_proj','gate_proj','silu_gate_product']},
        'qkv_share_input_exact':shared_qkv,'up_gate_share_input_exact':sibling_mlp,
        'warning':'Quantizer group ordering [k,v,q] does not serialize the network as k-output -> v-input -> q-input.'}
    dump('FORWARD_DEPENDENCY.json',forward)
    dump('SEQUENTIAL_QUANT_ORDER.json',{**json.loads((OUT/'SEQUENTIAL_QUANT_ORDER.json').read_text()),'forward_dependencies_separate_file':'FORWARD_DEPENDENCY.json'})

    # Hessian is built only from real k_proj inputs, with the native builder.
    target=layer.get_submodule('self_attn.k_proj');x=torch.cat(captured['self_attn.k_proj']).float()
    core_for_h=make_core(target,torch.zeros((896,896),device='cuda',dtype=torch.float32),ns=0)
    # GPTQ.process_batch establishes native accumulation and sample count.
    core_for_h.H=torch.zeros((896,896),device='cuda',dtype=torch.float32);core_for_h.nsamples=0
    for d in captured['self_attn.k_proj']:core_for_h.process_batch(d)
    H=core_for_h.H.clone();ns=core_for_h.nsamples
    w=target.weight.detach().clone()
    native=make_core(target,H,ns).quantize(blocksize=128)
    from native_m2 import native_dynamic_gptq
    restored=native_dynamic_gptq(w,H,nsamples=ns,alpha=1.0)
    xerr=x.float()@(native[0].float()-w.float()).T
    xerr2=x.float()@(restored.weight_fp16.float()-w.float()).T
    parity={'scope':'L0 self_attn.k_proj, full native target geometry 128x896, actual k_proj pre-hook input, frozen samples0-1 x64',
        'weight_exact':exact(native[0],restored.weight_fp16),'scale_exact':exact(native[1],restored.scales),
        'zero_exact':exact(native[2],restored.zeros),'g_idx_exact':exact(native[3],restored.g_idx),
        'output_mse_exact':exact(xerr.square().mean(),xerr2.square().mean()),
        'native_output_mse':float(xerr.square().mean()),'certified_output_mse':float(xerr2.square().mean()),
        'pristine_weight_sha256':sha(w),'undamped_H_sha256':sha(H),'activation_sha256':sha(x),
        'H_nsamples':ns,'pass':exact(native[0],restored.weight_fp16) and exact(native[1],restored.scales) and exact(native[2],restored.zeros) and exact(native[3],restored.g_idx) and exact(xerr.square().mean(),xerr2.square().mean())}
    dump('ALPHA1_PREFLIGHT_PARITY.json',parity)
    if not parity['pass']:raise RuntimeError('Tiny certified alpha=1 parity failed; STOP before candidates')

    torch.save({'docs':docs,'inputs':{n:captured[n] for n in TARGETS},'W_k':w,'H_k':H,'nsamples_k':ns,
        'native_baseline_k':tuple(t.detach().cpu() for t in native[:4]),
        'layer_state':{key:value.detach().cpu() for key,value in layer.state_dict().items()},
        'model_rev':REV,'cal_manifest_sha256':sha_file(manifest_path)},OUT/'smoke_state.pt')
    protocol=json.loads((V16/'PROTOCOL.json').read_text(encoding='utf-8'))
    dump('PROTOCOL.json',{'name':'v17.2 M2 Full-Model Smoke Resume','frozen_v16_protocol_sha256':sha_file(V16/'PROTOCOL.json'),
        'calibration_manifest_sha256':sha_file(manifest_path),'calibration_subset':'frozen samples 0,1; first 64 token ids each; no resampling',
        'model_revision':REV,'native':protocol['native'],'alpha_grid':GRID,'search_unit':'one shared alpha per output row over all seven G128 groups',
        'candidate_implementation':'certified native_m2 native GPTQ core; full target output-row geometry for each alpha; score 32 fixed rows',
        'primary_target':'L0 self_attn.k_proj first32 output rows','prohibited':'formal full-model search; formal PPL; full OmniQuant','expected_row_alpha_scores':320})
    print(json.dumps({'source_pass':True,'alpha1_pass':parity['pass'],'capture_pass':True,'qkv_exact':shared_qkv,
        'H_nsamples':ns,'W_shape':list(w.shape),'X_shape':list(torch.cat([d[0] for d in captured['self_attn.k_proj']]).shape)},indent=2),flush=True)

if __name__=='__main__':main()
