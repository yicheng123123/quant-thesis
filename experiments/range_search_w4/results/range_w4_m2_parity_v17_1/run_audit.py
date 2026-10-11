"""v17.1 bounded native parity audit; never executes a candidate search."""
from pathlib import Path
import copy, csv, hashlib, importlib.metadata, importlib.util, inspect, json, sys
import torch
from safetensors import safe_open
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
import gptqmodel.quantization.gptq as gptq_module
import gptqmodel.quantization.quantizer as quantizer_module
from upstream_probe import capture_native, make_native, cfg, REV
from native_m2 import native_dynamic_gptq
from operator_instrumentation import Recorder, make_instrumented, native_integer_observer

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];V16=OUT.parent/'range_w4_fullmodel_baseline_v16';V17=OUT.parent/'range_w4_fullmodel_smoke_v17'
def sha_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def meta(t):return dict(dtype=str(t.dtype),device=str(t.device),shape=list(t.shape),stride=list(t.stride()),contiguous=t.is_contiguous(),sha256=sha(t))
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def cmp(a,b):
    am,bm=meta(a),meta(b);a=a.detach().cpu();b=b.detach().cpu()
    if a.shape!=b.shape:return dict(exact=False,shape_mismatch=True,native=am,custom=bm)
    d=(a.double()-b.double()).abs();indices=(a!=b).nonzero();first=indices[0].tolist() if len(indices) else None
    entry=dict(exact=torch.equal(a,b),max_abs_diff=float(d.max()) if d.numel() else 0,
        mean_abs_diff=float(d.mean()) if d.numel() else 0,num_diff=int((a!=b).sum()),first_differing_index=first,native=am,custom=bm)
    if first is not None:
        ix=tuple(first);entry.update(native_value=float(a[ix]),custom_value=float(b[ix]))
        if a.dtype==b.dtype==torch.float32:
            entry.update(native_bits_hex=hex(int(a.contiguous().view(torch.int32)[ix])&0xffffffff),custom_bits_hex=hex(int(b.contiguous().view(torch.int32)[ix])&0xffffffff))
    return entry
def load_core():
    source=ROOT/'experiments/range_search_w4/dynamic_core.py';spec=importlib.util.spec_from_file_location('v17_1_legacy_core',source)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    (OUT/'legacy_dynamic_core.py').write_bytes(source.read_bytes());return mod
def layer_from_weight(w,bias=None):
    m=torch.nn.Linear(w.shape[1],w.shape[0],bias=bias is not None,device='meta',dtype=w.dtype)
    m.weight=torch.nn.Parameter(w.clone(),requires_grad=False)
    if bias is not None:m.bias=torch.nn.Parameter(bias.clone(),requires_grad=False)
    return m
def loss(x,w,q):return (x.float()@(q.float()-w.float()).T).square().mean()
def compare_result(w,x,n,c,rows=None,cols=None):
    ri=slice(None) if rows is None else rows;ci=slice(None) if cols is None else cols
    nw=n[0][ri,ci];cw=c.weight_fp16[ri,ci]
    ns=n[1][ri];cs=c.scales[ri];nz=n[2][ri];cz=c.zeros[ri]
    if cols is not None:
        group=cols.start//128;ns=ns[:,group:group+1];cs=cs[:,group:group+1];nz=nz[:,group:group+1];cz=cz[:,group:group+1]
    ww=w[ri,ci];xx=x[:,ci]
    result={'weight':cmp(nw,cw),'scale':cmp(ns,cs),'zero':cmp(nz,cz),
        'output_mse':cmp(loss(xx,ww,nw),loss(xx,ww,cw)),
        'native_output_mse':float(loss(xx,ww,nw)),'custom_output_mse':float(loss(xx,ww,cw))}
    result['pass']=all(result[k]['exact'] for k in ['weight','scale','zero','output_mse']);return result
def build_h(module,docs):
    g=GPTQ(module,qcfg=cfg());g.quantizer.configure(perchannel=True,bits=4,sym=True)
    for d in docs:g.process_batch(d)
    return g.H.clone(),g.nsamples
def settings():return dict(torch_version=torch.__version__,gptqmodel_version=importlib.metadata.version('gptqmodel'),
    transformers_version=importlib.metadata.version('transformers'),gpu=torch.cuda.get_device_name(0),
    allow_tf32=torch.backends.cuda.matmul.allow_tf32,matmul_precision=torch.get_float32_matmul_precision(),
    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),autocast_cuda=torch.is_autocast_enabled('cuda'))
def source_info(fn):
    fn=inspect.unwrap(fn);return dict(path=inspect.getsourcefile(fn),function=fn.__qualname__,first_line=inspect.getsourcelines(fn)[1],file_sha256=sha_file(inspect.getsourcefile(fn)))
def source_line(path,needle):return next(i+1 for i,s in enumerate(Path(path).read_text(encoding='utf-8').splitlines()) if needle in s)

def main():
    torch.manual_seed(20261005);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    legacy=load_core();state=torch.load(OUT/'repro_state.pt',map_location='cuda',weights_only=False)
    w,bias,H=state['W'],state['bias'],state['H'];docs=state['docs'];x=torch.cat([d[0] for d in docs]).float();module=layer_from_weight(w,bias)
    native,native_states=capture_native(make_native(module,H));old=state['custom'];raw_rebuilt,nsamples=build_h(copy.deepcopy(module),docs)
    # Controlled variants differ only in diagonal extraction/reduction or write syntax.
    mean_native=torch.mean(torch.diag(H));mean_view=H.diagonal().mean();variants=[];tensors={}
    for name,use_view,write_view in [('native_expression',False,False),('native_mean_view_write',False,True),('view_mean_index_write',True,False),('v17_expression',True,True)]:
        hh=H.clone();mean=hh.diagonal().mean() if use_view else torch.mean(torch.diag(hh));damp=.1*mean
        if write_view:hh.diagonal().add_(damp)
        else:
            diag=torch.arange(hh.shape[0],device=hh.device);hh[diag,diag]+=damp
        ch=torch.linalg.cholesky(hh);inv=make_native(module,H).block_cholesky_inverse(ch,block_size=896);factor=torch.linalg.cholesky(inv,upper=True)
        tensors[name]={'mean':mean.clone(),'damp':damp.clone(),'damped_H':hh,'chol':ch,'inverse':inv,'Hinv':factor}
        variants.append({'name':name,'mean':float(mean),'mean_exact_native':torch.equal(mean,mean_native),
            'damped_H':cmp(native_states['damped_H'],hh),'Hinv':cmp(native_states['Hinv'],factor)})
    first=dict(operation='FP32 mean reduction of the Hessian diagonal before damping',classification='H/Hinv preprocessing implementation mismatch (contiguous diag copy vs strided diagonal view)',
        native_source=source_info(GPTQ.quantize),custom_source={'path':str(V17/'m2_sequential_smoke.py'),'function':'main','file_sha256':sha_file(V17/'m2_sequential_smoke.py')},
        native_line=source_line(inspect.getsourcefile(inspect.unwrap(GPTQ.quantize)),'damp = damp_percent * torch.mean'),
        custom_line=source_line(V17/'m2_sequential_smoke.py','h.diagonal().add_'),
        native_expression='torch.mean(torch.diag(H))',custom_expression='h.diagonal().mean()',
        input_values_exact=cmp(torch.diag(H),H.diagonal()),native_reduction_input=meta(torch.diag(H)),custom_reduction_input=meta(H.diagonal()),
        output=cmp(mean_native,mean_view),dtype_before='torch.float32',dtype_after='torch.float32',shape_before=[896],shape_after=[],
        previous_numerical_state='undamped Hessian and diagonal element values exact; extraction layout differs',
        exact_values_tensor_file='FIRST_DIVERGENCE_TENSORS.pt',observed_native_variants=variants)
    dump('FIRST_DIVERGENCE.json',first)
    torch.save({'H':H.cpu(),'native_diagonal':torch.diag(H).cpu(),'custom_diagonal':H.diagonal().cpu(),
        'native_mean':mean_native.cpu(),'custom_mean':mean_view.cpu(),'native_damp':native_states['damp'].cpu(),
        'custom_damp':old['damp'].cpu(),'native_Hinv':native_states['Hinv'].cpu(),'custom_Hinv':old['Hinv'].cpu()},OUT/'FIRST_DIVERGENCE_TENSORS.pt')
    fixed=tensors['native_expression'];fixed_factor=fixed['Hinv']
    checks={key:cmp(native_states[key],old[key]) for key in ['damp','damped_H','chol','inverse','Hinv']}
    fixed_checks={key:cmp(native_states[key],fixed[key]) for key in ['damp','damped_H','chol','inverse','Hinv']}
    start={'original_start_pass':False,'original_comparison_stopped_before_group0':True,
        'W':cmp(module.weight,w),'H':cmp(H,raw_rebuilt),'dead_mask':cmp(torch.diag(H)==0,torch.diag(raw_rebuilt)==0),
        'order':list(range(896)),'config':dict(bits=4,sym=True,group_size=128,desc_act=False,block_size=128,damp_percent=.1,static_groups=False),
        'original_Hinv':checks['Hinv'],'corrected_Hinv':fixed_checks['Hinv'],
        'initial_quantizer_state':'native scale=0/zero=0 buffers before find_params; legacy uses None until group entry; neither is used for column quantization',
        'runtime_settings':settings(),'W_native_working':meta(w.float()),'W_custom_working':meta(w.float().clone())}
    qn=make_native(module,H).quantizer;qn.find_params(w.float()[:,:128],weight=True)
    cs,cz=legacy.dynamic_params(w.float()[:,:128],1.)
    start['first_used_scale']=cmp(qn.scale,cs);start['first_used_zero']=cmp(qn.zero,cz)
    start['corrected_start_pass']=all(v['exact'] for v in fixed_checks.values()) and start['H']['exact'] and start['W']['exact']
    dump('START_STATE_CHECK.json',start)
    dump('HINV_AUDIT.json',{'raw_H_rebuilt':cmp(H,raw_rebuilt),'old_stages':checks,'corrected_stages':fixed_checks,'controlled_variants':variants,
        'v17_external_hinv_hash_reproduced':sha(old['Hinv'])=='26171c3a97741883b151b69cfd4563197e8047900f3b7dd37ef4248fdca682fe',
        'old_group0_operator_comparison':'STOPPED: Hinv already unequal; cannot attribute original failure to compensation or range estimation'})
    if not start['corrected_start_pass']:raise RuntimeError('Corrected initial state still differs; stop')
    print('UPSTREAM_ROOT_CAUSE_CONFIRMED',flush=True)

    # Observe exact source copies only after giving both paths native-equal Hinv.
    nr=Recorder(OUT,'native',meta);cr=Recorder(OUT,'custom',meta)
    nfun=make_instrumented(GPTQ.quantize,nr,'native',OUT);cfun=make_instrumented(legacy.dynamic_gptq,cr,'custom',OUT)
    original_helper=quantizer_module.quantize;quantizer_module.quantize=native_integer_observer(nr,original_helper)
    try:
        with torch.inference_mode():observed_native=nfun(make_native(module,H),blocksize=128)
    finally:quantizer_module.quantize=original_helper
    with torch.no_grad():observed_custom=cfun(w.float(),fixed_factor,alpha=1.,group_size=128,block_size=128)
    plain_custom=legacy.dynamic_gptq(w.float(),fixed_factor,alpha=1.)
    nr.save();cr.save()
    def event_map(rec):return {(e['operation'],e['column']):e for e in rec.events}
    nm,cm=event_map(nr),event_map(cr);comparisons=[]
    for key,ne in nm.items():
        ce=cm.get(key)
        if ce is None:comparisons.append({'operation':key[0],'column':key[1],'missing_custom':True});continue
        for tensor,ni in ne['tensors'].items():
            if tensor not in ce['tensors']:continue
            ci=ce['tensors'][tensor];a=nr.tensors[ni['tensor_key']];b=cr.tensors[ci['tensor_key']]
            comparisons.append({'operation':key[0],'column':key[1],'tensor':tensor,'comparison':cmp(a,b),
                'native_source_line':ne['source_line'],'custom_source_line':ce['source_line']})
    divergent=[e for e in comparisons if e.get('missing_custom') or not e['comparison']['exact']]
    instrumented_match={'native_weight':cmp(native[0],observed_native[0]),'native_scale':cmp(native[1],observed_native[1]),
        'custom_weight':cmp(plain_custom.weight_fp32,observed_custom.weight_fp32),'custom_scale':cmp(plain_custom.scales,observed_custom.scales)}
    dump('COMPENSATION_AUDIT.json',{'condition':'common captured native Hinv; all 128 target rows; group0 observation only',
        'operation_comparisons':comparisons,'first_divergent_operation_with_common_Hinv':divergent[0] if divergent else None,
        'all_compared_group0_operations_exact':not divergent,'instrumentation_did_not_change_results':instrumented_match,
        'view_audit':'runtime storage pointers/strides/base flags for working blocks and updates are stored in both operator_trace JSON files',
        'original_vs_repaired_full_native_weight':cmp(native[0],plain_custom.weight_fp32.half())})
    dump('QUANTIZER_AUDIT.json',{'group0_scale':start['first_used_scale'],'group0_zero':start['first_used_zero'],
        'integer_and_dequant_operator_checks':[e for e in comparisons if e['operation'] in ['rounded_integer_q','dequantized_q']],
        'native_maxq':15,'native_sym':True,'native_round':'torch.round','native_clamp':'torch.clamp(round(x/scale)+zero,0,maxq)',
        'native_dequant':'scale*(q-zero)','legacy_dequant':'(q-zero)*scale','range_formula':'(xmax-xmin)/int64 maxq tensor; sym zero=8; native all-positive and all-zero rules retained',
        'root_cause_is_quantizer':False,'repaired_alpha1':'actual native Quantizer.find_params and Quantizer.quantize, with no range-modifying hook'})

    # Actual runtime dtype timeline, including final storage casts.
    timeline=[]
    for label,weight,hfactor,result,rec in [('native',w,native_states['Hinv'],native[0],nr),('legacy_custom',w,fixed_factor,plain_custom.weight_fp32,cr)]:
        rows=[('original_model_W',weight,'checkpoint converted to FP16 before GPTQ'),('working_W',weight.float(),'.float() before any range/GPTQ operation'),('H',H,'process_batch casts calibration input .float() before Gram accumulation'),('Hinv',hfactor,'native lower Cholesky -> native inverse helper -> upper Cholesky; FP32'),('final_returned_weight',result,'native type_as(original FP16) after full GPTQ; legacy returns FP32')]
        for stage,t,cast in rows:timeline.append(dict(path=label,stage=stage,**meta(t),cast_timing=cast,autocast=False))
        for operation in ['group_entry','dequantized_q','error_vector','compensation_product','updated_remaining_W']:
            e=next(e for e in rec.events if e['operation']==operation)
            for name,info in e['tensors'].items():
                timeline.append(dict(path=label,stage=operation+'/'+name,**{k:info[k] for k in ['dtype','device','shape','stride','contiguous','sha256']},cast_timing='no dtype cast in this operation',autocast=False))
    timeline.append(dict(path='legacy_custom',stage='final_stored_weight',**meta(plain_custom.weight_fp32.half()),cast_timing='.half() AFTER GPTQ, BEFORE output loss',autocast=False))
    with (OUT/'DTYPE_TIMELINE.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=timeline[0].keys());writer.writeheader();writer.writerows(timeline)

    # Native-vs-native controls and the exact old custom path, five independent starts.
    native_runs=[];legacy_runs=[];restored_runs=[]
    for repeat in range(5):
        n=make_native(module,H).quantize(blocksize=128);native_runs.append(n[:4])
        oldrun=legacy.dynamic_gptq(w[:32].float(),old['Hinv'],alpha=1.)
        legacy_runs.append((oldrun.weight_fp32.half(),oldrun.scales,oldrun.zeros))
        r=native_dynamic_gptq(w,H,nsamples=2,alpha=1.);restored_runs.append(r)
    def repeated_results(runs,original_w):
        reference=runs[0];cases=[]
        for i,r in enumerate(runs[1:],2):
            cases.append({'run':i,'weight':cmp(reference[0],r[0]),'scale':cmp(reference[1],r[1]),'zero':cmp(reference[2],r[2]),'output_mse':cmp(loss(x,original_w,reference[0]),loss(x,original_w,r[0]))})
        return {'runs':5,'all_bitwise_repeatable':all(c[k]['exact'] for c in cases for k in ['weight','scale','zero','output_mse']),
            'reference_hashes':{'weight':sha(reference[0]),'scale':sha(reference[1]),'zero':sha(reference[2])},'comparisons':cases,'runtime_settings':settings()}
    dump('NATIVE_REPEATABILITY.json',repeated_results(native_runs,w))
    new_tuples=[(r.weight_fp16,r.scales,r.zeros) for r in restored_runs]
    dump('CUSTOM_REPEATABILITY.json',{'legacy_v17_32row_manual_Hinv':repeated_results(legacy_runs,w[:32]),'restored_native_core_full_shape':repeated_results(new_tuples,w),
        'native_vs_restored':compare_result(w,x,native_runs[0],restored_runs[0])})
    legacy_fixed32=legacy.dynamic_gptq(w[:32].float(),fixed_factor,alpha=1.)
    dump('LEGACY_REPAIR_CONTROL.json',{'correcting_only_damping_all128_rows':cmp(native[0],plain_custom.weight_fp32.half()),
        'correcting_only_damping_first32_rows':cmp(native[0][:32],legacy_fixed32.weight_fp32.half()),
        'scale_all128':cmp(native[1],plain_custom.scales),'scale_first32':cmp(native[1][:32],legacy_fixed32.scales),
        'original_legacy32_vs_native32':cmp(native[0][:32],legacy_runs[0][0]),
        'note':'Production interface reuses native core with original target geometry; historical dynamic_core is preserved.'})
    print('REPEAT_CONTROLS_COMPLETE',flush=True)

    # Frozen ladder. Levels 1-3 share one complete alpha=1 native-core result.
    ref=native_runs[0];restored=native_dynamic_gptq(w,H,nsamples=2,alpha=1.,trace=True)
    l1=compare_result(w,x,ref,restored,rows=[17],cols=slice(512,640));l1.update(level=1,scope='reported FP16 mismatch row17 group4; original full target compensation geometry preserved')
    l1['group0_control']=compare_result(w,x,ref,restored,rows=[17],cols=slice(0,128))
    dump('PARITY_LEVEL1.json',l1)
    if not l1['pass']:raise RuntimeError('Level1 failed')
    l2=compare_result(w,x,ref,restored,rows=[17]);l2.update(level=2,scope='original mismatch row17, complete row')
    dump('PARITY_LEVEL2.json',l2)
    if not l2['pass']:raise RuntimeError('Level2 failed')
    l3=compare_result(w,x,ref,restored,rows=list(range(32)));l3.update(level=3,scope='first32 original smoke rows; all128 target rows executed to preserve native geometry')
    dump('PARITY_LEVEL3.json',l3)
    if not l3['pass']:raise RuntimeError('Level3 failed')

    # Four fixed L0 projections, actual projection input hooks, only two 64-token forwards.
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RotaryEmbedding
    mp=ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/REV
    config=AutoConfig.from_pretrained(str(mp),local_files_only=True);config._attn_implementation='sdpa'
    block=Qwen2DecoderLayer(config,layer_idx=0).half().to('cuda').eval()
    with safe_open(str(mp/'model.safetensors'),framework='pt',device='cpu') as f:
        weights={key[len('model.layers.0.'):]:f.get_tensor(key).half() for key in f.keys() if key.startswith('model.layers.0.')}
    block.load_state_dict(weights,strict=True);rotary=Qwen2RotaryEmbedding(config=config).to('cuda')
    inputs={name:[] for name in ['k_proj','v_proj','q_proj','o_proj']};handles=[]
    for name in inputs:
        def hook(m,args,selected=name):inputs[selected].append(args[0].detach().clone())
        handles.append(getattr(block.self_attn,name).register_forward_pre_hook(hook))
    with torch.inference_mode():
        for doc in docs:
            positions=torch.arange(64,device='cuda')[None];pe=rotary(doc,positions)
            block(doc,attention_mask=None,position_ids=positions,cache_position=positions[0],position_embeddings=pe,use_cache=False)
    for handle in handles:handle.remove()
    cases=[]
    for name in ['k_proj','v_proj','q_proj','o_proj']:
        projection=getattr(block.self_attn,name);hh,num=build_h(copy.deepcopy(projection),inputs[name]);nn=make_native(projection,hh).quantize(blocksize=128)
        rr=native_dynamic_gptq(projection.weight.detach(),hh,nsamples=num,alpha=1.)
        rows=[0,projection.out_features//2,projection.out_features-1];xx=torch.cat([d[0] for d in inputs[name]]).float()
        case=compare_result(projection.weight.detach(),xx,nn,rr,rows=rows);case.update(projection=name,rows=rows,input_hashes=[sha(d) for d in inputs[name]],module_shape=list(projection.weight.shape))
        cases.append(case)
    l4={'level':4,'scope':'fixed k/v/q/o projections, first/middle/last rows each; native target geometry preserved; actual native L0 projection inputs','cases':cases,'pass':all(c['pass'] for c in cases)}
    dump('PARITY_LEVEL4.json',l4)
    if not l4['pass']:raise RuntimeError('Level4 failed')
    l5=compare_result(w,x,make_native(module,H).quantize(blocksize=128),native_dynamic_gptq(w,H,nsamples=2,alpha=1.))
    l5.update(level=5,scope='complete original L0 k_proj, 128 rows x 896 columns, same v17 failing input')
    dump('PARITY_LEVEL5.json',l5)
    if not l5['pass']:raise RuntimeError('Level5 failed')

    sources={'native':source_info(GPTQ.quantize),'native_hessian_builder':source_info(GPTQ.process_batch),'native_inverse':source_info(GPTQ.block_cholesky_inverse),
        'native_quantizer_find_params':source_info(quantizer_module.Quantizer.find_params),'native_quantize_helper':source_info(original_helper),
        'old_custom':source_info(legacy.dynamic_gptq),'old_external_hinv':{'path':str(V17/'m2_sequential_smoke.py'),'file_sha256':sha_file(V17/'m2_sequential_smoke.py'),'function':'main'},
        'restored_interface':source_info(native_dynamic_gptq),'versions':settings()}
    dump('SOURCE_MAP.json',sources)
    (OUT/'SOURCE_MAP.md').write_text('# Exact loaded source map\n\n'+ '\n'.join(f'- {k}: `{v.get("path")}`; `{v.get("function")}`; SHA256 `{v.get("file_sha256")}`.' for k,v in sources.items() if isinstance(v,dict) and 'path' in v)+'\n\nRuntime package versions and CUDA settings are in SOURCE_MAP.json. Historical source files were read and copied for observation; no installed package or historical dynamic_core was edited.\n',encoding='utf-8')
    protocol=json.loads((V16/'PROTOCOL.json').read_text(encoding='utf-8'))
    dump('PROTOCOL.json',{'name':'v17.1 Native GPTQ Parity Root-Cause Audit','frozen_v16_protocol_sha256':sha_file(V16/'PROTOCOL.json'),'model_revision':REV,
        'calibration_manifest_sha256':sha_file(V16/'CALIBRATION_MANIFEST.json'),'native':protocol['native'],'alpha':1.,
        'primary_target':'L0 self_attn.k_proj','input':'exact v17 frozen first2 samples x64 tokens; original failure uses raw L0 block-entry embeddings',
        'projection_controls':['L0.k_proj','L0.v_proj','L0.q_proj','L0.o_proj'],'repeat_runs_each':5,'tolerance':'unchanged: exact FP16 weight and exact scale/zero',
        'stop_after_ladder':True,'grid_candidates_executed':0})
    with (OUT/'PATCH_MANIFEST.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['file','change','why','algorithm_semantic_change','historical_source_changed']);writer.writeheader()
        writer.writerow(dict(file='native_m2.py',change='New isolated M2 interface delegates all GPTQ damping/compensation/storage to GPTQModel; range hook only after dynamic find_params and only for alpha !=1',why='Remove v17 strided diagonal mean substitution and duplicate native loop; retain native full target geometry',algorithm_semantic_change='NO relative to frozen native plus scale clipping',historical_source_changed='NO'))
        writer.writerow(dict(file='operator_instrumentation.py',change='Observation copies compiled from loaded source AST; record actual intermediate tensors and split product evaluation for capture',why='Locate/compare operators without altering installed source; instrumented vs plain output checks included',algorithm_semantic_change='NO (audit instrumentation only)',historical_source_changed='NO'))
    nrpt=json.loads((OUT/'NATIVE_REPEATABILITY.json').read_text());crpt=json.loads((OUT/'CUSTOM_REPEATABILITY.json').read_text())
    restored_ok=all(c['pass'] for c in [l1,l2,l3,l4,l5]) and start['corrected_start_pass'] and not divergent and all(v['exact'] for v in instrumented_match.values()) and nrpt['all_bitwise_repeatable'] and crpt['restored_native_core_full_shape']['all_bitwise_repeatable'] and crpt['native_vs_restored']['pass']
    verdict='M2_NATIVE_PARITY_RESTORED' if restored_ok else 'M2_NATIVE_PARITY_NOT_RESTORED'
    dump('SUMMARY.json',{'verdict':verdict,'root_cause':first['classification'],'first_operation':first['operation'],
        'native_repeatable':nrpt['all_bitwise_repeatable'],'legacy_custom_repeatable':crpt['legacy_v17_32row_manual_Hinv']['all_bitwise_repeatable'],
        'restored_custom_repeatable':crpt['restored_native_core_full_shape']['all_bitwise_repeatable'],
        'parity_levels_pass':[l1['pass'],l2['pass'],l3['pass'],l4['pass'],l5['pass']],
        'group0_common_Hinv_operator_comparisons':len(comparisons),'group0_all_exact':not divergent,
        'search_candidates_executed':0,'additional_input_capture_issue':'v17 used raw block input for k_proj Hessian; native k_proj actually receives input_layernorm output. Original input retained for reproduction; Level4 uses actual projection hooks.'})
    (OUT/'REPORT.md').write_text(f'''# v17.1 Native GPTQ parity audit

Verdict: **{verdict}**. The audit stopped after the five-level parity ladder. No range-search grid, selected-weight commit, self-prefix experiment, OmniQuant, or PPL was run.

1. The first numerical divergence is **the FP32 diagonal mean reduction before damping**, not a range or compensation operation. Native computes `torch.mean(torch.diag(H))`; v17 computes `H.diagonal().mean()`. The diagonal values agree bitwise, but the native reduction input is contiguous and the other is a stride-897 view. The means are {float(mean_native):.17g} and {float(mean_view):.17g}, differing by {abs(float(mean_native-mean_view)):.17g}. Exact float bit patterns and full tensors are saved in FIRST_DIVERGENCE.json / FIRST_DIVERGENCE_TENSORS.pt.

2. The mismatch is upstream H/Hinv preprocessing. The external v17 factor hash was reproduced exactly. Capturing the actual native factor showed it was already different before group0. The original starting-state check therefore fails; a compensation comparison under that unequal starting state was stopped. HINV_AUDIT.json records the damping, lower Cholesky, inverse and upper-Cholesky comparisons.

3. Group0 range/scale comes from W, independently of Hinv, so it can agree while the divisor and compensation use a different factor. These small update changes alter the group1 current weights and therefore subsequent dynamic ranges.

4. Rounding decisions and repeated compensation turn initial FP32 differences into some different final FP16 weights. Under a captured common native Hinv, the group0 trace compares {len(comparisons)} tensor operations; all exact: {not divergent}. This controlled check removes the initial factor mismatch before testing the quantizer/error/product/update path.

5. Native-vs-native five-run bitwise repeatability: {nrpt['all_bitwise_repeatable']}.

6. The original custom path's five-run bitwise repeatability: {crpt['legacy_v17_32row_manual_Hinv']['all_bitwise_repeatable']}; the restored native-core path: {crpt['restored_native_core_full_shape']['all_bitwise_repeatable']}. These controls provide no evidence that backend nondeterminism explains this failure. No tolerance amendment is proposed.

7. The implementation mismatch is repairable. Controlled variants show that keeping the native mean but changing the diagonal write syntax reproduces native H/Hinv, whereas retaining the view reduction does not. The new isolated `native_m2.native_dynamic_gptq` accepts pristine FP16 W plus undamped FP32 H and runs the actual native core. Alpha=1 has no range modification. Other alpha values are an implemented range-hook branch, unexecuted in this alpha=1-only audit.

8. The restored interface meets exact weight/scale/zero/output-MSE parity at levels 1-5: reported failing group4/row17 (plus group0 control), complete row17, first32 original smoke rows, fixed k/v/q/o first-middle-last controls, and complete original k_proj. PARITY_LEVEL1..5.json contain exact checks. All native-core calls preserve the original full projection geometry; row subsets are taken from outputs. This audit does not qualify an independently sliced row-batch implementation for a later search. LEGACY_REPAIR_CONTROL.json shows that correcting only damping gives exact full128-row scale, but a separately executed 32-row legacy slice still differs at 24 scale elements (max 7.450580596923828e-09), even though its FP16 weights agree. The restored interface avoids that row-geometry substitution.

9. No unresolved mismatch is attributed to CUDA. Runtime TF32, autocast, deterministic settings and matmul precision are recorded. All source hashes and runtime dtype/storage/view observations are available in SOURCE_MAP, DTYPE_TIMELINE and operator trace artifacts. The historical source and v17 results are unchanged.

10. One additional implementation issue was found: the old v17 smoke used the **raw decoder-entry embedding input** as k_proj calibration input. Native k_proj receives RMSNorm output. Both sides used the same old input, so this does not explain parity failure. It must be corrected when re-entering v17 sequential smoke. Root-cause levels 1-3/5 deliberately retain the old input for exact reproduction; level4 captures real projection inputs from two native L0-only forwards.

Reproduce from the repository root with `.\\.venv-gptq\\Scripts\\python.exe experiments/range_search_w4/results/range_w4_m2_parity_v17_1/upstream_probe.py`, then the same Python executable with `run_audit.py`. Read the completed artifacts before rerunning; these commands are diagnostic only. The v17 smoke launcher remains frozen and still imports the old core. A future explicitly requested smoke must import the new interface and pass undamped H; this audit does not resume that smoke.
''',encoding='utf-8')
    print(json.dumps({'verdict':verdict,'levels':[l1['pass'],l2['pass'],l3['pass'],l4['pass'],l5['pass']],'trace_comparisons':len(comparisons),'trace_exact':not divergent},indent=2),flush=True)

if __name__=='__main__':main()
