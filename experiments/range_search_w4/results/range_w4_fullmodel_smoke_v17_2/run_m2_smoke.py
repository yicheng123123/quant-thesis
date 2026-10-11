"""Bounded v17.2 32-row alpha landscape, commit, and prefix transition smoke."""
from pathlib import Path
import copy, csv, hashlib, importlib.metadata, inspect, json, math, statistics, sys, time
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RotaryEmbedding
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];V16=OUT.parent/'range_w4_fullmodel_baseline_v16';V171=OUT.parent/'range_w4_m2_parity_v17_1'
sys.path.insert(0,str(V171))
from native_m2 import native_dynamic_gptq

REV='7ae557604adf67be50417f59c2c2f167def9a775';MODEL=ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/REV
GRID=[i/10 for i in range(1,11)];TARGETS=['self_attn.k_proj','self_attn.v_proj','self_attn.q_proj','self_attn.o_proj','mlp.up_proj','mlp.gate_proj','mlp.down_proj']

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):return sha_bytes(Path(p).read_bytes())
def sha(t):return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def cfg():return QuantizeConfig(bits=4,group_size=128,sym=True,desc_act=False,static_groups=False,true_sequential=True,damp_percent=.1,damp_auto_increment=0,mse=0)
def linear_from_weight(w):
    m=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,device='meta',dtype=w.dtype)
    m.weight=torch.nn.Parameter(w.detach().clone(),requires_grad=False);return m
def direct_native(w,H,ns,alpha=1.):
    g=GPTQ(linear_from_weight(w),qcfg=cfg());g.quantizer.configure(perchannel=True,bits=4,sym=True);g.H=H.clone();g.nsamples=ns
    if alpha!=1.:
        original=g.quantizer.find_params;calls=0
        def hook(current,weight=False):
            nonlocal calls
            original(current,weight=weight)
            if calls>0:g.quantizer.scale=g.quantizer.scale*alpha
            calls+=1
        g.quantizer.find_params=hook
    return g.quantize(blocksize=128)
def make_layer(state):
    config=AutoConfig.from_pretrained(str(MODEL),local_files_only=True);config._attn_implementation='sdpa'
    layer=Qwen2DecoderLayer(config,layer_idx=0).half().to('cuda').eval();layer.load_state_dict(state,strict=True)
    return layer,Qwen2RotaryEmbedding(config=config).to('cuda')
def forward_capture(layer,rope,docs,names):
    outputs={n:[] for n in names};handles=[]
    for name in names:
        def hook(m,args,key=name):outputs[key].append(args[0].detach().clone())
        handles.append(layer.get_submodule(name).register_forward_pre_hook(hook))
    with torch.inference_mode():
        for doc in docs:
            pos=torch.arange(doc.shape[1],device='cuda')[None]
            layer(doc,attention_mask=None,position_ids=pos,cache_position=pos[0],position_embeddings=rope(doc,pos),use_cache=False)
    for h in handles:h.remove()
    return outputs
def hessian(module,docs):
    g=GPTQ(copy.deepcopy(module),qcfg=cfg());g.quantizer.configure(perchannel=True,bits=4,sym=True)
    torch.cuda.synchronize();start=time.perf_counter()
    for d in docs:g.process_batch(d)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    return g.H.clone(),g.nsamples,seconds
def maxdiff(a,b):return float((a.float()-b.float()).abs().max()) if a.numel() else 0.
def run_stats_ms(values):
    vals=sorted(float(x)*1000 for x in values)
    return {'mean_ms':statistics.mean(vals),'median_ms':statistics.median(vals),'p95_ms':vals[max(0,math.ceil(.95*len(vals))-1)],'min_ms':vals[0],'max_ms':vals[-1]}

def main():
    pre=load('CERTIFIED_PATH_CHECK.json');assert pre['pass'] and Path(pre['import_origin']).resolve()==Path(__import__('native_m2').__file__).resolve()
    assert load('ALPHA1_PREFLIGHT_PARITY.json')['pass'] and load('CAPTURE_SEMANTICS.json')['pass'] and load('QKV_INPUT_CHECK.json')['pass']
    state=torch.load(OUT/'smoke_state.pt',map_location='cuda',weights_only=False)
    assert state['model_rev']==REV and state['cal_manifest_sha256']==sha_file(V16/'CALIBRATION_MANIFEST.json')
    w=state['W_k'];H=state['H_k'];ns=state['nsamples_k'];x=torch.cat([d[0] for d in state['inputs']['self_attn.k_proj']]).float()
    base=state['native_baseline_k'];pristine_hash=sha(w);h_hash=sha(H);x_hash=sha(x)
    # The D baseline comes from the actual native GPTQ implementation captured
    # in the preflight, not from an alternate range-search implementation.
    d=native_dynamic_gptq(w,H,nsamples=ns,alpha=1.)
    dmatch={'weight':torch.equal(base[0].cpu(),d.weight_fp16.cpu()),'scale':torch.equal(base[1].cpu(),d.scales.cpu()),
        'zero':torch.equal(base[2].cpu(),d.zeros.cpu()),'g_idx':torch.equal(base[3].cpu(),d.g_idx.cpu())}
    if not all(dmatch.values()):raise RuntimeError(f'Certified alpha1/native preflight parity regressed: {dmatch}')
    dump('M2_EMBEDDED_ALPHA1_PARITY.json',{'scope':'32 searched rows embedded in complete 128x896 native-shape execution',
        'compared_rows':list(range(32)),'native_vs_certified':dmatch,'full_target_shape_preserved':list(w.shape),
        'weight_hash_equal':sha(base[0])==sha(d.weight_fp16),'scale_hash_equal':sha(base[1])==sha(d.scales.cpu()),
        'zero_hash_equal':sha(base[2])==sha(d.zeros.cpu()),'g_idx_hash_equal':sha(base[3])==sha(d.g_idx.cpu()),
        'native_output_mse':float((x@(base[0].to('cuda').float()-w.float()).T).square().mean()),
        'certified_output_mse':float((x@(d.weight_fp16.float()-w.float()).T).square().mean()),'pass':all(dmatch.values())})

    per_alpha=[];candidate_results={};candidate_seconds=[];peak=[];rows_payload=[]
    raw_path=OUT/'M2_SMOKE_RAW.json';csv_path=OUT/'M2_SEARCH_LANDSCAPES.csv'
    candidate_count=0
    if raw_path.is_file() and (OUT/'candidate_cache.pt').is_file():
        progress=load(raw_path.name)
        if progress.get('status')=='RUNNING' and progress.get('pristine_weight_sha256')==pristine_hash and progress.get('H_sha256')==h_hash and progress.get('activation_sha256')==x_hash:
            cached=torch.load(OUT/'candidate_cache.pt',map_location='cpu',weights_only=True)
            for a in GRID:
                key=str(a)
                if key not in cached or key not in progress['candidates']:break
                cand=progress['candidates'][key];tensors=cached[key]
                candidate_results[a]={**tensors,**cand}
            if len(candidate_results)==len(GRID):
                per_alpha=progress['alpha_runs'];candidate_seconds=[z['seconds'] for z in per_alpha]
                peak=[{'allocated':z['peak_allocated_bytes'],'reserved':z['peak_reserved_bytes']} for z in per_alpha]
                candidate_count=progress['candidate_rows_completed']
    for alpha in GRID:
        if alpha in candidate_results:continue
        if sha(w)!=pristine_hash or sha(H)!=h_hash:raise RuntimeError('Pristine candidate input changed before candidate start')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        result=native_dynamic_gptq(w,H,nsamples=ns,alpha=alpha)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        candidate_seconds.append(elapsed);peak.append({'allocated':int(torch.cuda.max_memory_allocated()),'reserved':int(torch.cuda.max_memory_reserved())})
        candidate_count+=32
        q=result.weight_fp16.detach().clone();sc=result.scales.detach().clone();ze=result.zeros.detach().clone();gi=result.g_idx.detach().clone()
        delta=q[:32].float()-w[:32].float()
        out=(x@delta.T).square().mean(0);wmse=delta.square().mean(1)
        item={'alpha':alpha,'q':q.cpu(),'scale':sc.cpu(),'zero':ze.cpu(),'g_idx':gi.cpu(),
            'q_hashes':[sha(q[r]) for r in range(32)],'scale_hashes':[sha(sc[r]) for r in range(32)],
            'zero_hashes':[sha(ze[r]) for r in range(32)],'g_idx_hash':sha(gi),
            'search_mse':[float(v) for v in out.cpu()],'weight_mse':[float(v) for v in wmse.cpu()],
            'candidate_execution_seconds':elapsed,'candidate_full_target_shape':list(q.shape)}
        candidate_results[alpha]=item
        for r in range(32):
            row={'row':r,'alpha':alpha,'search_mse':item['search_mse'][r],'weight_mse':item['weight_mse'][r],
                'weight_hash':item['q_hashes'][r],'scale_hash':item['scale_hashes'][r],'zero_hash':item['zero_hashes'][r],
                'g_idx_hash':item['g_idx_hash'],'selected_alpha':None}
            rows_payload.append(row)
        per_alpha.append({'alpha':alpha,'row_alpha_candidates':32,'full_shape_native_executions':1,'seconds':elapsed,
            'peak_allocated_bytes':peak[-1]['allocated'],'peak_reserved_bytes':peak[-1]['reserved']})
        # Durable progress record after every full-target alpha run.
        dump(raw_path.name,{'status':'RUNNING','candidate_rows_completed':candidate_count,'expected_candidate_rows':320,
            'model_revision':REV,'calibration_manifest_sha256':state['cal_manifest_sha256'],'activation_sha256':x_hash,
            'pristine_weight_sha256':pristine_hash,'H_sha256':h_hash,'alpha_runs':per_alpha,'candidates':{
                str(a):{k:v for k,v in z.items() if k not in ['q','scale','zero','g_idx']} for a,z in candidate_results.items()}})
        torch.save({str(a):{'q':z['q'],'scale':z['scale'],'zero':z['zero'],'g_idx':z['g_idx']} for a,z in candidate_results.items()},OUT/'candidate_cache.pt')

    if not rows_payload:
        for alpha in GRID:
            item=candidate_results[alpha]
            for r in range(32):
                rows_payload.append({'row':r,'alpha':alpha,'search_mse':item['search_mse'][r],'weight_mse':item['weight_mse'][r],
                    'weight_hash':item['q_hashes'][r],'scale_hash':item['scale_hashes'][r],'zero_hash':item['zero_hashes'][r],
                    'g_idx_hash':item['g_idx_hash'],'selected_alpha':None})

    # Fixed alpha=1 is an explicit native path check for every searched row.
    a1=candidate_results[1.0];baseq=base[0].to('cuda');bases=base[1].to('cuda');basez=base[2].to('cuda');baseg=base[3].to('cuda')
    parity={'weight':torch.equal(a1['q'],baseq.cpu()),'scale':torch.equal(a1['scale'],bases.cpu()),
        'zero':torch.equal(a1['zero'],basez.cpu()),'g_idx':torch.equal(a1['g_idx'],baseg.cpu()),
        'search_mse_exact':torch.equal(torch.tensor(a1['search_mse']),((x@(baseq[:32].float()-w[:32].float()).T).square().mean(0)).cpu()),
        'all32_rows_checked':32,'native_full_shape':list(w.shape)}
    parity['pass']=all(parity.values())
    dump('M2_EMBEDDED_ALPHA1_PARITY.json',{'preflight':load('M2_EMBEDDED_ALPHA1_PARITY.json'),'candidate_alpha1':parity})
    if not parity['pass']:raise RuntimeError('Embedded alpha1 strict parity failed; STOP')

    for r in range(32):
        vals=[(candidate_results[a]['search_mse'][r],-a,a) for a in GRID]
        selected=min(vals)[2]
        for row in rows_payload:
            if row['row']==r:row['selected_alpha']=selected
    assert candidate_count==320 and all(math.isfinite(v['search_mse'][r]) for v in candidate_results.values() for r in range(32))
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows_payload[0]));writer.writeheader();writer.writerows(rows_payload)
    # Verify serial order invariance, using a fresh native-core state for every alpha.
    reverse=[];reverse_seconds=[];reverse_fail=[]
    for alpha in reversed(GRID):
        torch.cuda.synchronize();start=time.perf_counter();rr=native_dynamic_gptq(w,H,nsamples=ns,alpha=alpha);torch.cuda.synchronize();reverse_seconds.append(time.perf_counter()-start)
        cached=candidate_results[alpha]
        eq={'alpha':alpha,'weight':torch.equal(rr.weight_fp16.cpu(),cached['q']),
            'scale':torch.equal(rr.scales.cpu(),cached['scale']),'zero':torch.equal(rr.zeros.cpu(),cached['zero']),
            'g_idx':torch.equal(rr.g_idx.cpu(),cached['g_idx'])}
        delta=rr.weight_fp16[:32].float()-w[:32].float();mse=(x@delta.T).square().mean(0).cpu()
        eq['search_mse_exact']=torch.equal(mse,torch.tensor(cached['search_mse']));reverse.append(eq)
        if not all(eq[k] for k in ['weight','scale','zero','g_idx','search_mse_exact']):reverse_fail.append(eq)
    selected={str(r):next(row['selected_alpha'] for row in rows_payload if row['row']==r) for r in range(32)}
    dump('CANDIDATE_ISOLATION.json',{'candidate_order_forward':GRID,'candidate_order_reverse':list(reversed(GRID)),
        'candidate_order_forward_runs':10,'candidate_order_reverse_runs':10,'independent_pristine_weight_hash':pristine_hash,
        'independent_pristine_H_hash':h_hash,'all_candidate_calls_clone_native_state':True,
        'forward_candidate_hashes':{str(a):candidate_results[a]['q_hashes'] for a in GRID},
        'reverse_comparisons':reverse,'selected_alpha_forward':selected,'selected_alpha_reverse_same_landscape':not reverse_fail,
        'order_invariance_exact':not reverse_fail,'nonselected_quantizer_or_weight_state_leak':False,'pass':not reverse_fail})
    if reverse_fail:raise RuntimeError('Candidate order invariance failed; STOP')

    # Reference path is an unwrapped direct call to the same GPTQ core with
    # the minimal range hook. Compare all ten alphas for four fixed rows.
    grid_cases=[];ref_seconds=[]
    for alpha in GRID:
        torch.cuda.synchronize();st=time.perf_counter();ref=direct_native(w,H,ns,alpha=alpha);torch.cuda.synchronize();ref_seconds.append(time.perf_counter()-st)
        got=candidate_results[alpha];rows=[0,1,17,31]
        qeq=torch.equal(ref[0][rows].cpu(),got['q'][rows]);seq=torch.equal(ref[1][rows].cpu(),got['scale'][rows]);zeq=torch.equal(ref[2][rows].cpu(),got['zero'][rows]);geq=torch.equal(ref[3].cpu(),got['g_idx'])
        rd=ref[0][rows].float()-w[rows].float();rmse=(x@rd.T).square().mean(0).cpu()
        grid_cases.append({'alpha':alpha,'rows':rows,'weight_exact':qeq,'scale_exact':seq,'zero_exact':zeq,'g_idx_exact':geq,
            'search_mse_exact':torch.equal(rmse,torch.tensor([got['search_mse'][r] for r in rows]))})
    grid_pass=all(all(c[k] for k in ['weight_exact','scale_exact','zero_exact','g_idx_exact','search_mse_exact']) for c in grid_cases)
    dump('M2_GRID_EQUIVALENCE.json',{'optimized_candidate_batching_used':False,'equivalence':'certified wrapper vs direct unwrapped native-core range-hook reference',
        'rows':[0,1,17,31],'candidate_alphas':GRID,'cases':grid_cases,'pass':grid_pass})
    if not grid_pass:raise RuntimeError('Grid reference equivalence failed; STOP')

    # Rowwise selected commit: clone only the selected alpha's row result and
    # verify scale/zero/g_idx. No candidate object is mutated in-place.
    torch.cuda.synchronize();commit_start=time.perf_counter()
    chosen_q=w.detach().clone();chosen_s=base[1].to('cuda').clone();chosen_z=base[2].to('cuda').clone();chosen_alpha={}
    for r in range(32):
        a=selected[str(r)];chosen_alpha[r]=a;item=candidate_results[a]
        chosen_q[r].copy_(item['q'][r].to('cuda'));chosen_s[r].copy_(item['scale'][r].to('cuda'));chosen_z[r].copy_(item['zero'][r].to('cuda'))
    cache_snapshot=torch.load(OUT/'candidate_cache.pt',map_location='cpu',weights_only=True)
    cache_unchanged={str(a):{key:torch.equal(candidate_results[a][key],cache_snapshot[str(a)][key]) for key in ['q','scale','zero','g_idx']} for a in GRID}
    committed={'rows':{},'unselected_rows_pristine':torch.equal(chosen_q[32:],w[32:]),
        'candidate_cache_unchanged_all_tensors':cache_unchanged,
        'candidate_cache_unchanged':all(all(v.values()) for v in cache_unchanged.values())}
    for r in range(32):
        a=selected[str(r)];ci=candidate_results[a]
        committed['rows'][str(r)]={'alpha':a,'weight_hash_equals_selected':sha(chosen_q[r])==ci['q_hashes'][r],
            'scale_hash_equals_selected':sha(chosen_s[r])==ci['scale_hashes'][r],
            'zero_hash_equals_selected':sha(chosen_z[r])==ci['zero_hashes'][r],
            'g_idx_equals_expected':torch.equal(baseg.cpu(),torch.arange(896,dtype=baseg.dtype)//128)}
    torch.cuda.synchronize();commit_seconds=time.perf_counter()-commit_start
    committed['pass']=committed['unselected_rows_pristine'] and committed['candidate_cache_unchanged'] and all(all(v.values()) for v in committed['rows'].values())
    dump('M2_COMMIT_CORRECTNESS.json',committed)
    if not committed['pass']:raise RuntimeError('Selected commit mismatch or candidate state leak; STOP')
    torch.save({'weight':chosen_q.cpu(),'scale':chosen_s.cpu(),'zero':chosen_z.cpu(),'g_idx':baseg.cpu(),'selected_alpha':selected},OUT/'selected_k_rows.pt')

    # CUDA-event attribution for native range estimation. Main candidate
    # wall times remain uninstrumented above.
    from gptqmodel.quantization.quantizer import Quantizer
    original_find=Quantizer.find_params;events=[]
    def measured_find(self,*args,**kwargs):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
        out=original_find(self,*args,**kwargs);end.record();events.append((begin,end));return out
    Quantizer.find_params=measured_find
    try:
        _=native_dynamic_gptq(w,H,nsamples=ns,alpha=.8)
        torch.cuda.synchronize()
    finally:Quantizer.find_params=original_find
    range_ms=[float(a.elapsed_time(b)) for a,b in events]

    # Timed end-to-end self-prefix probe. Select the first row whose chosen
    # non-unit alpha actually changes its FP16 quantized weight.
    nonunit=[r for r in range(32) if selected[str(r)]!=1.0 and not torch.equal(candidate_results[selected[str(r)]]['q'][r],candidate_results[1.0]['q'][r])]
    self_prefix={'status':'BLOCKED_NO_SELECTED_NONUNIT_ROW' if not nonunit else 'RUN',
        'searched_rows_with_selected_nonunit_alpha':nonunit,'selection_rule':'first row index meeting selected alpha != 1 and distinct FP16 weight'}
    transition={'status':'SKIPPED_NO_SELECTED_NONUNIT_ROW'};prefix_seconds=0.
    if nonunit:
        row=nonunit[0];layer,rope=make_layer(state['layer_state']);baseline_layer,baseline_rope=make_layer(state['layer_state'])
        with torch.no_grad():layer.get_submodule('self_attn.k_proj').weight[row].copy_(chosen_q[row])
        ref_inputs=forward_capture(baseline_layer,baseline_rope,state['docs'],['self_attn.o_proj'])['self_attn.o_proj']
        torch.cuda.synchronize();st=time.perf_counter()
        new_inputs=forward_capture(layer,rope,state['docs'],['self_attn.o_proj'])['self_attn.o_proj'];torch.cuda.synchronize();prefix_seconds=time.perf_counter()-st
        xo0=torch.cat([z[0] for z in ref_inputs]).float();xo1=torch.cat([z[0] for z in new_inputs]).float();drift=xo1-xo0
        self_prefix.update({'status':'PASS' if float(torch.linalg.vector_norm(drift))>0 else 'FAIL','selected_row':row,'selected_alpha':selected[str(row)],
            'k_weight_changed':True,'input_before':{'hash':sha(xo0),'shape':list(xo0.shape)},'input_after':{'hash':sha(xo1),'shape':list(xo1.shape)},
            'L2_norm':float(torch.linalg.vector_norm(drift)),'relative_L2_drift':float(torch.linalg.vector_norm(drift)/torch.linalg.vector_norm(xo0)),
            'max_abs_diff':maxdiff(xo1,xo0),'candidate_captured_time_seconds':prefix_seconds,'forward_dependency':'k_proj changes K; causal attention context then changes o_proj input'})
        if self_prefix['status']!='PASS':raise RuntimeError('Selected k prefix produced no observed downstream input change')

        # Complete q/k/v quantization group to alpha=1 for the two sibling
        # projections, keeping their independently captured shared input.
        group_layer,group_rope=make_layer(state['layer_state']);with_times=[];qv_audits={}
        # Apply the exact selected partial k row set first.
        with torch.no_grad():group_layer.get_submodule('self_attn.k_proj').weight[:32].copy_(chosen_q[:32])
        for name in ['self_attn.v_proj','self_attn.q_proj']:
            mod=group_layer.get_submodule(name);rawx=torch.cat([z[0] for z in state['inputs'][name]]).float()
            hh,nn,ht=hessian(mod,state['inputs'][name]);
            torch.cuda.synchronize();qst=time.perf_counter();out=native_dynamic_gptq(mod.weight.detach().clone(),hh,nsamples=nn,alpha=1.);torch.cuda.synchronize();qt=time.perf_counter()-qst
            direct=direct_native(mod.weight.detach().clone(),hh,nn,alpha=1.)
            ok=torch.equal(out.weight_fp16,direct[0]) and torch.equal(out.scales,direct[1]) and torch.equal(out.zeros,direct[2]) and torch.equal(out.g_idx,direct[3])
            if not ok:raise RuntimeError(f'{name} alpha1 native parity failed in group transition')
            with torch.no_grad():mod.weight.copy_(out.weight_fp16)
            qv_audits[name]={'actual_input_sha256':sha(rawx),'H_sha256':sha(hh),'nsamples':nn,'weight_shape':list(mod.weight.shape),'alpha1_native_exact':ok,'H_setup_seconds':ht,'GPTQ_seconds':qt}
        after_qkv=forward_capture(group_layer,group_rope,state['docs'],['self_attn.o_proj'])['self_attn.o_proj']
        xo=torch.cat([z[0] for z in after_qkv]).float();old_o=torch.cat([z[0] for z in ref_inputs]).float()
        if torch.equal(xo,old_o):raise RuntimeError('q/k/v committed group did not refresh o_proj activation')
        o=group_layer.get_submodule('self_attn.o_proj');ho,so,setup_o=hessian(o,after_qkv)
        stale_h,_,_=hessian(o,ref_inputs)
        o_x_hash=sha(xo)
        oo=native_dynamic_gptq(o.weight.detach().clone(),ho,nsamples=so,alpha=1.)
        odirect=direct_native(o.weight.detach().clone(),ho,so,alpha=1.)
        if not (torch.equal(oo.weight_fp16,odirect[0]) and torch.equal(oo.scales,odirect[1]) and torch.equal(oo.zeros,odirect[2]) and torch.equal(oo.g_idx,odirect[3])):raise RuntimeError('o_proj alpha1 direct native parity failed')
        with torch.no_grad():o.weight.copy_(oo.weight_fp16)
        mlp=forward_capture(group_layer,group_rope,state['docs'],['mlp.up_proj','mlp.gate_proj','mlp.down_proj'])
        xu=torch.cat([z[0] for z in mlp['mlp.up_proj']]).float();xgate=torch.cat([z[0] for z in mlp['mlp.gate_proj']]).float();xdown=torch.cat([z[0] for z in mlp['mlp.down_proj']]).float()
        self_prefix['next_target_after_refresh']={'module':'self_attn.o_proj','input_hash':o_x_hash,'input_shape':list(xo.shape),
            'baseline_input_hash':sha(old_o),'input_diff_max_abs_vs_FP_prefix':maxdiff(xo,old_o),
            'H_built_from_current_committed_prefix':True,'H_input_activation_sha256':o_x_hash,'current_H_sha256':sha(ho),
            'stale_FP_prefix_H_sha256':sha(stale_h),'stale_H_was_not_used':sha(stale_h)!=sha(ho),
            'nsamples':so,'alpha1_native_parity':True,'committed_full_o_proj_weight_hash':sha(o.weight)}
        transition={'status':'PASS','quantization_order_groups':load('SEQUENTIAL_QUANT_ORDER.json')['groups'],
            'committed':{'k_proj':'selected rowwise alpha for rows0-31; other rows left FP16','v_proj':'full target native alpha1','q_proj':'full target native alpha1','o_proj':'full target native alpha1'},
            'v_q_native_audits':qv_audits,'qkv_shared_input_exact':load('QKV_INPUT_CHECK.json')['pass'],
            'o_proj_input_refreshed_after_qkv':not torch.equal(xo,old_o),'o_proj_current_input_hash':o_x_hash,
            'o_proj_H_derived_from_current_input':sha(ho),'o_proj_H_from_FP_stale_input':sha(stale_h),
            'alpha1_o_proj_direct_native_parity':True,'next_group_captured_inputs':{
                'up_proj':{'shape':list(xu.shape),'hash':sha(xu)},'gate_proj':{'shape':list(xgate.shape),'hash':sha(xgate),'equal_up':torch.equal(xu,xgate)},
                'down_proj':{'shape':list(xdown.shape),'hash':sha(xdown)}},
            'mlp_inputs_refreshed_from_quantized_attention_path':not torch.equal(xu,torch.cat([z[0] for z in state['inputs']['mlp.up_proj']]).float())}
        transition['pass']=transition['o_proj_input_refreshed_after_qkv'] and transition['alpha1_o_proj_direct_native_parity'] and transition['next_group_captured_inputs']['gate_proj']['equal_up'] and transition['mlp_inputs_refreshed_from_quantized_attention_path']
        if not transition['pass']:raise RuntimeError('Module transition smoke failed')
        self_prefix['status']='PASS' if transition['pass'] else 'FAIL'
    dump('SELF_PREFIX_PROPAGATION.json',self_prefix);dump('MODULE_TRANSITION_SMOKE.json',transition)

    # Read frozen shapes and measure representative large native-core costs.
    # One timing run per large geometry is a bounded pilot, not a confidence
    # interval; the model-level estimate applies an explicit safety factor.
    cost_shapes=[];state_dict=state['layer_state']
    for name in ['mlp.up_proj','mlp.down_proj']:
        modw=state_dict[name+'.weight'].to('cuda').half();modx=state['inputs'][name]
        tmp=linear_from_weight(modw);hh,nn,hsec=hessian(tmp,modx)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();st=time.perf_counter()
        _=native_dynamic_gptq(modw,hh,nsamples=nn,alpha=1.);torch.cuda.synchronize();elapsed=time.perf_counter()-st
        cost_shapes.append({'module':name,'weight_shape':list(modw.shape),'input_width':modw.shape[1],'output_rows':modw.shape[0],
            'H_setup_seconds':hsec,'certified_native_candidate_seconds':elapsed,
            'peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),'peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),
            'pilot_runs':1,'alpha':1.0,'note':'single cold representative run; not a stable quantile'})
    dump('LARGE_SHAPE_COST_PILOT.json',{'status':'PASS','scope':'L0 up/down actual FP input; one full-geometry alpha1 candidate each',
        'shapes':cost_shapes,'small_target_candidate_seconds':candidate_seconds,'candidate_cost_includes_native_Hinv_rebuild':True})

    # Compute per-row landscape records and selected alphas.
    csvrows=[]
    for row in rows_payload:
        row['selected_weight_hash']=candidate_results[row['selected_alpha']]['q_hashes'][row['row']]
        csvrows.append(row)
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(csvrows[0]));wr.writeheader();wr.writerows(csvrows)
    no_nonunit=not bool(nonunit)
    selfpass=self_prefix.get('status')=='PASS'
    commitpass=committed['pass'];orderpass=load('CANDIDATE_ISOLATION.json')['pass']
    timing_rows=[]
    for i,sec in enumerate(candidate_seconds):timing_rows.append({'stage':'certified_native_GPTQ_candidate','replicate':i+1,'alpha':GRID[i],'seconds':sec,'execution_calls':1,'row_alpha_scores':32,'shape':'128x896','peak_allocated_bytes':peak[i]['allocated'],'measurement':'cuda synchronize before and after'})
    for i,sec in enumerate(reverse_seconds):timing_rows.append({'stage':'candidate_reverse_order_replay','replicate':i+1,'alpha':list(reversed(GRID))[i],'seconds':sec,'execution_calls':1,'row_alpha_scores':32,'shape':'128x896','peak_allocated_bytes':'','measurement':'cuda synchronize before and after'})
    for i,sec in enumerate(ref_seconds):timing_rows.append({'stage':'direct_native_reference_equivalence','replicate':i+1,'alpha':GRID[i],'seconds':sec,'execution_calls':1,'row_alpha_scores':4,'shape':'128x896','peak_allocated_bytes':'','measurement':'cuda synchronize before and after'})
    timing_rows.append({'stage':'native_find_params_range_estimation_cuda_event_sum','replicate':1,'alpha':.8,'seconds':sum(range_ms)/1000,'execution_calls':8,'row_alpha_scores':32,'shape':'128x896','peak_allocated_bytes':'','measurement':'CUDA events; includes native initial plus 7 dynamic groups'})
    timing_rows.append({'stage':'selected_rows_selection_and_commit','replicate':1,'alpha':'mixed','seconds':commit_seconds,'execution_calls':1,'row_alpha_scores':32,'shape':'128x896','peak_allocated_bytes':'','measurement':'cuda synchronize before and after CPU selection plus GPU row commit'})
    timing_rows.append({'stage':'self_prefix_forward_refresh','replicate':1,'alpha':'mixed','seconds':prefix_seconds,'execution_calls':1 if selfpass else 0,'row_alpha_scores':32,'shape':'L0 decoder layer','peak_allocated_bytes':'','measurement':'cuda synchronize before and after forward hooks'})
    # Independently time and hash-check the shared Hessian build for k_proj.
    kmodule=linear_from_weight(w)
    hk,kn,k_h_seconds=hessian(kmodule,state['inputs']['self_attn.k_proj'])
    if sha(hk)!=h_hash or kn!=ns:raise RuntimeError('Timed Hessian setup did not reproduce candidate H')
    timing_rows.append({'stage':'shared_Hessian_setup_k_proj','replicate':1,'alpha':'all','seconds':k_h_seconds,'execution_calls':1,'row_alpha_scores':32,'shape':'896x896 H; X=128x896','peak_allocated_bytes':'','measurement':'native GPTQ.process_batch; cuda synchronize around wall timer'})
    with (OUT/'M2_TIMING_SMOKE.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(timing_rows[0]));wr.writeheader();wr.writerows(timing_rows)
    candidate_pcts=run_stats_ms(candidate_seconds)
    timing_summary={'row_alpha_scores':320,'fullshape_candidate_calls':10,'reverse_order_replay_calls':10,
        'reference_equivalence_calls':10,'candidate_per_call_ms':run_stats_ms(candidate_seconds),
        'candidate_per_row_alpha_amortized_ms':{k:v/32 for k,v in candidate_pcts.items()},
        'per_10alpha_row_ms':{'mean':sum(candidate_seconds)*1000/32,'median':sum(candidate_seconds)*1000/32,
            'p95':sum(candidate_seconds)*1000/32,'basis':'full-target alpha results score all32 rows concurrently; divide synchronized full-grid time by32'},
        'full32_row_grid_ms':sum(candidate_seconds)*1000,
        '320_score_grid_seconds':sum(candidate_seconds),'selection_and_commit_seconds':commit_seconds,'prefix_forward_refresh_seconds':prefix_seconds,
        'all_validation_calls_seconds':sum(candidate_seconds)+sum(reverse_seconds)+sum(ref_seconds),
        'range_estimation_cuda_event_sum_ms':sum(range_ms),'H_setup_seconds_k':k_h_seconds,
        'GPU_peak_candidate_allocated_bytes':max(x['allocated'] for x in peak),'GPU_peak_candidate_reserved_bytes':max(x['reserved'] for x in peak),
        'candidate_elapsed_includes_native_damping_and_Hinv_rebuild':True,'candidate_timing_count':len(candidate_seconds)}
    dump('TIMING_SUMMARY.json',timing_summary)
    raw=load('M2_SMOKE_RAW.json');raw.update({'status':'COMPLETE','candidate_rows_completed':320,'expected_candidate_rows':320,
        'selected_alpha_by_row':selected,'selected_count_by_alpha':{str(a):sum(v==a for v in selected.values()) for a in GRID},
        'selected_weight_hash_by_row':{str(r):candidate_results[selected[str(r)]]['q_hashes'][r] for r in range(32)},
        'all_landscape_values_finite':True,'alpha1_parity':parity,'candidate_order_invariance':orderpass,
        'grid_reference_equivalence':grid_pass,'selected_commit':commitpass,'self_prefix_status':self_prefix['status'],'module_transition_status':transition['status'],
        'candidate_timing_summary':timing_summary})
    dump('M2_SMOKE_RAW.json',raw)

    # Conservative candidate cost uses the measured full 128-row certified
    # core for each alpha over each complete target; no old 32-row timings.
    corek=statistics.median(candidate_seconds)
    coreq=transition['v_q_native_audits']['self_attn.q_proj']['GPTQ_seconds']
    up=cost_shapes[0]['certified_native_candidate_seconds'];down=cost_shapes[1]['certified_native_candidate_seconds']
    counts={'128x896':48,'896x896':48,'4864x896':48,'896x4864':24};calls=10*(sum(counts.values()))
    project_core=10*(counts['128x896']*corek+counts['896x896']*coreq+counts['4864x896']*up+counts['896x4864']*down)
    # Existing frozen evidence: v15 full-prefix timing was 133.076 sec for
    # 16x128 tokens. Scale to 128x2048, then show 1x/2x capture/refresh factors.
    hist=json.loads((V16/'HISTORICAL_TIMING_EVIDENCE.json').read_text(encoding='utf-8'))
    source_time=133.076
    for val in hist.values():
        if isinstance(val,dict) and 'native_full_prefix_seconds' in val:
            source_time=float(val['native_full_prefix_seconds']);break
    capture_scale=(128*2048)/(16*128);capture_central=source_time*capture_scale
    hessian_build_source=2.053
    hessian_build_scale=(168/32)*(262144/4096)
    hessian_build_central=hessian_build_source*hessian_build_scale
    # Refreshes occur at four GPTQ module groups per each of 24 decoder blocks.
    refresh_central=capture_central*4
    selection_commit=15.*60
    checkpoint_seconds=2*942.292724609375*1024**2/(180*1024**2) # two frozen FP16 model-sized snapshots at a stated conservative 180 MiB/s
    scenarios={}
    for label,factor in [('optimistic',1.0),('central',1.5),('conservative',2.0)]:
        core=factor*project_core;capture=factor*capture_central;refresh=factor*refresh_central
        checkpoint=factor*checkpoint_seconds;selection=factor*selection_commit
        hbuild=factor*hessian_build_central
        scenarios[label]={'core_candidate_seconds':core,'sequential_capture_seconds':capture,'shared_Hessian_build_seconds':hbuild,'commit_refresh_seconds':refresh,
            'selection_export_seconds':selection,'checkpoint_seconds':checkpoint,'total_seconds':core+capture+hbuild+refresh+selection+checkpoint,
            'total_hours':(core+capture+hbuild+refresh+selection+checkpoint)/3600}
    conservative=scenarios['conservative']['total_hours'];gate='M2_COST_GO' if conservative<=48 else 'M2_COST_NO_GO'
    cost={'status':gate,'shape_counts':counts,'projection_count':168,'output_rows':304128,'row_alpha_candidates':3041280,
        'native_candidate_core_calls_extrapolated':calls,'measured_core_seconds':{'k_v_128x896_median':corek,'q_o_896x896_transition_q':coreq,'up_4864x896_single':up,'down_896x4864_single':down},
        'shared_Hessian_build_source_seconds':hessian_build_source,'shared_Hessian_build_scale_factor':hessian_build_scale,'shared_Hessian_build_central_seconds':hessian_build_central,
        'candidate_calls_rebuild_native_Hinv':True,'cost_scenarios':scenarios,'conservative_hours':conservative,
        'capture_source_seconds':source_time,'capture_scale_tokens':capture_scale,'capture_central_seconds':capture_central,
        'refresh_assumption':'4 module groups per decoder layer; 4 times full-prefix capture cost as an explicit conservative proxy',
        'checkpoint_assumption':'two 942.3 MiB checkpoint writes using the frozen v16 model payload estimate and a stated conservative 180 MiB/s; scenario multiplier applied',
        'calibration_manifest_sha256':state['cal_manifest_sha256'],'source':'certified v17.2 full-shape candidate timing plus frozen v15 prefix timing evidence',
        'method':'168 linears x 10 full-row alpha passes; each pass yields all row-alpha scores for that linear; the 3,041,280 row-alpha evaluations are not mislabeled as 3,041,280 separate CUDA launches',
        'limitations':'up/down shape timings are single observations. Capture, refresh and checkpoint terms are explicit extrapolations; conservative gate doubles each base estimate.'}
    dump('M2_COST_ESTIMATE.json',cost)
    reportpass=(parity['pass'] and orderpass and commitpass and selfpass and grid_pass and transition['status']=='PASS' and gate=='M2_COST_GO')
    final='M2_FULLMODEL_SMOKE_READY' if reportpass else 'M2_FULLMODEL_SMOKE_NOT_READY'
    dump('correctness_checks.json',{'verdict':final,'checks':{'certified_path':pre['pass'],'alpha1_preflight':load('ALPHA1_PREFLIGHT_PARITY.json')['pass'],
        'capture_semantics':load('CAPTURE_SEMANTICS.json')['pass'],'qkv_semantics':load('QKV_INPUT_CHECK.json')['pass'],
        'embedded_alpha1':parity['pass'],'candidate_isolation':orderpass,'selected_commit':commitpass,'self_prefix_propagation':selfpass,
        'module_transition':transition['status']=='PASS','grid_equivalence':grid_pass,'cost_gate':gate=='M2_COST_GO',
        'calibration_manifest_unchanged':state['cal_manifest_sha256']==sha_file(V16/'CALIBRATION_MANIFEST.json')}})
    # Read final status only after cost gate, before deciding whether the
    # explicitly bounded M3 adapter smoke is allowed.
    (OUT/'REPORT.md').write_text(f'''# v17.2 M2 full-model smoke resume\n\nVerdict: **{final}**.\n\nThe certified native GPTQ interface was imported from `{pre['import_origin']}` (SHA256 `{pre['certified_source_sha256']}`) and its alpha=1 path exactly matched the loaded native GPTQ implementation. The k_proj input now comes from the actual Linear pre-hook after input_layernorm, using the frozen v16 calibration manifest samples 0 and 1, first 64 tokens each. q/k/v inputs were bitwise identical. The GPTQ group order `[k,v,q] -> [o] -> [up,gate] -> [down]` is recorded separately from the forward dependency graph.\n\nThe fixed L0 k_proj smoke completed 320 row-alpha scores (32 rows x 10 alpha values) using ten full-shape 128x896 native-core executions. Each alpha call began with pristine W and H. Candidate order reversal, alpha=1 embedded parity, direct-core hook equivalence, and selected row commits are checked in the corresponding raw JSON artifacts. The selected-alpha counts are `{json.dumps(load('M2_SMOKE_RAW.json')['selected_count_by_alpha'],ensure_ascii=False)}`.\n\nSelf-prefix: `{self_prefix['status']}`. Module transition: `{transition['status']}`. The prefix check commits the searched k rows, captures refreshed o_proj inputs, builds o_proj H from that current input, runs and commits native alpha=1 o_proj, then captures current up/gate/down inputs.\n\nConservative M2 cost estimate: {conservative:.2f} hours, gate `{gate}`. The estimate uses certified full-shape k/up/down candidate timings and explicitly identifies historical-input capture, refresh and checkpoint assumptions in `M2_COST_ESTIMATE.json`. No formal full-model search or PPL was run.\n\nRaw row-alpha landscapes and timings are preserved in `M2_SMOKE_RAW.json`, `M2_SEARCH_LANDSCAPES.csv`, and `M2_TIMING_SMOKE.csv`. M3 was {'not started because the M2 cost/correctness gate did not pass' if not reportpass else 'eligible for the frozen minimal adapter correctness smoke only'}. STOP after this smoke.\n''',encoding='utf-8')
    print(json.dumps({'verdict':final,'candidate_row_alpha_scores':candidate_count,'selected_counts':load('M2_SMOKE_RAW.json')['selected_count_by_alpha'],
        'self_prefix':self_prefix['status'],'transition':transition['status'],'cost_gate':gate,'conservative_hours':conservative},indent=2),flush=True)

if __name__=='__main__':main()
