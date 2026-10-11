"""L0/32-row sequential M2 smoke. Refuses any scope beyond the frozen 320 slots."""
from pathlib import Path
import hashlib, importlib.util, json, math, os, statistics, sys, time, csv, copy
import torch
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
V16=OUT.parent/'range_w4_fullmodel_baseline_v16'
REV='7ae557604adf67be50417f59c2c2f167def9a775'
GRID=[i/10 for i in range(1,11)]
DEVICE='cuda'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load_core():
    p=ROOT/'experiments/range_search_w4/dynamic_core.py'
    spec=importlib.util.spec_from_file_location('v17_dynamic_core',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def sync():torch.cuda.synchronize()
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
class StopAtFirstBlock(RuntimeError):pass

def main():
    for name in ['M2_SMOKE_RAW.json','M2_PARITY.json','M2_COMMIT_CORRECTNESS.json','M2_GRID_EQUIVALENCE.json','M2_TIMING_SMOKE.csv','SEQUENTIAL_MODULE_ORDER.json']:
        if (OUT/name).exists():raise RuntimeError(f'refusing to overwrite completed artifact: {name}')
    protocol=json.loads((V16/'PROTOCOL.json').read_text(encoding='utf-8'))
    calibration=json.loads((V16/'CALIBRATION_MANIFEST.json').read_text(encoding='utf-8'))
    assert protocol['search']['grid']==GRID
    assert len(calibration['samples'])==128 and all(len(s['token_ids'])==2048 for s in calibration['samples'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(20261005);torch.cuda.reset_peak_memory_stats()
    from transformers import AutoModelForCausalLM
    from gptqmodel.models.definitions.qwen2 import Qwen2GPTQ
    from gptqmodel.quantization.gptq import GPTQ
    from gptqmodel.quantization import QuantizeConfig
    core=load_core()
    model_path=ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/REV
    model=AutoModelForCausalLM.from_pretrained(str(model_path),torch_dtype=torch.float16,low_cpu_mem_usage=True,
        local_files_only=True,attn_implementation='sdpa').to(DEVICE).eval()
    ids=[torch.tensor(s['token_ids'][:64],device=DEVICE,dtype=torch.long)[None] for s in calibration['samples'][:2]]
    assert len(ids)==2 and all(x.shape==(1,64) for x in ids)
    captured=[]
    def stop_hook(module,args,kwargs):
        captured.append((args[0].detach().clone(),{k:v for k,v in kwargs.items()}));raise StopAtFirstBlock
    handle=model.model.layers[0].register_forward_pre_hook(stop_hook,with_kwargs=True)
    torch.cuda.reset_peak_memory_stats();sync();capture_start=time.perf_counter()
    with torch.inference_mode():
        for sample in ids:
            try:model(input_ids=sample,use_cache=False)
            except StopAtFirstBlock:pass
    sync();capture_seconds=time.perf_counter()-capture_start;capture_peak=(int(torch.cuda.max_memory_allocated()),int(torch.cuda.max_memory_reserved()));handle.remove()
    assert len(captured)==2 and all(torch.isfinite(x[0]).all() for x in captured)
    docs=[x[0][0] for x in captured]; x=torch.cat(docs).float(); block=model.model.layers[0]
    order=[list(g) for g in Qwen2GPTQ.layer_modules]
    expected=[['self_attn.k_proj','self_attn.v_proj','self_attn.q_proj'],['self_attn.o_proj'],['mlp.up_proj','mlp.gate_proj'],['mlp.down_proj']]
    assert order==expected
    dump(OUT/'SEQUENTIAL_MODULE_ORDER.json',dict(source='gptqmodel.models.definitions.qwen2.Qwen2GPTQ.layer_modules',
        groups=order,true_sequential=True,semantics='all group-member inputs captured at group entry; install the complete group before producing next-group activations',
        smoke_target='first member k_proj, first32 rows only; same-group v/q input is expected unchanged; check next-group o_proj input for changed activation',
        model_revision=REV))
    # Exact-seven discovery against official Qwen2 decoder and checkpoint shapes.
    target=block.self_attn.k_proj
    shape={'in_features':target.in_features,'out_features':target.out_features,
           'weight_shape':list(target.weight.shape),'bias_shape':list(target.bias.shape)}
    pristine=target.weight.detach().clone(); pristine_hash=tensor_sha(pristine)
    assert target.in_features==896 and target.out_features==128
    qcfg=QuantizeConfig(bits=4,group_size=128,sym=True,desc_act=False,static_groups=False,true_sequential=True,
                        damp_percent=.1,damp_auto_increment=0,mse=0)
    native=GPTQ(copy.deepcopy(target),qcfg=qcfg)
    native.quantizer.configure(perchannel=True,bits=4,sym=True)
    torch.cuda.reset_peak_memory_stats();sync();h_start=time.perf_counter()
    for doc in docs:native.process_batch(doc[None])
    h=native.H.clone(); dead=torch.diag(h)==0; dead_count=int(dead.sum())
    if dead_count:raise RuntimeError('smoke has dead columns: native dead-column path required before continuing')
    h.diagonal().add_(.1*h.diagonal().mean());chol=torch.linalg.cholesky(h)
    hinv_factor=torch.linalg.cholesky(native.block_cholesky_inverse(chol,block_size=h.shape[0]),upper=True)
    sync();h_seconds=time.perf_counter()-h_start;h_peak=(int(torch.cuda.max_memory_allocated()),int(torch.cuda.max_memory_reserved()))
    # Native alpha=1 parity over all 32 rows using the same pristine target and H.
    nweight,nscale,nzero,ngidx,*_=native.quantize(blocksize=128)
    native_half=nweight[:32].half().clone();native_scale=nscale[:32].clone();native_zero=nzero[:32].clone()
    W=pristine[:32].float().clone();Hinv=hinv_factor.clone();initial_hash=tensor_sha(W);hinv_hash=tensor_sha(Hinv)
    parity_start=time.perf_counter();custom1=core.dynamic_gptq(W,Hinv,alpha=1.0,group_size=128,block_size=128);sync()
    parity=dict(rows=32,weight_fp16_exact=torch.equal(custom1.weight_fp32.half(),native_half),
        scale_max_abs=float((custom1.scales-native_scale).abs().max()),zero_exact=torch.equal(custom1.zeros,native_zero),
        g_idx_expected=torch.equal(ngidx.cpu(),torch.arange(896,dtype=ngidx.dtype)//128),
        weight_max_abs=float((custom1.weight_fp32.half().float()-native_half.float()).abs().max()),
        calibration_local_mse_native=float(((x@(native_half.float()-W).T).square().mean()).item()),
        calibration_local_mse_custom=float(((x@(custom1.weight_fp32.half().float()-W).T).square().mean()).item()),
        pristine_hash=initial_hash,common_hinv_hash=hinv_hash)
    parity['pass']=bool(parity['weight_fp16_exact'] and parity['scale_max_abs']==0 and parity['zero_exact'] and parity['g_idx_expected'])
    if not parity['pass']:raise RuntimeError('M2 alpha=1 native parity failed; stop')
    parity['seconds']=time.perf_counter()-parity_start;dump(OUT/'M2_PARITY.json',parity)
    # Freeze the pre-commit next-group input. Same-group siblings share their captured input.
    qx=docs[0].unsqueeze(0); call_kwargs=captured[0][1]
    o_before=[];same_group_before=[]
    ho=block.self_attn.o_proj.register_forward_pre_hook(lambda m,a:o_before.append(a[0].detach().clone()))
    hv=block.self_attn.v_proj.register_forward_pre_hook(lambda m,a:same_group_before.append(a[0].detach().clone()))
    with torch.inference_mode():block(qx,**call_kwargs)
    ho.remove();hv.remove()
    # The model and compensation states remain pristine until all 10 independent candidates complete.
    results=[];candidate_seconds=[];candidate_halves=[];candidate_scales=[];candidate_zeros=[]
    torch.cuda.reset_peak_memory_stats();sync()
    for alpha in GRID:
        before_w=tensor_sha(W);before_h=tensor_sha(Hinv);sync();t=time.perf_counter()
        result=core.dynamic_gptq(W,Hinv,alpha=alpha,group_size=128,block_size=128)
        qhalf=result.weight_fp32.half().clone()
        loss=((x@(qhalf.float()-W).T).square().mean(0))
        sync();elapsed=time.perf_counter()-t
        if tensor_sha(W)!=initial_hash or tensor_sha(Hinv)!=hinv_hash:raise RuntimeError('candidate mutated pristine W/Hinv')
        candidate_seconds.append(elapsed);candidate_halves.append(qhalf);candidate_scales.append(result.scales.clone());candidate_zeros.append(result.zeros.clone())
        for row in range(32):
            results.append(dict(row=row,alpha=alpha,search_mse=float(loss[row]),
                weight_mse=float((qhalf[row].float()-W[row]).square().mean()),
                q_hash=tensor_sha(qhalf[row]),scale=result.scales[row].detach().cpu().tolist(),zero=result.zeros[row].detach().cpu().tolist()))
    candidate_peak=(int(torch.cuda.max_memory_allocated()),int(torch.cuda.max_memory_reserved()))
    torch.cuda.reset_peak_memory_stats();sync();selection_start=time.perf_counter()
    byrow=[]
    selected=[]
    for row in range(32):
        candidates=[r for r in results if r['row']==row]
        best=min(candidates,key=lambda r:(r['search_mse'],-r['alpha']))
        selected.append(best)
        byrow.append(dict(row=row,selected_alpha=best['alpha'],search_mse=best['search_mse'],candidate_hash=best['q_hash']))
    sync();selection_seconds=time.perf_counter()-selection_start;selection_peak=(int(torch.cuda.max_memory_allocated()),int(torch.cuda.max_memory_reserved()))
    # Serial row reference vs the exact B=32 production core for four prespecified rows.
    eqrows=[]
    for row in [0,10,20,31]:
        for ai,alpha in enumerate(GRID):
            serial=core.dynamic_gptq(W[row:row+1],Hinv,alpha=alpha,group_size=128,block_size=128)
            qeq=torch.equal(serial.weight_fp32.half(),candidate_halves[ai][row:row+1])
            se=float((serial.scales-candidate_scales[ai][row:row+1]).abs().max())
            ze=torch.equal(serial.zeros,candidate_zeros[ai][row:row+1])
            sl=float((x@(serial.weight_fp32.half().float()-W[row:row+1]).T).square().mean().item())
            eqrows.append(dict(row=row,alpha=alpha,weight_fp16_exact=qeq,scale_max_abs=se,zero_exact=ze,
                search_mse_abs=abs(sl-results[ai*32+row]['search_mse'])))
    grid_eq=dict(reference='each row and alpha rerun alone from identical pristine row/Hinv',optimized='B32 batched rows, serial alpha loop, full G128 GPTQ',
        comparisons=len(eqrows),pass_=all(a['weight_fp16_exact'] and a['zero_exact'] and a['scale_max_abs']==0 and a['search_mse_abs']<=1e-8 for a in eqrows),cases=eqrows)
    grid_eq['pass']=grid_eq.pop('pass_');dump(OUT/'M2_GRID_EQUIVALENCE.json',grid_eq)
    if not grid_eq['pass']:raise RuntimeError('B32 vs serial full-grid equivalence failed; stop')
    # Reordering candidates cannot contaminate the state because no candidate touches the model.
    alpha1_after=core.dynamic_gptq(W,Hinv,alpha=1.0,group_size=128,block_size=128)
    isolation=dict(pristine_weight_unchanged=tensor_sha(W)==initial_hash,hinv_unchanged=tensor_sha(Hinv)==hinv_hash,
        model_unmodified_before_commit=tensor_sha(target.weight[:32])==tensor_sha(pristine[:32]),
        alpha1_reordered_exact=torch.equal(alpha1_after.weight_fp32.half(),candidate_halves[9]),
        candidates_complete=len(results)==320,pass_=True)
    isolation['pass']=all([isolation['pristine_weight_unchanged'],isolation['hinv_unchanged'],isolation['model_unmodified_before_commit'],isolation['alpha1_reordered_exact'],isolation['candidates_complete']])
    # Commit selected rows only; do not copy any candidate's mutable compensation workspace.
    chosen_row=next((r['row'] for r in selected if r['alpha']!=1.0),0)
    chosen=selected[chosen_row];alpha=chosen['alpha'];ai=GRID.index(alpha);candidate=candidate_halves[ai][chosen_row].clone()
    assert tensor_sha(candidate)==chosen['candidate_hash']
    torch.cuda.reset_peak_memory_stats();sync();commit_start=time.perf_counter()
    with torch.no_grad():target.weight[chosen_row].copy_(candidate.to(target.weight.dtype))
    installed_hash=tensor_sha(target.weight[chosen_row]);commit_hash_match=installed_hash==chosen['candidate_hash']
    o_after=[];same_group_after=[]
    ho=block.self_attn.o_proj.register_forward_pre_hook(lambda m,a:o_after.append(a[0].detach().clone()))
    hv=block.self_attn.v_proj.register_forward_pre_hook(lambda m,a:same_group_after.append(a[0].detach().clone()))
    with torch.inference_mode():block(qx,**call_kwargs)
    ho.remove();hv.remove()
    o_delta=float((o_after[0].float()-o_before[0].float()).abs().max())
    v_delta=float((same_group_after[0].float()-same_group_before[0].float()).abs().max())
    sync();commit_seconds=time.perf_counter()-commit_start;commit_peak=(int(torch.cuda.max_memory_allocated()),int(torch.cuda.max_memory_reserved()))
    commit=dict(selected_row=chosen_row,selected_alpha=alpha,requires_non1=True,any_non1_selected=any(r['selected_alpha']!=1 for r in byrow),
        selected_state_hash=chosen['candidate_hash'],installed_weight_hash=installed_hash,installed_matches_candidate=commit_hash_match,
        all_unselected_rows_unchanged=torch.equal(torch.cat((target.weight[:chosen_row],target.weight[chosen_row+1:32])),
                                                   torch.cat((pristine[:chosen_row],pristine[chosen_row+1:32]))),
        same_group_v_input_max_abs=v_delta,next_native_group_o_input_max_abs=o_delta,
        same_group_v_input_unchanged=v_delta==0,next_group_activation_changed=o_delta>0,
        state_isolation_pass=isolation['pass'])
    commit['pass']=bool(commit_hash_match and commit['all_unselected_rows_unchanged'] and commit['state_isolation_pass'] and commit['next_group_activation_changed'] and commit['any_non1_selected'])
    dump(OUT/'M2_COMMIT_CORRECTNESS.json',commit)
    if not commit['pass']:raise RuntimeError('selected commit/self-prefix activation check failed; stop')
    # Timings: actual stage wall seconds; candidate per-row shares are explicitly attributed.
    row_slots=[t/32 for t in candidate_seconds for _ in range(32)]
    row_fullgrid=[sum(candidate_seconds)/32 for _ in range(32)]
    rows=[]
    for stage,seconds,units,peak in [('activation_capture',capture_seconds,2,capture_peak),('hessian_build_and_damp_inverse',h_seconds,1,h_peak),
            ('candidate_full_gptq_plus_loss',sum(candidate_seconds),320,candidate_peak),('selection',selection_seconds,32,selection_peak),
            ('selected_weight_commit_and_next_group_forward',commit_seconds,1,commit_peak)]:
        rows.append(dict(scope='stage',stage=stage,elapsed_seconds=seconds,work_units=units,
            mean_seconds='',median_seconds='',p95_seconds='',wall_clock_gpu_synchronized=True,
            peak_allocated_bytes=peak[0],peak_reserved_bytes=peak[1]))
    for scope,values in [('row_alpha_attributed',row_slots),('row_full_10_alpha_attributed',row_fullgrid),
                         ('32_row_block_per_alpha',candidate_seconds),('full_32_row_10_alpha_block',[sum(candidate_seconds)])]:
        summary=stats(values)
        rows.append(dict(scope=scope,stage='candidate_full_gptq_plus_loss',elapsed_seconds=sum(values),work_units=len(values),
            mean_seconds=summary['mean'],median_seconds=summary['median'],p95_seconds=summary['p95'],
            wall_clock_gpu_synchronized=True,peak_allocated_bytes=candidate_peak[0],peak_reserved_bytes=candidate_peak[1]))
    with (OUT/'M2_TIMING_SMOKE.csv').open('w',newline='',encoding='utf-8') as f:
        wri=csv.DictWriter(f,fieldnames=rows[0].keys());wri.writeheader();wri.writerows(rows)
    def stats(v):return dict(mean=statistics.mean(v),median=statistics.median(v),p95=sorted(v)[math.ceil(.95*len(v))-1])
    raw=dict(status='PASS',scope='L0 k_proj first32 output rows; 2x64 frozen calibration; exactly 320 row-alpha candidates',
        protocol_sha256=sha(V16/'PROTOCOL.json'),calibration_manifest_sha256=sha(V16/'CALIBRATION_MANIFEST.json'),
        checkpoint_revision=REV,module_shape=shape,docs=2,tokens_per_doc=64,output_row_limit=32,grid=GRID,
        group_order=order,activation_capture_seconds=capture_seconds,hessian_build_seconds=h_seconds,
        native_alpha1_parity=parity,candidate_seconds_by_alpha=candidate_seconds,candidate_elapsed_seconds=sum(candidate_seconds),
        candidate_logical_evaluations=len(results),candidate_alpha_distribution={str(a):sum(r['selected_alpha']==a for r in byrow) for a in GRID},
        selected_rows=byrow,candidate_row_alpha_attributed_seconds=stats(row_slots),row_full_grid_attributed_seconds=stats(row_fullgrid),
        row_block_32_seconds=sum(candidate_seconds),state_isolation=isolation,commit=commit,
        peak_allocated_bytes=max(v[0] for v in [capture_peak,h_peak,candidate_peak,selection_peak,commit_peak]),
        peak_reserved_bytes=max(v[1] for v in [capture_peak,h_peak,candidate_peak,selection_peak,commit_peak]),
        device=torch.cuda.get_device_name(0))
    dump(OUT/'M2_SMOKE_RAW.json',raw)
    print(json.dumps(dict(candidate_evaluations=len(results),candidate_seconds=sum(candidate_seconds),parity=parity,
        selection=raw['candidate_alpha_distribution'],commit=commit,grid_equivalence=grid_eq['pass'],
        activation_capture_seconds=capture_seconds,hessian_seconds=h_seconds,
        peak_allocated=raw['peak_allocated_bytes'],peak_reserved=raw['peak_reserved_bytes']),indent=2))

if __name__=='__main__':main()
