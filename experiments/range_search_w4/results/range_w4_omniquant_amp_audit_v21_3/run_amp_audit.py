"""Frozen-source audit and bounded disposable AMP diagnostics. No formal resume."""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
AUDIT12 = RESULTS/'range_w4_omniquant_nan_audit_v21_2'
UP = RESULTS/'range_w4_fullmodel_baseline_v16/official_source'
PARENT = RESULTS/'range_w4_omniquant_full_v21_1'
M3 = PARENT/'M3'
sys.path.insert(0,str(AUDIT12))
import audit_common as ac
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters, smooth_and_quant_temporary


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name,value):
    ac.write_json(HERE/name,value)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def upstream_scaler():
    spec=importlib.util.spec_from_file_location('v21_3_exact_upstream_utils',UP/'utils.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NativeScalerWithGradNormCount


def state_summary(opt):
    tensors={}
    for index,p in enumerate(p for group in opt.param_groups for p in group['params']):
        for key,value in opt.state.get(p,{}).items():
            tensors[f'{index}:{key}'] = ({'sha256':ac.sha_tensor(value),'stats':ac.stats(value)}
                                        if isinstance(value,torch.Tensor) else value)
    encoded=json.dumps(tensors,sort_keys=True,allow_nan=False).encode()
    return {'entries':len(opt.state),'sha256':hashlib.sha256(encoded).hexdigest(),
            'tensors':tensors,'all_finite':all(v['stats']['all_finite'] for v in tensors.values() if isinstance(v,dict))}


def observe_wrapper(wrapper,optimizer,owned):
    """Wrap calls only to read states; invoke original native methods exactly once."""
    event={}
    scaler=wrapper._scaler
    original_unscale,original_step,original_update=scaler.unscale_,scaler.step,scaler.update
    def grad_stats():
        return {name:ac.stats(p.grad) for name,p in owned}
    def unscale(opt):
        event['scaled_gradients']=grad_stats()
        result=original_unscale(opt)
        event['unscaled_gradients']=grad_stats()
        state=scaler._per_optimizer_states[id(opt)]
        event['found_inf_per_device']={str(device):float(v.item()) for device,v in state['found_inf_per_device'].items()}
        event['found_inf']=sum(event['found_inf_per_device'].values())
        event['stage_after_unscale']=str(state['stage'])
        return result
    def step(opt,*args,**kwargs):
        result=original_step(opt,*args,**kwargs)
        event['stage_after_step']=str(scaler._per_optimizer_states[id(opt)]['stage'])
        return result
    def update(*args,**kwargs):
        result=original_update(*args,**kwargs)
        event['optimizer_tracking_cleared_after_update']=len(scaler._per_optimizer_states)==0
        return result
    scaler.unscale_,scaler.step,scaler.update=unscale,step,update
    def restore():
        scaler.unscale_,scaler.step,scaler.update=original_unscale,original_step,original_update
    return event,restore


def execute_step(layer,args,fixture,sample,optimizer,wrapper):
    owned=[(name,p) for name,p in layer.named_parameters() if 'bound_factor' in name]
    opt_ids={id(p) for group in optimizer.param_groups for p in group['params']}
    assert opt_ids=={id(p) for _,p in owned}
    before={name:p.detach().clone() for name,p in owned}
    state_before=state_summary(optimizer)
    scale_before=wrapper.state_dict()
    calls=[]
    handle=optimizer.register_step_pre_hook(lambda *_:calls.append('optimizer.step'))
    inp=ac.tree(fixture['inputs'][sample:sample+1],ac.DEVICE)
    target=ac.tree(fixture['targets'][sample:sample+1],ac.DEVICE)
    kwargs=ac.tree(fixture['kwargs'][sample],ac.DEVICE)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.cuda.amp.autocast():
        smooth_and_quant_temporary(layer,args,False)
        output=layer(inp,**kwargs)[0]
        loss=torch.nn.MSELoss()(target,output)
    if not math.isfinite(float(loss.detach())):
        raise FloatingPointError('diagnostic loss is nonfinite; stop without an optimizer call')
    optimizer.zero_grad()
    event,restore=observe_wrapper(wrapper,optimizer,owned)
    try:
        norm=wrapper(loss,optimizer,parameters=get_omni_parameters(layer,False))
    finally:
        restore();handle.remove()
    torch.cuda.synchronize()
    changed=[name for name,p in owned if not torch.equal(p.detach(),before[name])]
    state_after=state_summary(optimizer)
    post_finite=all(bool(torch.isfinite(p.detach()).all()) for p in layer.parameters())
    record={'layer':21,'epoch':0,'sample':sample,'loss':float(loss.detach()),
            'loss_finite':True,'forward_output':ac.stats(output),'scale_before':float(scale_before['scale']),
            'scaler_state_before':scale_before,'scaler_state_after':wrapper.state_dict(),
            'scale_after':float(wrapper.state_dict()['scale']),'optimizer_step_executed':bool(calls),
            'optimizer_step_call_count':len(calls),'optimizer_step_skipped':not bool(calls),
            'optimizer_state_before':state_before,'optimizer_state_after':state_after,
            'optimizer_state_changed':state_before['sha256']!=state_after['sha256'],
            'LWC_parameters_changed':bool(changed),'changed_parameter_names':changed,
            'LWC_finite':all(bool(torch.isfinite(p.detach()).all()) for _,p in owned),
            'all_named_parameters_finite':post_finite,
            'norm':float(norm) if bool(torch.isfinite(norm)) else None,
            'norm_finite':bool(torch.isfinite(norm)),**event,
            'wall_seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'formal_committed_updates':0,
            'disposable_optimizer_updates':len(calls)}
    assert record['LWC_finite'] and post_finite and state_after['all_finite']
    assert math.isfinite(record['scale_after'])
    assert (record['found_inf']>0)==record['optimizer_step_skipped']
    if record['found_inf']>0:
        assert not changed and not record['optimizer_state_changed']
    return record


def micro_test(NativeScaler):
    p=torch.nn.Parameter(torch.ones((1,1),device='cuda',dtype=torch.float32))
    optimizer=torch.optim.AdamW([p],lr=0.01,weight_decay=0.0)
    wrapper=NativeScaler()
    records=[]
    for index,value in enumerate((512.0,0.001)):
        optimizer.zero_grad()
        before=p.detach().clone();state_before=state_summary(optimizer)
        scale_before=wrapper.state_dict()
        calls=[];handle=optimizer.register_step_pre_hook(lambda *_:calls.append(1))
        with torch.cuda.amp.autocast():
            output=torch.nn.functional.linear(torch.full((1,1),value,device='cuda'),p)
            loss=output.float().square().mean()
        event,restore=observe_wrapper(wrapper,optimizer,[('weight',p)])
        try:
            norm=wrapper(loss,optimizer,parameters=[p])
        finally:
            restore();handle.remove()
        record={'iteration':index,'input_value':value,'loss':float(loss.detach()),
                'scale_before':float(scale_before['scale']),'scale_after':wrapper.state_dict()['scale'],
                'optimizer_step_executed':bool(calls),'parameter_changed':not torch.equal(before,p.detach()),
                'parameter_finite':bool(torch.isfinite(p).all()),
                'optimizer_state_before':state_before,'optimizer_state_after':state_summary(optimizer),**event}
        records.append(record)
    assert records[0]['found_inf']>0 and not records[0]['optimizer_step_executed']
    assert not records[0]['parameter_changed'] and records[0]['scale_after']==records[0]['scale_before']*0.5
    assert records[1]['found_inf']==0 and records[1]['optimizer_step_executed'] and records[1]['parameter_changed']
    assert all(r['parameter_finite'] for r in records)
    return {'status':'PASS','torch_version':torch.__version__,'device':torch.cuda.get_device_name(0),
            'implementation':'exact frozen upstream NativeScalerWithGradNormCount; default dynamic GradScaler',
            'iterations':records,'purpose':'independent finite-loss overflow -> skip/backoff -> next finite update',
            'manually_set_loss_scale':False,'formal_committed_updates':0}


def history_audit():
    rows=list(csv.DictReader((M3/'logs/training_steps.csv').open(encoding='utf-8',newline='')))
    stop=load(M3/'STOP_DIAGNOSTIC.json');attempt=stop['attempt_id']
    kept=[r for r in rows if r['attempt_id']==attempt and int(r['layer'])<=20]
    order=[(int(r['layer']),int(r['epoch_index']),int(r['sample_index'])) for r in kept]
    expected=[(layer,epoch,sample) for layer in range(21) for epoch in range(20) for sample in range(128)]
    assert order==expected
    assert all(r['lwc_grad_finite']=='True' and int(r['lwc_grad_param_count'])==14 and math.isfinite(float(r['loss'])) for r in kept)
    epochs=list(csv.DictReader((M3/'logs/training_epochs.csv').open(encoding='utf-8',newline='')))
    kept_epochs=[r for r in epochs if r['attempt_id']==attempt]
    assert len(kept_epochs)==420 and all(math.isfinite(float(r['norm_mean'])) and r['lwc_finite']=='True' for r in kept_epochs)
    return {'status':'PASS','checkpoint_attempt_id':attempt,'committed_prefix_iterations':len(kept),
            'inferred_successful_optimizer_steps':len(kept),'inference_basis':'all 14 optimizer-owned LWC gradients finite AFTER native scaler on every prefix row; optimizer has only LWC (LET off), native scaler steps when found_inf is zero',
            'historical_optimizer_step_hooks_or_found_inf_saved':False,
            'ordering_unique_complete':True,'prefix_nonfinite_LWC_gradient_rows':0,
            'finite_epoch_norm_rows':len(kept_epochs),'all_attempt_rows':dict(Counter(r['attempt_id'] for r in rows)),
            'excluded_failed_attempt_iterations':2560,'excluded_failed_L21_iterations':1,
            'manifest_count_formula':'completed_layers * epochs * samples, not an actual step counter',
            '61440_definition':'24*20*128 scheduled iterations; equal successful steps only when no skips',
            'no_missing_sample_retries_added':True}


def source_audit():
    manifest=load(RESULTS/'range_w4_fullmodel_baseline_v16/OFFICIAL_SOURCE_SHA256.json')
    protected={str(p):sha(p) for p in AUDIT12.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    protected.update(ac.protected_hashes())
    for relative,expected in manifest.items():
        p=UP/relative
        assert sha(p)==expected,relative
        protected[str(p)]=expected
    import torch.amp.grad_scaler as gs
    paths=[UP/'quantize/omniquant.py',UP/'utils.py',UP/'main.py',M3/'run_formal_m3.py',
           M3/'official_port/quantize/omniquant.py',M3/'official_port/utils.py',
           Path(inspect.getsourcefile(gs)),Path(inspect.getsourcefile(torch.cuda.amp.GradScaler))]
    write('SOURCE_SHA256.json',{str(p):sha(p) for p in paths})
    write('PROTECTED_HASHES.json',protected)
    assert sha(UP/'utils.py')==sha(M3/'official_port/utils.py')
    write('SOURCE_SECTIONS.json',{
        'official_commit':'feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14',
        'upstream_case':'A: dynamic GradScaler',
        'verified_upstream_manifest_files':len(manifest),
        'upstream_training_loop':{'file':str(UP/'quantize/omniquant.py'),'function':'omniquant','lines':[243,274]},
        'upstream_wrapper':{'file':str(UP/'utils.py'),'class':'NativeScalerWithGradNormCount','lines':[27,53]},
        'runner_callback_placement':{'file':str(M3/'official_port/quantize/omniquant.py'),'lines':[397,403],
                                     'STOP_is_after_native_scaler_step_and_update':True},
        'runner_gradient_gate':{'file':str(M3/'run_formal_m3.py'),'lines':[424,442]},
        'other_overstrict_gates':{'epoch_norm_finite':[414,422],'final_all_finite_grad_count':[573,580]},
        'installed_torch_behavior':'local torch.amp.grad_scaler.py source + runtime micro-test; not inferred from tutorials'})


def main():
    if (HERE/'OFFICIAL_OVERFLOW_HANDLING_REPLAY.json').exists():
        raise RuntimeError('bounded diagnostic already cached; do not overwrite or rerun')
    HERE.mkdir(parents=True,exist_ok=True)
    source_audit()
    assert load(AUDIT12/'correctness_checks.json')['status']=='PASS'
    config,lock,tokens,checkpoint_path,resume=ac.context()
    fixture_path=AUDIT12/'fixtures/actual_l21.pt'
    fixture_manifest=load(AUDIT12/'FIXTURE_MANIFEST.json')
    expected=next(x['sha256'] for x in fixture_manifest['files'] if x['path'].replace('\\','/')=='fixtures/actual_l21.pt')
    assert sha(fixture_path)==expected
    fixture=torch.load(fixture_path,map_location='cpu',weights_only=False)
    assert fixture['layer']==21
    write('PROTOCOL.json',{'name':'v21.3 Official AMP / GradScaler Semantics Audit',
        'official_commit':lock['official_commit'],'model_revision':lock['model_revision'],
        'checkpoint':str(checkpoint_path),'checkpoint_sha256':sha(checkpoint_path),
        'fixture':str(fixture_path),'fixture_sha256':expected,
        'fixture_provenance':'v21.2 native resume reconstruction from verified block20, captured L21 optimizer-init boundary; reused without redoing completed prefix',
        'calibration_manifest_sha256':lock['calibration_manifest_sha256'],
        'native_scaler_source':str(UP/'utils.py'),'default_scale_is_dynamic':True,
        'single_step_boundary':'L21/epoch0/sample0 through upstream scaler.step/update, then stop',
        'conditional_sequence_max_samples':4,'conditional_sequence_samples':[0,1,2,3],
        'sequence_preserves_optimizer_scaler_parameters_between_samples':True,
        'all_updates_disposable':True,'formal_committed_updates':0,
        'precision':'formal autocast FP16, FP32 LWC; no forced FP32 or manually fixed scale',
        'forbidden':['formal resume','full epoch','LR/optimizer/quantizer change','sample retry','export','WT2/C4','T4'],
        'new_writes':str(HERE)})
    NativeScaler=upstream_scaler()
    write('GRADSCALER_BEHAVIOR_CHECK.json',micro_test(NativeScaler))
    write('HISTORICAL_UPDATE_COUNT_AUDIT.json',history_audit())
    layer,args=ac.new_layer(fixture)
    optimizer=torch.optim.AdamW([{'params':let_parameters(layer,False),'lr':args.let_lr},
                                {'params':lwc_parameters(layer),'lr':args.lwc_lr}],weight_decay=args.wd)
    wrapper=NativeScaler()
    record=execute_step(layer,args,fixture,0,optimizer,wrapper)
    assert record['loss']==0.15326565504074097
    recoverable=(record['found_inf']>0 and record['optimizer_step_skipped'] and
                 record['scale_after']==record['scale_before']*0.5 and not record['LWC_parameters_changed'] and
                 not record['optimizer_state_changed'] and record['LWC_finite'])
    write('OFFICIAL_OVERFLOW_HANDLING_REPLAY.json',{'status':'PASS',
        'classification':'RECOVERABLE_DYNAMIC_LOSS_SCALE_OVERFLOW' if recoverable else 'FATAL_UNDER_OFFICIAL_AMP_SEMANTICS',
        'diagnostic_only':True,'native_wrapper_sha256':sha(UP/'utils.py'),
        'record':record,'sample1_entered':False,'formal_committed_updates':0})
    single_record=record
    print('SINGLE',record['loss'],record['found_inf'],record['scale_before'],record['scale_after'],flush=True)
    del layer,optimizer,wrapper;gc.collect();torch.cuda.empty_cache()
    if not recoverable:
        raise RuntimeError('official single-step is not recoverable; no sequence permitted')
    layer,args=ac.new_layer(fixture)
    optimizer=torch.optim.AdamW([{'params':let_parameters(layer,False),'lr':args.let_lr},
                                {'params':lwc_parameters(layer),'lr':args.lwc_lr}],weight_decay=args.wd)
    wrapper=NativeScaler();sequence=[]
    for sample in range(4):
        record=execute_step(layer,args,fixture,sample,optimizer,wrapper)
        sequence.append(record)
        write(f'SEQUENCE_SAMPLE_{sample}.json',record)
        print('SEQUENCE',sample,record['loss'],record['scale_before'],record['found_inf'],record['scale_after'],record['LWC_parameters_changed'],flush=True)
    fields=['sample','loss','scale_before','found_inf','optimizer_step_skipped','scale_after','LWC_finite',
            'LWC_parameters_changed','optimizer_state_changed','optimizer_step_call_count','formal_committed_updates']
    with (HERE/'DYNAMIC_SCALE_RECOVERY_TRACE.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in sequence)
    write('DYNAMIC_SCALE_RECOVERY_SUMMARY.json',{'status':'PASS','samples':4,
        'overflow_events':sum(r['found_inf']>0 for r in sequence),
        'disposable_successful_optimizer_steps':sum(r['optimizer_step_call_count'] for r in sequence),
        'subsequent_finite_update_observed':any(r['sample']>0 and r['found_inf']==0 and r['LWC_parameters_changed'] for r in sequence),
        'formal_committed_updates':0,'full_epoch_executed':False})
    protected=load(HERE/'PROTECTED_HASHES.json')
    assert all(sha(Path(path))==expected for path,expected in protected.items())
    write('PROTECTED_ARTIFACT_CHECK.json',{'status':'PASS','files':len(protected),'all_hashes_unchanged':True})
    write('AUDIT_COST.json',{'single_sample_replays':1,'continuous_sequence_samples':4,'micro_test_iterations':2,
        'sum_layer_replay_seconds':load(HERE/'OFFICIAL_OVERFLOW_HANDLING_REPLAY.json')['record']['wall_seconds']+sum(r['wall_seconds'] for r in sequence),
        'max_layer_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in [single_record]+sequence),
        'max_layer_peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in [single_record]+sequence),
        'formal_committed_updates':0,'cost_scope':'instrumented layer replays only; excludes loading, CPU source/hash checks and IO; not a calibration speed benchmark'})
    print('BOUNDED DIAGNOSTIC COMPLETE; NO FORMAL RESUME',flush=True)


if __name__=='__main__':
    main()
