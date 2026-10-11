"""Bounded full-matrix mixed-precision Gram comparison; no model commits/PPL."""
import time
import csv,gc,hashlib,importlib.util,json,math,os,sys,threading,traceback
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[2]
V46=BASE/'v46_m1_single_integer_repair_fullmodel';V48=BASE/'v48_fast_numerical_acceptance';V49=BASE/'v49_fullmodel_fast_integer_repair'
CAP=1200;SHADOW_ROWS=[0,224,448,672];GPU_START=None;RESULTS=[];SHADOW=[];COMPARE=[]
def dump(n,v):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def hashX(X):return hashlib.sha256(memoryview(X)).hexdigest()
def elapsed():return 0 if GPU_START is None else time.time()-GPU_START
def deadline(reserve=0):
    if GPU_START is not None and elapsed()>=CAP-reserve:raise TimeoutError('v50 cumulative 1200-second cap')
def loadmod(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(V46));sys.path.insert(0,str(V48))
rt=loadmod('run_v42',V46/'runtime_v46.py');rt.OUT=OUT;rt.MODE='v50_preparation';rt.LAYER=0;rt.deadline=deadline
old=loadmod('repair_control',V46/'repair_v45.py');pc=sys.modules['precision_check']
base=loadmod('v48_baseline_v50',V48/'fast_accept.py');mg=loadmod('mixed_gram_v50',OUT/'mixed_gram.py')

def grid(o,c,W,device='cuda'):
    sc=o['scales'].to(device)[:,o['g_idx'].long().to(device)]
    q=(sc*(c.to(device).float()-8)).half().float()
    return q.double()-W.to(device).double(),sc

def cpu_tests():
    original=rt.timer;rt.timer=lambda s:nullcontext();checks=[]
    try:
        engine=loadmod('v48_toy_v50',V48/'fast_accept.py')
        X=torch.eye(2,dtype=torch.float64);C=X.T@X/2;h=torch.ones(2,dtype=torch.float64)
        stats=dict(h=h,rho=1.0);mg.attach(engine,stats,pc)
        E=torch.tensor([[2.,0.],[0.,0.]],dtype=torch.float64);z=torch.zeros(2,dtype=torch.long)
        code=torch.full((2,2),8,dtype=torch.long);SC=torch.ones((2,2));p=E@C
        st=dict(code=code,Q=torch.zeros((2,2)),E=E,p=p,B=E.abs()@C.abs(),loss=(E*p).sum(1),updates=z.clone(),evaluations=z.clone(),status=z.clone(),rejections=z.clone(),blocked=torch.zeros((2,2,2),dtype=torch.bool))
        before={k:v.clone() for k,v in st.items()};rr=torch.tensor([0,1]);col=rr*0;off=torch.tensor([-1,1]);dv=off.double()
        pred=2*dv*p[rr,col]+dv.square()*C.diag()[col];pr=old.propose(st,rr,col,off,SC,pred)
        a,b,_,_,ck=engine.settle(st,pr,C,SC,X)
        assert a.tolist()==[True,False] and len(ck)==2
        assert all(torch.equal(st[k][1],before[k][1]) for k in ['code','E','p','loss','updates'])
        assert st['blocked'][1,0,1]
        checks+=['raw-X branch accepts genuine descent and rejects positive increment','rejection state/blacklist restored']
        Ct=C.clone();Ct[0,0]+=.01;Ee=E[:1];D=torch.tensor([[-1.,0.]],dtype=torch.float64);pp=Ee@Ct;pd=(2*D*pp).sum(1)+(D*(D@Ct)).sum(1)
        r=engine.verify(X,Ee,D,Ct,pp,pd);assert r['direct64'][0]<-r['allowed_error'][0]
        checks+=['mixed-Gram discrepancy distinct from raw sign tolerance']
        for which in ['cache','formula']:
            try:
                engine.verify(X,Ee,D,Ct,pp+.01 if which=='cache' else pp,pd+.01 if which=='formula' else pd)
            except ArithmeticError:checks.append(which+' corruption detected')
            else:raise AssertionError('corruption undetected')
        old.retire_caps(st,256,32);st['updates'][0]=32;old.retire_caps(st,256,32);assert st['status'][0]==2
        st['evaluations'][1]=256;old.retire_caps(st,256,32);assert st['status'][1]==3
        checks+=['original 32-accept and candidate-cap semantics']
        xx=torch.tensor([[((i*17+j*11)%29-14)/32 for j in range(7)] for i in range(128)],dtype=torch.float16)
        a32=(xx.float().T@xx.float()).double()/len(xx);a64=xx.double().T@xx.double()/len(xx);hh=xx.double().square().mean(0).sqrt();rho=mg.gamma32(len(xx))+8*pc.gamma(len(xx)+9)
        assert bool(((a32-a64).abs()<=rho*hh[:,None]*hh[None,:]+1e-15).all())
        checks+=['fixed constructed mixed Gram vs raw FP64 reference']
        assert not torch.cuda.is_initialized()
        dump('CPU_TESTS.json',dict(pass_all=True,checks=checks,no_GPU=True))
    finally:rt.timer=original

def instrument(engine,stats):
    original=engine.settle;records=[]
    def settle(st,pr,C,SC,X):
        rr,col,off=[pr[k] for k in ['rows','col','off']]
        if stats is None:ug=torch.zeros_like(pr['pred'])
        else:
            w=(st['E'][rr].abs()*stats['h']).sum(1);dv=pr['delta']
            ug=stats['rho']*(2*dv.abs()*stats['h'][col]*w+dv.square()*stats['h'][col].square())
        out=original(st,pr,C,SC,X);accepted,rejected,used,U,checks=out
        records.append(dict(attempt=len(records)+1,rows=rr.cpu(),columns=col.cpu(),offsets=off.cpu(),prediction=pr['pred'].cpu(),used_prediction=used.cpu(),accepted=accepted.cpu(),total_uncertainty=U.cpu(),gram_uncertainty=ug.cpu()))
        return out
    engine.settle=settle
    return records

def terminal(X,C,W,o,result,stats):
    with rt.timer('independent_all_row_terminal_FP64'):
        e0,_=grid(o,o['codes'],W);ef,_=grid(o,result['codes'],W);E=torch.cat([e0,ef]);r=len(e0)
        s=torch.zeros(2*r,device='cuda',dtype=torch.float64)
        for lo in range(0,len(X),4096):
            deadline();x=torch.from_numpy(np.array(X[lo:lo+4096],copy=True)).cuda().double();y=x@E.T;s+=y.square().sum(0)
        raw=s/len(X);quad=(E*(E@C)).sum(1);op=(E.abs()*(E.abs()@C.abs())).sum(1)
        oldtol=64*pc.gamma(len(X)+C.shape[0]+math.ceil(len(X)/4096)+32)*op+64*2**-53*raw
        gram=torch.zeros_like(raw) if stats is None else stats['rho']*(E.abs()*stats['h']).sum(1).square()
        tol=oldtol+gram;err=(raw-quad).abs()
        assert bool(torch.isfinite(raw).all() and torch.isfinite(quad).all() and (err<=tol).all()),'terminal Gram/raw-X discrepancy'
        assert bool((raw[r:]<=raw[:r]).all()),'confirmed all-row terminal objective ascent'
        assert bool(((result['final_mse'].cuda()-quad[r:]).abs()<=oldtol[r:]).all()),'same-C loss bookkeeping error'
        return dict(initial=raw[:r].cpu(),final=raw[r:].cpu(),coverage_rows=r,max_error=float(err.max()),max_error_ratio=float((err/tol.clamp_min(1e-300)).max()),max_gram_allowance=float(gram.max()))

def shadow(X,C,W,o,result,engine,stats,path,module):
    code=o['codes'].long().clone();es=[];ds=[];ps=[];keys=[]
    for mv in result['moves']:
        for row in SHADOW_ROWS:
            ix=(mv['rows']==row).nonzero().flatten()
            if len(ix):
                k=int(ix[0]);col=int(mv['columns'][k]);off=int(mv['offsets'][k]);sc=o['scales'][row,o['g_idx'].long()]
                q=(sc*(code[row].float()-8)).half().float();e=q.double()-W[row].double();dv=(sc[col]*(code[row,col].float()+off-8)).half().float()-q[col]
                D=torch.zeros_like(e);D[col]=dv.double();es.append(e);ds.append(D);ps.append(mv['predicted_delta'][k]);keys.append((row,mv['round'],col,off))
        code[mv['rows'],mv['columns']]+=mv['offsets']
    assert torch.equal(code.byte(),result['codes']), 'accepted trajectory reconstructs final code'
    with rt.timer('validation_shadow_complete_sampled_trajectories'):
        E=torch.stack(es).cuda();D=torch.stack(ds).cuda();pred=torch.stack(ps).cuda()
        check=engine.verify(X,E,D,C,E@C,pred)
        assert bool((check['direct64'] < -check['allowed_error']).all()),'sampled update raw-X not certified negative by fixed empirical criterion'
    return [dict(path=path,module=module,row=k[0],attempt=k[1],coordinate=k[2],offset=k[3],prediction=float(pred[j]),raw_FP64_increment=float(check['direct64'][j]),absolute_error=float(check['absolute_error'][j]),raw_rounding_allowance=float(check['allowed_error'][j]),gram_allowance=float(check.get('gram_allowance',torch.zeros_like(pred))[j]),strict_raw_descent=True) for j,k in enumerate(keys)]

def proposal_comparison(a,b,module):
    def flatten(records):
        out={}
        for r in records:
            for j,row in enumerate(r['rows'].tolist()):out[(r['attempt'],row)]=(int(r['columns'][j]),int(r['offsets'][j]),bool(r['accepted'][j]),float(r['prediction'][j]))
        return out
    aa=flatten(a);bb=flatten(b);common=aa.keys()&bb.keys();diff=[]
    for key in sorted(common):
        x,y=aa[key],bb[key]
        if x[:3]!=y[:3]:diff.append(dict(module=module,attempt=key[0],row=key[1],baseline_coordinate=x[0],mixed_coordinate=y[0],baseline_offset=x[1],mixed_offset=y[1],baseline_accept=x[2],mixed_accept=y[2],baseline_prediction=x[3],mixed_prediction=y[3]))
    rt.csvwrite(f'reports/SELECTED_PROPOSAL_DIFFERENCES_{module.replace(".","_")}.csv',diff)
    return dict(baseline_proposals=len(aa),mixed_proposals=len(bb),common_attempt_row_proposals=len(common),different_selected_coordinates_or_offsets=sum(aa[k][:2]!=bb[k][:2] for k in common),different_acceptance_flags=sum(aa[k][2]!=bb[k][2] for k in common),baseline_only_attempt_row_proposals=len(aa.keys()-bb.keys()),mixed_only_attempt_row_proposals=len(bb.keys()-aa.keys()),scope='selected proposals at matching attempt/row; not every candidate score')

def C_validation(C,reference,stats,module):
    with rt.timer('validation_reference_C_error'):
        ref=reference.cuda();err=C-ref;h=stats['h'];maximum=0.;ratio=0.
        for lo in range(0,len(C),256):
            bound=stats['rho']*h[lo:lo+256,None]*h[None,:]
            maximum=max(maximum,float(bound.max()));ratio=max(ratio,float((err[lo:lo+256].abs()/bound.clamp_min(1e-300)).max()))
        assert ratio<=1,'mixed Gram discrepancy exceeds frozen assumption-based envelope'
        result=dict(module=module,max_abs=float(err.abs().max()),max_diag_abs=float(err.diag().abs().max()),fro_abs=float(torch.linalg.vector_norm(err)),relative_fro=float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(ref)),max_error_over_gram_proxy=ratio,max_proxy_bound=maximum,rho=stats['rho'],zero_variance_columns=stats['zero_variance_columns'].tolist())
    return result

def main():
    global GPU_START
    assert not (OUT/'GPU_START.json').exists(),'refuse silent budget reset or second timed run'
    protocol=json.loads((OUT/'PROTOCOL.json').read_text());assert sha(OUT/'run_v50.py')==protocol['runner_sha256'] and sha(OUT/'mixed_gram.py')==protocol['new_path_sha256'] and sha(OUT/'NUMERICAL_RULES.md')==protocol['rules_sha256']
    assert sha(V48/'fast_accept.py')==protocol['baseline_sha256']
    cpu_tests();GPU_START=time.time();rt.WALL_START=GPU_START;rt.START=time.perf_counter()
    dump('GPU_START.json',dict(start_epoch=GPU_START,cap_seconds=CAP,pid=os.getpid(),protocol_sha256=sha(OUT/'PROTOCOL.json'),rules_sha256=sha(OUT/'NUMERICAL_RULES.md'),runner_sha256=sha(OUT/'run_v50.py'),requested_model='Sol high; not verifiable through interface'))
    def hard_stop():dump('STATUS.json',dict(status='HARD_CAP',category='D',elapsed_seconds=elapsed(),completed=RESULTS));os._exit(124)
    watchdog=threading.Timer(CAP,hard_stop);watchdog.daemon=True;watchdog.start()
    try:
      with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
        with rt.timer('warmup'):z=torch.zeros((64,64),device='cuda');z@z
        capture_peak=0;inputs={};source_inputs=[]
        with rt.timer('locked_v49_model_load'):model=AutoModelForCausalLM.from_pretrained(str(V49/'R1/checkpoint/final'),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()
        hidden=rt.embed(model)
        for group,module in [('up_gate','mlp.up_proj'),('down','mlp.down_proj')]:
            rt.MODULE=module;X=rt.capture(model,0,hidden,group);xh=hashX(X);ck=torch.load(V49/f'formal/groups/layer_00_{group}.pt',map_location='cpu',weights_only=False)
            assert xh==ck['input_sha256'] and ck['committed'] and ck['protocol_sha256']==sha(V49/'PROTOCOL.json')
            with rt.timer('input_cache_IO'):p=OUT/'inputs'/f'{group}.npy';np.save(p,X)
            inputs[module]=p;source_inputs.append(dict(module=module,group=group,input_sha256=xh,N=len(X),d=X.shape[1],v49_C_sha256=ck['C_sha256']))
            del X,ck;gc.collect()
        capture_peak=torch.cuda.max_memory_allocated();del model,hidden,z;gc.collect();torch.cuda.empty_cache()
        dump('INPUT_SOURCE_CHECKS.json',dict(pass_all=True,locked_model_sha256=json.loads((V49/'CHECKPOINT_LOCK.json').read_text())['final_model_sha256'],inputs=source_inputs,TF32=torch.backends.cuda.matmul.allow_tf32,autocast=torch.is_autocast_enabled()))
        original=torch.load(V49/'original_codes/layer_00.pt',map_location='cpu',weights_only=True);gram_errors=[];max_alloc=capture_peak;max_reserved=0
        for item in source_inputs:
            module=item['module'];X=np.load(inputs[module],mmap_mode='r');o=original[module]
            with safe_open(str(rt.FP/'model.safetensors'),framework='pt',device='cpu') as f:W=f.get_tensor('model.layers.0.'+module+'.weight').half()
            with safe_open(str(rt.V18/'M1/checkpoint/final/model.safetensors'),framework='pt',device='cpu') as f:m1w=f.get_tensor('model.layers.0.'+module+'.weight')
            assert torch.equal(m1w.view(torch.int16),o['weight'].view(torch.int16));assert torch.equal((o['scales'][:,o['g_idx'].long()]*(o['codes'].float()-8)).half().view(torch.int16),m1w.view(torch.int16))
            assert (o['codes']<=15).all() and (o['zeros']==8).all() and torch.equal(o['g_idx'].long(),torch.arange(X.shape[1])//128)
            assert torch.equal(rt.unpack(rt.pack(o['codes'])),o['codes'])
            original_hashes={k:rt.th(o[k]) for k in ['codes','scales','zeros','g_idx']};wh=rt.th(W);pair={};records={};reference=None
            for name in ['baseline','mixed']:
                deadline(60);rt.MODE=name;rt.MODULE=module;torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();total_t=time.perf_counter()
                c_t=time.perf_counter()
                if name=='baseline':C=rt.build_C(X);stats=None;engine=loadmod('v48_baseline_v50_'+module.replace('.','_'),V48/'fast_accept.py')
                else:C,stats=mg.build(X,rt,pc);engine=mg.attach(loadmod('v48_mixed_v50_'+module.replace('.','_'),V48/'fast_accept.py'),stats,pc)
                torch.cuda.synchronize();c_seconds=time.perf_counter()-c_t;ch=rt.th(C.cpu());records[name]=instrument(engine,stats)
                if name=='baseline':assert ch==item['v49_C_sha256']
                r_t=time.perf_counter();res=engine.batch_repair(o['codes'],o['scales'],o['g_idx'],W,X,C,module);torch.cuda.synchronize();repair_seconds=time.perf_counter()-r_t
                t_t=time.perf_counter();ter=terminal(X,C,W,o,res,stats);torch.cuda.synchronize();terminal_seconds=time.perf_counter()-t_t
                pipeline=time.perf_counter()-total_t;peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();max_alloc=max(max_alloc,peak);max_reserved=max(max_reserved,reserved)
                assert all(torch.equal(res[k],o[k]) for k in ['scales','zeros','g_idx']) and torch.equal(rt.unpack(rt.pack(res['codes'])),res['codes'])
                assert (res['updates']<=32).all() and (res['evaluations']<=128*C.shape[0]).all()
                if name=='baseline':
                    ck=torch.load(V49/f'formal/groups/layer_00_{item["group"]}.pt',map_location='cpu',weights_only=False)['modules'][module]
                    assert torch.equal(res['codes'],ck['codes']) and torch.equal(res['evaluations'],ck['evaluations']),'baseline v49 replay mismatch'
                    del ck
                stat=dict(path=name,module=module,rows=len(res['codes']),d=C.shape[0],C_seconds=c_seconds,repair_seconds=repair_seconds,terminal_seconds=terminal_seconds,pipeline_seconds=pipeline,accepted_updates=int(res['updates'].sum()),candidate_evaluations=int(res['evaluations'].sum()),rejected_candidates=int(res['rejections'].sum()),rechecks=len(res['recheck_events']),initial_raw64_sum=float(ter['initial'].sum()),final_raw64_sum=float(ter['final'].sum()),terminal_max_error=ter['max_error'],terminal_max_error_ratio=ter['max_error_ratio'],max_p_drift=res['max_p_drift'],peak_allocated_bytes=peak,peak_reserved_bytes=reserved,C_sha256=ch,zero_variance_columns=[] if stats is None else stats['zero_variance_columns'].tolist())
                RESULTS.append(stat);dump('PARTIAL_RESULTS.json',RESULTS);rt.csvwrite('MATRIX_TIMINGS.csv',RESULTS)
                with rt.timer('local_evidence_IO'):
                    torch.save(res,OUT/'local_states'/f'LOCAL_{name}_{module.replace(".","_")}.pt');torch.save(records[name],OUT/'local_states'/f'PROPOSALS_{name}_{module.replace(".","_")}.pt')
                SHADOW.extend(shadow(X,C,W,o,res,engine,stats,name,module));rt.csvwrite('SHADOW_TRAJECTORIES.csv',SHADOW)
                pair[name]=(res,ter,stat)
                if name=='baseline':reference=C.cpu()
                else:gram_errors.append(C_validation(C,reference,stats,module));dump('C_ERROR.json',gram_errors)
                assert rt.th(C.cpu())==ch and rt.th(W)==wh and original_hashes=={k:rt.th(o[k]) for k in original_hashes}
                del C,stats,engine;gc.collect();torch.cuda.empty_cache()
            a,ta,sa=pair['baseline'];b,tb,sb=pair['mixed'];pcmp=proposal_comparison(records['baseline'],records['mixed'],module)
            cmp=dict(module=module,different_codes=int((a['codes']!=b['codes']).sum()),different_rows=int((a['codes']!=b['codes']).any(1).sum()),different_accept_counts_rows=int((a['updates']!=b['updates']).sum()),different_eval_counts_rows=int((a['evaluations']!=b['evaluations']).sum()),mixed_minus_baseline_raw64_final_sum=float((tb['final']-ta['final']).sum()),worst_row_mixed_minus_baseline=float((tb['final']-ta['final']).max()),C_speedup=sa['C_seconds']/sb['C_seconds'],pipeline_speedup=sa['pipeline_seconds']/sb['pipeline_seconds'],**pcmp)
            COMPARE.append(cmp);dump('COMPARISON.json',COMPARE)
            rr=[]
            for path,(res,ter,stat) in pair.items():
                rr.extend(dict(path=path,module=module,row=i,updates=int(res['updates'][i]),evaluations=int(res['evaluations'][i]),rejections=int(res['rejections'][i]),status=int(res['status'][i]),initial_raw64_mse=float(ter['initial'][i]),final_raw64_mse=float(ter['final'][i])) for i in range(len(res['codes'])))
            rt.csvwrite(f'ROWS_{module.replace(".","_")}.csv',rr)
            assert hashX(X)==item['input_sha256'],'X pollution'
            del X,W,m1w,o,pair,a,b,ta,tb,records,reference,res;gc.collect();torch.cuda.empty_cache()
        sources=json.loads((OUT/'SOURCE_AUDIT.json').read_text());assert all(sha(x['path'])==x['sha256'] for x in sources),'historical source changed'
        dump('SOURCE_IMMUTABILITY.json',dict(pass_all=True,files=sources))
        dump('STATUS.json',dict(status='COMPLETE',category='A' if all(c['pipeline_speedup']>1 for c in COMPARE) else 'B',independent_path=True,numerical_checks_pass=True,elapsed_seconds=elapsed(),cap_seconds=CAP,peak_allocated_bytes=max_alloc,peak_reserved_bytes=max_reserved,shadow_accepted_updates=len(SHADOW),full_terminal_rows_per_path=5760,model_committed=False,model_checkpoint=False,held=False,PPL=False))
    except BaseException as ex:
        dump('STATUS.json',dict(status='FAILED_STOP',category='D' if isinstance(ex,TimeoutError) else 'C',error=repr(ex),traceback=traceback.format_exc(),elapsed_seconds=elapsed(),completed=RESULTS));raise
    finally:watchdog.cancel()
if __name__=='__main__':main()
