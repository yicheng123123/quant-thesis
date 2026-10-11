"""Bounded diagnostic + full-matrix v48 comparison. No commits/PPL/held."""
import time
START=time.perf_counter()
import csv,gc,hashlib,importlib.util,json,math,os,sys,threading,traceback
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM
ROOT=Path(r'E:\quant-thesis');BASE=ROOT/'experiments/range_search_w4/results';OUT=Path(__file__).resolve().parent
V46=BASE/'v46_m1_single_integer_repair_fullmodel';V45=BASE/'v45_m2_single_integer_repair_fullmodel';V44=BASE/'v44_m2_single_integer_repair_fullmodel';V43=BASE/'v43_m2_single_integer_repair_fullmodel'
CAP=1200;SHADOW_ROWS=[0,224,448,672]
def dump(n,v): (OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def elapsed():return time.perf_counter()-START
def deadline(reserve=0):
    if elapsed()>=CAP-reserve:raise TimeoutError('v48 cumulative1200s cap')
def loadmod(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(V46))
rt=loadmod('run_v42',V46/'runtime_v46.py');rt.OUT=OUT;rt.MODE='v48_preflight';rt.LAYER=-1;rt.START=START;rt.WALL_START=time.time()-elapsed();rt.deadline=deadline
old=loadmod('repair_control',V46/'repair_v45.py');pc=sys.modules['precision_check']
sys.path.insert(0,str(OUT));fast=loadmod('fast_accept',OUT/'fast_accept.py')
def hashX(X):
    with rt.timer('input_hash'):return hashlib.sha256(memoryview(X)).hexdigest()
def loadmodel(p):
    with rt.timer('model_load'):return AutoModelForCausalLM.from_pretrained(str(p),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()
def targetW(li,module):
    with safe_open(str(rt.FP/'model.safetensors'),framework='pt',device='cpu') as f:return f.get_tensor(f'model.layers.{li}.{module}.weight').half()
def gridE(o,c,W):
    sc=o['scales'].cuda()[:,o['g_idx'].long().cuda()];q=(sc*(c.cuda().float()-8)).half().float();return q.double()-W.cuda().double(),sc
def check_candidates(label,X,C,E,p,rr,col,off,SC,code,saved_pred,saved_actual):
    Q=(SC*(code.cuda().float()-8)).half().float();dv=(SC[rr,col]*(code.cuda()[rr,col].float()+off-8)).half().float()-Q[rr,col]
    D=torch.zeros_like(E[rr]);D[torch.arange(len(rr),device='cuda'),col]=dv.double()
    pred=2*dv.double()*p[rr,col]+dv.double().square()*C.diag()[col]
    with rt.timer('historical_rawX_FP64_check'):ck=pc.verify(X,E[rr],D,C,p[rr],pred)
    newE=E.clone();newE[rr,col]+=dv.double()
    oldloss=rt.direct(X,E);newloss=rt.direct(X,newE);actual=(newloss-oldloss)[rr]
    assert torch.equal(actual.cpu(),saved_actual),'original full-shape FP32 delta mismatch'
    st={'E':E,'B':E.abs()@C.abs()}
    U=fast.envelope(st,C,rr,col,dv.double(),len(X))
    rows=[]
    for j,r in enumerate(rr):
        rows.append(dict(source=label,row=int(r),coordinate=int(col[j]),offset=int(off[j]),saved_prediction=float(saved_pred[j]),fresh_prediction=float(pred[j]),FP32_delta=float(actual[j]),rawX_FP64_delta=float(ck['direct64'][j]),error=float(ck['absolute_error'][j]),allowed_error=float(ck['allowed_error'][j]),empirical_envelope=float(U[j]),fast_clear_descent=bool(pred[j]<-U[j]),original_FP32_reject=bool(actual[j]>=0),dtype='Xfp16/E,C,p,delta,rawreferencefp64',N=len(X)))
    return rows
def historical():
    records=[];rt.LAYER=3
    model=loadmodel(rt.V18/'M2/checkpoint/final');hidden=rt.embed(model)
    with torch.no_grad():
        for li in range(4):
            for g in rt.GROUPS:
                path=V44/f'formal/groups/layer_{li:02d}_{g}.pt'
                if path.exists():
                    ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['committed']
                    for n,s in ck['modules'].items():model.model.layers[li].get_submodule(n).weight.copy_(s['weight'].cuda())
            if li<3:hidden=rt.advance_block(model,li,hidden)
    for g,label in [('up_gate','v43'),('down','v44')]:
        rt.MODULE=label;X=rt.capture(model,3,hidden,g);diag=json.loads(((V43 if label=='v43' else V44)/'NUMERICAL_DIAGNOSIS.json').read_text());assert hashX(X)==diag['input_sha256']
        C=rt.build_C(X);assert rt.th(C.cpu())==diag['C_sha256']
        module='mlp.up_proj' if label=='v43' else 'mlp.down_proj';W=targetW(3,module);o=torch.load(V45/'original_codes/layer_03.pt',weights_only=True)[module]
        if label=='v43':
            f=torch.load(V43/'NUMERICAL_FAILURE_formal_3_mlp_up_proj.pt',weights_only=True);code=f['code'];E,SC=gridE(o,code,W);p=E@C;rr=f['rows'].cuda();col=torch.full_like(rr,490);off=torch.tensor([1,-1,-1],device='cuda');saved_pred=f['pred'];saved_actual=f['actual']
        else:
            f=torch.load(V44/'CRITICAL_FAILURE_formal_3_mlp_down_proj.pt',weights_only=False);st=f['state'];pr=f['proposal'];code=st['code'];E,SC=gridE(o,code,W)
            assert torch.equal(E.cpu(),st['E']) and torch.equal((SC*(code.cuda().float()-8)).half().float().cpu(),st['Q'])
            p=st['p'].cuda();pc.cache_check(E,C,p);ix=(pr['rows']==211).nonzero().flatten();rr=pr['rows'][ix].cuda();col=pr['col'][ix].cuda();off=pr['off'][ix].cuda();saved_pred=pr['pred'][ix];saved_actual=(f['direct_new_loss']-st['loss'])[rr.cpu()]
            assert torch.equal((pr['E']-st['E'])[rr.cpu()].cuda(),torch.nn.functional.one_hot(col,C.shape[0]).double()*pr['delta'][ix].cuda()[:,None])
        records+=check_candidates(label,X,C,E,p,rr,col,off,SC,code,saved_pred,saved_actual)
        del X,C,E,p,SC,f,o,W;gc.collect();torch.cuda.empty_cache()
    del model,hidden;gc.collect();torch.cuda.empty_cache()
    model=loadmodel(V46/'R1/checkpoint/final');hidden=rt.embed(model)
    for li in range(3):hidden=rt.advance_block(model,li,hidden)
    rt.MODULE='v46_rejections';X=rt.capture(model,3,hidden,'up_gate');ck=torch.load(V46/'formal/groups/layer_03_up_gate.pt',weights_only=False);assert hashX(X)==ck['input_sha256'];C=rt.build_C(X);assert rt.th(C.cpu())==ck['C_sha256']
    orig=torch.load(V46/'original_codes/layer_03.pt',weights_only=True)
    for module in ['mlp.up_proj','mlp.gate_proj']:
        rt.MODULE=module;s=ck['modules'][module];o=orig[module];W=targetW(3,module)
        for event in s['rejection_events']:
            code=o['codes'].long().clone()
            for mv in s['moves']:
                if mv['round']<event['attempt']:code[mv['rows'],mv['columns']]+=mv['offsets']
            E,SC=gridE(o,code,W);p=E@C;rr=event['rows'].cuda();col=event['columns'].cuda();off=event['offsets'].cuda()
            records+=check_candidates('v46_'+module,X,C,E,p,rr,col,off,SC,code,event['predicted_delta'],event['direct_delta'])
    assert len(records)==13
    rt.csvwrite('HISTORICAL_CANDIDATE_CHECKS.csv',records);dump('HISTORICAL_GATE.json',dict(pass_all=True,candidates=13,v43=3,v44=1,v46=9,FP32_original_batch_delta_exact=True,independent_rawX_FP64_pass=True,policy_change='FP64 descent may accept historical FP32 rejection; historical states remain unchanged'))
    del model,hidden,X,C,E,p,SC,ck,orig;gc.collect();torch.cuda.empty_cache()

def fullterminal(X,C,W,o,result):
    with rt.timer('independent_full_terminal_FP64'):
        e0,_=gridE(o,o['codes'],W);ef,_=gridE(o,result['codes'],W);EE=torch.cat([e0,ef]);r=e0.shape[0];s=torch.zeros(2*r,device='cuda',dtype=torch.float64)
        for lo in range(0,len(X),4096):
            xx=torch.from_numpy(np.array(X[lo:lo+4096],copy=True)).cuda().double();y=xx@EE.T;s+=y.square().sum(0)
        raw=s/len(X);quad=(EE*(EE@C)).sum(1);op=(EE.abs()*(EE.abs()@C.abs())).sum(1)
        tol=64*pc.gamma(len(X)+C.shape[0]+math.ceil(len(X)/4096)+32)*op+64*2**-53*raw
        assert bool(torch.isfinite(raw).all() and ((raw-quad).abs()<=tol).all()),'terminal raw-X/quadratic failure'
        if result['acceptance_revision'].startswith('v48'):
            assert bool((result['final_mse'].cuda()-quad[r:]).abs().le(tol[r:]).all()),'increment bookkeeping failure'
        assert bool((raw[r:]<=raw[:r]).all()),'confirmed terminal objective ascent'
        return dict(initial=raw[:r].cpu(),final=raw[r:].cpu(),max_error=float((raw-quad).abs().max()),max_error_ratio=float(((raw-quad).abs()/tol.clamp_min(1e-300)).max()),coverage_rows=r)
def shadow(X,C,W,o,result,path,module):
    code=o['codes'].long().clone();Es=[];Ds=[];preds=[];keys=[]
    for mv in result['moves']:
        for row in SHADOW_ROWS:
            ii=(mv['rows']==row).nonzero().flatten()
            if len(ii):
                j=int(ii[0]);col=int(mv['columns'][j]);off=int(mv['offsets'][j]);sc=o['scales'][row,o['g_idx'].long()]
                q=(sc*(code[row].float()-8)).half().float();e=q.double()-W[row].double();dv=(sc[col]*(code[row,col].float()+off-8)).half().float()-q[col]
                D=torch.zeros_like(e);D[col]=dv.double();Es.append(e);Ds.append(D);preds.append(mv['predicted_delta'][j]);keys.append((row,mv['round'],col,off))
        code[mv['rows'],mv['columns']]+=mv['offsets']
    if not Es:return []
    with rt.timer('shadow_rawX_full_trajectories'):
        E=torch.stack(Es).cuda();D=torch.stack(Ds).cuda();pred=torch.stack(preds).cuda();check=pc.verify(X,E,D,C,E@C,pred)
        assert bool((check['direct64']<0).all()),'shadow confirmed accepted objective non-descent'
    return [dict(path=path,module=module,row=k[0],attempt=k[1],coordinate=k[2],offset=k[3],prediction=float(pred[j]),rawX_FP64_increment=float(check['direct64'][j]),error=float(check['absolute_error'][j]),allowed_error=float(check['allowed_error'][j]),pass_all=True) for j,k in enumerate(keys)]

def cpu_tests():
    from contextlib import nullcontext
    orig_timer=rt.timer;rt.timer=lambda s:nullcontext();checks=[]
    try:
        # Fixed exactly representable math; no historical-error calibration.
        X=torch.eye(2,dtype=torch.float64);C=X.T@X/2;sc=torch.ones((2,2));code=torch.full((2,2),8,dtype=torch.long);E=torch.tensor([[2.,0.],[0.,0.]],dtype=torch.float64);z=torch.zeros(2,dtype=torch.long)
        st=dict(code=code,Q=torch.zeros((2,2)),E=E,p=E@C,B=E.abs()@C.abs(),loss=(E*(E@C)).sum(1),updates=z.clone(),evaluations=z.clone(),status=z.clone(),rejections=z.clone(),blocked=torch.zeros((2,2,2),dtype=torch.bool))
        before={k:v.clone() for k,v in st.items()};rr=torch.tensor([0,1]);col=torch.tensor([0,0]);off=torch.tensor([-1,1]);dv=off.double();pred=2*dv*st['p'][rr,col]+dv.square()*C.diag()[col]
        pr=old.propose(st,rr,col,off,sc,pred);a,b,*_=fast.settle(st,pr,C,sc,X);assert a.tolist()==[True,False]
        assert all(torch.equal(st[k][1],before[k][1]) for k in ['code','E','p','loss','updates']);checks+=['mixed accept/reject restores state','uncertain branch raw-X positive rejects']
        assert st['blocked'][1,0,1];old.choose(st,sc,C,256);assert st['evaluations'].tolist()==[4,3];checks+=['blacklist no repeat and eval ledger']
        st['updates'][0]=32;old.retire_caps(st,256,32);assert st['status'][0]==2
        st['status'][1]=0;st['evaluations'][1]=255;old.score_and_retire(st,sc,C,256,torch.ones(2)*1e-14);assert st['evaluations'][1]==256 and st['status'][1]==3;checks+=['caps and budget']
        try:pc.cache_check(E,C,E@C+1e-3);raise RuntimeError('corruption undetected')
        except ArithmeticError:checks+=['cache corruption hard failure']
        D=torch.tensor([[-1.,0.],[1.,0.]],dtype=torch.float64);p=E@C;pd=(2*D*p).sum(1)+(D*(D@C)).sum(1)
        ck=pc.verify(X,E,D,C,p,pd)
        try:pc.verify(X,E,D,C,p,pd+1e-4);raise RuntimeError('formula corruption undetected')
        except ArithmeticError:checks+=['formula corruption hard failure']
        assert ck['direct64'].tolist()==[-1.5,.5];checks+=['fixed analytic raw-X reference']
        # Actual shared search batching, fixed ties, legal boundaries.
        st['status'][:]=0;st['evaluations'][:]=0;before={k:v.clone() for k,v in st.items()};f=old.choose(st,sc,C,256)
        for row in range(2):
            one={k:v[row:row+1].clone() for k,v in before.items()};g=old.choose(one,sc[row:row+1],C,256);assert torch.equal(f[0][row:row+1],g[0]) and torch.equal(f[1][row:row+1],g[1])
        checks+=['batch and row candidate reference exact']
        st['blocked'][:]=True;st['status'][:]=0;old.score_and_retire(st,sc,C,256,torch.ones(2)*1e-14);assert bool((st['status']==1).all());checks+=['all blocked candidates stop']
        xx=torch.tensor([[((i*13+j*19)%37-18)/16 for j in range(4)] for i in range(64)],dtype=torch.float32);cc=xx.double().T@xx.double()/64
        codes=torch.full((3,4),8,dtype=torch.uint8);scales=torch.full((3,1),.03125);g=torch.zeros(4,dtype=torch.int32)
        w=torch.tensor([[.035,.018,-.017,.002],[-.013,.01,.02,-.04],[.015,.05,-.026,-.015]],dtype=torch.float16)
        batch=fast.batch_repair(codes,scales,g,w,xx,cc,'toy_batch')
        for i in range(3):
            one=fast.batch_repair(codes[i:i+1],scales[i:i+1],g,w[i:i+1],xx,cc,'toy_row')
            assert all(torch.equal(batch[k][i:i+1],one[k]) for k in ['codes','updates','evaluations','status','rejections'])
        checks+=['complete fast batch matches sequential row reference']
        dump('CPU_TESTS.json',dict(pass_all=True,checks=checks))
    finally:rt.timer=orig_timer

def main():
    assert not (OUT/'GPU_START.json').exists(),'do not silently rerun/reset GPU budget'
    watched=[V46/'repair_v45.py',V46/'precision_check.py',V46/'runtime_v46.py',V46/'PROTOCOL.json',V46/'R1/checkpoint/final/model.safetensors',V43/'NUMERICAL_FAILURE_formal_3_mlp_up_proj.pt',V43/'RUN_STATUS.json',V44/'CRITICAL_FAILURE_formal_3_mlp_down_proj.pt',V44/'RUN_STATUS.json',V46/'original_codes/layer_00.pt',V46/'original_codes/layer_03.pt',rt.V18/'CALIBRATION_MANIFEST.json',BASE/'v47_integer_repair_cost_audit/STAGE_TIMINGS.csv']
    audit=[dict(path=str(p),sha256=sha(p)) for p in watched];dump('SOURCE_AUDIT.json',audit)
    assert sha(V46/'repair_v45.py')==rt.P['repair_source']['repair_v45_sha256'];assert sha(V46/'precision_check.py')==rt.P['repair_source']['precision_check_sha256']
    assert sha(rt.V18/'CALIBRATION_MANIFEST.json')==rt.P['initialization']['calibration_sha256']
    cpu_tests();dump('GPU_START.json',dict(cap_seconds=CAP,process_start_epoch=rt.WALL_START,elapsed_before_GPU=elapsed(),pid=os.getpid(),protocol_sha256=sha(OUT/'ACCEPTANCE_RULES.md'),source_sha256=sha(OUT/'run_v48.py'),acceptance_sha256=sha(OUT/'fast_accept.py'),estimate_seconds=1000,model_setting='Sol high requested; interface cannot confirm'))
    def stop():
        dump('STATUS.json',dict(status='HARD_CAP',category='D',elapsed_seconds=elapsed()));os._exit(124)
    watchdog=threading.Timer(max(0,CAP-elapsed()),stop);watchdog.daemon=True;watchdog.start()
    results=[];shadowrows=[]
    try:
        with torch.inference_mode():
            with rt.timer('warmup'):z=torch.zeros((64,64),device='cuda');z@z
            historical()
            rt.LAYER=0;rt.MODE='v48_matrix';model=loadmodel(V46/'R1/checkpoint/final');hidden=rt.embed(model);orig=torch.load(V46/'original_codes/layer_00.pt',weights_only=True)
            for group,module in [('up_gate','mlp.up_proj'),('down','mlp.down_proj')]:
                deadline(250);rt.MODULE=module;ck=torch.load(V46/f'formal/groups/layer_00_{group}.pt',weights_only=False);X=rt.capture(model,0,hidden,group);xh=hashX(X);assert xh==ck['input_sha256'];C=rt.build_C(X);ch=rt.th(C.cpu());assert ch==ck['C_sha256'];W=targetW(0,module);o=orig[module]
                baseline_hashes=dict(X=xh,C=ch,W=rt.th(W),scales=rt.th(o['scales']),gidx=rt.th(o['g_idx']),zeros=rt.th(o['zeros']))
                assert bool((o['codes']<=15).all() and (o['zeros']==8).all()) and torch.equal(o['g_idx'].long(),torch.arange(C.shape[0])//128)
                assert torch.equal(rt.unpack(rt.pack(o['codes'])),o['codes']),'initial packing roundtrip'
                pair={}
                for name,engine in [('control',old),('fast',fast)]:
                    rt.MODE=name;torch.cuda.synchronize();t=time.perf_counter()
                    result=engine.batch_repair(o['codes'],o['scales'],o['g_idx'],W,X,C,module,**({'save_rows':False} if name=='control' else {}))
                    terminal=fullterminal(X,C,W,o,result);torch.cuda.synchronize();seconds=time.perf_counter()-t
                    assert all(torch.equal(result[k],o[k]) for k in ['scales','zeros','g_idx']);assert torch.equal(rt.unpack(rt.pack(result['codes'])),result['codes'])
                    if name=='control':assert torch.equal(result['codes'],ck['modules'][module]['codes']) and torch.equal(result['evaluations'],ck['modules'][module]['evaluations'])
                    stat=dict(path=name,module=module,rows=len(result['codes']),input_dim=C.shape[0],complete_repair_with_terminal_seconds=seconds,accepted_updates=int(result['updates'].sum()),candidate_evaluations=int(result['evaluations'].sum()),rejected_candidates=int(result['rejections'].sum()),FP64_rechecked_candidates=len(result['recheck_events']),initial_raw64_mse_sum=float(terminal['initial'].sum()),final_raw64_mse_sum=float(terminal['final'].sum()),terminal_max_error=terminal['max_error'],terminal_max_error_ratio=terminal['max_error_ratio'],terminal_coverage_rows=terminal['coverage_rows'],max_p_drift=result['max_p_drift'])
                    pair[name]=(result,terminal,stat);results.append(stat);dump('PARTIAL_RESULTS.json',results)
                    # Local evidence states, not deployable model checkpoint.
                    torch.save(result,OUT/f'LOCAL_{name}_{module.replace(".","_")}.pt')
                    shadowrows+=shadow(X,C,W,o,result,name,module);rt.csvwrite('SHADOW_TRAJECTORIES.csv',shadowrows)
                a,ta,sa=pair['control'];b,tb,sb=pair['fast'];dump(f'COMPARE_{module.replace(".","_")}.json',dict(control=sa,fast=sb,speedup=sa['complete_repair_with_terminal_seconds']/sb['complete_repair_with_terminal_seconds'],different_codes=int((a['codes']!=b['codes']).sum()),different_rows=int((a['codes']!=b['codes']).any(1).sum()),fast_minus_control_raw64_mse_sum=float(tb['final'].sum()-ta['final'].sum()),worst_row_fast_minus_control=float((tb['final']-ta['final']).max()),inputs=baseline_hashes))
                rr=[]
                for name,(res,ter,stat) in pair.items():
                    rr.extend(dict(path=name,module=module,row=i,updates=int(res['updates'][i]),evaluations=int(res['evaluations'][i]),rejections=int(res['rejections'][i]),status=int(res['status'][i]),initial_raw64_mse=float(ter['initial'][i]),final_raw64_mse=float(ter['final'][i])) for i in range(len(res['codes'])))
                rt.csvwrite(f'ROWS_{module.replace(".","_")}.csv',rr)
                assert hashX(X)==xh and rt.th(C.cpu())==ch and rt.th(W)==baseline_hashes['W'],'source pollution'
                del X,C,W,pair,a,b,ta,tb,result,ck;gc.collect();torch.cuda.empty_cache()
        after=[dict(path=r['path'],sha256=sha(r['path'])) for r in audit];assert after==audit;dump('SOURCE_IMMUTABILITY.json',dict(pass_all=True,files=after))
        rt.csvwrite('MATRIX_TIMINGS.csv',results);dump('STATUS.json',dict(status='COMPLETE',category='A' if all(results[i]['complete_repair_with_terminal_seconds']>results[i+1]['complete_repair_with_terminal_seconds'] for i in [0,2]) else 'B',elapsed_seconds=elapsed(),cap_seconds=CAP,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,full_terminal_rows_per_path=5760,shadow_rows_per_path=8,shadow_updates=len(shadowrows),new_model_checkpoint=False,held=False,PPL=False,committed=False))
    except BaseException as ex:
        dump('STATUS.json',dict(status='FAILED',category='D' if isinstance(ex,TimeoutError) else 'C',error=repr(ex),traceback=traceback.format_exc(),elapsed_seconds=elapsed(),completed=results));raise
    finally:watchdog.cancel()
if __name__=='__main__':main()
