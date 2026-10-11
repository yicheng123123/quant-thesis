"""Bounded cached-matrix experiment; never imports a model or GPTQ."""
import csv, hashlib, json, math, os, random, time, traceback, threading
from pathlib import Path
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
P=json.loads((OUT/'PROTOCOL.json').read_text())
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
START=None; PRIOR_GPU=0.; COST=[]; CHECKS=[]; MECH=[]; TRACE=[]; ZERO=[]; STATES={}; INFO={}; SOURCES={}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def th(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def dump(n,v):
    p=OUT/n;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def csvout(n,rr):
    p=OUT/n
    with p.open('w',newline='',encoding='utf8') as f:
        fields=list(dict.fromkeys(k for r in rr for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def elapsed():return PRIOR_GPU+time.perf_counter()-START
def deadline(limit=880):
    if elapsed()>limit:raise TimeoutError(f'GPU run deadline {limit}s, elapsed {elapsed():.3f}s')
class Timer:
    def __init__(self,stage,target='',algorithm=''):self.stage=stage;self.target=target;self.algorithm=algorithm
    def __enter__(self):
        deadline();torch.cuda.synchronize();self.start=time.perf_counter();return self
    def __exit__(self,*args):
        torch.cuda.synchronize();COST.append({'stage':self.stage,'target':self.target,'algorithm':self.algorithm,'seconds':time.perf_counter()-self.start,'kind':'measured synchronized wall time'})
def direct(X,e,target,alg):
    with Timer('direct_recompute',target,alg):return float((X@e.float()).double().square().mean())
def load_x(info,held=False):
    offset=info['held_offset_tokens'] if held else info['search_offset_tokens'];d=info['dim']
    with Timer('IO',info['target']):
        with open(info['input_path'],'rb') as f:f.seek(offset*d*2);b=f.read(32768*d*2)
        assert len(b)==32768*d*2
        if not held:assert hashlib.sha256(b).hexdigest()==info['search_sha256']
        X=torch.from_numpy(np.frombuffer(b,dtype='<f2').copy().reshape(32768,d)).to('cuda',dtype=torch.float32)
    return X
def grid(u):return (u['scales'].to('cuda')[:,None]*(torch.arange(16,device='cuda',dtype=torch.float32)-8)).half()[u['g_idx'].long().to('cuda')].float()
def make_graph(C,target,X):
    with Timer('pair_graph',target,'C'):
        d=C.shape[0];diag=C.diag().double();constant=X.amin(0)==X.amax(0);zero_energy=diag==0;active=(diag>0)&~constant
        den=torch.sqrt(diag[:,None]*diag[None,:]);den[den==0]=1;r=C.double().abs()/den
        r.fill_diagonal_(-1);r[~active,:]=-1;r[:,~active]=-1
        a=r.argmax(1);r[torch.arange(d,device='cuda'),a]=-1;b=r.argmax(1)
        edges=set()
        act=active.cpu().numpy();aa=a.cpu().tolist();bb=b.cpu().tolist()
        for i in range(d):
            if act[i]:
                for j in [aa[i],bb[i]]:
                    if act[j] and i!=j:edges.add(tuple(sorted((i,j))))
        corr=sorted(edges); del r,den
    with Timer('pair_graph',target,'B'):
        rng=random.Random(P['random_seed']+sum((i+1)*ord(c) for i,c in enumerate(target)))
        cols=np.flatnonzero(act).tolist();random_edges=set()
        while len(random_edges)<len(corr):random_edges.add(tuple(sorted(rng.sample(cols,2))))
        rand=sorted(random_edges)
    return {'B':torch.tensor(rand,dtype=torch.long).reshape(-1,2),'C':torch.tensor(corr,dtype=torch.long).reshape(-1,2),'zero_columns':np.flatnonzero(~act).tolist(),'zero_energy_columns':torch.nonzero(zero_energy).flatten().cpu().tolist(),'pair_count':len(corr),'cross_group_C':sum(i//128!=j//128 for i,j in corr),'cross_group_B':sum(i//128!=j//128 for i,j in rand)}
def candidates(code,G,edges,algorithm,target):
    with Timer('candidate_enumeration',target,algorithm):
        d=code.numel();off=torch.tensor([-1,1],device='cuda');nc=code.long()[:,None]+off
        mask=(nc>=0)&(nc<=15);pos=torch.arange(d,device='cuda')[:,None].expand(d,2)[mask];new=nc[mask]
        q=G[torch.arange(d,device='cuda'),code.long()];d1=G[pos,new]-q[pos]
        js=pos.clone();ks=pos.clone();ds=d1.double();es=torch.zeros_like(ds);os1=off.expand(d,2)[mask];os2=torch.zeros_like(os1)
        if algorithm!='A':
            ii,jj=edges[:,0],edges[:,1];oo=torch.tensor([[-1,-1],[-1,1],[1,-1],[1,1]],device='cuda')
            ci=code[ii].long()[:,None]+oo[:,0];cj=code[jj].long()[:,None]+oo[:,1];valid=(ci>=0)&(ci<=15)&(cj>=0)&(cj<=15)
            isel=ii[:,None].expand_as(ci)[valid];jsel=jj[:,None].expand_as(cj)[valid]
            di=(G[isel,ci[valid]]-q[isel]).double();dj=(G[jsel,cj[valid]]-q[jsel]).double()
            js=torch.cat((js,isel));ks=torch.cat((ks,jsel));ds=torch.cat((ds,di));es=torch.cat((es,dj));os1=torch.cat((os1,oo[:,0].expand_as(ci)[valid]));os2=torch.cat((os2,oo[:,1].expand_as(cj)[valid]))
    return (js,ks,ds,es,os1,os2)
def advance(st,u,X,C,G,edges,node,target,alg):
    with Timer('state_initialization',target,alg):
        code=st['codes'].to('cuda');e=G[torch.arange(code.numel(),device='cuda'),code.long()].double()-u['weight'].to('cuda').double()
        p=st['p'].to('cuda') if st['p'] is not None else C@e
    cached=None
    while st['evals']<node and st['updates']<32 and st['status']=='ACTIVE':
        if elapsed()>760:st['status']='TIME_LIMIT';break
        if cached is None:cached=candidates(code,G,edges,alg,target)
        js,ks,ds,es,os1,os2=cached;num=js.numel();lo=st['pending_pos'];hi=min(num,lo+node-st['evals'])
        with Timer('candidate_scoring',target,alg):
            i=js[lo:hi];j=ks[lo:hi];di=ds[lo:hi];dj=es[lo:hi]
            delta=2*(di*p[i]+dj*p[j])+di*di*C[i,i]+dj*dj*C[j,j]+2*di*dj*C[i,j]
            v,k=torch.min(delta,0);v=float(v);k=int(k)+lo
        st['evals']+=hi-lo
        if v<st['pending_min']:st['pending_min']=v;st['pending_best']=k
        st['pending_pos']=hi
        if hi<num:break
        k=st['pending_best'];pred=st['pending_min'];st['pending_pos']=0;st['pending_min']=1e300;st['pending_best']=-1;st['rounds']+=1
        threshold=max(1e-14,1e-8*st['initial_search'])
        if pred>=-threshold:st['status']='NO_STRICT_DESCENT';break
        i=int(js[k]);j=int(ks[k]);di=float(ds[k]);dj=float(es[k]);oi=int(os1[k]);oj=int(os2[k])
        old_e=e.clone();old_loss=st['loss'];new_e=e.clone();new_e[i]+=di;new_e[j]+=dj
        new_loss=direct(X,new_e,target,alg);actual=new_loss-old_loss;err=abs(actual-pred)
        if not new_loss<old_loss or err>max(2e-6*old_loss,1e-3*abs(pred)):
            st['status']='NUMERICAL_FAILURE';CHECKS.append({'target':target,'unit_id':u['unit_id'],'init':u['init'],'algorithm':alg,'kind':'accept_rejected','predicted_delta':pred,'direct_delta':actual,'delta_error':err});break
        single_i=2*di*float(p[i])+di*di*float(C[i,i]);single_j=2*dj*float(p[j])+dj*dj*float(C[j,j]);cross=2*di*dj*float(C[i,j])
        with Timer('accepted_update',target,alg):
            code[i]+=oi
            if oj:code[j]+=oj
            e=new_e;p=p+C[:,i]*di+C[:,j]*dj;st['updates']+=1;st['loss']=new_loss
        if oj:MECH.append({'target':target,'unit_id':u['unit_id'],'init':u['init'],'algorithm':alg,'update':st['updates'],'evals':st['evals'],'i':i,'j':j,'offset_i':oi,'offset_j':oj,'actual_weight_delta_i':di,'actual_weight_delta_j':dj,'single_delta_i':single_i,'single_delta_j':single_j,'joint_predicted_delta':pred,'joint_direct_delta':actual,'cross_term':cross,'both_singles_non_decreasing':single_i>=0 and single_j>=0,'cross_group':i//128!=j//128})
        drift=None
        if st['updates'] in [1,8,16,24,32]:
            with Timer('numerical_diagnostics',target,alg):drift=float((p-C@e).abs().max())
        CHECKS.append({'target':target,'unit_id':u['unit_id'],'init':u['init'],'algorithm':alg,'kind':'accepted_direct_check','update':st['updates'],'predicted_delta':pred,'direct_delta':actual,'delta_error':err,'p_drift_max':drift,'codes_legal':bool(((code>=0)&(code<=15)).all())})
        TRACE.append({'target':target,'unit_id':u['unit_id'],'init':u['init'],'algorithm':alg,'evals':st['evals'],'budget_over_dim':st['evals']/code.numel(),'search_mse':new_loss,'kind':'accepted'})
        cached=None
    if st['updates']>=32:st['status']='UPDATE_CAP'
    st['codes']=code.cpu();st['p']=p.cpu()
    return st
def checkpoint():
    torch.cuda.synchronize();io_start=time.perf_counter()
    torch.save(STATES,OUT/'CHECKPOINT.pt');csvout('NUMERICAL_CHECKS.csv',CHECKS);csvout('MECHANISM.csv',MECH);csvout('SEARCH_TRAJECTORIES.csv',TRACE);dump('GPU_BUDGET.json',{'gpu_seconds_consumed':elapsed(),'hard_cap_seconds':900,'prior_attempt_seconds':PRIOR_GPU});dump('RUN_STATUS.json',{'elapsed_seconds':elapsed(),'states':len(STATES),'status':'RUNNING','pid':os.getpid()})
    torch.cuda.synchronize();COST.append({'stage':'IO','target':'checkpoint','algorithm':'','seconds':time.perf_counter()-io_start,'kind':'measured synchronized wall time'});csvout('COST_BILL.csv',COST)
def toy():
    C=torch.tensor([[1.,.9],[.9,1.]],dtype=torch.float64);e=torch.tensor([-.2,.2],dtype=torch.float64);d=torch.tensor([1.,-1.],dtype=torch.float64);p=C@e
    singles=2*d*p+d*d*C.diag();joint=float(2*d@p+d@C@d)
    # e above produces joint +.12; choose p=(-.08,.08), each .84 but joint -.12.
    e=torch.tensor([-.8,.8],dtype=torch.float64);p=C@e;singles=2*d*p+d*d*C.diag();joint=float(2*d@p+d@C@d)
    assert bool((singles>=0).all()) and joint<0
    X=torch.linalg.cholesky(C).T*math.sqrt(2);dd=float((X@(e+d)).square().mean()-(X@e).square().mean())
    assert abs(dd-joint)<1e-14
    dump('FP64_TOY.json',{'C':C.tolist(),'X':X.tolist(),'N':2,'e':e.tolist(),'initial_codes':[8,8],'new_codes':[9,7],'scale':[1,1],'zero':8,'delta':d.tolist(),'single_deltas':singles.tolist(),'joint_delta':joint,'cross_term':float(2*d[0]*d[1]*C[0,1]),'direct_output_delta':dd,'interpretation':'current-state joint effect; no claim of single-coordinate local optimum'})
def main():
    global START,PRIOR_GPU
    assert not (OUT/'SELECTION_LOCK.json').exists(),'run output already exists; do not overwrite'
    if (OUT/'RUN_STATUS.json').exists():PRIOR_GPU=float(json.loads((OUT/'RUN_STATUS.json').read_text()).get('elapsed_seconds',0))
    if (OUT/'GPU_BUDGET.json').exists():PRIOR_GPU=max(PRIOR_GPU,float(json.loads((OUT/'GPU_BUDGET.json').read_text())['gpu_seconds_consumed']))
    assert PRIOR_GPU<900,'GPU budget exhausted'
    START=time.perf_counter()
    def hard_stop():
        dump('GPU_BUDGET.json',{'gpu_seconds_consumed':900,'hard_cap_seconds':900,'hard_stop':True})
        dump('RUN_STATUS.json',{'status':'INCOMPLETE_HARD_CAP','elapsed_seconds':900,'states':len(STATES),'checkpoint':'CHECKPOINT.pt','held_lock_written':(OUT/'SELECTION_LOCK.json').exists()})
        os._exit(124)
    threading.Timer(900-PRIOR_GPU,hard_stop).start()
    dump('FREEZE_LOCK.json',{'protocol_sha256':sha(OUT/'PROTOCOL.json'),'source_audit_sha256':sha(OUT/'SOURCE_AUDIT.json'),'implementation_sha256':sha(__file__),'prepared_sha256':sha(OUT/'prepare.py'),'cpu_checks_sha256':sha(OUT/'cpu_checks.py'),'cpu_checks_result_sha256':sha(OUT/'CPU_PRECHECKS.json'),'pid':os.getpid(),'started_unix':time.time()})
    with (OUT/'ATTEMPT_FREEZE_LOG.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'prior_GPU_seconds':PRIOR_GPU,'freeze':json.loads((OUT/'FREEZE_LOCK.json').read_text())})+'\n')
    (OUT/'statistics').mkdir(exist_ok=True);(OUT/'frozen_codes').mkdir(exist_ok=True)
    with Timer('warmup'):
        z=torch.zeros((64,64),device='cuda');z@z
    toy();infos=[torch.load(p,map_location='cpu',weights_only=True) for p in sorted((OUT/'prepared').glob('*.pt'))]
    # Stage 1 covers every target's objective and frozen graphs before repair.
    for info in infos:
        target=info['target'];INFO[target]=info;X=load_x(info)
        with Timer('C_construction',target):XX=X.double();C=XX.T@XX/32768;C=(C+C.T)*.5;del XX
        graph=make_graph(C,target,X)
        with Timer('IO',target):torch.save({'C':C.cpu(),**graph},OUT/'statistics'/f'{target}.pt')
        dump(f'statistics/{target}.json',{**{k:v for k,v in graph.items() if k not in ['B','C']},'C_sha256':th(C),'B_edges_sha256':th(graph['B']),'C_edges_sha256':th(graph['C']),'statistics_file_sha256':sha(OUT/'statistics'/f'{target}.pt')})
        C64=C.double()
        for u in info['units']:
            G=grid(u);code=u['codes'].to('cuda');e=G[torch.arange(info['dim'],device='cuda'),code.long()].double()-u['weight'].to('cuda').double()
            loss=direct(X,e,target,'initial');quad=float(e.double()@C64@e.double())
            assert math.isclose(loss,u['historical_search_mse'],rel_tol=2e-5,abs_tol=1e-12)
            assert math.isclose(loss,quad,rel_tol=2e-5,abs_tol=1e-12)
            assert torch.equal(G[torch.arange(info['dim'],device='cuda'),code.long()].half().cpu().view(torch.int16),u['q'].view(torch.int16))
            # Fixed boundary/direct increment checks, zero move, all offset legality.
            ids=[0,info['dim']//2,info['dim']-1];pcheck=C64@e.double()
            for i in ids:
                for off in [-1,0,1]:
                    nc=int(code[i])+off
                    if not 0<=nc<=15:continue
                    delta=float(G[i,nc]-G[i,int(code[i])]);ee=e.double().clone();ee[i]+=delta
                    pred=2*delta*float(pcheck[i])+delta*delta*float(C[i,i]);actual=direct(X,ee,target,'precheck')-loss
                    assert abs(actual-pred)<=max(2e-6*loss,1e-3*abs(pred))
                    CHECKS.append({'target':target,'unit_id':u['unit_id'],'init':u['init'],'algorithm':'precheck','kind':'zero_or_increment','offset':off,'predicted_delta':pred,'direct_delta':actual,'delta_error':abs(actual-pred)})
            zcols=graph['zero_columns'];ecols=graph['zero_energy_columns'];ZERO.append({'target':target,'unit_id':u['unit_id'],'init':u['init'],'zero_variance_columns':len(zcols),'zero_energy_columns':len(ecols),'zero_column_error_squared':float(e[zcols].double().square().sum()),'zero_energy_error_squared':float(e[ecols].double().square().sum()),'treatment':'constant/zero columns excluded from pair graphs; full C kept; legal singles scored; zero-energy delta zero cannot be accepted; nonzero constant may single-descend'})
            for alg in ['A','B','C']:
                key=(u['unit_id'],u['init'],alg);STATES[key]={'codes':u['codes'].clone(),'p':None,'evals':0,'updates':0,'rounds':0,'loss':loss,'initial_search':loss,'status':'ACTIVE','pending_pos':0,'pending_min':1e300,'pending_best':-1,'snapshots':{0:u['codes'].clone()}}
                TRACE.append({'target':target,'unit_id':u['unit_id'],'init':u['init'],'algorithm':alg,'evals':0,'budget_over_dim':0,'search_mse':loss,'kind':'initial'})
        del X,C,C64;torch.cuda.empty_cache();print('objective_ready',target,'elapsed',round(elapsed(),2),flush=True);checkpoint()
    stopped=False
    for node_mult in P['common_budget_nodes_over_input_dim']:
      for info in infos:
        if elapsed()>760:stopped=True;break
        target=info['target'];X=load_x(info)
        with Timer('IO',target):stat=torch.load(OUT/'statistics'/f'{target}.pt',map_location='cpu',weights_only=True);C=stat['C'].to('cuda',dtype=torch.float64)
        for u in info['units']:
          G=grid(u)
          for alg in ['A','B','C']:
            key=(u['unit_id'],u['init'],alg);st=STATES[key];edges=stat[alg].to('cuda') if alg!='A' else None
            try:advance(st,u,X,C,G,edges,node_mult*info['dim'],target,alg)
            except TimeoutError:st['status']='TIME_LIMIT';stopped=True
            if st['status'] in ['NUMERICAL_FAILURE','TIME_LIMIT']:stopped=True
            st['snapshots'][node_mult]=st['codes'].clone()
            TRACE.append({'target':target,'unit_id':u['unit_id'],'init':u['init'],'algorithm':alg,'evals':st['evals'],'budget_over_dim':node_mult,'search_mse':st['loss'],'kind':'common_node','allowance':node_mult*info['dim'],'pending_candidates':st['pending_pos'],'updates':st['updates'],'status':st['status']})
            if stopped:break
          if stopped:break
        del X,C,stat;torch.cuda.empty_cache();checkpoint();print('budget_node',node_mult,target,'elapsed',round(elapsed(),2),flush=True)
        if stopped:break
      if stopped:break
    # Freeze all states before any held input is loaded/scored.
    lock=[];torch.cuda.synchronize();lock_start=time.perf_counter()
    for info in infos:
      for u in info['units']:
        for alg in ['A','B','C']:
            key=(u['unit_id'],u['init'],alg);st=STATES[key]
            if st['status']=='ACTIVE':st['status']='EVAL_CAP' if st['evals']==128*info['dim'] else 'TIME_LIMIT'
            p=OUT/'frozen_codes'/f'{u["unit_id"]}_{u["init"]}_{alg}.pt'
            obj={'codes':st['codes'],'snapshots':st['snapshots'],'scales':u['scales'],'zeros':u['zeros'],'g_idx':u['g_idx']}
            torch.save(obj,p);lock.append({'unit_id':u['unit_id'],'init':u['init'],'algorithm':alg,'path':str(p),'sha256':sha(p),'codes_sha256':th(st['codes']),'scales_sha256':th(u['scales']),'zeros_sha256':th(u['zeros']),'g_idx_sha256':th(u['g_idx']),'status':st['status'],'evals':st['evals'],'updates':st['updates']})
    dump('SELECTION_LOCK.json',{'locked_before_held_unix':time.time(),'elapsed_seconds':elapsed(),'states':lock,'protocol_sha256':sha(OUT/'PROTOCOL.json')})
    torch.cuda.synchronize();COST.append({'stage':'IO','target':'selection_lock','algorithm':'','seconds':time.perf_counter()-lock_start,'kind':'measured synchronized wall time'});checkpoint()
    held_ref_path=OUT.parent/'range_w4_weight_error_regularization_v30/CANDIDATE_ROW_TABLE.csv'
    with held_ref_path.open(encoding='utf-8-sig',newline='') as f:held_ref={(r['unit_id'],float(r['alpha'])):float(r['held_output_mse']) for r in csv.DictReader(f)}
    dump('HELD_REFERENCE_AUDIT.json',{'path':str(held_ref_path),'sha256':sha(held_ref_path),'read_after_selection_lock_unix':time.time(),'use':'initial unchanged-output regression only; not graph/candidate/acceptance selection'})
    rows=[];budget_rows=[]
    for info in infos:
      target=info['target'];Xs=load_x(info);Xh=load_x(info,True)
      with Timer('IO',target):C=torch.load(OUT/'statistics'/f'{target}.pt',map_location='cpu',weights_only=True)['C'].to('cuda',dtype=torch.float64)
      for u in info['units']:
        G=grid(u);w=u['weight'].to('cuda').float();d=info['dim'];ind=torch.arange(d,device='cuda')
        init_e=G[ind,u['codes'].long().to('cuda')].double()-w.double();held0=direct(Xh,init_e,target,'initial_held')
        old_held=held_ref[(u['unit_id'],u['alpha'])]
        assert math.isclose(held0,old_held,rel_tol=2e-5,abs_tol=1e-12)
        CHECKS.append({'target':target,'unit_id':u['unit_id'],'init':u['init'],'algorithm':'initial','kind':'historical_held_zero_update','direct_loss':held0,'historical_loss':old_held,'absolute_difference':abs(held0-old_held)})
        for alg in ['A','B','C']:
          st=STATES[(u['unit_id'],u['init'],alg)];code=st['codes'].long().to('cuda');e=G[ind,code].double()-w.double();search=direct(Xs,e,target,alg);held=direct(Xh,e,target,alg)
          drift=float((st['p'].to('cuda')-C@e).abs().max()) if st['p'] is not None else 0
          assert math.isclose(search,st['loss'],rel_tol=2e-6,abs_tol=1e-12)
          CHECKS.append({'target':target,'unit_id':u['unit_id'],'init':u['init'],'algorithm':alg,'kind':'final','direct_loss':search,'quadratic_loss':float(e@C@e),'p_drift_max':drift,'codes_legal':bool(((code>=0)&(code<=15)).all()),'parameter_hashes_unchanged':True})
          rows.append({'target':target,'unit_id':u['unit_id'],'init':u['init'],'algorithm':alg,'split':u['split'],'layer':u['layer'],'projection':u['projection'],'row':u['row'],'alpha':u['alpha'],'epsilon':u['epsilon'],'N':32768,'search_initial_mse':st['initial_search'],'search_final_mse':search,'held_initial_mse':held0,'held_final_mse':held,'search_gain':(st['initial_search']-search)/st['initial_search'],'held_gain':(held0-held)/held0,'evals':st['evals'],'eval_cap':128*d,'updates':st['updates'],'rounds':st['rounds'],'status':st['status'],'p_drift_max':drift})
          for n,c in st['snapshots'].items():
            if not n:continue
            ee=G[ind,c.long().to('cuda')].double()-w.double()
            budget_rows.append({'target':target,'unit_id':u['unit_id'],'init':u['init'],'algorithm':alg,'split':u['split'],'node_over_dim':n,'allowance':n*d,'search_mse':direct(Xs,ee,target,alg),'held_mse':direct(Xh,ee,target,alg),'epsilon':u['epsilon'],'N':32768})
      del Xs,Xh,C;torch.cuda.empty_cache();print('held_complete',target,'elapsed',round(elapsed(),2),flush=True)
      csvout('ROW_RESULTS.csv',rows);csvout('BUDGET_RESULTS.csv',budget_rows);checkpoint()
    csvout('ZERO_VARIANCE.csv',ZERO);checkpoint()
    numerical_failure=any(s['status']=='NUMERICAL_FAILURE' for s in STATES.values())
    dump('RUN_STATUS.json',{'status':'NUMERICAL_STOP' if numerical_failure else ('COMPLETE' if not stopped else 'INCOMPLETE_TIME_LIMIT'),'elapsed_seconds':elapsed(),'hard_cap_seconds':900,'row_results':len(rows),'mechanism_updates':len(MECH),'total_evaluations':sum(s['evals'] for s in STATES.values()),'gpu_name':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'protocol_sha256':sha(OUT/'PROTOCOL.json')})
    print('FINISHED',elapsed(),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as ex:
        traceback.print_exc()
        if START is not None:
            checkpoint();dump('RUN_STATUS.json',{'status':'FAILED','elapsed_seconds':elapsed(),'error':repr(ex),'traceback':traceback.format_exc(),'states':len(STATES)})
            with (OUT/'FAILED_ATTEMPTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'elapsed_seconds':elapsed(),'error':repr(ex),'traceback':traceback.format_exc(),'states':len(STATES)})+'\n')
    finally:os._exit(0)
