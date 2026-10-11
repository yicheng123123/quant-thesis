"""v48 acceptance revision. Search functions are imported, never rewritten."""
import math
import torch
import run_v42 as rt
import repair_control as old
from precision_check import gamma, cache_check, verify

def envelope(st,C,rr,col,dv,N):
    T=2*dv.abs()*st['B'][rr,col]+dv.square()*C.diag()[col].abs()
    K=(64*gamma(C.shape[0]+32)+64*2**-53)*st['E'][rr].abs().sum(1)*C.abs().amax(0)[col]
    return 64*gamma(N+C.shape[0]+math.ceil(N/4096)+32)*T+2*dv.abs()*K+32*2**-53*T

def settle(st,pr,C,SC,X):
    rr,col,off,dv,pred=[pr[k] for k in ['rows','col','off','delta','pred']]
    assert bool(torch.isfinite(pred).all() and torch.isfinite(dv).all())
    assert bool(((pr['code']>=0)&(pr['code']<=15)).all())
    assert torch.equal(((SC[rr,col]*(pr['code'][rr,col].float()-8)).half().float()-st['Q'][rr,col]).double(),dv)
    old_p=st['p'][rr].clone()
    U=envelope(st,C,rr,col,dv,len(X));assert bool(torch.isfinite(U).all())
    uncertain=pred>=-U;accepted=~uncertain;used=pred.clone();rechecks=[]
    if bool(uncertain.any()):
        tr=rr[uncertain];tc=col[uncertain];td=dv[uncertain]
        with rt.timer('fast_uncertain_refresh_and_rawX'):
            fresh,cache=cache_check(st['E'][tr],C,st['p'][tr]);st['p'][tr]=fresh
            pp=2*td*fresh[torch.arange(len(tr),device=C.device),tc]+td.square()*C.diag()[tc]
            D=torch.zeros_like(st['E'][tr]);D[torch.arange(len(tr),device=C.device),tc]=td
            check=verify(X,st['E'][tr],D,C,fresh,pp)
            accepted[uncertain]=check['direct64'] < -check['allowed_error'];used[uncertain]=pp
            rechecks=[{'row':int(r),'direct64':float(check['direct64'][j]),'prediction':float(pp[j]),'allowed_error':float(check['allowed_error'][j]),'accepted':bool(accepted[uncertain][j])} for j,r in enumerate(tr)]
    rejected=~accepted;ar=rr[accepted];jr=rr[rejected]
    st['p'][jr]=old_p[rejected]
    snapshot={k:st[k][jr].clone() for k in ['code','E','p','loss','updates']}
    pr['code'][jr]=st['code'][jr];pr['E'][jr]=st['E'][jr]
    st['code']=pr['code'];st['E']=pr['E']
    if len(ar):
        ac=col[accepted];dd=dv[accepted]
        st['Q'][ar,ac]=(SC[ar,ac]*(st['code'][ar,ac].float()-8)).half().float()
        st['p'][ar]+=C[ac,:]*dd[:,None];st['B'][ar]+=C[ac,:].abs()*dd.abs()[:,None]
        st['loss'][ar]+=used[accepted];st['updates'][ar]+=1;st['blocked'][ar]=False
    if len(jr):
        st['blocked'][jr,col[rejected],(off[rejected]+1)//2]=True;st['rejections'][jr]+=1
        assert all(torch.equal(st[k][jr],v) for k,v in snapshot.items()),'rejection state restore'
    return accepted,rejected,used,U,rechecks

def batch_repair(codes,scales,gidx,W,X,C,target):
    rt.MODULE=target;rows,d=codes.shape;device=C.device;cap=128*d
    with rt.timer('fast_initialization'):
        code=codes.to(device,dtype=torch.long).clone();SC=scales.to(device)[:,gidx.long().to(device)]
        Q=(SC*(code.float()-8)).half().float();E=Q.double()-W.to(device).double();p=E@C
        assert bool(torch.isfinite(E).all() and torch.isfinite(p).all() and torch.isfinite(SC).all())
        z=torch.zeros(rows,device=device,dtype=torch.long)
        st=dict(code=code,Q=Q,E=E,p=p,B=E.abs()@C.abs(),updates=z.clone(),evaluations=z.clone(),status=z.clone(),rejections=z.clone(),blocked=torch.zeros((rows,d,2),device=device,dtype=torch.bool))
        initial_quad=(E*p).sum(1);st['loss']=initial_quad.clone()
    initial=rt.direct(X,E);assert bool(torch.isclose(initial,initial_quad,rtol=2e-5,atol=1e-12).all())
    threshold=torch.maximum(initial*1e-8,torch.full_like(initial,1e-14));moves=[];rechecks=[];attempt=0;max_drift=0
    while bool((st['status']==0).any()):
        rt.deadline();attempt+=1;assert attempt<=cap+1
        old.retire_caps(st,cap,32)
        if not bool((st['status']==0).any()):break
        with rt.timer('fast_candidate_scoring'):
            pred,best,take=old.score_and_retire(st,SC,C,cap,threshold)
        if not bool(take.any()):continue
        rr=take.nonzero().flatten();col=best[rr]//2;off=best[rr]%2*2-1
        with rt.timer('fast_propose_and_settle'):
            pr=old.propose(st,rr,col,off,SC,pred[rr]);accepted,rejected,used,U,checks=settle(st,pr,C,SC,X)
        ar=rr[accepted];rechecks.extend(checks)
        if len(ar):
            moves.append(dict(round=attempt,rows=ar.cpu(),columns=col[accepted].cpu(),offsets=off[accepted].cpu(),evaluations=st['evaluations'][ar].cpu(),predicted_delta=used[accepted].cpu(),updates_after=st['updates'][ar].cpu(),envelope=U[accepted].cpu()))
        if len(ar) and bool(torch.isin(st['updates'][ar],torch.tensor([1,8,16,24,32],device=device)).any()):
            with rt.timer('fast_p_check_refresh'):
                fresh,cache=cache_check(st['E'],C,st['p']);max_drift=max(max_drift,cache['cache_max_absolute_error']);st['p']=fresh;st['B']=st['E'].abs()@C.abs()
        print('FAST',target,attempt,'accepted',len(ar),'rejected',int(rejected.sum()),flush=True)
    with rt.timer('fast_final_cache_grid_pack'):
        fresh,cache=cache_check(st['E'],C,st['p']);max_drift=max(max_drift,cache['cache_max_absolute_error'])
        c=st['code'].byte().cpu();q=st['Q'].half().cpu();packed=rt.pack(c)
        assert bool(((st['code']>=0)&(st['code']<=15)).all()) and torch.equal(rt.unpack(packed),c)
        assert torch.equal((scales[:,gidx.long()]*(rt.unpack(packed).float()-8)).half().view(torch.int16),q.view(torch.int16))
    return dict(codes=c,weight=q,scales=scales.clone(),g_idx=gidx.clone(),zeros=torch.full_like(scales,8),updates=st['updates'].cpu(),evaluations=st['evaluations'].cpu(),status=st['status'].cpu(),rejections=st['rejections'].cpu(),initial_mse=initial.cpu(),final_mse=st['loss'].cpu(),moves=moves,recheck_events=rechecks,max_p_drift=max_drift,acceptance_revision='v48 empirical FP64 envelope and independent raw-X recheck')
