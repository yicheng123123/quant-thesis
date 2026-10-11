"""Explicit v44 rejection revision. Original FP32 full-row direct scorer is reused."""
import csv, math
import torch
import run_v42 as rt

def choose(st,SC,C,cap):
    code=st['code'];rows,d=code.shape;device=code.device
    pred=torch.full((rows,),float('inf'),device=device,dtype=torch.float64);best=torch.zeros(rows,device=device,dtype=torch.long)
    full=torch.zeros(rows,device=device,dtype=torch.bool);remaining_candidate=torch.zeros_like(full)
    for lo in range(0,rows,256):
        rr=torch.arange(lo,min(lo+256,rows),device=device);rr=rr[st['status'][rr]==0]
        if not rr.numel():continue
        cc=code[rr];legal=torch.stack((cc>0,cc<15),-1);eligible=legal&~st['blocked'][rr]
        n=eligible.sum((1,2));remaining_candidate[rr]=n>0;remaining=cap-st['evaluations'][rr]
        rank=eligible.flatten(1).long().cumsum(1).reshape_as(eligible);scored=eligible&(rank<=remaining[:,None,None])
        full[rr]=n<=remaining
        # Compute only scored logical candidates; no unbudgeted score slots.
        dv=torch.stack(((SC[rr]*(cc.float()-9)).half().float()-st['Q'][rr],(SC[rr]*(cc.float()-7)).half().float()-st['Q'][rr]),-1).double()
        inds=scored.nonzero();score=torch.full(dv.shape,float('inf'),device=device,dtype=torch.float64)
        if inds.numel():
            a,b,k=inds.unbind(1);delta=dv[a,b,k];v=2*delta*st['p'][rr[a],b]+delta.square()*C.diag()[b]
            assert bool(torch.isfinite(v).all()),'nonfinite candidate increment'
            score[a,b,k]=v
        v,k=score.flatten(1).min(1);pred[rr]=v;best[rr]=k;st['evaluations'][rr]+=scored.sum((1,2))
    assert bool((st['evaluations']<=cap).all()),'evaluation budget exceeded'
    return pred,best,full,remaining_candidate

def propose(st,rr,col,off,SC,pred):
    k=(off+1)//2
    assert not bool(st['blocked'][rr,col,k].any()),'repeated rejected candidate in unchanged row state'
    code=st['code'].clone();code[rr,col]+=off
    assert bool(((code>=0)&(code<=15)).all()),'illegal tentative code'
    dv=(SC[rr,col]*(code[rr,col].float()-8)).half().float()-st['Q'][rr,col]
    E=st['E'].clone();E[rr,col]+=dv.double()
    return {'rows':rr,'col':col,'off':off,'delta':dv.double(),'pred':pred,'code':code,'E':E}

def retire_caps(st,cap,accept_cap):
    st['status'][(st['status']==0)&(st['updates']>=accept_cap)]=2
    st['status'][(st['status']==0)&(st['evaluations']>=cap)]=3

def score_and_retire(st,SC,C,cap,threshold):
    pred,best,full,available=choose(st,SC,C,cap);active=st['status']==0
    st['status'][active&~available]=1
    st['status'][active&available&~full]=3
    st['status'][active&available&full&(pred>=-threshold)]=1
    take=(st['status']==0)&full&available&(pred < -threshold)
    return pred,best,take

def settle(st,proposal,newloss,C,SC):
    rr,col,off,dv,pred=[proposal[k] for k in ['rows','col','off','delta','pred']]
    assert bool(torch.isfinite(newloss).all() and torch.isfinite(pred).all()),'nonfinite direct loss/prediction'
    old=st['loss'][rr].clone();actual=newloss[rr]-old;error=(actual-pred).abs();tol=torch.maximum(old*2e-6,pred.abs()*1e-3)
    if not bool((error<=tol).all()):raise ArithmeticError('material numerical inconsistency beyond original delta tolerance')
    accepted=actual<0;rejected=~accepted;ar=rr[accepted];jr=rr[rejected]
    old_code=st['code'][jr].clone();old_p=st['p'][jr].clone();old_loss=st['loss'][jr].clone();old_updates=st['updates'][jr].clone()
    before_updates=st['updates'][rr].clone()
    # Restore rejected tentative state before assigning the proposal arrays.
    proposal['code'][jr]=st['code'][jr];proposal['E'][jr]=st['E'][jr]
    st['code']=proposal['code'];st['E']=proposal['E']
    if ar.numel():
        ac=col[accepted];st['Q'][ar,ac]=(SC[ar,ac]*(st['code'][ar,ac].float()-8)).half().float()
        st['p'][ar]+=C[ac,:]*dv[accepted,None];st['loss'][ar]=newloss[ar];st['updates'][ar]+=1
        st['blocked'][ar]=False
    if jr.numel():
        st['blocked'][jr,col[rejected],(off[rejected]+1)//2]=True;st['rejections'][jr]+=1
        assert torch.equal(st['code'][jr],old_code) and torch.equal(st['p'][jr],old_p) and torch.equal(st['loss'][jr],old_loss) and torch.equal(st['updates'][jr],old_updates),'rejection state restore failure'
    return accepted,rejected,actual,error,tol,before_updates

def batch_repair(codes,scales,gidx,W,X,C,target,save_rows=True,fixture_accept_cap=None,fixture_original_initial_loss=None):
    """Batched independent rows; never changes direct scorer batch shape or precision."""
    rt.MODULE=target;rows,d=codes.shape;device=C.device;cap=128*d;accept_cap=32 if fixture_accept_cap is None else fixture_accept_cap
    assert 1<=accept_cap<=32
    with rt.timer('repair_initialization'):
        code=codes.to(device,dtype=torch.long).clone();SC=scales.to(device)[:,gidx.long().to(device)]
        Q=(SC*(code.float()-8)).half().float();E=Q.double()-W.to(device).double();p=E@C
        assert bool(torch.isfinite(E).all() and torch.isfinite(p).all() and torch.isfinite(SC).all()),'nonfinite grid/state'
        z=torch.zeros(rows,device=device,dtype=torch.long)
        st={'code':code,'Q':Q,'E':E,'p':p,'updates':z.clone(),'evaluations':z.clone(),'status':z.clone(),'rejections':z.clone(),'blocked':torch.zeros((rows,d,2),device=device,dtype=torch.bool)}
    initial=rt.direct(X,E);st['loss']=initial.clone();assert bool(torch.isclose(initial,(E*p).sum(1),rtol=2e-5,atol=1e-12).all())
    threshold_source=initial if fixture_original_initial_loss is None else fixture_original_initial_loss.to(device)
    assert fixture_original_initial_loss is None or fixture_accept_cap is not None
    threshold=torch.maximum(threshold_source*1e-8,torch.full_like(initial,1e-14));moves=[];events=[];score_passes=[];max_drift=0.;attempt=0
    while bool((st['status']==0).any()):
        rt.deadline(360 if rt.MODE=='formal' else 0);attempt+=1
        assert attempt<=cap+1,'unexpected infinite retry'
        retire_caps(st,cap,accept_cap)
        if not bool((st['status']==0).any()):break
        with rt.timer('candidate_scoring'):
            before=st['evaluations'].clone();pred,best,take=score_and_retire(st,SC,C,cap,threshold)
            score_passes.append({'attempt':attempt,'evaluations_added':(st['evaluations']-before).cpu(),'status_after_scoring':st['status'].cpu().clone()})
        if not bool(take.any()):continue
        rr=take.nonzero().flatten();col=best[rr]//2;off=best[rr]%2*2-1
        proposal=propose(st,rr,col,off,SC,pred[rr]);newloss=rt.direct(X,proposal['E'])
        try:
            with rt.timer('accepted_update_and_rejection_restore'):
                accepted,rejected,actual,error,tol,before_updates=settle(st,proposal,newloss,C,SC)
        except Exception:
            torch.save({'state':{k:v.cpu() for k,v in st.items()},'proposal':{k:v.cpu() if torch.is_tensor(v) else v for k,v in proposal.items()},'direct_new_loss':newloss.cpu()},rt.OUT/f'CRITICAL_FAILURE_{rt.MODE}_{rt.LAYER}_{target.replace(".","_")}.pt');raise
        ar=rr[accepted];jr=rr[rejected]
        rt.note('proposal_settlement',attempt=attempt,proposed_rows=rr.numel(),accepted_rows=ar.numel(),rejected_rows=jr.numel(),max_delta_error=float(error.max()),max_error_over_allowed=float((error/tol).max()),accepted_strictly_decrease=bool((actual[accepted]<0).all()),rejected_state_bitwise_restored=True,pass_all=True)
        if ar.numel():
            moves.append({'round':attempt,'rows':ar.cpu(),'columns':col[accepted].cpu(),'offsets':off[accepted].cpu(),'evaluations':st['evaluations'][ar].cpu(),'predicted_delta':pred[ar].cpu(),'direct_delta':actual[accepted].cpu(),'updates_after':st['updates'][ar].cpu()})
        if jr.numel():
            entry={'attempt':attempt,'rows':jr.cpu(),'columns':col[rejected].cpu(),'offsets':off[rejected].cpu(),'state_accept_count':before_updates[rejected].cpu(),'evaluations':st['evaluations'][jr].cpu(),'predicted_delta':pred[jr].cpu(),'direct_delta':actual[rejected].cpu(),'old_loss':st['loss'][jr].cpu(),'restored':True};events.append(entry)
            with (rt.OUT/'REJECTION_EVENTS.csv').open('a',encoding='utf8',newline='') as f:
                fields=['mode','layer','module','attempt','row','coordinate','offset','state_accept_count','evaluations','predicted_delta','direct_delta','old_loss','relative_predicted_delta','relative_direct_delta','restored'];writer=csv.DictWriter(f,fieldnames=fields)
                if f.tell()==0:writer.writeheader()
                for j,r in enumerate(jr.cpu().tolist()):
                    old=float(entry['old_loss'][j]);pd=float(entry['predicted_delta'][j]);ad=float(entry['direct_delta'][j]);writer.writerow(dict(zip(fields,[rt.MODE,rt.LAYER,target,attempt,r,int(entry['columns'][j]),int(entry['offsets'][j]),int(entry['state_accept_count'][j]),int(entry['evaluations'][j]),pd,ad,old,pd/old if old else 0,ad/old if old else 0,True])))
        if ar.numel() and bool(torch.isin(st['updates'][ar],torch.tensor([1,8,16,24,32],device=device)).any()):
            with rt.timer('p_drift_check'):
                drift=float((st['p']-st['E']@C).abs().max());max_drift=max(max_drift,drift);rt.note('p_drift',attempt=attempt,max_abs=drift)
        print(rt.MODE,rt.LAYER,target,'attempt',attempt,'accepted',ar.numel(),'rejected',jr.numel(),'elapsed',round(rt.elapsed(),2),flush=True)
    final=rt.direct(X,st['E']);assert bool(torch.isclose(final,st['loss'],rtol=2e-6,atol=1e-12).all()) and bool((final<=initial).all())
    with rt.timer('final_correctness'):
        max_drift=max(max_drift,float((st['p']-st['E']@C).abs().max()));c=st['code'].byte().cpu();q=st['Q'].half().cpu();packed=rt.pack(c)
        assert bool(((st['code']>=0)&(st['code']<=15)).all()) and torch.equal(rt.unpack(packed),c)
        dq=(scales[:,gidx.long()]*(rt.unpack(packed).float()-8)).half();assert torch.equal(dq.view(torch.int16),q.view(torch.int16))
        rt.note('final',rows=rows,input_dim=d,initial_pooled_mse=float(initial.sum()),final_pooled_mse=float(final.sum()),max_p_drift=max_drift,codes_legal=True,pack_reload_bitwise=True,scale_sha256=rt.th(scales),g_idx_sha256=rt.th(gidx),rejections=int(st['rejections'].sum()))
    result={'codes':c,'packed_u4':packed,'weight':q,'scales':scales.clone(),'g_idx':gidx.clone(),'zeros':torch.full_like(scales,8),'updates':st['updates'].cpu(),'evaluations':st['evaluations'].cpu(),'initial_mse':initial.cpu(),'final_mse':final.cpu(),'status':st['status'].cpu(),'moves':moves,'rejections':st['rejections'].cpu(),'rejection_events':events,'score_passes':score_passes,'max_p_drift':max_drift,'acceptance_revision':'v44 state-specific reject and continue','initial_fixture_codes':codes.clone() if fixture_accept_cap else None}
    if save_rows:
        rt.STATS.append({'mode':'formal','layer':rt.LAYER,'module':target,'rows':rows,'input_dim':d,'accepted_updates':int(st['updates'].sum()),'modified_coordinates':int((c!=codes).sum()),'candidate_evaluations':int(st['evaluations'].sum()),'initial_search_mse_sum':float(initial.sum()),'final_search_mse_sum':float(final.sum()),'N':len(X),'max_p_drift':max_drift,'rejected_candidates':int(st['rejections'].sum()),'rows_with_rejection':int((st['rejections']>0).sum()),'complete':True});rt.csvwrite('MODULE_RESULTS.csv',rt.STATS)
    return result
