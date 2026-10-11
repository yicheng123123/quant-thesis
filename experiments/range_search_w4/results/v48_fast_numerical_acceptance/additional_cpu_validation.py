"""CPU settlement branch checks; frozen rules unchanged, no GPU calls."""
import json,sys,time
from pathlib import Path
from contextlib import nullcontext
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_v48 as r
r.rt.timer=lambda stage:nullcontext()
def state(E,C):
    rows,d=E.shape;z=torch.zeros(rows,dtype=torch.long);SC=torch.ones((rows,d));code=torch.full((rows,d),8,dtype=torch.long)
    return dict(code=code,Q=torch.zeros((rows,d)),E=E,p=E@C,B=E.abs()@C.abs(),loss=(E*(E@C)).sum(1),updates=z.clone(),evaluations=z.clone(),status=z.clone(),rejections=z.clone(),blocked=torch.zeros((rows,d,2),dtype=torch.bool)),SC
def step(st,SC,C,X,col,off):
    rr=torch.arange(st['code'].shape[0]);col=torch.full_like(rr,col);off=torch.full_like(rr,off);dv=off.double()
    pred=2*dv*st['p'][rr,col]+dv.square()*C.diag()[col]
    pr=r.old.propose(st,rr,col,off,SC,pred);return r.fast.settle(st,pr,C,SC,X)
def main():
    t=time.perf_counter();checks=[]
    X=torch.tensor([[1.,-1e-13],[0.,1.]],dtype=torch.float64);C=X.T@X/2;E=torch.tensor([[-.5,1.]],dtype=torch.float64)
    st,SC=state(E,C);a,b,p,U,ck=step(st,SC,C,X,0,1)
    assert len(ck)==1 and a.tolist()==[True] and st['updates'].tolist()==[1] and ck[0]['direct64']< -ck[0]['allowed_error'];checks.append('uncertain negative independent raw-X acceptance')
    X=torch.tensor([[1.,-1e-16],[0.,1.]],dtype=torch.float64);C=X.T@X/2;st,SC=state(E,C);snap={k:v.clone() for k,v in st.items()};a,b,p,U,ck=step(st,SC,C,X,0,1)
    assert len(ck)==1 and b.tolist()==[True]
    assert all(torch.equal(st[k],snap[k]) for k in ['code','E','p','loss','updates','evaluations']);checks.append('uncertain negative within reference uncertainty rejects with exact state restore')
    X=torch.eye(2,dtype=torch.float64);C=X.T@X/2;st,SC=state(torch.tensor([[2.,0.]],dtype=torch.float64),C)
    step(st,SC,C,X,1,1);assert st['blocked'][0,1,1] and st['rejections'][0]==1
    st['evaluations'][0]=11;step(st,SC,C,X,0,-1);assert not st['blocked'].any() and st['updates'][0]==1 and st['evaluations'][0]==11;checks.append('rejected coordinate blacklist cleared only after accepted state change; budget unchanged')
    result=dict(pass_all=True,checks=checks,seconds=time.perf_counter()-t,device='CPU only',scope='deterministic settlement tests, not candidate-search acceptance or model quality; thresholds/rules unchanged',fixed_X_cross_terms=[-1e-13,-1e-16])
    (r.OUT/'ADDITIONAL_CPU_TESTS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(result)
if __name__=='__main__':main()
