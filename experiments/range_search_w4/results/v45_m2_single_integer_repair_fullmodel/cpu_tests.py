"""Deterministic failure cases, independent reference and batched/row equivalence."""
import json,time
from contextlib import nullcontext
from decimal import Decimal,localcontext
import torch
import repair_v45 as r
import precision_check as pc
from pathlib import Path
OUT=Path(__file__).resolve().parent
r.rt.timer=lambda stage:nullcontext()
def fixture(rows=1,d=2):
 code=torch.full((rows,d),8,dtype=torch.long);z=torch.zeros(rows,dtype=torch.long)
 st={'code':code,'Q':torch.zeros((rows,d)),'E':torch.full((rows,d),2.,dtype=torch.float64),'p':torch.full((rows,d),2e-8,dtype=torch.float64),'loss':torch.ones(rows,dtype=torch.float64),'updates':z.clone(),'evaluations':z.clone(),'status':z.clone(),'rejections':z.clone(),'blocked':torch.zeros((rows,d,2),dtype=torch.bool)}
 return st,torch.ones((rows,d)),torch.eye(d,dtype=torch.float64)*1e-8
def step(st,sc,C,newloss,X=None):
 pred,best,take=r.score_and_retire(st,sc,C,256,torch.full_like(st['loss'],1e-8));rr=take.nonzero().flatten();proposal=r.propose(st,rr,best[rr]//2,best[rr]%2*2-1,sc,pred[rr]);return r.settle(st,proposal,newloss,C,sc,X)
def main():
 checks=[];st,sc,C=fixture(2);before={k:v.clone() for k,v in st.items()};a,b,*_=step(st,sc,C,torch.tensor([1-3e-8,1+1e-9],dtype=torch.float64))
 assert a.tolist()==[True,False]
 for k in ['code','p','loss','updates']:assert torch.equal(st[k][1],before[k][1])
 assert st['updates'].tolist()==[1,0] and st['blocked'][1,0,0];checks.append('normal accept plus independent FP32 rejection restores exact row state')
 _,best,_,_=r.choose(st,sc,C,256);assert best[1]!=0 and st['evaluations'].tolist()==[8,7];checks.append('state blacklist and repeated-neighborhood evaluation ledger')
 st,sc,C=fixture();step(st,sc,C,torch.tensor([1.01],dtype=torch.float64));step(st,sc,C,torch.tensor([1.01],dtype=torch.float64));_,_,take=r.score_and_retire(st,sc,C,256,torch.full_like(st['loss'],1e-8));assert not take.any() and st['status'][0]==1 and st['rejections'][0]==2;checks.append('FP32 rise always rejects, including large cross-precision discrepancy; all candidates rejected stops')
 st,sc,C=fixture();_,_,take=r.score_and_retire(st,sc,C,3,torch.full_like(st['loss'],1e-8));assert st['evaluations'][0]==3 and st['status'][0]==3 and not take.any();checks.append('budget exhaustion consumes exact remaining evaluations, no partial-best accept')
 st,sc,C=fixture();step(st,sc,C,torch.tensor([1+1e-9],dtype=torch.float64));step(st,sc,C,torch.tensor([1-3e-8],dtype=torch.float64));assert st['updates'][0]==1 and not st['blocked'].any();assert st['evaluations'][0]==7;checks.append('acceptance clears blacklist but never resets budget')
 st,sc,C=fixture();st['updates'][0]=32;r.retire_caps(st,256,32);assert st['status'][0]==2;checks.append('32-update cap')
 st,sc,C=fixture();st['code'][0]=torch.tensor([0,15]);st['Q']=(sc*(st['code'].float()-8)).half().float();r.choose(st,sc,C,256);assert st['evaluations'][0]==2;checks.append('code boundaries')
 st,sc,C=fixture();before={k:v.clone() for k,v in st.items()}
 try:step(st,sc,C,torch.tensor([float('nan')],dtype=torch.float64));raise RuntimeError('nonfinite not stopped')
 except AssertionError:pass
 for k in ['code','p','loss','updates']:assert torch.equal(st[k],before[k])
 checks.append('nonfinite hard error leaves current state unchanged')
 X=torch.eye(2,dtype=torch.float64)*(2e-8)**0.5
 st,sc,C=fixture();a,b,*rest=step(st,sc,C,torch.tensor([1-5e-6],dtype=torch.float64),X)
 assert a.tolist()==[True] and len(rest[-1])==1 and rest[-1][0]['same_precision_pass'];checks.append('FP32 decrease plus old-check failure invokes independent FP64 verification and accepts')
 st,sc,C=fixture();st['p'][0,0]+=1e-4;before={k:v.clone() for k,v in st.items()}
 try:step(st,sc,C,torch.tensor([1-5e-6],dtype=torch.float64),X);raise RuntimeError('corrupted cache not stopped')
 except ArithmeticError as ex:assert str(ex)=='FP64_CACHE_ERROR'
 for k in ['code','p','loss','updates']:assert torch.equal(st[k],before[k])
 checks.append('injected real cache error stops, not rejected or accepted')
 # Independently computed Decimal raw-X increment, fixed binary inputs and scales.
 X=torch.tensor([[((i*7+j*11)%23-11)/16 for j in range(5)] for i in range(17)],dtype=torch.float64)
 E=torch.tensor([[1/8,-1/4,3/16,-7/32,1/64],[8.,-8.,1e-8,-1e-8,2.]],dtype=torch.float64)
 D=torch.tensor([[0.,1/64,0.,0.,0.],[-1e-8,0.,0.,0.,0.]],dtype=torch.float64)
 C=X.T@X/len(X);p=E@C;pred=(2*D*p).sum(1)+(D*(D@C)).sum(1)
 check=pc.verify(X,E,D,C,p,pred)
 with localcontext() as ctx:
  ctx.prec=80;refs=[]
  for row in range(2):
   total=Decimal(0)
   for xx in X.tolist():
    a=sum(Decimal.from_float(x)*Decimal.from_float(e) for x,e in zip(xx,E[row].tolist()))
    dv=sum(Decimal.from_float(x)*Decimal.from_float(dd) for x,dd in zip(xx,D[row].tolist()))
    total+=2*a*dv+dv*dv
   refs.append(float(total/Decimal(len(X))))
 assert bool((check['direct64']-torch.tensor(refs,dtype=torch.float64)).abs().le(check['allowed_error']).all());checks.append('fixed Decimal80 raw-X reference including cancellation and near-zero changes')
 try:pc.verify(X,E,D,C,p,pred+1e-4);raise RuntimeError('corrupt formula not stopped')
 except ArithmeticError as ex:assert str(ex)=='FP64_FORMULA_DIRECT_ERROR'
 checks.append('injected formula scalar error stops')
 allchecks=[pc.verify(X,E[i:i+1],D[i:i+1],C,p[i:i+1],pred[i:i+1]) for i in range(2)]
 assert bool(torch.isclose(check['direct64'],torch.cat([v['direct64'] for v in allchecks]),rtol=1e-13,atol=1e-18).all());checks.append('row-batched independent FP64 verification matches row reference within frozen precision criterion')
 # Actual single-point neighborhood search batching and tie-breaking vs row reference.
 st,sc,C=fixture(4);st['p'][1,1]=3e-8;st['code'][2,0]=0;st['Q'][2,0]=-8;st['blocked'][3,0,0]=True
 before={k:v.clone() for k,v in st.items()};full=r.choose(st,sc,C,256)
 for i in range(4):
  one={k:v[i:i+1].clone() for k,v in before.items()};ref=r.choose(one,sc[i:i+1],C,256)
  assert torch.equal(full[0][i:i+1],ref[0]) and torch.equal(full[1][i:i+1],ref[1]) and torch.equal(st['evaluations'][i:i+1],one['evaluations'])
 checks.append('batched candidate argmin, boundaries, blacklist and exact tie order match per-row reference')
 # Full small repair path vs sequential row reference, both direct scorers retain all3 output rows.
 r.rt.WALL_START=json.loads((OUT/'GPU_START.json').read_text())['unix'];r.rt.deadline=lambda reserve=0:None;r.rt.MODE='cpu_test';r.rt.LAYER=-1
 x=torch.tensor([[((i*13+j*19)%37-18)/16 for j in range(4)] for i in range(64)],dtype=torch.float32);c=x.double().T@x.double()/64
 codes=torch.full((3,4),8,dtype=torch.uint8);scales=torch.full((3,1),.03125);g=torch.zeros(4,dtype=torch.int32)
 w=torch.tensor([[.035,.018,-.017,.002],[-.013,.01,.02,-.04],[.015,.05,-.026,-.015]],dtype=torch.float16)
 batch=r.batch_repair(codes,scales,g,w,x,c,'toy_full_reference',False)
 sc=scales[:,g.long()];q=(sc*(codes.float()-8)).half().float();e=q.double()-w.double();z=torch.zeros(3,dtype=torch.long)
 st={'code':codes.long().clone(),'Q':q.clone(),'E':e.clone(),'p':e@c,'updates':z.clone(),'evaluations':z.clone(),'status':z.clone(),'rejections':z.clone(),'blocked':torch.zeros((3,4,2),dtype=torch.bool),'loss':r.rt.direct(x,e)}
 threshold=torch.maximum(st['loss']*1e-8,torch.full_like(st['loss'],1e-14))
 for row in range(3):
  # Only reference tests use Python row loops; scoring still has the original full-output shape.
  st['status'][:]=1;st['status'][row]=0
  while st['status'][row]==0:
   r.retire_caps(st,512,32)
   if st['status'][row]!=0:break
   pred,best,take=r.score_and_retire(st,sc,c,512,threshold)
   if not take.any():continue
   rr=take.nonzero().flatten();prop=r.propose(st,rr,best[rr]//2,best[rr]%2*2-1,sc,pred[rr]);r.settle(st,prop,r.rt.direct(x,prop['E']),c,sc,x)
 assert torch.equal(batch['codes'],st['code'].byte()) and torch.equal(batch['updates'],st['updates']) and torch.equal(batch['evaluations'],st['evaluations']) and torch.equal(batch['rejections'],st['rejections'])
 checks.append('complete small batched repair terminal codes, accepted/evaluated/rejected counts match sequential row reference with same full FP32 scoring shape')
 result={'pass':True,'checks':checks,'fixed_reference':{'N':17,'d':5,'rows':2,'precision':'Decimal80','maximum_rawX_FP64_minus_Decimal':float((check['direct64']-torch.tensor(refs,dtype=torch.float64)).abs().max()),'same_precision_allowed_errors':check['allowed_error'].tolist()},'scope':'controlled FP32 scores only for branch transactions; actual independent raw-X math checked against Decimal and full small repair vs reference; no model quality evidence'}
 (OUT/'CPU_TESTS.json').write_text(json.dumps(result,indent=2));print('CPU_TESTS_PASS',len(checks),flush=True)
if __name__=='__main__':main()
