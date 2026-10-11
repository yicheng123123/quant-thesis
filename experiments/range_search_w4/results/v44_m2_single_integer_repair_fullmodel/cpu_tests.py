"""Deterministic controlled-scorer state-machine tests, not model evidence."""
import json,torch
from pathlib import Path
import rejection_repair as r
OUT=Path(__file__).resolve().parent
def fixture(rows=1,d=2):
 c=torch.full((rows,d),8,dtype=torch.long);z=torch.zeros(rows,dtype=torch.long)
 st={'code':c,'Q':torch.zeros((rows,d)),'E':torch.full((rows,d),2.,dtype=torch.float64),'p':torch.full((rows,d),2e-8,dtype=torch.float64),'loss':torch.ones(rows,dtype=torch.float64),'updates':z.clone(),'evaluations':z.clone(),'status':z.clone(),'rejections':z.clone(),'blocked':torch.zeros((rows,d,2),dtype=torch.bool)}
 return st,torch.ones((rows,d)),torch.eye(d,dtype=torch.float64)*1e-8
def step(st,sc,C,newloss):
 pred,best,take=r.score_and_retire(st,sc,C,256,torch.full_like(st['loss'],1e-8));rr=take.nonzero().flatten();proposal=r.propose(st,rr,best[rr]//2,best[rr]%2*2-1,sc,pred[rr]);return r.settle(st,proposal,newloss,C,sc)
def main():
 checks=[];st,sc,C=fixture(2);before={k:v.clone() for k,v in st.items()};a,b,*_=step(st,sc,C,torch.tensor([1-3e-8,1+1e-9],dtype=torch.float64))
 assert a.tolist()==[True,False] and st['code'][0,0]==7 and st['updates'].tolist()==[1,0]
 for k in ['code','p','loss','updates']:assert torch.equal(st[k][1],before[k][1])
 assert st['blocked'][1,0,0] and torch.equal(st['evaluations'],torch.tensor([4,4]));checks.append('independent acceptance and bitwise rejection restoration')
 pred,best,full,avail=r.choose(st,sc,C,256);assert best[1]!=0 and st['evaluations'].tolist()==[8,7];checks.append('blacklist forbids repeated rejected coordinate/offset; reevaluations counted')
 st,sc,C=fixture();step(st,sc,C,torch.tensor([1+1e-9],dtype=torch.float64));step(st,sc,C,torch.tensor([1+1e-9],dtype=torch.float64));pred,best,take=r.score_and_retire(st,sc,C,256,torch.full_like(st['loss'],1e-8))
 assert not take.any() and st['status'][0]==1 and st['rejections'][0]==2 and st['updates'][0]==0;checks.append('all predicted-descent candidates rejected; finite no-candidate termination')
 st,sc,C=fixture();pred,best,take=r.score_and_retire(st,sc,C,3,torch.full_like(st['loss'],1e-8));assert st['evaluations'][0]==3 and st['status'][0]==3 and not take.any() and st['updates'][0]==0;checks.append('partial neighborhood consumes exact remaining budget; no partial-best acceptance')
 st,sc,C=fixture();step(st,sc,C,torch.tensor([1+1e-9],dtype=torch.float64));step(st,sc,C,torch.tensor([1-3e-8],dtype=torch.float64));assert st['updates'][0]==1 and not st['blocked'].any();pred,best,full,avail=r.choose(st,sc,C,256);assert best[0]==0;checks.append('legal accepted state change clears blacklist without resetting evaluation budget')
 st,sc,C=fixture();st['updates'][0]=32;r.retire_caps(st,256,32);assert st['status'][0]==2 and st['evaluations'][0]==0;checks.append('32 accepted-update cap')
 for newval in [float('nan'),1.001]:
  st,sc,C=fixture();before={k:v.clone() for k,v in st.items()}
  try:step(st,sc,C,torch.tensor([newval],dtype=torch.float64));raise RuntimeError('critical error was not stopped')
  except (AssertionError,ArithmeticError):pass
  for k in ['code','p','loss','updates']:assert torch.equal(st[k],before[k])
 checks.append('nonfinite and material tolerance errors stop rather than reject-and-hide')
 st,sc,C=fixture();st['code'][0]=torch.tensor([0,15]);st['Q']=(sc*(st['code'].float()-8)).half().float();r.choose(st,sc,C,256);assert st['evaluations'][0]==2;checks.append('legal unsigned4 boundary filtering')
 (OUT/'CPU_TESTS.json').write_text(json.dumps({'pass':True,'checks':checks,'scope':'controlled synthetic scorer values; not empirical model quality'},indent=2));print('CPU_TESTS_PASS',len(checks))
if __name__=='__main__':main()
