import contextlib,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O));import run_v52 as r
h=r.fast.initialization_review;rt=r.rt;rt.timer=lambda stage:contextlib.nullcontext();rt.deadline=lambda *a:None

def state(X,C):
 codes=torch.full((2,X.shape[1]),8,dtype=torch.long);SC=torch.full(codes.shape,.25);W=torch.ones_like(SC)
 Q=(SC*(codes.float()-8)).half().float();E=Q.double()-W.double();p=E@C;z=torch.zeros(2,dtype=torch.long)
 st=dict(code=codes,Q=Q,E=E,p=p,B=E.abs()@C.abs(),updates=z.clone(),evaluations=z.clone(),status=z.clone(),rejections=z.clone(),blocked=torch.zeros((2,X.shape[1],2),dtype=torch.bool));q=(E*p).sum(1);st['loss']=q.clone()
 return st,SC,W,q

def fp(st):return {k:h.tensor_hash(v) for k,v in st.items()}
X=torch.eye(4,dtype=torch.float32);C=X.double().T@X.double()/len(X);st,SC,W,q=state(X,C);results=[]
def case(name,initial,XX=X,CC=C,expect_error=False,callback=None):
 ss,sc,ww,qq=state(XX,CC);before=fp(ss);h.SOURCE_CHECK=callback or (lambda:{'pass':True})
 try:
  info=h.review(ss,sc,ww,XX,CC,initial,qq,name);assert not expect_error
  out=dict(name=name,pass_=True,review=info)
 except AssertionError as ex:
  assert expect_error;out=dict(name=name,pass_=True,detected_error=str(ex))
 assert fp(ss)==before;out['all_state_bytes_unchanged']=True;results.append(out)
case('normal',q.clone())
case('cross_precision',q*1.0001)
case('wrong_C_independent_reference',q*1.0001,CC=C*1.5,expect_error=True)
r.rt.LAYER=0;key=(r.RUN_ID,r.PHASE,0,'up_gate');r.SOURCE_RECORDS[key]=dict(input_sha256=r.xhash(X),input_shape=list(X.shape),input_dtype=str(X.dtype),C_sha256=r.tensor_hash(C),C_shape=list(C.shape),C_dtype=str(C.dtype))
case('wrong_input_source',q*1.0001,XX=X*2,expect_error=True,callback=lambda:r.initial_source_check(st['code'],SC,torch.zeros(4,dtype=torch.long),W,X*2,C,'mlp.up_proj'))
case('wrong_C_source',q*1.0001,CC=C*2,expect_error=True,callback=lambda:r.initial_source_check(st['code'],SC,torch.zeros(4,dtype=torch.long),W,X,C*2,'mlp.up_proj'))
case('nonfinite_loss',torch.full_like(q,float('nan')),expect_error=True)
xx=X.clone();xx[0,0]=float('nan');case('nonfinite_raw_input',q*1.0001,XX=xx,expect_error=True)
base=r.loadmod('v48_cpu_reference',r.V48/'fast_accept.py')
g=torch.zeros(4,dtype=torch.int32);c=torch.full((2,4),8,dtype=torch.uint8);s=torch.full((2,1),.25,dtype=torch.float16);ww=torch.ones((2,4),dtype=torch.float16)
a=base.batch_repair(c,s,g,ww,X,C,'cpu_reference');b=r.fast.batch_repair(c,s,g,ww,X,C,'cpu_v52')
for k in ['codes','weight','scales','g_idx','zeros','updates','evaluations','status','rejections','initial_mse','final_mse']:
 assert torch.equal(a[k],b[k]),k
assert len(a['moves'])==len(b['moves'])
for ma,mb in zip(a['moves'],b['moves']):
 for k in ma:
  assert torch.equal(ma[k],mb[k]) if torch.is_tensor(ma[k]) else ma[k]==mb[k]
results.append(dict(name='normal_path_exact_candidate_trajectory',pass_=True,accepted_updates=int(b['updates'].sum()),evaluation_count=int(b['evaluations'].sum()),all_outputs_and_moves_equal=True))
(O/'CPU_TESTS.json').write_text(json.dumps(dict(**{'pass':True},CPU_only=True,cases=results,core_sha256=r.sha(O/'fast_accept_v52.py'),review_sha256=r.sha(O/'initialization_review.py')),indent=2)+'\n')
print('V52_CPU_TESTS_PASS',len(results))
