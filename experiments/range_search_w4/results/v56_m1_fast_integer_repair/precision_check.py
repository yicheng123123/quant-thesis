"""Independent raw-X FP64 increment and frozen scale-aware verification."""
import math
import numpy as np
import torch
POLICY={'unit_roundoff':2.0**-53,'increment_scale_factor':32,'loss_floor_factor':64,'cache_scale_factor':64,'chunk_tokens':4096,'max_accepts':32,'reference_rationale':'Empirical FP64 verification criterion using operation magnitudes and conservative serial accumulation lengths; not a strict floating-point error theorem.'}
def gamma(k):
 u=POLICY['unit_roundoff'];assert k*u<1
 return k*u/(1-k*u)
def cache_check(E,C,p,fresh=None):
 assert E.dtype==C.dtype==p.dtype==torch.float64
 assert bool(torch.isfinite(E).all() and torch.isfinite(C).all() and torch.isfinite(p).all())
 fresh=E@C if fresh is None else fresh
 # Bound abs(E)@abs(C) by row l1 norm times each C-column's largest magnitude.
 scale=E.abs().sum(1)[:,None]*C.abs().amax(0)[None,:]
 tol=(64*gamma(E.shape[1]+32)+64*POLICY['unit_roundoff'])*scale
 err=(fresh-p).abs()
 if not bool((err<=tol).all()):raise ArithmeticError('FP64_CACHE_ERROR')
 return fresh,{'cache_max_absolute_error':float(err.max()),'cache_max_error_over_tolerance':float((err/torch.clamp(tol,min=torch.finfo(torch.float64).tiny)).max())}
def independent_delta(X,E,D):
 """Uses only raw X, E and actual dense D. No C or p in this computation."""
 assert E.dtype==D.dtype==torch.float64 and E.shape==D.shape
 n=len(X);r,d=E.shape;assert n>0 and X.shape[1]==d
 acc=torch.zeros(r,dtype=torch.float64,device=E.device);op=acc.clone();loss=acc.clone()
 for lo in range(0,n,4096):
  xx=X[lo:lo+4096]
  if not torch.is_tensor(xx):xx=torch.from_numpy(np.array(xx,copy=True))
  xx=xx.to(E.device,dtype=torch.float64);assert bool(torch.isfinite(xx).all())
  a=xx@E.T;dv=xx@D.T
  aa=xx.abs()@E.abs().T;dd=xx.abs()@D.abs().T
  acc+=(2*a*dv+dv.square()).sum(0)
  op+=(2*aa*dd+dd.square()).sum(0);loss+=(a.square()+dv.square()).sum(0)
 direct=acc/n;op/=n;loss/=n
 g=gamma(d+n+math.ceil(n/4096));rtol=32*g
 atol=32*g*op+64*POLICY['unit_roundoff']*loss
 assert bool(torch.isfinite(direct).all() and torch.isfinite(atol).all())
 return direct,atol,rtol,op,loss
def verify(X,E,D,C,p,pred):
 assert bool(torch.isfinite(pred).all()) and pred.dtype==torch.float64
 direct,atol,rtol,op,loss=independent_delta(X,E,D)
 fresh,cache=cache_check(E,C,p)
 formula=(2*D*p).sum(1)+(D*(D@C)).sum(1)
 fresh_formula=(2*D*fresh).sum(1)+(D*(D@C)).sum(1)
 tol=atol+rtol*torch.maximum(pred.abs(),direct.abs())
 err=(pred-direct).abs()
 # Selected candidate scalar and full delta formula must agree as well.
 selected_error=(pred-formula).abs()
 if not bool((err<=tol).all() and (selected_error<=tol).all()):raise ArithmeticError('FP64_FORMULA_DIRECT_ERROR')
 return {'direct64':direct,'predict64':pred,'formula64':formula,'fresh_predict64':fresh_formula,'absolute_error':err,'absolute_tolerance':atol,'relative_tolerance':rtol,'allowed_error':tol,'operation_scale':op,'loss_scale':loss,**cache}
