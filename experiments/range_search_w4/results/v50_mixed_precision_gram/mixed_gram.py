"""One v50 Gram construction and independent uncertainty extension.

Search implementation is imported unchanged. No reference C64 is an input.
"""
import math
import numpy as np
import torch

CHUNK=4096
def gamma32(k):
    u=2.0**-24
    assert k*u<1
    return k*u/(1-k*u)

def build(X,rt,pc):
    with rt.timer('C_mixed_construction_and_statistics'):
        assert X.dtype==np.float16 and not torch.is_autocast_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32
        n,d=X.shape;C=torch.zeros((d,d),device='cuda',dtype=torch.float64)
        ss=torch.zeros(d,device='cuda',dtype=torch.float64)
        with torch.autocast(device_type='cuda',enabled=False):
            for lo in range(0,n,CHUNK):
                rt.deadline()
                x=torch.from_numpy(np.array(X[lo:lo+CHUNK],copy=True)).to('cuda',dtype=torch.float32)
                assert bool(torch.isfinite(x).all())
                C.add_((x.T@x).double())
                ss.add_(x.double().square().sum(0))
        C/=n;C=(C+C.T)*.5;ss/=n
        g64=pc.gamma(n+math.ceil(n/CHUNK)+8)
        h=(ss/(1-g64)).sqrt()
        g32=gamma32(min(n,CHUNK));rho=g32+8*g64*(1+g32)
        assert bool(torch.isfinite(C).all() and torch.isfinite(h).all() and (ss>=0).all())
        return C,dict(h=h,energy=ss,rho=rho,gamma32=g32,gamma64=g64,N=n,chunk=CHUNK,
                      zero_variance_columns=(ss==0).nonzero().flatten().cpu())

def delta_allowance(E,D,stats):
    h=stats['h'];aa=(E.abs()*h).sum(1);bb=(D.abs()*h).sum(1)
    return stats['rho']*(2*aa*bb+bb.square())

def attach(engine,stats,pc):
    original_envelope=engine.envelope
    def envelope(st,C,rr,col,dv,N):
        w=(st['E'][rr].abs()*stats['h']).sum(1)
        ug=stats['rho']*(2*dv.abs()*stats['h'][col]*w+dv.square()*stats['h'][col].square())
        return original_envelope(st,C,rr,col,dv,N)+ug
    def verify(X,E,D,C,p,pred):
        direct,atol,rtol,op,loss=pc.independent_delta(X,E,D)
        fresh,cache=pc.cache_check(E,C,p)
        formula=(2*D*p).sum(1)+(D*(D@C)).sum(1)
        rawtol=atol+rtol*torch.maximum(pred.abs(),direct.abs())
        ug=delta_allowance(E,D,stats)
        selected_error=(pred-formula).abs();err=(pred-direct).abs()
        if not bool((selected_error<=rawtol).all()):raise ArithmeticError('MIXED_SAME_C_FORMULA_ERROR')
        if not bool((err<=rawtol+ug).all()):raise ArithmeticError('MIXED_GRAM_RAW_X_ERROR')
        return dict(direct64=direct,predict64=pred,formula64=formula,absolute_error=err,
                    allowed_error=rawtol,gram_allowance=ug,comparison_tolerance=rawtol+ug,
                    operation_scale=op,loss_scale=loss,**cache)
    engine.envelope=envelope;engine.verify=verify
    return engine
