"""v52: state-preserving fallback for the initial loss comparison only."""
import hashlib, math
import numpy as np
import torch
import run_v42 as rt

SOURCE_CHECK = None

def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def review(st, SC, W, X, C, initial, initial_quad, target):
    passed = torch.isclose(initial, initial_quad, rtol=2e-5, atol=1e-12)
    failed = (~passed).nonzero().flatten()
    info = dict(target=target, original_failed_rows=len(failed), original_check_pass=bool(passed.all()), fallback=False, state_unchanged=True, events=[])
    if not len(failed):
        return info
    versions = {k:(id(v),v._version) for k,v in st.items() if torch.is_tensor(v)}
    before = {k:tensor_hash(v[failed]) for k,v in st.items() if torch.is_tensor(v)}
    initial_before = tensor_hash(initial); quad_before = tensor_hash(initial_quad)
    with rt.timer('initial_rawX_FP64_review'):
        assert bool(torch.isfinite(initial).all() and torch.isfinite(initial_quad).all()), 'nonfinite initial loss'
        assert SOURCE_CHECK is not None, 'missing independent initialization source binding'
        provenance = SOURCE_CHECK()
        assert provenance.get('pass'), 'initialization source mismatch'
        cc = st['code'][failed]
        assert bool(((cc>=0)&(cc<=15)).all()), 'illegal initial code'
        real_grid = (SC[failed]*(cc.float()-8)).half().float()
        independent_e = real_grid.double()-W.to(C.device).double()[failed]
        assert torch.equal(real_grid,st['Q'][failed]) and torch.equal(independent_e,st['E'][failed]), 'initial state differs from true grid and pristine weight'
        assert bool(torch.isfinite(independent_e).all() and torch.isfinite(C).all()), 'nonfinite initialization state'
        total = torch.zeros(len(failed),device=C.device,dtype=torch.float64)
        for lo in range(0,len(X),4096):
            rt.deadline()
            if torch.is_tensor(X):
                xx=X[lo:lo+4096].to(device=C.device,dtype=torch.float64)
            else:
                xx=torch.from_numpy(np.array(X[lo:lo+4096],copy=True)).to(device=C.device,dtype=torch.float64)
            assert bool(torch.isfinite(xx).all()), 'nonfinite raw input'
            yy=xx@independent_e.T
            total+=yy.square().sum(0)
        raw=total/len(X)
        assert bool(torch.isfinite(raw).all()), 'nonfinite raw-X FP64 loss'
        tol=initial_quad[failed].abs()*2e-5+1e-12
        assert bool(torch.isclose(raw,initial_quad[failed],rtol=2e-5,atol=1e-12).all()), 'independent raw-X FP64 initial loss disagrees with quadratic'
    assert all((id(st[k]),st[k]._version)==v for k,v in versions.items()), 'initial review changed state identity/version'
    assert all(tensor_hash(st[k][failed])==v for k,v in before.items()), 'initial review changed failed-row state'
    assert tensor_hash(initial)==initial_before and tensor_hash(initial_quad)==quad_before, 'initial review replaced original loss'
    info.update(fallback=True,source=provenance)
    for j,r in enumerate(failed):
        info['events'].append(dict(row=int(r),fp32_loss=float(initial[r]),quadratic_loss=float(initial_quad[r]),rawX_fp64_loss=float(raw[j]),original_abs_error=float((initial[r]-initial_quad[r]).abs()),FP64_abs_error=float((raw[j]-initial_quad[r]).abs()),allowed_error=float(tol[j]),state_unchanged=True))
    rt.note('initialization_cross_precision_review',**info)
    return info
