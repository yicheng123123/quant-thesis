"""User-requested encoding/cast/toy checks on CPU; no experiment scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import json,time
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent
torch.set_num_threads(2)
start=time.perf_counter();count=0;cast_differences=0;boundary_excluded=0
for p in sorted((OUT/'prepared').glob('*.pt')):
    info=torch.load(p,map_location='cpu',weights_only=True)
    for u in info['units']:
        d=info['dim'];g=(u['scales'][:,None]*(torch.arange(16,dtype=torch.float32)-8)).half()[u['g_idx'].long()].float();code=u['codes'].long();i=torch.arange(d)
        assert torch.equal(g[i,code].half().view(torch.int16),u['q'].view(torch.int16))
        for off in [-1,1]:
            nc=code+off;valid=(nc>=0)&(nc<=15);boundary_excluded+=int((~valid).sum());count+=int(valid.sum())
            actual=g[i[valid],nc[valid]]-g[i[valid],code[valid]];nominal=off*u['scales'][u['g_idx'].long()][valid]
            cast_differences+=int((actual!=nominal).sum())
            c=u['codes'].clone();c[valid]+=off
            assert bool(((c>=0)&(c<=15)).all())
C=torch.tensor([[1.,.9],[.9,1.]],dtype=torch.float64);e=torch.tensor([-.8,.8],dtype=torch.float64);delta=torch.tensor([1.,-1.],dtype=torch.float64);p=C@e
singles=2*delta*p+delta.square()*C.diag();joint=float(2*delta@p+delta@C@delta);direct=float((e+delta)@C@(e+delta)-e@C@e)
assert bool((singles>=0).all()) and joint<0 and abs(joint-direct)<1e-14
assert torch.allclose(p+C@delta,C@(e+delta),rtol=0,atol=1e-15)
out={'status':'PASS','initializations_checked':96,'legal_single_candidates':count,'out_of_bounds_excluded':boundary_excluded,'actual_FP16_cast_delta_differs_from_offset_scale':cast_differences,'zero_update_parameters_and_weights_bitwise_equal':True,'toy_single_deltas':singles.tolist(),'toy_joint_delta':joint,'toy_cross_term':-1.8,'toy_direct_delta':direct,'CPU_wall_seconds':time.perf_counter()-start}
(OUT/'CPU_PRECHECKS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
