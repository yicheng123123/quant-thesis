import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from decoder_interface import extract_hidden,write_document
O=Path(__file__).resolve().parent;t=time.perf_counter();torch.set_num_threads(4);cases=[]
def err(name,fn):
 try:fn()
 except (TypeError,ValueError,IndexError):cases.append(dict(name=name,pass_=True));return
 raise AssertionError(f'{name} did not reject')
x=torch.arange(2*5*7,dtype=torch.float16).reshape(2,5,7);before=x.clone();ver=x._version
for name,result in [('Tensor_B2',x),('legacy_tuple_B2',(x,))]:
 y=extract_hidden(result,expected_shape=(2,5,7));assert y is x and torch.equal(y,before);assert not torch.equal(y[:,0],y[:,1]);cases.append(dict(name=name,pass_=True,identity_preserved=True,token_axis_distinct=True))
err('first_token_tuple',(lambda:extract_hidden((x[0,0],),expected_shape=(2,5,7))))
err('missing_batch_axis',lambda:extract_hidden(x[0],expected_shape=(2,5,7)))
err('wrong_token_count',lambda:extract_hidden(x,expected_shape=(2,4,7)))
err('wrong_feature_count',lambda:extract_hidden(x,expected_shape=(2,5,8)))
for name,obj in [('list',[x]),('dict',{'hidden_states':x}),('None',None),('integer',1),('empty_tuple',()),('extra_tuple_items',(x,None)),('wrong_tuple_first',('hidden',))]:err('unsupported_'+name,lambda v=obj:extract_hidden(v,expected_shape=(2,5,7)))
y=x[:1].clone();cache=np.zeros((2,5,7),dtype=np.float16);write_document(cache,1,y);assert np.array_equal(cache[1],y[0].numpy());cases.append(dict(name='strict_cache_success',pass_=True))
snap=cache.copy()
for name,v in [('cache_first_token',y[0,0]),('cache_missing_batch',y[0]),('cache_wrong_batch',x),('cache_wrong_shape',y[:,:4]),('cache_wrong_dtype',y.float())]:
 err(name,lambda z=v:write_document(cache,0,z));assert np.array_equal(cache,snap)
err('cache_out_of_range',lambda:write_document(cache,2,y));assert np.array_equal(cache,snap)
n=y.clone();n[0,1,2]=float('nan');err('nonfinite_hidden',lambda:extract_hidden(n,expected_shape=(1,5,7)))
assert torch.equal(x,before) and x._version==ver
r={'pass':True,'cases':cases,'count':len(cases),'CPU_seconds':time.perf_counter()-t,'GPU_calls':0,'all_original_tensor_bytes_and_versions_unchanged':True,'rejecting_cache_calls_leave_destination_unchanged':True}
(O/'DETERMINISTIC_TESTS.json').write_text(json.dumps(r,indent=2)+'\n');print('DETERMINISTIC_TESTS_PASS',len(cases))
