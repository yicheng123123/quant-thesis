import hashlib, json, time
import torch
from safetensors import safe_open
OUT=None
def run_preflight(rt, repair, model):
    global OUT
    OUT=rt.OUT
    rt.MODE='preflight';rt.LAYER=0
    ids=rt.zero_model_check(model)
    with torch.inference_mode():before=model(ids,use_cache=False).logits.detach().cpu()
    hidden=rt.embed(model)
    state=torch.load(OUT/'original_codes/layer_00.pt',map_location='cpu',weights_only=True)
    name='self_attn.q_proj';s=state[name];source_codes=s['codes'].clone()
    src_hash=hashlib.sha256(source_codes.contiguous().numpy().tobytes()).hexdigest()
    X=rt.capture(model,0,hidden,'qkv');C=rt.build_C(X)
    with safe_open(str(rt.FP/'model.safetensors'),framework='pt',device='cpu') as fp:
        W=fp.get_tensor('model.layers.0.self_attn.q_proj.weight')[:128].half()
    result=repair.batch_repair(s['codes'][:128],s['scales'][:128],s['g_idx'],W,X,C,'preflight_M1_L0_q_proj_rows0_127',save_rows=False)
    assert bool((result['final_mse']<=result['initial_mse']).all())
    assert bool(((result['codes']>=0)&(result['codes']<=15)).all())
    assert bool((result['updates']<=32).all()) and bool((result['evaluations']<=128*896).all())
    assert torch.equal(s['codes'],source_codes)
    assert hashlib.sha256(s['codes'].contiguous().numpy().tobytes()).hexdigest()==src_hash
    q=model.model.layers[0].self_attn.q_proj.weight.detach().cpu()
    assert torch.equal(q.view(torch.int16),s['weight'].view(torch.int16))
    with torch.inference_mode():after=model(ids,use_cache=False).logits.detach().cpu()
    assert torch.equal(before,after),'preflight row repair mutated M1 model'
    report={'pass':True,'scope':'M1 source integration and one uncommitted fixed 128-row search block',
      'M1_zero_pack_roundtrip_and_sample_output':True,'sample_logits_bitwise_equal':True,
      'all168_M1_weights_rebuilt_from_unique_codes_and_native_grid':True,
      'fixed_block':{'layer':0,'module':name,'rows':'0:128','input_dim':896,'N':len(X),'C_dtype':str(C.dtype),'C_undamped':True,
       'accepted_updates':int(result['updates'].sum()),'candidate_evaluations':int(result['evaluations'].sum()),
       'initial_search_mse_sum':float(result['initial_mse'].sum()),'final_search_mse_sum':float(result['final_mse'].sum()),
       'source_codes_unchanged':True,'model_weights_unchanged':True},
      'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'completed_unix':time.time()}
    rt.dump('validation/PREFLIGHT_GPU.json',report)
    del X,C,hidden
    torch.cuda.empty_cache()
    return ids,report
