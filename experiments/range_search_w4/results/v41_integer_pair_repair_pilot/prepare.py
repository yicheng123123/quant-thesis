"""CPU-only provenance preparation. No quantizer/model imports or CUDA calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RES = OUT.parent
V24 = RES/'range_w4_structured_group_e2_v24'
V30 = RES/'range_w4_weight_error_regularization_v30'
V23 = RES/'range_w4_block_objective_e1_v23'
torch.set_num_threads(2)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def th(t): return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def dump(n,x): (OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def readcsv(p):
    with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    start=time.perf_counter(); sources=[]; units=[]; audits=[]
    table=readcsv(V30/'SEARCH_CANDIDATE_TABLE.csv')
    eps={r['unit_id']:float(r['unit_tie_epsilon_mse']) for r in readcsv(V24/'HELD_RESULTS.csv') if r['method']=='M2_original_scalar10'}
    model=ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors'
    assert sha(model)==table[0]['pristine_checkpoint_file_sha256']
    sources.append({'path':str(model),'sha256':table[0]['pristine_checkpoint_file_sha256'],'role':'pristine BF16 checkpoint cast to frozen FP16'})
    (OUT/'prepared').mkdir(exist_ok=True)
    for layer in [0,12,23]:
      for proj in ['up_proj','down_proj']:
        dim=896 if proj=='up_proj' else 4864
        cache=(V23 if proj=='up_proj' else V24)/'cache'/f'input_L{layer}_{proj}.f16'
        relevant=[r for r in table if int(r['layer'])==layer and r['projection']==proj]
        assert cache.stat().st_size==64*2048*dim*2
        assert sha(cache)==relevant[0]['input_file_sha256']
        sources.append({'path':str(cache),'sha256':relevant[0]['input_file_sha256'],'role':'historical fixed M2-prefix FP16 inputs'})
        with safe_open(str(model),framework='pt',device='cpu') as f: w=f.get_tensor(f'model.layers.{layer}.mlp.{proj}.weight').half().contiguous()
        assert th(w)==relevant[0]['pristine_weight_tensor_sha256']
        for split in ['S0','S1']:
          bank={}; selected=[]
          for r in [r for r in relevant if r['split_id']==split]:
            stem=r['source_candidate_stem']
            if stem not in bank:
                p=V24/'raw/full_candidates'/f'{stem}.pt'; m=p.with_suffix('.json')
                assert sha(p)==r['candidate_tensor_file_sha256'] and sha(m)==r['candidate_metadata_file_sha256']
                q=torch.load(p,map_location='cpu',weights_only=True); meta=json.loads(m.read_text(encoding='utf-8-sig'))
                assert set(q)=={'weight_fp16','scales','zeros','g_idx'}
                for k,hk in [('weight_fp16','weight_sha256'),('scales','scales_sha256'),('zeros','zeros_sha256'),('g_idx','g_idx_sha256')]:assert th(q[k])==meta[hk]
                assert bool((q['zeros']==8).all()) and bool((q['scales']>0).all())
                assert torch.equal(q['g_idx'],torch.arange(dim,dtype=torch.int32)//128)
                # Exact inverse lookup in the known native grid, not weight/scale rounding.
                grid=(q['scales'][:,:,None]*(torch.arange(16,dtype=torch.float32)-8)).half()
                assert bool((grid[:,:,1:].view(torch.int16)!=grid[:,:,:-1].view(torch.int16)).all())
                vals=grid[:,q['g_idx'].long(),:]
                matches=vals.view(torch.int16)==q['weight_fp16'].view(torch.int16)[:,:,None]
                counts=matches.sum(-1)
                assert bool((counts==1).all()),f'grid not uniquely invertible: {stem}'
                codes=matches.to(torch.int32).argmax(-1).to(torch.uint8)
                reconstructed=vals.gather(2,codes.long()[:,:,None]).squeeze(2)
                assert torch.equal(reconstructed.view(torch.int16),q['weight_fp16'].view(torch.int16))
                q['codes']=codes;bank[stem]=q
                audits.append({'stem':stem,'coordinates':codes.numel(),'unique_exact_grid_matches':int((counts==1).sum()),'original_codes_saved':False,'code_provenance':'unique bitwise inverse of all 16 native FP32 scale*(code-8) then FP16 grid values','zero_update_bitwise_equal':True,'source_sha256':sha(p)})
          for j,row in enumerate([0,224,448,672]):
            uid=f'L{layer}_{proj}_R{row}_{split}'
            rr=[r for r in relevant if r['unit_id']==uid]
            winner=min(rr,key=lambda r:(float(r['search_output_mse']),-float(r['alpha'])))
            assert winner['is_original_search_selected']=='True'
            for init,r in [('alpha1',next(r for r in rr if float(r['alpha'])==1)),('scalar10',winner)]:
                q=bank[r['source_candidate_stem']]
                selected.append({'unit_id':uid,'layer':layer,'projection':proj,'split':split,'row':row,'init':init,'alpha':float(r['alpha']),'epsilon':eps[uid],'historical_search_mse':float(r['search_output_mse']), 'weight':w[row].clone(),'q':q['weight_fp16'][j].clone(),'codes':q['codes'][j].clone(),'scales':q['scales'][j].clone(),'zeros':q['zeros'][j].clone(),'g_idx':q['g_idx'].clone(),'source_stem':r['source_candidate_stem']})
          token_start=0 if split=='S0' else 32*2048
          with open(cache,'rb') as f:
              f.seek(token_start*dim*2);block=f.read(16*2048*dim*2)
          search_hash=hashlib.sha256(block).hexdigest()
          assert search_hash==next(r for r in relevant if r['split_id']==split)['search_input_bytes_sha256']
          target=f'L{layer}_{proj}_{split}'
          torch.save({'target':target,'dim':dim,'input_path':str(cache),'search_offset_tokens':token_start,'held_offset_tokens':token_start+16*2048,'search_sha256':search_hash,'units':selected},OUT/'prepared'/f'{target}.pt')
          units.append({'target':target,'prepared_sha256':sha(OUT/'prepared'/f'{target}.pt'),'input_path':str(cache),'search_sha256':search_hash})
          print('prepared',target,flush=True)
    for p in [V30/'SEARCH_CANDIDATE_TABLE.csv',V24/'HELD_RESULTS.csv',V24/'PROTOCOL.json',ROOT/'.venv-gptq/Lib/site-packages/gptqmodel/quantization/quantizer.py',ROOT/'.venv-gptq/Lib/site-packages/gptqmodel/quantization/gptq.py',RES/'range_w4_m2_runtime_v17_3/optimized_native_m2.py']:
        sources.append({'path':str(p),'sha256':sha(p),'role':'frozen semantics/provenance; HELD_RESULTS used only for historical epsilon'})
    dump('SOURCE_AUDIT.json',{'status':'PASS','candidate_banks':len(audits),'row_split_initializations':96,'audits':audits,'sources':sources,'prepared':units,'cpu_prep_wall_seconds':time.perf_counter()-start,'original_integer_code_history_claim':False,'exact_effective_codes':'All cached weights uniquely identify legal native codes bitwise; forward quantizer code trajectory was not stored and is not claimed recovered.'})
if __name__=='__main__':main()
