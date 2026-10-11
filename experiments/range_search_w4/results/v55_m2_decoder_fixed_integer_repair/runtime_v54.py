"""v46 R1 fixed-grid single-coordinate repair; batched output-row search.

Historical artifacts are opened read-only. No GPTQ or alpha search is imported.
All mutations and checkpoints are confined to this script's result directory.
"""
import csv, gc, hashlib, importlib.util, json, math, os, sys, threading, time, traceback
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from decoder_interface import extract_hidden, write_document
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from audit_sources import OUT, ROOT, V18, V41, FP, sha, th, pack, unpack
sys.path.insert(0,str(ROOT))
from benchmark.eval.ppl import evaluate_ppl

P=json.loads((OUT/'PROTOCOL.json').read_text())
CAL=json.loads((V18/'CALIBRATION_MANIFEST.json').read_text())
GROUPS={'qkv':['self_attn.k_proj','self_attn.v_proj','self_attn.q_proj'],'o':['self_attn.o_proj'],'up_gate':['mlp.up_proj','mlp.gate_proj'],'down':['mlp.down_proj']}
START=None; WALL_START=None; MODE='precheck'; LAYER=-1; MODULE=''; TIMINGS=[]; STATS=[]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
def dump(n,v):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def csvwrite(n,rr):
    if not rr:return
    with (OUT/n).open('w',encoding='utf8',newline='') as f:
        fields=list(dict.fromkeys(k for r in rr for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
def note(kind,**kw):
    with (OUT/'CORRECTNESS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'mode':MODE,'layer':LAYER,'module':MODULE,'kind':kind,**kw},allow_nan=False)+'\n')
def elapsed():return time.time()-WALL_START
def deadline(reserve=0):
    if elapsed()>5400-reserve:raise TimeoutError(f'v42 hard budget/reserve exhausted: {elapsed()} s, reserve {reserve}')
@contextmanager
def timer(stage):
    deadline();torch.cuda.synchronize();t=time.perf_counter()
    try:yield
    finally:
        torch.cuda.synchronize();r={'mode':MODE,'layer':LAYER,'module':MODULE,'stage':stage,'seconds':time.perf_counter()-t,'elapsed_seconds':elapsed(),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'type':'measured synchronized wall'}
        TIMINGS.append(r)
        with (OUT/'STAGE_TIMINGS.csv').open('a',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(r));
            if f.tell()==0:w.writeheader()
            w.writerow(r)
def direct(X,E):
    """FP32 output accumulation and FP64 SSE, streaming fixed input chunks."""
    with timer('direct_recompute'):
        if torch.is_tensor(X):return (X@E.float().T).double().square().mean(0)
        s=torch.zeros(E.shape[0],device='cuda',dtype=torch.float64);n=len(X);ef=E.float()
        for lo in range(0,n,4096):
            xx=torch.from_numpy(np.array(X[lo:lo+4096],copy=True)).to('cuda',dtype=torch.float32)
            y=xx@ef.T;s+=y.double().square().sum(0)
        return s/n
def batch_repair(codes,scales,gidx,W,X,C,target,save_rows=True):
    """One Python loop per complete neighborhood, vectorized independent rows."""
    global MODULE
    MODULE=target
    with timer('repair_initialization'):
        code=codes.to('cuda',dtype=torch.long).clone();SC=scales.to('cuda')[:,gidx.long().to('cuda')]
        Q=(SC*(code.float()-8)).half().float();E=Q.double()-W.to('cuda').double();pc=E@C
        rows,d=code.shape;rid=torch.arange(rows,device='cuda');diag=C.diag();updates=torch.zeros(rows,device='cuda',dtype=torch.long)
        counts=updates.clone();active=torch.ones(rows,device='cuda',dtype=torch.bool);status=torch.zeros_like(updates)
    loss=direct(X,E);initial=loss.clone();quad=(E*pc).sum(1)
    assert bool(torch.isclose(loss,quad,rtol=2e-5,atol=1e-12).all()),'initial direct/quadratic regression'
    initial_code=code.clone();moves=[];max_drift=0.;threshold=torch.maximum(initial*1e-8,torch.full_like(initial,1e-14))
    for round_idx in range(32):
        deadline(360 if MODE=='formal' else 0)
        if not bool(active.any()):break
        with timer('candidate_scoring'):
            pred=torch.full((rows,),float('inf'),device='cuda',dtype=torch.float64);best=torch.zeros(rows,device='cuda',dtype=torch.long)
            for lo in range(0,rows,256):
                rr=rid[lo:lo+256];rr=rr[active[rr]]
                if not rr.numel():continue
                cc=code[rr];valid=torch.stack((cc>0,cc<15),-1)
                deltas=torch.stack(((SC[rr]*(cc.float()-9)).half().float()-Q[rr],(SC[rr]*(cc.float()-7)).half().float()-Q[rr]),-1).double()
                delta=2*deltas*pc[rr,:,None]+deltas.square()*diag[None,:,None]
                delta.masked_fill_(~valid,float('inf'));v,k=delta.flatten(1).min(1)
                pred[rr]=v;best[rr]=k;counts[rr]+=valid.sum((1,2))
            assert bool((counts<=128*d).all()),'candidate cap'
            accept=active&(pred < -threshold);status[active&~accept]=1;active&=accept
        if not bool(accept.any()):break
        rr=rid[accept];col=best[rr]//2;off=(best[rr]%2)*2-1
        dv=(SC[rr,col]*(code[rr,col].float()+off.float()-8)).half().float()-Q[rr,col]
        proposed=E.clone();proposed[rr,col]+=dv.double()
        newloss=direct(X,proposed);actual=newloss[rr]-loss[rr]
        discrepancy=(actual-pred[rr]).abs();tol=torch.maximum(loss[rr]*2e-6,pred[rr].abs()*1e-3)
        ok=(newloss[rr]<loss[rr])&(discrepancy<=tol)
        note('accepted_direct_check',round=round_idx+1,accepted_rows=rr.numel(),max_delta_error=float(discrepancy.max()),max_error_over_allowed=float((discrepancy/tol).max()),strict_decrease_all=bool((newloss[rr]<loss[rr]).all()),pass_all=bool(ok.all()))
        if not bool(ok.all()):
            bad=~ok;torch.save({'rows':rr[bad].cpu(),'pred':pred[rr][bad].cpu(),'actual':actual[bad].cpu(),'old_loss':loss[rr][bad].cpu(),'new_loss':newloss[rr][bad].cpu(),'code':code.cpu()},OUT/f'NUMERICAL_FAILURE_{MODE}_{LAYER}_{target.replace(".","_")}.pt')
            raise ArithmeticError('direct loss descent/discrepancy failure; no tolerance change')
        with timer('accepted_update'):
            code[rr,col]+=off;Q[rr,col]=(SC[rr,col]*(code[rr,col].float()-8)).half().float()
            E=proposed;pc[rr]+=C[col,:]*dv.double()[:,None];updates[rr]+=1;loss=newloss
        moves.append({'round':round_idx+1,'rows':rr.cpu(),'columns':col.cpu(),'offsets':off.cpu(),'evaluations':counts[rr].cpu(),'predicted_delta':pred[rr].cpu(),'direct_delta':actual.cpu()})
        if round_idx+1 in [1,8,16,24,32]:
            with timer('p_drift_check'):
                drift=float((pc-E@C).abs().max());max_drift=max(max_drift,drift)
                note('p_drift',round=round_idx+1,max_abs=drift)
        print(MODE,LAYER,target,'round',round_idx+1,'rows',rr.numel(),'elapsed',round(elapsed(),2),flush=True)
    final=direct(X,E);assert bool(torch.isclose(final,loss,rtol=2e-6,atol=1e-12).all())
    assert bool((final<=initial).all()),'final increase'
    with timer('final_correctness'):
        drift=float((pc-E@C).abs().max());max_drift=max(max_drift,drift)
        assert bool(((code>=0)&(code<=15)).all())
        c=code.byte().cpu();q=Q.half().cpu();packed=pack(c);assert torch.equal(unpack(packed),c)
        sc=scales.cpu();gg=gidx.cpu();reload_q=(sc[:,gg.long()]*(unpack(packed).float()-8)).half()
        assert torch.equal(reload_q.view(torch.int16),q.view(torch.int16))
        note('final',rows=rows,input_dim=d,initial_pooled_mse=float(initial.sum()),final_pooled_mse=float(final.sum()),max_p_drift=max_drift,codes_legal=True,pack_reload_bitwise=True,scale_sha256=th(sc),g_idx_sha256=th(gg))
    status[updates==32]=2
    result={'codes':c,'packed_u4':packed,'weight':q,'scales':scales.clone(),'g_idx':gidx.clone(),'zeros':torch.full_like(scales,8),'updates':updates.cpu(),'evaluations':counts.cpu(),'initial_mse':initial.cpu(),'final_mse':final.cpu(),'status':status.cpu(),'moves':moves,'max_p_drift':max_drift}
    if save_rows:
        with (OUT/f'{MODE}_ROW_RESULTS.csv').open('a',encoding='utf8',newline='') as f:
            fields=['layer','module','row','input_dim','initial_search_mse','final_search_mse','updates','evaluations','eval_cap','status'];w=csv.DictWriter(f,fieldnames=fields)
            if f.tell()==0:w.writeheader()
            for r in range(rows):w.writerow(dict(zip(fields,[LAYER,target,r,d,float(initial[r]),float(final[r]),int(updates[r]),int(counts[r]),128*d,{1:'NO_STRICT_DESCENT',2:'UPDATE_CAP'}[int(status[r])]])))
        STATS.append({'mode':MODE,'layer':LAYER,'module':target,'rows':rows,'input_dim':d,'accepted_updates':int(updates.sum()),'modified_coordinates':int((code!=initial_code).sum()),'candidate_evaluations':int(counts.sum()),'initial_search_mse_sum':float(initial.sum()),'final_search_mse_sum':float(final.sum()),'N':len(X),'max_p_drift':max_drift,'complete':True})
        csvwrite('MODULE_RESULTS.csv',STATS)
    return result

def reproduce_v41():
    global MODE,LAYER,MODULE
    MODE='v41_regression';results=[]
    lock=json.loads((V41/'SELECTION_LOCK.json').read_text());lm={(r['unit_id'],r['init'],r['algorithm']):r for r in lock['states']}
    for p in sorted((V41/'prepared').glob('*.pt')):
        info=torch.load(p,map_location='cpu',weights_only=True);d=info['dim']
        with timer('IO'):
            with open(info['input_path'],'rb') as f:f.seek(info['search_offset_tokens']*d*2);b=f.read(32768*d*2)
            assert hashlib.sha256(b).hexdigest()==info['search_sha256']
            X=torch.from_numpy(np.frombuffer(b,dtype='<f2').copy().reshape(32768,d)).to('cuda',dtype=torch.float32)
            statp=V41/'statistics'/f'{info["target"]}.pt';side=json.loads(statp.with_suffix('.json').read_text());assert sha(statp)==side['statistics_file_sha256']
            C=torch.load(statp,map_location='cpu',weights_only=True)['objective_C'].to('cuda')
        u=info['units'];LAYER=u[0]['layer'];MODULE=info['target']
        r=batch_repair(torch.stack([a['codes'] for a in u]),torch.stack([a['scales'] for a in u]),u[0]['g_idx'],torch.stack([a['weight'] for a in u]),X,C,info['target'],False)
        for j,a in enumerate(u):
            frozen=lm[(a['unit_id'],a['init'],'A')];assert sha(frozen['path'])==frozen['sha256']
            expected=torch.load(frozen['path'],map_location='cpu',weights_only=True)
            equal=torch.equal(r['codes'][j],expected['codes'])
            assert equal and int(r['updates'][j])==frozen['updates'] and int(r['evaluations'][j])==frozen['evals'],'v41 frozen A mismatch'
            for node in [16,32,64,96,128]:
                snap=a['codes'].clone()
                for move in r['moves']:
                    hits=torch.nonzero(move['rows']==j).flatten()
                    if hits.numel():
                        k=int(hits[0])
                        if int(move['evaluations'][k])<=node*d:snap[int(move['columns'][k])]+=int(move['offsets'][k])
                assert torch.equal(snap,expected['snapshots'][node]),'v41 common-budget selection path mismatch'
            results.append({'unit_id':a['unit_id'],'init':a['init'],'codes_equal':equal,'all5_budget_selection_states_equal':True,'updates':frozen['updates'],'evaluations':frozen['evals']})
        # Small independent sequential reference uses the actual historical A code.
        if info['target'] in ['L0_up_proj_S0','L0_down_proj_S0']:
            spec=importlib.util.spec_from_file_location('v41_readonly_reference',V41/'run_pilot.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref);ref.START=time.perf_counter()
            for j in range(2):
                a=u[j];G=ref.grid(a);e=G[torch.arange(d,device='cuda'),a['codes'].long().to('cuda')].double()-a['weight'].to('cuda').double();ls=float((X@e.float()).double().square().mean())
                state={'codes':a['codes'].clone(),'p':None,'evals':0,'updates':0,'rounds':0,'loss':ls,'initial_search':ls,'status':'ACTIVE','pending_pos':0,'pending_min':1e300,'pending_best':-1}
                for node in [16,32,64,96,128]:ref.advance(state,a,X,C,G,None,node*d,info['target'],'A')
                assert torch.equal(state['codes'],r['codes'][j]),'batch/actual sequential reference mismatch'
                note('batch_reference',unit_id=a['unit_id'],init=a['init'],codes_equal=True,reference='unchanged v41 advance A')
        del X,C;torch.cuda.empty_cache()
    assert len(results)==96;dump('V41_A_REPRODUCTION.json',{'pass':True,'trajectories':results,'all96_terminal_codes_updates_evals_equal':True,'sequential_reference_trajectories':4,'inputs':'read-only fixed v41 search caches; no held reads'})

def model_load():
    with timer('model_load'):
        return AutoModelForCausalLM.from_pretrained(str(V18/'M1/checkpoint/final'),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()
def zero_model_check(model):
    global MODULE
    ids=torch.tensor(CAL['samples'][0]['token_ids'][:64],device='cuda')[None]
    with timer('zero_model_output'):
        with torch.inference_mode():before=model(ids,use_cache=False).logits.cpu()
    with timer('unpack_pack_zero_commit'):
      for li in range(24):
        src=torch.load(OUT/f'original_codes/layer_{li:02d}.pt',map_location='cpu',weights_only=True)
        for name,s in src.items():
            cc=unpack(pack(s['codes']));assert torch.equal(cc,s['codes'])
            assert torch.equal(s['packed_u4'],pack(cc))
            q=(s['scales'][:,s['g_idx'].long()]*(cc.float()-8)).half();m=model.model.layers[li].get_submodule(name)
            assert torch.equal(m.weight.detach().cpu().view(torch.int16),q.view(torch.int16))
            with torch.no_grad():m.weight.copy_(q.to('cuda'))
    with timer('zero_model_output'):
        with torch.inference_mode():after=model(ids,use_cache=False).logits.cpu()
    assert torch.equal(before,after),'zero model output differs'
    note('zero_model',all168_weights_bitwise=True,all_grid_parameters_unchanged=True,logits_bitwise=True,max_abs_logit_difference=0.,rtol=1e-5,atol=1e-6)
    return ids
def embed(model):
    with timer('initial_embedding_capture'):
        hidden=np.empty((128,2048,896),dtype=np.float16)
        with torch.inference_mode():
          for i,s in enumerate(CAL['samples']):write_document(hidden,i,model.model.embed_tokens(torch.tensor(s['token_ids'],device='cuda')[None]))
        return hidden
def capture(model,li,hidden,group):
    layer=model.model.layers[li];names=GROUPS[group];dim=layer.get_submodule(names[0]).in_features
    with timer('dependency_input_capture'):
        X=np.empty((128,2048,dim),dtype=np.float16);positions=torch.arange(2048,device='cuda')[None];active={};handles=[]
        def hook(module,args):
            if 'x' not in active:active['x']=args[0].detach();write_document(X,active['doc'],args[0])
            else:assert torch.equal(active['x'],args[0]),'sibling input mismatch'
            active['count']+=1
        try:
            if group!='qkv':handles=[layer.get_submodule(n).register_forward_pre_hook(hook) for n in names]
            with torch.inference_mode():
              for i in range(128):
                h=torch.from_numpy(hidden[i].copy()).cuda()[None];active.clear();active.update(doc=i,count=0)
                if group=='qkv':write_document(X,i,layer.input_layernorm(h))
                else:
                    layer(h,attention_mask=None,position_ids=positions,cache_position=positions[0],position_embeddings=model.model.rotary_emb(h,positions),use_cache=False)
                    assert active['count']==len(names)
        finally:
            for handle in handles:handle.remove()
        return X.reshape(-1,dim)
def build_C(X):
    with timer('C_construction'):
        d=X.shape[1];C=torch.zeros((d,d),device='cuda',dtype=torch.float64)
        for lo in range(0,len(X),4096):
            deadline();xx=torch.from_numpy(X[lo:lo+4096].copy()).to('cuda',dtype=torch.float64);C.add_(xx.T@xx)
        C/=len(X);C=(C+C.T)*.5
        assert bool(torch.isfinite(C).all());return C
def advance_block(model,li,hidden):
    with timer('down_commit_prefix_refresh'):
        out=np.empty_like(hidden);positions=torch.arange(2048,device='cuda')[None];layer=model.model.layers[li]
        with torch.inference_mode():
          for i in range(128):
            h=torch.from_numpy(hidden[i].copy()).cuda()[None]
            result=layer(h,attention_mask=None,position_ids=positions,cache_position=positions[0],position_embeddings=model.model.rotary_emb(h,positions),use_cache=False)
            states=extract_hidden(result,expected_shape=(1,hidden.shape[1],hidden.shape[2]))
            write_document(out,i,states)
        return out
def run_layer(model,li,hidden):
    global LAYER,MODULE
    LAYER=li;src=torch.load(OUT/f'original_codes/layer_{li:02d}.pt',map_location='cpu',weights_only=True)
    directory=OUT/MODE/'groups';directory.mkdir(parents=True,exist_ok=True)
    for group,names in GROUPS.items():
        MODULE=group;checkpoint=directory/f'layer_{li:02d}_{group}.pt'
        if checkpoint.exists():
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['committed'] and ck['protocol_sha256']==sha(OUT/'PROTOCOL.json')
            for n,s in ck['modules'].items():
                with torch.no_grad():model.model.layers[li].get_submodule(n).weight.copy_(s['weight'].cuda())
            print('resume_skip_committed',MODE,li,group,flush=True);continue
        X=capture(model,li,hidden,group)
        with timer('input_hash'):xh=hashlib.sha256(X.tobytes()).hexdigest()
        C=build_C(X);results={}
        with safe_open(str(FP/'model.safetensors'),framework='pt',device='cpu') as pristine:
          for name in names:
            s=src[name];W=pristine.get_tensor(f'model.layers.{li}.{name}.weight').half()
            result=batch_repair(s['codes'],s['scales'],s['g_idx'],W,X,C,name)
            assert torch.equal(result['zeros'],s['zeros']) and torch.equal(result['scales'],s['scales']) and torch.equal(result['g_idx'],s['g_idx'])
            results[name]=result
        MODULE=group
        with timer('whole_group_commit'):
            for name,r in results.items():
                m=model.model.layers[li].get_submodule(name)
                with torch.no_grad():m.weight.copy_(r['weight'].cuda())
                assert torch.equal(m.weight.detach().cpu(),r['weight'])
        with timer('packing_checkpoint_IO'):
            tmp=checkpoint.with_suffix('.pt.tmp');torch.save({'mode':MODE,'layer':li,'group':group,'input_sha256':xh,'C_sha256':th(C.cpu()),'protocol_sha256':sha(OUT/'PROTOCOL.json'),'modules':results,'committed':True,'elapsed_seconds':elapsed(),'immutable_grid':True},tmp);tmp.replace(checkpoint)
            dump(f'{MODE}/COMMIT_STATUS.json',{'last_committed_layer':li,'group':group,'group_checkpoint':str(checkpoint),'sha256':sha(checkpoint),'input_sha256':xh,'elapsed_seconds':elapsed()})
        del X,C,results;gc.collect();torch.cuda.empty_cache()
    MODULE='block';return advance_block(model,li,hidden)

class CapturePPL(torch.nn.Module):
    def __init__(self,model,n):super().__init__();self.inner=model;self.n=n;self.rows=[]
    def forward(self,*,input_ids,labels,use_cache=False,**kw):
        deadline();o=self.inner(input_ids=input_ids,labels=labels,use_cache=use_cache,**kw);idx=len(self.rows);begin=idx*512;end=min(begin+2048,self.n);valid=int((labels[:,1:]!=-100).sum());loss=float(o.loss.float())
        self.rows.append({'window_index':idx,'begin':begin,'end':end,'valid_tokens':valid,'window_loss':loss,'window_nll':loss*valid});return o
def evaluate(model):
    global MODE,MODULE
    MODE='evaluation'
    for ds,manifest,key in [('wikitext','WIKITEXT_TEST_MANIFEST.json','v16_wikitext_manifest_sha256'),('c4_fixed_budget','C4_EVAL_MANIFEST.json','v16_c4_manifest_sha256')]:
        path=V18/'evaluation/manifests'/manifest;assert sha(path)==P['v18']['source_protocol'][key]
        tokens=json.loads(path.read_text())['token_ids'];ids=torch.tensor(tokens)[None];wrapped=CapturePPL(model,len(tokens));MODULE=ds
        assert not (OUT/f'R1_{ds}.json').exists(),'one evaluation only'
        with timer('PPL_evaluation'):ppl,nll,n=evaluate_ppl(wrapped,ids,max_length=2048,stride=512,progress_every=25)
        assert n==len(tokens)-1==sum(r['valid_tokens'] for r in wrapped.rows)
        assert len(wrapped.rows)==max(1,math.ceil((len(tokens)-2048)/512)+1)
        assert abs(nll-sum(r['window_nll'] for r in wrapped.rows)/n)<1e-10
        csvwrite(f'R1_{ds}_WINDOWS.csv',wrapped.rows)
        dump(f'R1_{ds}.json',{'status':'PASS','ppl':ppl,'nll':nll,'scored_tokens':n,'input_tokens':len(tokens),'manifest_sha256':sha(path),'token_ids_sha256_i64le':hashlib.sha256(np.asarray(tokens,dtype='<i8').tobytes()).hexdigest(),'model_sha256':sha(OUT/'R1/checkpoint/final/model.safetensors'),'max_length':2048,'stride':512,'add_special_tokens':False,'chat_template':False,'representation':'dense-dequant FP16','windows':len(wrapped.rows)})
def final_save_check(model,ids):
    global MODE,MODULE
    MODULE='final_model'
    with timer('final_save'):
        dst=OUT/'R1/checkpoint/final';dst.mkdir(parents=True,exist_ok=True);model.save_pretrained(dst,safe_serialization=True,max_shard_size='5GB')
        with torch.inference_mode():before=model(ids,use_cache=False).logits.cpu()
    with timer('pack_reload_all_grid_check'):
        with safe_open(str(dst/'model.safetensors'),framework='pt',device='cpu') as sf:
          for ck in sorted((OUT/'formal/groups').glob('*.pt')):
            state=torch.load(ck,map_location='cpu',weights_only=False)
            for name,s in state['modules'].items():
                orig=torch.load(OUT/f'original_codes/layer_{state["layer"]:02d}.pt',map_location='cpu',weights_only=True)[name]
                for k in ['scales','zeros','g_idx']:assert torch.equal(s[k],orig[k])
                c=unpack(s['packed_u4']);assert torch.equal(c,s['codes'])
                q=(s['scales'][:,s['g_idx'].long()]*(c.float()-8)).half()
                saved=sf.get_tensor(f'model.layers.{state["layer"]}.{name}.weight')
                assert torch.equal(q.view(torch.int16),saved.view(torch.int16))
        dump('CHECKPOINT_LOCK.json',{'final_model_sha256':sha(dst/'model.safetensors'),'config_sha256':sha(dst/'config.json'),'protocol_sha256':sha(OUT/'PROTOCOL.json'),'group_checkpoints':[{'path':str(p),'sha256':sha(p)} for p in sorted((OUT/'formal/groups').glob('*.pt'))],'locked_before_PPL_unix':time.time()})
    del model;gc.collect();torch.cuda.empty_cache()
    with timer('reload'):
        loaded=AutoModelForCausalLM.from_pretrained(str(dst),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()
        with torch.inference_mode():after=loaded(ids,use_cache=False).logits.cpu()
        assert torch.allclose(before,after,rtol=1e-5,atol=1e-6)
        note('final_reload',pass_all=True,max_abs_logit_difference=float((before-after).abs().max()),rtol=1e-5,atol=1e-6,all168_code_grid_checks=True)
    return loaded
def main():
    global START,WALL_START,MODE,LAYER,MODULE
    assert json.loads((OUT/'SOURCE_AUDIT.json').read_text())['pass']
    resume='--resume' in sys.argv
    if resume:
        old=json.loads((OUT/'GPU_START.json').read_text());WALL_START=old['unix']
        assert old['implementation_sha256']==sha(__file__) and old['protocol_sha256']==sha(OUT/'PROTOCOL.json'),'resume frozen identity mismatch'
        assert not (OUT/'CHECKPOINT_LOCK.json').exists(),'locked/evaluated final model must not be repaired again'
        if (OUT/'MODULE_RESULTS.csv').exists():
            # Retain only statistics backed by completed group checkpoints.
            for r in csv.DictReader((OUT/'MODULE_RESULTS.csv').open()):
                group=next(g for g,names in GROUPS.items() if r['module'] in names)
                if (OUT/r['mode']/f'groups/layer_{int(r["layer"]):02d}_{group}.pt').exists():
                    STATS.append({k:(v if k in ['mode','module'] else (v=='True' if k=='complete' else float(v) if k in ['initial_search_mse_sum','final_search_mse_sum','max_p_drift'] else int(v))) for k,v in r.items()})
        dump('RESUME_LOG.json',{'unix':time.time(),'original_gpu_start_unix':WALL_START,'elapsed_seconds':elapsed(),'completed_group_repair_is_skipped':True})
    else:
        assert not (OUT/'GPU_START.json').exists(),'preserve existing run; use --resume with identical frozen implementation'
        WALL_START=time.time()
        dump('GPU_START.json',{'unix':WALL_START,'hard_cap_seconds':5400,'pid':os.getpid(),'implementation_sha256':sha(__file__),'protocol_sha256':sha(OUT/'PROTOCOL.json')})
    START=time.perf_counter();deadline()
    def hard_stop():
        dump('RUN_STATUS.json',{'status':'INCOMPLETE_HARD_CAP','elapsed_seconds':elapsed(),'category':'D','complete_model':False});os._exit(124)
    threading.Timer(5400-elapsed(),hard_stop).start()
    with timer('warmup'):z=torch.zeros((64,64),device='cuda');z@z
    if not (OUT/'V41_A_REPRODUCTION.json').exists():reproduce_v41()
    if resume and (OUT/'COST_PROJECTION.json').exists():
        projection=json.loads((OUT/'COST_PROJECTION.json').read_text());assert projection['go'],'pilot budget stop is terminal'
        pilot_seconds=projection['measured_complete_block0_seconds'];reserve=360
    else:
        run_pilot_and_project()
        projection=json.loads((OUT/'COST_PROJECTION.json').read_text())
        if not projection['go']:
            dump('RUN_STATUS.json',{'status':'PILOT_COMPLETE_BUDGET_STOP','category':'D','elapsed_seconds':elapsed(),'complete_model':False,'PPL_run':False});return
        pilot_seconds=projection['measured_complete_block0_seconds'];reserve=360
    MODE='formal';LAYER=-1;model=model_load();ids=torch.tensor(CAL['samples'][0]['token_ids'][:64],device='cuda')[None];hidden=embed(model)
    for li in range(24):
        remaining=24-li
        if elapsed()+remaining*pilot_seconds*1.15+reserve>5400:raise TimeoutError('remaining actual runtime no longer supports complete model and eval reserve')
        hidden=run_layer(model,li,hidden)
        dump('formal/PREFIX_STATUS.json',{'completed_layer':li,'hidden_sha256':hashlib.sha256(hidden.tobytes()).hexdigest(),'elapsed_seconds':elapsed()})
    assert len([s for s in STATS if s['mode']=='formal'])==168
    model=final_save_check(model,ids);evaluate(model)
    dump('RUN_STATUS.json',{'status':'COMPLETE','elapsed_seconds':elapsed(),'complete_model':True,'modules':168,'output_rows':304128,'PPL_run':True,'gpu_name':torch.cuda.get_device_name(),'torch':torch.__version__})
def run_pilot_and_project():
    global MODE,LAYER
    MODE='pilot';LAYER=0;model=model_load();ids=zero_model_check(model);hidden=embed(model)
    torch.cuda.reset_peak_memory_stats();pilot_start=time.perf_counter();hidden=run_layer(model,0,hidden);pilot_seconds=time.perf_counter()-pilot_start
    assert len([s for s in STATS if s['mode']=='pilot'])==7
    # All 24 blocks have audited identical actual matrix shapes; scale complete block timing.
    actual=list(csv.DictReader((OUT/'SOURCE_MODULES.csv').open()))
    shape0=sorted((r['module'],r['rows'],r['input_dim']) for r in actual if r['layer']=='0')
    assert all(sorted((r['module'],r['rows'],r['input_dim']) for r in actual if r['layer']==str(i))==shape0 for i in range(24))
    predicted=pilot_seconds*24*1.35+30;reserve=360;available=5400-elapsed();go=predicted+reserve<=available
    dump('COST_PROJECTION.json',{'type':'projection, not measured full-model runtime','measured_complete_block0_seconds':pilot_seconds,'actual_block_shapes_identical':True,'blocks':24,'conservative_multiplier':1.35,'model_load_embedding_overhead_seconds':30,'predicted_formal_repair_seconds':predicted,'save_reload_and_PPL_reserved_seconds':reserve,'remaining_hard_budget_seconds':available,'go':go,'measured_pilot_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'measured_pilot_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    print('PILOT_COMPLETE_COST_GATE',go,'pilot_seconds',pilot_seconds,'predicted',predicted,'remaining',available,flush=True)
    del model,hidden;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
    try:main()
    except Exception as ex:
        traceback.print_exc();dump('RUN_STATUS.json',{'status':'FAILED_STOP','category':'D','error':repr(ex),'traceback':traceback.format_exc(),'elapsed_seconds':elapsed() if WALL_START else 0,'complete_model':False,'PPL_run':False})
    finally:os._exit(0)
