"""Unchanged extracted v44 CPU audit functions, not its main."""
import csv,json,math,torch
from pathlib import Path
from audit_sources import OUT,V18,sha
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def csvread(n):return list(csv.DictReader((OUT/n).open(encoding='utf8')))
def csvwrite(n,rr):
 if not rr:return
 with (OUT/n).open('w',encoding='utf8',newline='') as fp:
  w=csv.DictWriter(fp,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def verify_delta(s,loss,rr,pred,actual,accept):
 threshold=torch.maximum(s['initial_mse'][rr]*1e-8,torch.full_like(pred,1e-14))
 assert bool(torch.isfinite(pred).all() and torch.isfinite(actual).all())
 assert bool((pred < -threshold).all())
 assert bool(((actual<0) if accept else (actual>=0)).all())
 assert bool(((actual-pred).abs()<=torch.maximum(loss[rr]*2e-6,pred.abs()*1e-3)).all())

def replay_legacy(s,o):
 code=o['codes'].long().clone();rows,d=code.shape
 counts=torch.zeros(rows,dtype=torch.long);updates=counts.clone();loss=s['initial_mse'].clone();active=torch.ones(rows,dtype=torch.bool)
 moves={m['round']:m for m in s['moves']};assert len(moves)==len(s['moves'])
 for attempt in range(1,33):
  if not active.any():break
  counts[active]+=(code[active]>0).sum(1)+(code[active]<15).sum(1)
  m=moves.get(attempt);active[:]=False
  if m is None:break
  rr,col,off=m['rows'],m['columns'],m['offsets'];assert rr.unique().numel()==rr.numel()
  assert torch.equal(m['evaluations'],counts[rr]);verify_delta(s,loss,rr,m['predicted_delta'],m['direct_delta'],True)
  code[rr,col]+=off;loss[rr]+=m['direct_delta'];updates[rr]+=1;active[rr]=True
 assert torch.equal(code.byte(),s['codes']) and torch.equal(counts,s['evaluations']) and torch.equal(updates,s['updates'])
 assert bool(torch.isclose(loss,s['final_mse'],rtol=2e-6,atol=1e-12).all())
 return counts,updates,torch.zeros_like(counts)

def replay_revision(s,o):
 code=o['codes'].long().clone();rows,d=code.shape;cap=128*d
 counts=torch.zeros(rows,dtype=torch.long);updates=counts.clone();rejections=counts.clone();status=counts.clone();loss=s['initial_mse'].clone()
 blocked=torch.zeros((rows,d,2),dtype=torch.bool)
 moves={m['round']:m for m in s['moves']};events={e['attempt']:e for e in s['rejection_events']}
 assert len(moves)==len(s['moves']) and len(events)==len(s['rejection_events'])
 attempts=set();rejection_rows=[]
 for sp in s['score_passes']:
  attempt=sp['attempt'];assert attempt not in attempts;attempts.add(attempt)
  status[(status==0)&(updates>=32)]=2;status[(status==0)&(counts>=cap)]=3
  active=status==0;legal=torch.stack((code>0,code<15),-1);eligible=(legal&~blocked).sum((1,2));remaining=cap-counts
  expected=torch.where(active,torch.minimum(eligible,remaining),torch.zeros_like(counts))
  assert torch.equal(expected,sp['evaluations_added']),'actual reevaluation accounting mismatch'
  counts+=expected;after=sp['status_after_scoring']
  assert bool(((after==0)|(after==1)|(after==2)|(after==3)).all())
  assert torch.equal(after[~active],status[~active])
  assert bool((after[active&(eligible==0)]==1).all())
  assert bool((after[active&(eligible>remaining)]==3).all())
  assert bool(((after[active&(eligible>0)&(eligible<=remaining)]==0)|(after[active&(eligible>0)&(eligible<=remaining)]==1)).all())
  m=moves.get(attempt);e=events.get(attempt)
  accepted=m['rows'] if m is not None else torch.empty(0,dtype=torch.long)
  rejected=e['rows'] if e is not None else torch.empty(0,dtype=torch.long)
  proposed=torch.cat((accepted,rejected));assert proposed.unique().numel()==proposed.numel()
  assert torch.equal(torch.sort(proposed).values,(after==0).nonzero().flatten())
  # Both events originate from the same independent-row tentative batch.
  for obj,rr,is_accept in [(m,accepted,True),(e,rejected,False)]:
   if obj is None:continue
   col,off=obj['columns'],obj['offsets'];assert bool(((off==-1)|(off==1)).all())
   kk=(off+1)//2;assert not bool(blocked[rr,col,kk].any())
   assert bool(legal[rr,col,kk].all()) and torch.equal(obj['evaluations'],counts[rr])
   verify_delta(s,loss,rr,obj['predicted_delta'],obj['direct_delta'],is_accept)
   if is_accept:
    code[rr,col]+=off;loss[rr]+=obj['direct_delta'];updates[rr]+=1;blocked[rr]=False
    assert torch.equal(obj['updates_after'],updates[rr])
   else:
    assert obj['restored'] and torch.equal(obj['old_loss'],loss[rr]) and torch.equal(obj['state_accept_count'],updates[rr])
    blocked[rr,col,kk]=True;rejections[rr]+=1
    for k,row in enumerate(rr.tolist()):
     old=float(loss[row]);pred=float(obj['predicted_delta'][k]);actual=float(obj['direct_delta'][k])
     rejection_rows.append({'attempt':attempt,'row':row,'coordinate':int(col[k]),'offset':int(off[k]),'state_accept_count':int(updates[row]),'evaluations':int(counts[row]),'predicted_delta':pred,'direct_delta':actual,'old_loss':old,'relative_predicted_delta':pred/old if old else 0,'relative_direct_delta':actual/old if old else 0,'restored':True})
  status=after.clone()
  assert bool(((code>=0)&(code<=15)).all() and (counts<=cap).all() and (updates<=32).all())
 status[(status==0)&(updates>=32)]=2;status[(status==0)&(counts>=cap)]=3
 assert set(moves)|set(events)<=attempts
 assert torch.equal(code.byte(),s['codes']) and torch.equal(counts,s['evaluations']) and torch.equal(updates,s['updates']) and torch.equal(rejections,s['rejections']) and torch.equal(status,s['status'])
 assert not bool((status==0).any()) and bool(torch.isclose(loss,s['final_mse'],rtol=2e-6,atol=1e-12).all())
 return counts,updates,rejections,rejection_rows

def ppl_audit():
 baseline=[]
 for m in ['M1','M2']:
  for ds in ['wikitext','c4_fixed_budget']:
   path=V18/f'evaluation/raw/{m}_{ds}.json';r=json.loads(path.read_text());assert r['status']=='PASS'
   for wf in r['model_weight_files']:assert sha(Path(r['model_source'])/wf['name'])==wf['sha256']
   wf=V18/r['window_csv'];wr=list(csv.DictReader(wf.open()));nt=sum(int(w['valid_tokens']) for w in wr);nll=sum(float(w['window_nll']) for w in wr)/nt
   assert nt==r['evaluated_tokens']==r['input_tokens']-1 and abs(nll-r['average_nll'])<1e-10 and abs(math.exp(nll)-r['perplexity'])<1e-10
   mf=V18/'evaluation/manifests'/('WIKITEXT_TEST_MANIFEST.json' if ds=='wikitext' else 'C4_EVAL_MANIFEST.json');assert sha(mf)==r['evaluation_manifest_sha256']
   baseline.append({'method':m,'dataset':ds,'ppl':r['perplexity'],'raw_sha256':sha(path),'windows_sha256':sha(wf),'manifest_sha256':sha(mf),'model_files_verified':True})
 dump('reports/BASELINE_AUDIT.json',{'pass':True,'baselines':baseline});comparison=[]
 for ds in ['wikitext','c4_fixed_budget']:
  p=OUT/f'R_{ds}.json'
  if not p.exists():continue
  r=json.loads(p.read_text());wr=csvread(f'R_{ds}_WINDOWS.csv');nt=sum(int(w['valid_tokens']) for w in wr);nll=sum(float(w['window_nll']) for w in wr)/nt
  assert nt==r['scored_tokens']==r['input_tokens']-1 and len(wr)==r['windows'] and abs(nll-r['nll'])<1e-10 and abs(math.exp(nll)-r['ppl'])<1e-10
  ready=read('DEPLOYMENT_LOCK.json');assert r['model_sha256']==ready['final_model_sha256']==sha(OUT/'R/checkpoint/final/model.safetensors')
  mf=V18/'evaluation/manifests'/('WIKITEXT_TEST_MANIFEST.json' if ds=='wikitext' else 'C4_EVAL_MANIFEST.json');manifest=json.loads(mf.read_text());assert sha(mf)==r['manifest_sha256']
  expected=[];prev=0
  for begin in range(0,r['input_tokens'],512):
   end=min(begin+2048,r['input_tokens']);target=end-prev;valid=target-1 if begin==0 else target
   expected.append((begin,end,valid));prev=end
   if end==r['input_tokens']:break
  assert [(int(w['begin']),int(w['end']),int(w['valid_tokens'])) for w in wr]==expected
  assert r['max_length']==2048 and r['stride']==512 and not r['add_special_tokens'] and not r['chat_template']
  assert len(manifest['token_ids'])==r['input_tokens']
  mm={b['method']:b['ppl'] for b in baseline if b['dataset']==ds}
  comparison.append({'dataset':ds,'M1_ppl':mm['M1'],'M2_ppl':mm['M2'],'R_ppl':r['ppl'],'R_minus_M1':r['ppl']-mm['M1'],'R_vs_M1_relative_percent':100*(r['ppl']/mm['M1']-1),'R_minus_M2':r['ppl']-mm['M2'],'R_vs_M2_relative_percent':100*(r['ppl']/mm['M2']-1),'improved_vs_M2':r['ppl']<mm['M2'],'scored_tokens':nt,'windows':len(wr),'raw_window_NLL_reaggregation_pass':True})
 csvwrite('reports/PPL_COMPARISON.csv',comparison);return comparison
