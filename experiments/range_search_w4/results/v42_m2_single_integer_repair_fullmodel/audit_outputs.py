"""CPU-only persisted checkpoint/move replay and historical immutability audit."""
import csv,json
import torch
from audit_sources import OUT,V18,V41,sha,th,pack,unpack,dump
torch.set_num_threads(4)
def main():
 source=json.loads((OUT/'SOURCE_AUDIT.json').read_text());start=json.loads((OUT/'GPU_START.json').read_text())
 assert sha(OUT/'run_v42.py')==start['implementation_sha256']==json.loads((OUT/'IMPLEMENTATION_LOCK.json').read_text())['implementation_sha256']
 assert sha(OUT/'PROTOCOL.json')==start['protocol_sha256']
 for s in source['sources']:assert sha(s['source'])==s['source_sha256'];assert sha(s['derived'])==s['derived_sha256']
 assert sha(V18/'M2/checkpoint/final/model.safetensors')==source['model_sha256']
 assert sha(V41/'PROTOCOL.json')==source['v41_protocol_sha256'] and sha(V41/'run_pilot.py')==source['v41_implementation_sha256']
 status=json.loads((OUT/'RUN_STATUS.json').read_text());phase='formal' if status['status']=='COMPLETE' else 'pilot'
 records=[];module_rows=[]
 for path in sorted((OUT/phase/'groups').glob('*.pt')):
  ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['committed'] and ck['protocol_sha256']==sha(OUT/'PROTOCOL.json')
  orig=torch.load(OUT/f'original_codes/layer_{ck["layer"]:02d}.pt',map_location='cpu',weights_only=True)
  for name,s in ck['modules'].items():
   o=orig[name];c=s['codes'];assert bool((c<=15).all());assert torch.equal(unpack(s['packed_u4']),c) and torch.equal(pack(c),s['packed_u4'])
   for key in ['scales','zeros','g_idx']:assert torch.equal(s[key],o[key])
   dq=(s['scales'][:,s['g_idx'].long()]*(c.float()-8)).half();assert torch.equal(dq.view(torch.int16),s['weight'].view(torch.int16))
   assert bool((s['final_mse']<=s['initial_mse']).all())
   replay=o['codes'].long().clone();rows,dim=replay.shape;counts=torch.zeros(rows,dtype=torch.long);updates=counts.clone();loss=s['initial_mse'].clone();active=torch.ones(rows,dtype=torch.bool)
   moves={m['round']:m for m in s['moves']}
   for roundnum in range(1,33):
    if not active.any():break
    counts[active]+=((replay[active]>0).sum(1)+(replay[active]<15).sum(1))
    m=moves.get(roundnum);active[:]=False
    if m is None:break
    rr,col,off=m['rows'],m['columns'],m['offsets'];assert rr.unique().numel()==rr.numel() and bool(((off==-1)|(off==1)).all())
    assert torch.equal(m['evaluations'],counts[rr])
    pred,actual=m['predicted_delta'],m['direct_delta'];threshold=torch.maximum(s['initial_mse'][rr]*1e-8,torch.full_like(pred,1e-14))
    assert bool((pred < -threshold).all() and (actual<0).all())
    assert bool(((actual-pred).abs()<=torch.maximum(loss[rr]*2e-6,pred.abs()*1e-3)).all())
    replay[rr,col]+=off;assert bool(((replay>=0)&(replay<=15)).all());loss[rr]+=actual;updates[rr]+=1;active[rr]=True
   assert torch.equal(replay.byte(),c) and torch.equal(counts,s['evaluations']) and torch.equal(updates,s['updates'])
   assert bool((counts<=128*dim).all() and (updates<=32).all())
   assert torch.isclose(loss,s['final_mse'],rtol=2e-6,atol=1e-12).all()
   records.append({'layer':ck['layer'],'module':name,'rows':rows,'input_dim':dim,'codes_sha256':th(c),'packed_sha256':th(s['packed_u4']),'grid_unchanged':True,'reload_bitwise':True,'all_moves_replayed':True,'actual_budget_verified':True,'updates':int(updates.sum()),'evaluations':int(counts.sum())})
   module_rows.append(rows)
  records.append({'checkpoint':str(path),'sha256':sha(path),'input_sha256':ck['input_sha256'],'C_sha256':ck['C_sha256'],'committed':True})
 if phase=='pilot':assert len(module_rows)==7 and sum(module_rows)==12672
 else:assert len(module_rows)==168 and sum(module_rows)==304128
 assert status['elapsed_seconds']<5400
 if status['status']=='PILOT_COMPLETE_BUDGET_STOP':
  assert not (OUT/'R').exists() and not list(OUT.glob('R_*WINDOWS.csv'))
 dump('OUTPUT_AUDIT.json',{'pass':True,'phase':phase,'module_count':len(module_rows),'output_rows':sum(module_rows),'persisted_codes_and_grid_verified':True,'all_move_sequences_replayed':True,'per_row_actual_evals_and_caps_verified':True,'historical_source_hashes_unchanged':True,'implementation_protocol_frozen_hashes_unchanged':True,'records':records})
 print('OUTPUT_AUDIT_PASS',phase,len(module_rows),sum(module_rows))
if __name__=='__main__':main()
