"""CPU-only user-requested final code, lock and budget correctness audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import csv,json,hashlib
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def th(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def read(n):
    with (OUT/n).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
lock=json.loads((OUT/'SELECTION_LOCK.json').read_text());held=json.loads((OUT/'HELD_REFERENCE_AUDIT.json').read_text())
assert held['read_after_selection_lock_unix']>lock['locked_before_held_unix']
assert sha(OUT/'PROTOCOL.json')==lock['protocol_sha256']
freeze=json.loads((OUT/'FREEZE_LOCK.json').read_text());assert sha(OUT/'run_pilot.py')==freeze['implementation_sha256']
prepared={}
for p in (OUT/'prepared').glob('*.pt'):
    info=torch.load(p,map_location='cpu',weights_only=True)
    for u in info['units']:prepared[(u['unit_id'],u['init'])]=u
assert len(prepared)==96 and len(lock['states'])==288
for s in lock['states']:
    assert sha(s['path'])==s['sha256']
    obj=torch.load(s['path'],map_location='cpu',weights_only=True);u=prepared[(s['unit_id'],s['init'])]
    assert th(obj['codes'])==s['codes_sha256'] and bool((obj['codes']<=15).all())
    for k,h in [('scales','scales_sha256'),('zeros','zeros_sha256'),('g_idx','g_idx_sha256')]:
        assert torch.equal(obj[k],u[k]) and th(obj[k])==s[h]
    assert set(obj['snapshots'])=={0,16,32,64,96,128}
    assert torch.equal(obj['snapshots'][0],u['codes']) and torch.equal(obj['snapshots'][128],obj['codes'])
rows=read('ROW_RESULTS.csv');budget=read('BUDGET_RESULTS.csv');trace=read('SEARCH_TRAJECTORIES.csv');checks=read('NUMERICAL_CHECKS.csv')
assert len(rows)==288 and len(budget)==1440
lookup={(r['unit_id'],r['init'],r['algorithm']):r for r in rows}
for r in rows:assert int(r['evals'])<=int(r['eval_cap']) and int(r['updates'])<=32
for r in checks:
    assert r['kind']!='accept_rejected'
    if r['kind']=='accepted_direct_check':assert float(r['direct_delta'])<0
    if r.get('codes_legal'):assert r['codes_legal']=='True'
for r in trace:
    if r['kind']=='common_node':
        key=(r['unit_id'],r['init'],r['algorithm']);st=lookup[key]
        assert int(r['evals'])<=int(r['allowance'])
        if int(r['budget_over_dim']) in [16,32]:assert int(r['evals'])==int(r['allowance'])
for info in sorted((OUT/'statistics').glob('*.json')):
    meta=json.loads(info.read_text());assert sha(info.with_suffix('.pt'))==meta['statistics_file_sha256']
    obj=torch.load(info.with_suffix('.pt'),map_location='cpu',weights_only=True);C=obj['objective_C'];assert C.ndim==2 and C.shape[0]==C.shape[1] and C.dtype==torch.float64
    assert th(C)==meta['C_sha256'] and torch.equal(C,C.T)
    assert obj['B'].shape==obj['C'].shape==(obj['pair_count'],2)
    for k,h in [('B','B_edges_sha256'),('C','C_edges_sha256')]:
        assert th(obj[k])==meta[h] and len(set(map(tuple,obj[k].tolist())))==obj['pair_count']
comparisons=[]
for init in ['alpha1','scalar10']:
  for unit in sorted(u for u,i in prepared if i==init):
    for obj in ['search','held']:
      for alg,basealg in [('A','init'),('B','init'),('C','init'),('C','A'),('C','B')]:
        r=lookup[(unit,init,alg)];b=float(r[f'{obj}_initial_mse']) if basealg=='init' else float(lookup[(unit,init,basealg)][f'{obj}_final_mse']);n=float(r[f'{obj}_final_mse']);eps=float(r['epsilon']);gain=(b-n)/b
        comparisons.append({'unit_id':unit,'target':r['target'],'init':init,'split':r['split'],'objective':obj,'comparison':f'{alg}_vs_{basealg}','base_mse':b,'new_mse':n,'base_SSE':b*32768,'new_SSE':n*32768,'gain_percent':gain*100,'historical_epsilon_mse':eps,'W_T_L':'W' if b-n>eps else ('L' if n-b>eps else 'T'),'held_degradation_gt10percent':obj=='held' and gain<-.1})
with (OUT/'ROW_COMPARISONS.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
result={'status':'PASS','locked_states':288,'all_budget_states':1440,'initializations':96,'every_scale_zero_gidx_unchanged':True,'all_codes_legal':True,'all_accepted_direct_deltas_strictly_negative':True,'held_reference_read_after_lock':True,'two_low_common_nodes_actual_evaluations_equal_allowances_for_every_algorithm':True,'all_graph_pairs_unique_and_B_C_counts_equal':True,'all_C_fresh_full_FP64_symmetric':True,'row_comparisons':len(comparisons),'status_counts':dict(__import__('collections').Counter(r['status'] for r in rows))}
(OUT/'OUTPUT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
