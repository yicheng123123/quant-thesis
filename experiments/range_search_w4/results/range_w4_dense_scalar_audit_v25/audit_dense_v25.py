"""v25 CPU artifact audit. Never invokes model forwards/GPTQ/CUDA/PPL runners."""
import csv, hashlib, json, math, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
import torch
from safetensors import safe_open
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
RESULTS=OUT.parent
V18=RESULTS/'range_w4_fullmodel_v18'
V23=RESULTS/'range_w4_block_objective_e1_v23'
V24=RESULTS/'range_w4_structured_group_e2_v24'
V3=RESULTS/'range_w4_omniquant_finalize_v21_5'
REV='7ae557604adf67be50417f59c2c2f167def9a775'
SOURCE_HASHES={};CHECKS={};EXPECTED_SOURCES={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def track(p):
 p=Path(p).resolve();key=str(p.relative_to(ROOT));value=sha(p)
 if key in EXPECTED_SOURCES and value!=EXPECTED_SOURCES[key]:raise RuntimeError('Frozen source changed: '+key)
 SOURCE_HASHES[key]=value;return p
def read(p):return json.loads(track(p).read_text('utf-8-sig'))
def rows(p):
 with track(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8');tmp.replace(p)
def table(name,data):
 p=OUT/name;tmp=p.with_suffix('.csv.tmp');fields=list(dict.fromkeys(k for r in data for k in r))
 with tmp.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in data:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
 tmp.replace(p)
def check(name,condition):
 CHECKS[name]=bool(condition)
 if not condition:raise AssertionError(name)
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-15)
def thash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def select(bank,loss,grid):return min((r for r in bank if r['alpha'] in grid),key=lambda r:(r[loss],-r['alpha']))
def aggregate(units,label):
 base=math.fsum(r['scalar10_held_sse'] for r in units);new=math.fsum(r['dense46_held_sse'] for r in units)
 eps=math.fsum(r['epsilon_sse'] for r in units);wtl=Counter(r['outcome'] for r in units)
 o10=math.fsum(r['scalar10_oracle_held_sse'] for r in units);o46=math.fsum(r['dense46_oracle_held_sse'] for r in units)
 worst=min(units,key=lambda r:r['held_gain_fraction'])
 return dict(scope=label,units=len(units),scalar10_held_sse=base,dense46_held_sse=new,delta_sse=base-new,
  pooled_gain_fraction=(base-new)/base,epsilon_sse=eps,improved_beyond_epsilon=base-new>eps,
  wins=wtl['win'],ties=wtl['tie'],losses=wtl['loss'],median_unit_gain_fraction=statistics.median(r['held_gain_fraction'] for r in units),
  mean_unit_gain_fraction=statistics.mean(r['held_gain_fraction'] for r in units),worst_unit=worst['unit_id'],worst_gain_fraction=worst['held_gain_fraction'],
  scalar10_oracle_sse=o10,dense46_oracle_sse=o46,scalar10_selection_regret_sse=base-o10,dense46_selection_regret_sse=new-o46,
  scalar10_selection_regret_fraction=(base-o10)/o10,dense46_selection_regret_fraction=(new-o46)/o46,oracle_capacity_gain_fraction=(o10-o46)/o10,
  search_scalar10_sse=math.fsum(r['scalar10_search_mse']*r['elements'] for r in units),search_dense46_sse=math.fsum(r['dense46_search_mse']*r['elements'] for r in units),
  changed_alpha_units=sum(r['alpha10']!=r['alpha46'] for r in units))
def scalar_audit():
 p=read(V24/'PROTOCOL.json');docs=read(V24/'E2_DOCUMENT_MANIFEST.json');p18=read(V18/'PROTOCOL.json');cal=read(V18/'CALIBRATION_MANIFEST.json');lock=read(V24/'SEARCH_SELECTION_LOCK.json')
 for file,key in [('SEARCH_RESULTS.csv','search_results_sha256'),('E2_DOCUMENT_MANIFEST.json','e2_document_manifest_sha256'),('PROTOCOL.json','protocol_sha256')]:check('lock_'+file,sha(V24/file)==lock[key])
 check('same_revision',p['model']['revision']==p18['model']['revision']==REV)
 for rel,expected in p['frozen_source_sha256'].items():
  source=ROOT/rel;key=str(source.relative_to(ROOT))
  if key not in SOURCE_HASHES:track(source)
  check('v24_frozen_source_'+rel,SOURCE_HASHES[key]==expected)
 g10=p['methods']['M2_original_scalar']['alpha_grid'];g46=sorted(set(g10+[.1+.9*j/37 for j in range(1,37)]))
 check('original46_grid',len(g46)==46 and set(g10).issubset(g46))
 samples=docs['samples'];check('token_sequence_hashes_v24',all(r['token_sha256'] in {hashlib.sha256(json.dumps(r['token_ids']).encode()).hexdigest(),hashlib.sha256(json.dumps(r['token_ids'],separators=(',',':')).encode()).hexdigest()} for r in samples))
 check('token_sequence_hashes_formal',all(hashlib.sha256(json.dumps(r['token_ids']).encode()).hexdigest()==r['token_sha256'] for r in cal['samples']))
 check('ordered_v24_i64le_hash',thash(torch.tensor([t for r in samples for t in r['token_ids']],dtype=torch.int64))==docs['ordered_token_ids_sha256_i64le'])
 check('frozen_split_labels',[r['split'] for r in samples]==['S0_search']*16+['S0_held']*16+['S1_search']*16+['S1_held']*16)
 check('64_complete_disjoint_documents',len({r['article_sha256'] for r in samples})==64 and all(len(r['token_ids'])==2048 for r in samples))
 check('no_formal_calibration_overlap',not({r['article_sha256'] for r in samples}&{r['article_sha256'] for r in cal['samples']}))
 historical=read(V24/'correctness_checks.json');check('v24_historical_correctness',historical['status']=='PASS');read(V18/'correctness_checks.json')
 sc=[r for r in rows(V24/'SEARCH_RESULTS.csv') if r['candidate_method']=='scalar_dense_bank'];lookup={(r['unit_id'],float(r['alpha'])):r for r in sc}
 held_csv={(r['unit_id'],r['method']):r for r in rows(V24/'HELD_RESULTS.csv')};oracle_csv={r['unit_id']:r for r in rows(V24/'ORACLE_CAPACITY_ANALYSIS.csv')}
 read(V24/'HELD_RESULTS_RAW.json');read(V24/'E2_SUMMARY.json');read(V24/'RUN_REPORT.json');track(V24/'REPORT.md')
 check('2208_unique_search_positions',len(sc)==len(lookup)==2208)
 bank=defaultdict(list);parameters={};meta_groups=defaultdict(list);execution_rows=[]
 base_model=track(ROOT/'.hf_cache/hub'/f'models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/{REV}/model.safetensors')
 with safe_open(str(base_model),framework='pt',device='cpu') as f:original={(li,pr):f.get_tensor(f'model.layers.{li}.mlp.{pr}.weight').float() for li in p['targets']['layers'] for pr in p['targets']['projections']}
 for path in sorted((V24/'raw/full_candidates').glob('*.json')):
  m=read(path);li,pr,sp,a=m['layer'],m['projection'],m['split'],float(m['alpha']);h=read(V24/'raw/held_full_scores'/path.name)
  lo,hi=p['documents']['split_ids'][sp]['held_positions'];check('held_indices_'+path.stem,h['held_doc_indices']==list(range(lo,hi+1)))
  check('held_provenance_'+path.stem,h['candidate_weight_sha256']==m['weight_sha256'] and h['search_selection_sha256']==lock['search_selection_sha256'])
  obj=torch.load(track(path.with_suffix('.pt')),map_location='cpu',weights_only=True)
  for field,key in [('weight_fp16','weight_sha256'),('scales','scales_sha256'),('zeros','zeros_sha256'),('g_idx','g_idx_sha256')]:check('tensor_'+path.stem+'_'+field,thash(obj[field])==m[key])
  check('finite_'+path.stem,all(torch.isfinite(obj[k]).all().item() for k in ['weight_fp16','scales','zeros']))
  check('actual_pristine_'+path.stem,thash(original[(li,pr)].half())==m['pristine_weight_sha256'])
  check('scale_zero_shape_'+path.stem,obj['scales'].shape==obj['zeros'].shape==torch.Size([len(m['rows']),original[(li,pr)].shape[1]//128]))
  check('zero8_positive_scale_'+path.stem,torch.all(obj['zeros']==8).item() and torch.all(obj['scales']>0).item())
  check('g_idx_'+path.stem,torch.equal(obj['g_idx'],torch.arange(original[(li,pr)].shape[1],dtype=obj['g_idx'].dtype)//128))
  meta_groups[(li,pr,sp)].append(m)
  execution_rows.append(dict(layer=li,projection=pr,split_id=sp,alpha=a,original10=a in g10,rows_full_linear=original[(li,pr)].shape[0],input_dim=original[(li,pr)].shape[1],
   full_linear_executions=1,search_row_scores=len(m['rows']),gptq_seconds=m['gptq_seconds'],search_scoring_seconds=m['search_score_seconds'],checkpoint_io_seconds=m['checkpoint_io_seconds'],held_scoring_seconds=h['held_score_seconds']))
  for i,row in enumerate(m['rows']):
   uid=f'L{li}_{pr}_R{row}_{sp}';rec=dict(unit_id=uid,layer=li,projection=pr,row=row,split_id=sp,alpha=a,search_mse=m['search_mse_by_row'][str(row)],held_mse=h['held_mse_by_row'][str(row)])
   check(f'search_match_{uid}_{a}',close(rec['search_mse'],lookup[(uid,a)]['search_mse']));bank[uid].append(rec)
   parameters[(uid,a)]=dict(scale=obj['scales'][i].tolist(),zero=obj['zeros'][i].tolist(),g_idx_sha256=m['g_idx_sha256'],quantized_weight_row_sha256=thash(obj['weight_fp16'][i]),weight_mse=float((obj['weight_fp16'][i].float()-original[(li,pr)][row]).square().mean()))
 check('552_full_linear_candidates',len(execution_rows)==552)
 for k,v in meta_groups.items():
  lo,hi=p['documents']['split_ids'][k[2]]['search_positions'];check('search_indices_'+str(k),v[0]['search_doc_indices']==list(range(lo,hi+1)))
  check('same_inputs_H_pristine_'+str(k),len(v)==46 and len({(r['hessian_sha256'],r['pristine_weight_sha256'],tuple(r['search_doc_indices']),r['e2_manifest_sha256']) for r in v})==1)
 units=[];chosen=[]
 for uid,b in sorted(bank.items()):
  check('all46_unique_'+uid,len(b)==46 and {r['alpha'] for r in b}==set(g46))
  s10,s46=select(b,'search_mse',g10),select(b,'search_mse',g46);o10,o46=select(b,'held_mse',g10),select(b,'held_mse',g46)
  c10,c46=held_csv[(uid,'M2_original_scalar10')],held_csv[(uid,'dense_scalar46')];eps=float(c10['unit_tie_epsilon_mse']);elements=int(c10['held_doc_count'])*docs['tokens_per_document']
  check('selected_match_'+uid,s10['alpha']==float(c10['alpha_A']) and s46['alpha']==float(c46['alpha_A']) and close(s10['held_mse'],c10['held_local_mse']) and close(s46['held_mse'],c46['held_local_mse']))
  check('oracle_match_'+uid,close(o10['held_mse'],oracle_csv[uid]['scalar10_held_oracle_mse']) and close(o46['held_mse'],oracle_csv[uid]['dense46_held_oracle_mse']))
  check('search_superset_'+uid,s46['search_mse']<=s10['search_mse'])
  d=s10['held_mse']-s46['held_mse'];u={k:s10[k] for k in ['unit_id','layer','projection','row','split_id']}
  u.update(alpha10=s10['alpha'],alpha46=s46['alpha'],elements=elements,scalar10_search_mse=s10['search_mse'],dense46_search_mse=s46['search_mse'],scalar10_held_mse=s10['held_mse'],dense46_held_mse=s46['held_mse'],
   scalar10_held_sse=s10['held_mse']*elements,dense46_held_sse=s46['held_mse']*elements,scalar10_oracle_alpha=o10['alpha'],dense46_oracle_alpha=o46['alpha'],scalar10_oracle_held_sse=o10['held_mse']*elements,dense46_oracle_held_sse=o46['held_mse']*elements,
   held_gain_fraction=d/s10['held_mse'],delta_sse=d*elements,epsilon_mse=eps,epsilon_sse=eps*elements,outcome='win' if d>eps else 'loss' if d< -eps else 'tie',scalar10_weight_mse=parameters[(uid,s10['alpha'])]['weight_mse'],dense46_weight_mse=parameters[(uid,s46['alpha'])]['weight_mse'])
  units.append(u);chosen.append(dict(unit_id=uid,scalar10=parameters[(uid,s10['alpha'])],dense46=parameters[(uid,s46['alpha'])]))
 check('48_units_32768_elements',len(units)==48 and all(u['elements']==32768 for u in units))
 aggregates=[aggregate(units,'ALL')]+[aggregate([u for u in units if u['split_id']==s],s) for s in ['S0','S1']]
 for li in p['targets']['layers']:
  aggregates.append(aggregate([u for u in units if u['layer']==li],f'L{li}'))
  for pr in p['targets']['projections']:
   cell=[u for u in units if u['layer']==li and u['projection']==pr];aggregates.append(aggregate(cell,f'L{li}/{pr}'))
   aggregates.extend(aggregate([u for u in cell if u['split_id']==s],f'L{li}/{pr}/{s}') for s in ['S0','S1'])
 aggregates.extend(aggregate([u for u in units if u['projection']==pr],pr) for pr in p['targets']['projections'])
 positive=sorted([u for u in units if u['delta_sse']>0],key=lambda u:u['delta_sse'],reverse=True);net=math.fsum(u['delta_sse'] for u in units);pos=math.fsum(u['delta_sse'] for u in positive)
 concentration=dict(net_delta_sse=net,positive_delta_sse=pos,negative_delta_sse=math.fsum(min(0,u['delta_sse']) for u in units),top_units=[])
 for k in [1,3,5]:
  ids={u['unit_id'] for u in positive[:k]};d=math.fsum(u['delta_sse'] for u in positive[:k]);concentration['top_units'].append(dict(k=k,unit_ids=sorted(ids),delta_sse=d,fraction_positive=d/pos,fraction_net=d/net,remaining=aggregate([u for u in units if u['unit_id'] not in ids],f'exclude_top_{k}')))
 concentration['leave_target_out']=[aggregate([u for u in units if not(u['layer']==li and u['projection']==pr)],f'exclude_L{li}/{pr}') for li in p['targets']['layers'] for pr in p['targets']['projections']]
 pairs=defaultdict(dict)
 for u in units:pairs[(u['layer'],u['projection'],u['row'])][u['split_id']]=u
 pair_rows=[dict(layer=k[0],projection=k[1],row=k[2],S0_outcome=v['S0']['outcome'],S1_outcome=v['S1']['outcome'],S0_gain=v['S0']['held_gain_fraction'],S1_gain=v['S1']['held_gain_fraction']) for k,v in pairs.items()]
 alpha_rows=[]
 for scope in ['ALL','S0','S1']+[f'L{li}/{pr}' for li in p['targets']['layers'] for pr in p['targets']['projections']]:
  sub=units if scope=='ALL' else [u for u in units if (u['split_id']==scope if scope in ['S0','S1'] else f"L{u['layer']}/{u['projection']}"==scope)]
  for method,key in [('scalar10','alpha10'),('dense46','alpha46')]:
   alpha_rows.extend(dict(scope=scope,method=method,alpha=a,units=n,fraction=n/len(sub)) for a,n in sorted(Counter(u[key] for u in sub).items()))
 result=dict(comparison='DIRECT dense46 versus scalar10',grids=dict(scalar10=g10,dense46=g46),aggregates=aggregates,concentration=concentration,pair_outcomes=dict(Counter(r['S0_outcome']+'/'+r['S1_outcome'] for r in pair_rows)),migration_split_gate_pass=all(r['improved_beyond_epsilon'] for r in aggregates if r['scope'] in ['S0','S1']),prefix='fixed v18 M2 checkpoint inputs, not dense46 own-prefix trajectory',oracle_use='diagnostic only',historical_correctness=historical)
 table('DIRECT_UNIT_RESULTS.csv',units);table('DIRECT_BREAKDOWN.csv',aggregates);table('ALPHA_DISTRIBUTION.csv',alpha_rows);table('PAIRED_SPLIT_OUTCOMES.csv',pair_rows);table('V24_FULL_LINEAR_EXECUTIONS.csv',execution_rows)
 write('SELECTED_QUANTIZATION_PARAMETERS.json',chosen);write('DIRECT_SCALAR_AUDIT.json',result);write('cache/SCALAR_BANK.json',dict(source='v24 raw scalar cache, no structured inference',candidate_row_scores=[r for b in bank.values() for r in b]))
 return result,execution_rows
def baseline_audit():
 methods=[];token_hashes=defaultdict(set)
 for method in ['M0','M1','M2','M3']:
  for ds in ['wikitext','c4_fixed_budget']:
   if method=='M3':
    final=read(V3/('WT2_M3_FINAL.json' if ds=='wikitext' else 'C4_M3_FINAL.json'));raw_path=Path(final['raw_source']);window_path=Path(final['window_source'])
   else:final=None;raw_path=V18/f'evaluation/raw/{method}_{ds}.json';window_path=V18/f'evaluation/windows/{method}_{ds}.csv'
   raw=read(raw_path);w=rows(window_path);n=sum(int(r['valid_tokens']) for r in w);nll=math.fsum(float(r['window_nll']) for r in w);avg=nll/n;ppl=math.exp(avg)
   check(f'baseline_windows_{method}_{ds}',[int(r['window_index']) for r in w]==list(range(len(w))))
   check(f'baseline_ppl_{method}_{ds}',close(ppl,raw['perplexity']) and n==raw['evaluated_tokens'])
   check(f'baseline_revision_{method}_{ds}',raw['model_revision']==REV and raw['max_length']==2048 and raw['stride']==512)
   if final:
    check('M3_hash_'+ds,sha(raw_path)==final['raw_json_sha256'] and sha(window_path)==final['window_csv_sha256']);check('M3_aggregate_'+ds,close(ppl,final['ppl']) and n==final['targets'])
   token_hashes[ds].add(raw['evaluation_token_ids_sha256_i64le'])
   for file in raw['model_weight_files']:
    actual=Path(raw['model_source'])/file['name'];key=str(actual.relative_to(ROOT))
    if key not in SOURCE_HASHES:track(actual)
    check(f'checkpoint_{method}_{ds}',SOURCE_HASHES[key]==file['sha256'])
   methods.append(dict(method=method,dataset=ds,targets=n,windows=len(w),nll=avg,ppl=ppl,evaluation_seconds=raw['evaluation_wall_seconds'],new_evaluation_run=False,evaluation_token_ids_sha256=raw['evaluation_token_ids_sha256_i64le']))
 check('same_evaluation_tokens',all(len(v)==1 for v in token_hashes.values()))
 for item in methods:
  base={r['method']:r for r in methods if r['dataset']==item['dataset']}
  for m in ['M1','M2','M3']:item['ppl_delta_vs_'+m]=item['ppl']-base[m]['ppl']
 read(V3/'PROTOCOL.json');read(V3/'correctness_checks.json');read(V3/'EVALUATION_CORRECTNESS.json');track(V3/'REPORT.md')
 table('BASELINES_REUSED.csv',methods);write('BASELINE_RECOMPUTATION.json',dict(status='PASS',scope='raw window aggregation and existing checkpoint/hash validation, no rerun',methods=methods))
 return methods

def e1_audit():
 read(V23/'PROTOCOL.json');s=read(V23/'E1_SUMMARY.json');read(V23/'correctness_checks.json');track(V23/'REPORT.md');group=defaultdict(list)
 for r in rows(V23/'HELD_RESULTS.csv'):group[r['unit_id']].append(r)
 details=[]
 for uid,b in group.items():
  l=[x for x in b if x['alpha_local_selected']=='True'];q=[x for x in b if x['alpha_block_selected']=='True'];check('E1_unique_'+uid,len(l)==len(q)==1)
  details.append(dict(unit_id=uid,local_selector_held_block_sse=float(l[0]['block_held_sse']),block_selector_held_block_sse=float(q[0]['block_held_sse'])))
 l=math.fsum(r['local_selector_held_block_sse'] for r in details);b=math.fsum(r['block_selector_held_block_sse'] for r in details)
 check('E1_pooled_recompute',math.isclose((l-b)/l,s['selector_comparison']['pooled_sse_improvement_fraction'],abs_tol=1e-13))
 obj=dict(status='PASS',historical_verdict=s['verdict'],units=len(details),local_selector_held_block_sse=l,block_selector_held_block_sse=b,pooled_gain_fraction=(l-b)/l,new_E1_execution=False)
 write('E1_RECOMPUTATION.json',obj);return obj

def cost_audit(executions):
 timing=[r for r in rows(V18/'FORMAL_TIMING.csv') if r['method']=='M2'];runtime=read(V18/'FORMAL_RUNTIME.json');read(V24/'RUNTIME_PREFLIGHT.json');stage=defaultdict(list)
 for r in timing:stage[r['stage']].append(r)
 totals={k:dict(count=len(v),seconds=math.fsum(float(x['wall_seconds']) for x in v)) for k,v in stage.items()}
 table('HISTORICAL_M2_STAGE_COST.csv',[dict(stage=k,**v) for k,v in totals.items()])
 cand=stage['full_linear_alpha_candidate_gptq'];by_module=defaultdict(lambda:defaultdict(list))
 for typ,data in [('gptq',cand),('score',stage['all_row_direct_output_scoring']),('row_select',stage['streaming_per_row_argmin_update'])]:
  for r in data:by_module[r['module']][typ].append(float(r['wall_seconds']))
 shape_times=defaultdict(lambda:defaultdict(list));module_rows=[]
 for name,v in sorted(by_module.items()):
  matching=[r for r in cand if r['module']==name];shape=tuple(json.loads(matching[0]['detail'])['shape'])
  check('10_actual_fullshape_calls_'+name,len(matching)==10 and all(len(v[k])==10 for k in ['gptq','score','row_select']))
  shape_times[shape]['modules'].append(name)
  row=dict(module=name,rows=shape[0],input_dim=shape[1],historical_calls=10,dense46_calls=46,historical_row_scores=10*shape[0],dense46_row_scores=46*shape[0])
  for typ in ['gptq','score','row_select']:
   total=math.fsum(v[typ]);row['historical_'+typ+'_seconds']=total;row['projected_dense46_'+typ+'_seconds']=total/10*46;shape_times[shape][typ].extend(v[typ])
  module_rows.append(row)
 check('1680_calls_not3041280',len(cand)==1680 and len(module_rows)==168);check('real_shapes_row_sum',sum(r['rows'] for r in module_rows)==304128)
 recent=defaultdict(list)
 for r in executions:recent[(r['rows_full_linear'],r['input_dim'])].append(r['gptq_seconds'])
 shape_rows=[]
 for shape,v in sorted(shape_times.items()):
  n=len(v['modules']);shape_rows.append(dict(rows=shape[0],input_dim=shape[1],linear_count=n,scalar10_executions=10*n,dense46_executions=46*n,scalar10_row_scores=10*n*shape[0],dense46_row_scores=46*n*shape[0],old_gptq_mean_seconds=statistics.mean(v['gptq']),old_gptq_max_seconds=max(v['gptq']),old_allrow_score_mean_seconds=statistics.mean(v['score']),v24_fullshape_candidate_count=len(recent[shape]),v24_gptq_mean_seconds=statistics.mean(recent[shape]) if recent[shape] else None,projected_dense_gptq_seconds=n*46*statistics.mean(v['gptq']),projected_dense_score_seconds=n*46*statistics.mean(v['score'])))
 table('COST_BY_REAL_LINEAR.csv',module_rows);table('COST_BY_SHAPE.csv',shape_rows)
 variable=math.fsum(totals[x]['seconds'] for x in ['full_linear_alpha_candidate_gptq','all_row_direct_output_scoring','streaming_per_row_argmin_update']);observed=runtime['m2']['wall_seconds_including_load_replay_save'];fixed=observed-variable;projected=fixed+4.6*variable
 mixed_gptq=math.fsum(46*len(v['modules'])*statistics.mean(recent[shape] or v['gptq']) for shape,v in shape_times.items())
 mixed=fixed+mixed_gptq+4.6*(totals['all_row_direct_output_scoring']['seconds']+totals['streaming_per_row_argmin_update']['seconds'])
 bygrid={}
 for label,predicate in [('original10',lambda r:r['original10']),('additional36',lambda r:not r['original10']),('dense46',lambda r:True)]:
  sub=[r for r in executions if predicate(r)];bygrid[label]=dict(full_linear_executions=len(sub),row_scores=sum(r['search_row_scores'] for r in sub),gptq_seconds=math.fsum(r['gptq_seconds'] for r in sub),search_score_seconds=math.fsum(r['search_scoring_seconds'] for r in sub),held_score_seconds=math.fsum(r['held_scoring_seconds'] for r in sub),checkpoint_io_seconds=math.fsum(r['checkpoint_io_seconds'] for r in sub),scope='measured shared bank stages, not independent total method wall')
 table('V24_SCALAR_BANK_COST.csv',[dict(grid=k,**v) for k,v in bygrid.items()])
 stage_sum=math.fsum(v['seconds'] for v in totals.values());deps=['qkv_commit_to_o_input','o_commit_to_up_gate_input','up_gate_commit_to_down_input','down_commit_to_next_layer_input']
 pilot0=[r for r in module_rows if r['module'].startswith('model.layers.0.')];pilot_est=sum(r['projected_dense46_'+x+'_seconds'] for r in pilot0 for x in ['gptq','score','row_select'])+sum(float(r['wall_seconds']) for r in timing if int(r['layer']) in [-1,0] and r['stage'] not in ['full_linear_alpha_candidate_gptq','all_row_direct_output_scoring','streaming_per_row_argmin_update','final_dense_model_save'])+(observed-stage_sum)/24
 v24runtime=read(V24/'RUN_REPORT.json')['runtime']
 components=[]
 categories={'candidate_search':['full_linear_alpha_candidate_gptq','all_row_direct_output_scoring','streaming_per_row_argmin_update'],'H_Hinv_preparation':['unique_group_hessian_build','shared_native_hinv_prepare'],'initial_prefix_capture':['initial_embedding_capture','qkv_shared_input_capture'],'dependency_capture_refresh':deps,'stitch_commit':['full_linear_row_stitch','whole_linear_commit'],'checkpoint_IO':['atomic_layer_checkpoint_io','final_dense_model_save'],'parity_references':['alpha1_native_parity_reference'],'model_load':['model_load']}
 for name,stages in categories.items():
  observed_component=sum(totals[x]['seconds'] for x in stages);components.append(dict(component=name,historical_M2_seconds=observed_component,projected_dense46_seconds=observed_component*(4.6 if name=='candidate_search' else 1),measurement='historical synchronized complete stages'))
 components.append(dict(component='unattributed_overhead',historical_M2_seconds=observed-stage_sum,projected_dense46_seconds=observed-stage_sum,measurement='wall minus measured stages; hash/validation/CPU overhead, not assigned to GPTQ'))
 table('FORMAL_COST_COMPONENTS.csv',components)
 obj=dict(status='HISTORICAL_FULL_SHAPE_COST_AUDIT_ONLY',candidate_calls=dict(scalar10=1680,dense46=7728,additional=6048),row_alpha_scores=dict(scalar10=3041280,dense46=13989888,additional=10948608),v18_calibration_wall_seconds=observed,v18_measured_stages=totals,v18_stage_sum_seconds=stage_sum,v18_unattributed_overhead_seconds=observed-stage_sum,fixed_and_overhead_seconds=fixed,variable_search_seconds=variable,
  search_gptq_plus_scoring_fraction=(totals['full_linear_alpha_candidate_gptq']['seconds']+totals['all_row_direct_output_scoring']['seconds'])/observed,dependency_refresh_fraction=sum(totals[x]['seconds'] for x in deps)/observed,
  historical_dense46_projection_seconds=projected,historical_dense46_projection_hours=projected/3600,historical_1_5x_projection_hours=1.5*projected/3600,mixed_recent_shape_projection_seconds=mixed,mixed_recent_1_5x_hours=1.5*mixed/3600,projected_block0_pilot_seconds=pilot_est,projected_block0_1_5x_seconds=1.5*pilot_est,budget_seconds=21600,pilot_budget_seconds=1800,
  cost_gate='NOT_RUN: historical estimates do not replace fresh pilot measurement',
  caveats=['All168 historical full shapes; not row-time extrapolation.','Recent v24 matching up/down GPTQ replaces only measured shapes in mixed model; scoring uses formal v18 ALL rows and128 docs.','Different own-prefix H/inputs and timing variability need fresh pilot if consistency passes.','Failed cross-split gate prevents optimization/profiling now.'],v24_grid_cost=bygrid,
  peaks=dict(v18_torch_allocated_bytes=runtime['cuda_peak_allocated_bytes'],v18_torch_reserved_bytes=runtime['cuda_peak_reserved_bytes'],new_cuda_peak_bytes=None,v24_torch_allocated_bytes=v24runtime['peak_cuda']['allocated_bytes'],v24_torch_reserved_bytes=v24runtime['peak_cuda']['reserved_bytes']),
  dependency_boundaries=dict(shared_groups_per_block=4,H_preparations=96,full_linear_commits=168,initial_embedding_captures=1,initial_qkv_norm_capture=1,qkv_o_refreshes=24,o_upgate_refreshes=24,upgate_down_refreshes=24,down_next_refreshes=23,total_calibration_block_forwards=95*128,extra_row_batch_refreshes=0,note='next qkv norm fused with down->next pass; L23 no next calibration input needed'),
  actual_repetition=dict(one_fullshape_candidate_all_rows=True,shared_X_H_inverse_middle=True,inverse_upper_factor_fully_cached=False,repeated_cholesky='Native initial damped-H Cholesky and final upper Cholesky each candidate; middle inverse cached',separate_factorization_profile_seconds=None,repair_applied=False,unused_forward_suffix='run_block_capture executes full block although qkv->o and o->up-gate need only prefix to captured input; unused suffix exists but no row/sibling input recapture',decision='No repeated per-row calls/sibling refresh found. Real factorization repetition unprofiled; failed quality gate prevents new optimization.'),cost_components=components,v24_entire_E2_wall_seconds=v24runtime['total_wall_seconds'],v24_entire_E2_wall_scope='includes structured search, preparation, correctness and six failed recovery attempts; not scalar-only',runner_optimization='NONE; historical algorithm unchanged')
 write('COST_AUDIT.json',obj);return obj
def code_audit():
 paths=[V18/'run_formal.py',RESULTS/'range_w4_m2_runtime_v17_3/optimized_native_m2.py',RESULTS/'range_w4_m2_parity_v17_1/native_m2.py',V24/'run_e2_v24.py',ROOT/'.venv-gptq/Lib/site-packages/gptqmodel/quantization/gptq.py',ROOT/'.venv-gptq/Lib/site-packages/gptqmodel/quantization/quantizer.py']
 needles={0:['for alpha in GRID:','rp.score_rows','best_q[better]','best_scale[better]','best_zero[better]','module.weight.copy_(stitched_q)','for group in ('],1:['def prepare_native_state','inverse_before_final_cholesky','core.H = prepared.hessian.clone()','if calls > 0:','core.quantizer.scale = core.quantizer.scale * alpha','core.quantize(blocksize=128)'],3:['def get_full_candidate','def score_selected_rows','for alpha in DENSE_GRID:','scalar_scores =','dense_scores ='],4:['self.quantizer.find_params(W[:,','W1[:, i:] -=','W[:, i2:] -=','torch.linalg.cholesky(H, upper=True)'],5:['torch.round(x / scale) + zero','torch.full_like(self.scale, (self.maxq + 1) / 2)']}
 anchors=[]
 for i,path in enumerate(paths):
  text=track(path).read_text('utf-8')
  for needle in needles.get(i,[]):
   found=[j for j,line in enumerate(text.splitlines(),1) if needle in line];check('source_'+path.name+'_'+needle,bool(found));anchors.append(dict(file=str(path.relative_to(ROOT)),anchor=needle,lines=found))
 table('SOURCE_CODE_AUDIT.csv',anchors);write('ALGORITHM_AUDIT.json',dict(status='SOURCE_AND_ARTIFACT_AUDIT',format='W4A16 symmetric G128; code [0,15],zero8; FP32 PyTorch round/clamp -> FP16 W',range_semantics='Native current compensated W group range; only dynamic group scale multiplied by alpha. alpha1 no hook',candidate_semantics='Pristine W and fresh GPTQ state; full Linear all output rows per candidate; formal allrow scoring and row stitch; shared g_idx required identical',v24_scope='full candidate generation but only four diagnostic rows retained/scored',grouping='qkv and up/gate shared X/H; no sibling forward dependency',newly_optimized=False,anchors=anchors))

def report(result,cost,baselines,e1,elapsed):
 ag={g['scope']:g for g in result['aggregates']};a,s0,s1=ag['ALL'],ag['S0'],ag['S1']
 lines=['# v25 稠密标量搜索收益、成本与迁移审计','', '**COMPLETE；跨 split 迁移 gate FAIL，STOP新增GPU试运行与整模型量化。**','', '## 直接比较','',
  f"dense46相对scalar10 pooled held SSE改善 {a['pooled_gain_fraction']:.6%}；S0 {s0['pooled_gain_fraction']:.6%}，S1 {s1['pooled_gain_fraction']:.6%}。S1失败，不能以总体正值替代用户要求的跨split一致性。",
  f"scalar10/dense46绝对pooled SSE {a['scalar10_held_sse']:.12g}/{a['dense46_held_sse']:.12g}，下降 {a['delta_sse']:.12g}。48单位W/T/L={a['wins']}/{a['ties']}/{a['losses']}；median {a['median_unit_gain_fraction']:.6%}；worst {a['worst_gain_fraction']:.6%} ({a['worst_unit']})。",
  f"S0 W/T/L={s0['wins']}/{s0['ties']}/{s0['losses']}；S1={s1['wins']}/{s1['ties']}/{s1['losses']}。epsilon沿用v24，不重新调阈值。每单位16篇×2048元素，SSE=MSE×32768；此处绝对SSE不混称MSE总和。",'']
 for li in [0,12,23]:
  for pr in ['up_proj','down_proj']:
   k=f'L{li}/{pr}';g=ag[k];lines.append(f"- {k}：pooled {g['pooled_gain_fraction']:.6%}；S0/S1 {ag[k+'/S0']['pooled_gain_fraction']:.6%}/{ag[k+'/S1']['pooled_gain_fraction']:.6%}；W/T/L {g['wins']}/{g['ties']}/{g['losses']}；median {g['median_unit_gain_fraction']:.6%}；worst {g['worst_gain_fraction']:.6%}。")
 lines+=['','## 集中度与oracle','']
 for t in result['concentration']['top_units']:lines.append(f"- top{t['k']}正收益单位解释正收益 {t['fraction_positive']:.4%}、净收益 {t['fraction_net']:.4%}；移除后pooled gain {t['remaining']['pooled_gain_fraction']:.6%}。")
 lines += ['净收益主要由L23/down驱动：去掉整个L23/down后仅约0.0753%；前三个正贡献是L23_down_R0_S0、L23_down_R224_S0和L23_up_R224_S0，全部来自S0。L12/down与L23/down的两split方向都正，其他四个target方向跨split改变；不据此挑层推广。',
  f"跨split配对outcome：`{json.dumps(result['pair_outcomes'],ensure_ascii=False)}`。top名单、逐target移除敏感度和全部负单位见DIRECT_SCALAR_AUDIT及DIRECT_UNIT_RESULTS。",
  f"scalar10/dense46 held oracle SSE {a['scalar10_oracle_sse']:.12g}/{a['dense46_oracle_sse']:.12g}；oracle capacity改善 {a['oracle_capacity_gain_fraction']:.6%}。search选择相对各自oracle regret {a['scalar10_selection_regret_fraction']:.6%}/{a['dense46_selection_regret_fraction']:.6%}。"]
 for s in ['S0','S1']:
  g=ag[s];lines.append(f"- {s}：scalar10/dense46 selection regret {g['scalar10_selection_regret_fraction']:.6%}/{g['dense46_selection_regret_fraction']:.6%}；oracle capacity {g['oracle_capacity_gain_fraction']:.6%}。")
 lines+=['','Oracle仅回溯诊断，不设计选择规则或挑层。更多候选的search最优不劣是集合包含关系；不能保证held/PPL更好。oracle存在潜力与search选择不能稳定泛化是两个观察，不构成因果机制证明。','', '## 实际协议、实现和正确性','',
  f'Qwen2.5-0.5B-Instruct revision `{REV}`，FP16/SDPA/RTX4060 Laptop，TF32 off；W4A16 symmetric G128，block128、damping.1、desc_act=False、static_groups=False、true_sequential=True、mse=0，H/GPTQ FP32。',
  '原生code[0,15]、zero8，round后clamp并反量化。group入口按当前compensated W估计range，再乘scale alpha；alpha1无hook。保留原补偿。10点=.1..1.0；46点=原10 union {.1+.9*j/37,j=1..36}，不是均匀.02网格。',
  '两种scalar共用同一完整Linear候选、输入、pristine W、H及选择目标，唯一差异是网格。直接读取552个scalar候选/held sidecar，重算2208个行分数的选择与oracle；独立核对CSV及W/scale/zero/g_idx tensor hash，不通过structured反推。',
  '代表层0/12/23，up/down都固定rows [0,224,448,672]（up不是四分位抽样，不能外推全行分布）。正式calibration为v18冻结128篇WT2 train×2048、seed20261005；本局部诊断沿用v22/v23/v24的64完整文章，每篇2048tokens；S0 search0..15/held16..31，S1 search32..47/held48..63。与正式128篇calibration hash无交集，不拆token、不重抽样。',
  '输入是固定v18 M2 checkpoint前缀：v23捕获up输入，v24 replay block捕获down输入。目标W pristine FP16。不是dense46自身prefix，也不是正式dense模型迁移。',
  'v18已经每alpha一次完整Linear产生所有行、全行评分、流式选择及W/scale/zero拼接，g_idx一致检查、一次完整提交；candidate独立clone。未发现逐行重复GPTQ或每32行full-prefix刷新。因此没有编造加速修复。',
  '共享X/H和逆运算中间结果，但完整逆Cholesky因子尚未全缓存：core每candidate仍做damped-H lower Cholesky与inverse upper Cholesky，中间inverse已缓存。这是实际重复计算，但本轮未单独profile；不能将整个GPTQ时间当作可节省时间。gate失败后不增加新runtime优化。',
  'run_block_capture当前运行完整block，所以qkv->o和o->up/gate捕获后的未使用suffix forward确实存在；依赖边界次数已合理，仍有partial-forward工程潜力。没有独立profile，不估造可节省比例。',
  'qkv、up/gate各共享一个输入/H，原生处理order不产生sibling forward dependency。每block四input/H组；完整group提交后qkv->o、o->up/gate、up/gate->down；完整down提交后仅为下一block刷新。24blocks：96次H准备、168次Linear提交、72次block内capture refresh、23次down->next，128×95=12160次block forward。初始embedding/一次qkv norm另计，next qkv norm融合在down->next，最后block无需下游input，零row-batch refresh。',
  '本轮correctness PASS仅指CPU原始数据/参数/协议审计；v18/v24历史runtime PASS可复用。新dense46 own-prefix alpha1/order/commit/propagation试运行是NOT_RUN，不冒充PASS。没有优化前后测速/张量对照，因为未改实现。','', '## 真实成本','',
  '正式10点1680完整Linear executions、3,041,280 row-alpha scores；46点7728 executions、13,989,888 scores。增加6048 calls/10,948,608 scores，候选执行数量4.6倍；35次历史native parity参考额外列账。']
 for label,g in cost['v24_grid_cost'].items():lines.append(f"- v24 {label}：{g['full_linear_executions']}完整GPTQ / {g['row_scores']}行search评分；GPTQ {g['gptq_seconds']:.3f}s，search {g['search_score_seconds']:.3f}s，held {g['held_score_seconds']:.3f}s，candidate IO {g['checkpoint_io_seconds']:.3f}s。")
 lines+=['','这是共享bank实测阶段，非两个方法独立总wall；v24只评分4行×16篇，不能用其评分时间外推全行×128篇。v24失败重试与预处理保留原RUN_REPORT，不计入scalar专属candidate时间。',
  f"正式M2校准wall {cost['v18_calibration_wall_seconds']:.3f}s：GPTQ+全行评分占 {cost['search_gptq_plus_scoring_fraction']:.2%}，dependency refresh占 {cost['dependency_refresh_fraction']:.2%}。分阶段实测和未归因overhead详见COST_AUDIT，全部168个矩阵/形状详见COST_BY_REAL_LINEAR与COST_BY_SHAPE。",
  f"按已完成全部真实矩阵计时，仅候选/评分/选择放大46/10、固定成本保留：历史dense投影 {cost['historical_dense46_projection_hours']:.3f}h；1.5倍保守 {cost['historical_1_5x_projection_hours']:.3f}h。较新v24匹配up/down完整形状实测GPTQ替换对应均值的混合1.5倍估计 {cost['mixed_recent_1_5x_hours']:.3f}h。均非本轮实测或runtime cost gate PASS；own-prefix H、环境时变需新pilot核验。",
  f"预定block0 pilot历史估计 {cost['projected_block0_pilot_seconds']:.1f}s，1.5倍 {cost['projected_block0_1_5x_seconds']:.1f}s（上限1800s）；正式新量化上限21600s不变。不是代表行时间乘全模型行数。",
  f"历史M2 peak allocated/reserved {cost['peaks']['v18_torch_allocated_bytes']/2**30:.3f}/{cost['peaks']['v18_torch_reserved_bytes']/2**30:.3f}GiB；本轮无CUDA，新GPU峰值NOT_RUN。",'','## 门槛与正式PPL','',
  'PROTOCOL先冻结复算规则、跨split gate与条件pilot（完整block0覆盖896/4864输入维度和实际依赖）。S1失败，按用户“收益不稳定则停止重计算”停止：pilot、新held、own-prefix提交传播、dense整模型量化/PPL均NOT_RUN。成本不是停止原因；正确性审计PASS也不能覆盖质量gate FAIL。',
  '以下已有正式基线从原始windows按valid_tokens加权复算，并核查revision、token hash和checkpoint SHA，未重复评测：','']
 for m in ['M0','M1','M2','M3']:
  w=next(r for r in baselines if r['method']==m and r['dataset']=='wikitext');c=next(r for r in baselines if r['method']==m and r['dataset']=='c4_fixed_budget');lines.append(f"- {m} WT2 PPL {w['ppl']:.8f}；C4 fixed-budget PPL {c['ppl']:.8f}。")
 lines += ['','C4是固定131071计分tokens validation流，不是标准/full C4。dense46 PPL和相对M1/M2/M3变化未测，不推断。M3 asymmetric LWC梯度block目标无GPTQ，与M2不只网格不同，非纯搜索因果对照。校准wall和evaluation wall分账；M3精确校准timer缺失不作速度比。',
  f"E1 raw复算pooled held block gain {e1['pooled_gain_fraction']:.8%}，verdict {e1['historical_verdict']}；E2仍保留不支持结果，两路线不扩展。",'','## 下一实验与研究/工程结论','',
  '建议结束当前扩大scalar网格的质量扩展。现有小幅pooled局部收益跨split不稳定，不能支撑新方法优势或PPL迁移；也不意味着所有scalar网格都无效。',
  '一个有依据的下一实验：冻结10点M2，在完整up/down矩阵与正式128篇输入上做<10分钟的阶段profile，分别量两次Cholesky、逐列补偿、全行直接评分与IO。只有factorization确是主要成本，再验证完整upper-factor缓存parity及同协议计时。无需新held/PPL、不扩大grid、不用oracle制定选择规则；本轮未执行。',
  '工程亮点是执行单位/完整矩阵成本审计、native dynamic候选隔离、row stitch/真实dependency、hash恢复、跨split负结果与oracle regret追踪。离线校准优化不叫在线推理加速。','', '## 复现','',
  '```powershell',f"& 'E:/quant-thesis/.venv-gptq/Scripts/python.exe' -B '{OUT.as_posix()}/audit_dense_v25.py'",'```','',
  f'仅CPU解析和hash验证，所有新增产物限v25；本次CPU分析wall {elapsed:.3f}s，不混入历史GPU实验cost。原始CSV/JSON、correctness、输入哈希、冻结协议与cache manifest齐全。CPU审计增加哈希检查时有一次编码定义错误，见AUDIT_ATTEMPT_FAILURE_01.json，已按历史JSON编码修正，数据未改；该失败pass计时未记录，因此不把本次成功pass时间声称为所有开发重试总wall。']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n','utf-8')

def main():
 start=time.perf_counter()
 prior=OUT/'cache/CACHE_MANIFEST.json'
 if prior.exists():
  old=json.loads(prior.read_text('utf-8'));EXPECTED_SOURCES.update(old['source_sha256']);check('existing_cache_bank_hash',sha(OUT/'cache/SCALAR_BANK.json')==old['bank_sha256'])
 read(OUT/'PROTOCOL.json');write('RUN_STATUS.json',dict(status='RUNNING',scope='CPU artifact audit'))
 code_audit();result,execution_rows=scalar_audit();print('DIRECT_SCALAR_RECOMPUTED',flush=True)
 baseline=baseline_audit();e1=e1_audit();cost=cost_audit(execution_rows)
 check('stop_condition_observed',not result['migration_split_gate_pass'])
 check('all_sources_unchanged',all(sha(ROOT/p)==h for p,h in SOURCE_HASHES.items()));elapsed=time.perf_counter()-start
 write('correctness_checks.json',dict(status='PASS',scope='CPU artifact/source audit; NOT fresh GPU candidate/stitch/commit correctness',checks=CHECKS,historical_runtime='v18/v24 PASS reused; fresh migration NOT_RUN'))
 write('INPUT_PROVENANCE.json',dict(source_sha256=SOURCE_HASHES,source_unchanged=True,script_sha256=sha(__file__),protocol_sha256=sha(OUT/'PROTOCOL.json')))
 write('cache/CACHE_MANIFEST.json',dict(status='COMPLETE',bank_sha256=sha(OUT/'cache/SCALAR_BANK.json'),source_sha256=SOURCE_HASHES,script_sha256=sha(__file__),resume='rerun CPU audit, verify source hashes; never invokes historical runners'))
 summary=dict(status='COMPLETE',verdict='DENSE46_CROSS_SPLIT_GATE_FAIL_STOP',correctness='PASS_ARTIFACT_AUDIT',migration_gate_pass=result['migration_split_gate_pass'],core=[r for r in result['aggregates'] if r['scope'] in ['ALL','S0','S1']],new_pilot='NOT_RUN_SPLIT_GATE_FAIL',new_fullmodel='NOT_RUN_SPLIT_GATE_FAIL',new_dense46_PPL=None,new_gptq_executions=0,new_model_forwards=0,new_held_scores=0,new_ppl_evaluations=0,new_analysis_wall_seconds=elapsed,cost_projection_hours=cost['historical_dense46_projection_hours'],recommendation='stop dense-grid quality expansion; optional frozen10 fullshape profiling next, not executed')
 write('SUMMARY.json',summary);report(result,cost,baseline,e1,elapsed)
 write('RUN_REPORT.json',dict(**summary,script_sha256=sha(__file__),protocol_sha256=sha(OUT/'PROTOCOL.json'),actual_new_cost=dict(final_successful_cpu_pass_seconds=elapsed,initial_successful_cpu_pass_seconds=21.176530299999285,failed_cpu_pass_seconds=None,failed_cpu_pass_note='token hash encoding assumption failed before candidate scoring; timer not recorded, do not fabricate total audit wall',new_gpu_calibration_seconds=0,new_gpu_evaluation_seconds=0,gpu_allocated_peak_bytes=None,gpu_reserved_peak_bytes=None),cost_projection=cost,historical_results_preserved=True,distinction='artifact correctness PASS; cross-split benefit FAIL; new runtime/PPL NOT_RUN'))
 write('RUN_STATUS.json',dict(status='COMPLETE',verdict=summary['verdict'],migration='STOP',analysis_correctness='PASS'));print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
