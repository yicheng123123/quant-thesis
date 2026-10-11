import ast,csv,json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parent
def rs(n):return list(csv.DictReader((O/n).open(encoding='utf8')))
def save(n,rows):
 fields=list(dict.fromkeys(k for x in rows for k in x))
 with (O/n).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
loc=rs('CALLSITE_INVENTORY.csv');project=[];vendor=[]
for x in loc:
 if 'site-packages' in x['file']:x['classification']='unrelated_vendor_non_Qwen2_not_on_project_call_chain';vendor.append(x)
 else:
  x['role']='historical_archival_copy_not_runtime_entry' if '\\evidence\\' in x['file'] else 'project_runtime_or_legacy_diagnostic_entry';project.append(x)
# Track the dimensionality-preserving norm between the mistaken decoder slice and later batch slice.
for D in [B/'range_w4_fullmodel_v18',B/'range_w4_winner_union_v40/A',B/'range_w4_winner_union_v40/B',B/'range_w4_winner_union_v40/PILOT']:
 p=D/'run_precheck.py'
 if not p.exists():continue
 src=p.read_text(encoding='utf-8-sig');tree=ast.parse(src)
 for fn in ast.walk(tree):
  if not isinstance(fn,ast.FunctionDef):continue
  sliced={};normalized={}
  for n in ast.walk(fn):
   if isinstance(n,ast.Assign) and isinstance(n.value,ast.Subscript) and isinstance(n.value.slice,ast.Constant) and n.value.slice.value==0 and isinstance(n.value.value,ast.Call) and ast.unparse(n.value.value.func)=='layer':
    for name in n.targets:
     if isinstance(name,ast.Name):sliced[name.id]=n.lineno
  for n in ast.walk(fn):
   if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and 'input_layernorm' in ast.unparse(n.value.func) and any(isinstance(a,ast.Name) and a.id in sliced for a in n.value.args):
    root=next(a.id for a in n.value.args if isinstance(a,ast.Name) and a.id in sliced)
    for name in n.targets:
     if isinstance(name,ast.Name):normalized[name.id]=(sliced[root],n.lineno)
  for n in ast.walk(fn):
   if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id in normalized and isinstance(n.slice,ast.Constant) and n.slice.value==0:
    first,norm=normalized[n.value.id];project.append({'file':str(p),'function':fn.name,'line':first,'kind':'decoder_zero_then_layernorm_then_zero_cache_slice','expression':'decoder[0] -> input_layernorm -> normalized[0]','followup_zero_lines':str(n.lineno),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':'project_legacy_precheck_boundary'})
save('CALLSITE_INVENTORY.csv',project);save('EXCLUDED_VENDOR_MATCHES.csv',vendor)
impact=rs('HISTORICAL_IMPACT.csv')
def extra(v,dirname,route,envrel=None):
 D=B/dirname;evidence=[str(D/route.split(' -> ')[0])];version='UNKNOWN'
 if envrel and (D/envrel).exists():
  e=json.loads((D/envrel).read_text());version=e.get('transformers','UNKNOWN');evidence.append(str(D/envrel))
 impact.append(dict(version=v,actual_call_route=route,same_decoder_index_present=True,historical_return_evidence=f'historical runtime metadata={version}; direct per-call return trace absent',historical_return_type='tuple API supported by recorded4.49; direct type UNKNOWN' if version=='4.49.0' else 'UNKNOWN',actually_executed_erroneous_Tensor_branch='NOT_EVIDENCED' if version=='4.49.0' else 'UNKNOWN',affected_scope='prefix/boundary consumers of decoder[0] under Tensor; exact stages listed in call inventory',calibration_protocol_validity='SUPPORTED_BY_RECORDED4.49; no wholehistorical independent revalidation' if version=='4.49.0' else 'UNKNOWN',checkpoint_provenance='existing source records retained; not re-generated or quality re-evaluated',historical_PPL='not independently re-audited in this expanded call-chain scope',quality_and_timing_interpretation='no blanket invalidation from current replay; scope and runtime evidence must accompany historical comparison',action='retain artifacts with evidence status; current5.x legacy entry must use checked extraction and strict cache writes before reuse',evidence=';'.join(evidence)))
extra('v28','range_w4_quadratic_full_m2_v28','run_v28.py -> run_block_to_next/replay_completed_prefix','validation/FORMAL_RUNTIME.json')
extra('v29','range_w4_quadratic_hybrid_m2_v29','run_v29.py -> run_block_to_next/replay_completed_prefix','validation/FORMAL_RUNTIME.json')
extra('v36','range_w4_awq_gptq_v36','run_v36.py -> prepare_prefix_inputs/run_full_mlp_and_boundary')
extra('v37','range_w4_awq_gptq_v37','run_v37_correctness.py -> capture_fresh_p2;repair_s0_held.py -> replay_full_block')
extra('v40_A','range_w4_winner_union_v40/A','run_formal.py -> run_block_to_next/replay_completed_prefix','FORMAL_RUNTIME.json')
extra('v40_B','range_w4_winner_union_v40/B','run_formal.py -> run_block_to_next/replay_completed_prefix','FORMAL_RUNTIME.json')
extra('v40_PILOT','range_w4_winner_union_v40/PILOT','run_formal.py -> run_block_to_next/replay_completed_prefix')
save('HISTORICAL_IMPACT.csv',impact)
summary=json.loads((O/'CPU_AUDIT_SUMMARY.json').read_text());summary.update(project_callsite_risk_locations=len(project),excluded_unrelated_vendor_matches=len(vendor),additional_norm_alias_sites=len(project)-len(loc)+len(vendor),call_chain_expanded_versions=['v28','v29','v36','v37','v40 A/B/PILOT'])
(O/'CPU_AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
(O/'LEGACY_CALLSITE_HANDLING.md').write_text('''# Scope and routing of checked decoder extraction

The shared advance_block body is identical in the v42-v45 runtime copies and v46 runtime; v49/v51/v52 import the v46 runtime. The isolated tested replacement is runtime_v53.py. Historical files remain unchanged. Future formal integer repair must load this local runtime (with its decoder_interface module) before importing the unchanged v52 fast core/initialization review. Replace the old driver's single loadmod runtime path in its new isolated copy; let its provenance wrappers bind the corrected capture/advance implementation. Keep the original C builder and candidate/acceptance code.

Legacy v18 (and v28/v29/v40 copies) use split result[0] then output[0], or decoder[0] followed by input_layernorm and a batch slice. They are safe only with the recorded tuple API. Before any future5.x rerun, the reviewed replacement at each listed site is full=extract_hidden(result, expected_shape=tuple(input_hidden.shape)), then write_document(target_cache, doc_index, full); next-layer normalized cache also receives the full3D result and strict write_document. Do not globally replace every[0][0]: hook args[0][0] is a tuple-input plus batch slice, not a decoder-return bug, but its cache write must still verify shape.

v36/v37 similar prefix/boundary code is included in the inventory and impact table. No current version executed these legacy experiment entries during v53. Their historical environment is UNKNOWN unless its own artifact records it. v48 L3 diagnostic replay traverses advance_block; L0 local matrix paths in v47/v48/v50 capture hooks and ignore decoder output, so direct-return risk is absent on those local paths.

The current v53 implementation (three changed runtime definitions only) has been independently validated with a native full-forward reference. This document does not claim all archived legacy drivers were edited or fully retested. They remain read-only and must not be executed unadapted under the Tensor-return API.
''',encoding='utf8')
print('CALLSITE_SCOPE_COMPLETE',len(project),'project sites,',len(vendor),'vendor exclusions,',len(impact),'impact rows')
