"""Freeze the v26 scope before GPU work."""
import hashlib,json
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[3];V=O.parent
names=['range_w4_fullmodel_v18/PROTOCOL.json','range_w4_fullmodel_v18/CALIBRATION_MANIFEST.json','range_w4_fullmodel_v18/CALIBRATION_VALIDATION.json','range_w4_fullmodel_v18/run_formal.py','range_w4_fullmodel_v18/run_precheck.py','range_w4_fullmodel_v18/M2/checkpoint/final/model.safetensors','range_w4_m2_runtime_v17_3/optimized_native_m2.py','range_w4_m2_parity_v17_1/native_m2.py','range_w4_structured_group_e2_v24/correctness_checks.json','range_w4_structured_group_e2_v24/RUNTIME_PREFLIGHT.json','range_w4_dense_scalar_audit_v25/REPORT.md','range_w4_dense_scalar_audit_v25/COST_AUDIT.json']
paths=[V/x for x in names]+[R/'.venv-gptq/Lib/site-packages/gptqmodel/quantization/gptq.py',R/'.venv-gptq/Lib/site-packages/gptqmodel/quantization/quantizer.py']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
cal=json.loads((V/names[1]).read_text('utf-8'));v18=json.loads((V/names[0]).read_text('utf-8'))
assert (cal['nsamples'],cal['seqlen'],cal['seed'])==(128,2048,20261005)
assert v18['m2']['alpha_grid']==[i/10 for i in range(1,11)]
p={'name':'v26 frozen M2 L0 factor-cache profiling','status':'FROZEN_BEFORE_GPU','model':v18['model'],'calibration':v18['calibration'],'quantization':v18['quantization'],'m2':v18['m2'],
 'targets':['L0 mlp.up_proj [4864,896]','L0 mlp.down_proj [896,4864]'],
 'input':'Replay L0 of frozen v18 M2 final checkpoint on exact 128x2048 formal calibration documents; capture up/down pre-hooks; base checkpoint pristine target W',
 'hessian':'v18 run_precheck.build_hessian from all 128 complete inputs; FP32 GPTQ.process_batch',
 'score':'v18 run_precheck.score_rows: all rows, all 128x2048 input, FP32 X@delta.T and FP64 accumulation in 4-document chunks',
 'profile':{'alphas':[1.0,0.8],'warmups_per_matrix':1,'repeats_per_alpha_per_mode':2,'sync':'stage boundaries, CUDA events for nested GPU operations, no per-column synchronize','gpu_budget_seconds':600,'no_input_subsampling':True,
 'stages':['capture/transfer','H','dead/damp','lower Cholesky','inverse cache','upper Cholesky','dynamic find_params','compensation and other','score/selection','clone/IO/unattributed']},
 'decision':{'factor_share_min':0.05,'factor_savings_vs_warmed_range_min':2.0,'rule':'Both matrices must exceed both thresholds, including one-time full-factor build; otherwise stop after profiling'},
 'cache':{'default':'off; original v17.3','enabled':'opt-in v26 wrapper, native GPTQ.quantize unchanged, exact-input checked two Cholesky intercepts','key':['H SHA256','nsamples128','FP32','device','dead mask','damping .1','desc_act false','static_groups false','identity column order','block128'],'invalidation':'any key difference; cache read-only during candidates'},
 'parity':{'alphas':[1.0,0.8],'weight_zero_g_idx':'exact','scale':'v17.3 rtol1e-6 atol1e-9','score':'same full-row function and frozen numerical standard','candidate_order':['forward','reverse'],'alpha1_native':'exact','block_propagation':'NOT_RUN'},
 'limits':['no new search','no held-out','no PPL','no full-model quantization','no partial forward or score approximation','historical artifacts read-only'],
 'input_sha256':{str(q.resolve().relative_to(R)):sha(q.resolve()) for q in paths},'new_artifacts':'v26 directory only'}
f=O/'PROTOCOL.json'
if f.exists():assert json.loads(f.read_text('utf-8'))==p
else:f.write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n','utf-8')
print('FROZEN',f)
