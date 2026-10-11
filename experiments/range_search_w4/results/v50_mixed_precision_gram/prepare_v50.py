"""CPU-only source/timing audit and pre-GPU freezing for v50."""
import csv,hashlib,json,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE.parents[2]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def main():
    assert not (OUT/'GPU_START.json').exists()
    v49=BASE/'v49_fullmodel_fast_integer_repair';v48=BASE/'v48_fast_numerical_acceptance';v46=BASE/'v46_m1_single_integer_repair_fullmodel';v18=BASE/'range_w4_fullmodel_v18'
    lock=json.loads((v49/'CHECKPOINT_LOCK.json').read_text());proto=json.loads((v49/'PROTOCOL.json').read_text())
    files=[v49/'STAGE_TIMINGS.csv',v49/'BLOCK_TIMINGS.csv',v49/'PROTOCOL.json',v49/'CHECKPOINT_LOCK.json',v49/'R1/checkpoint/final/model.safetensors',v49/'original_codes/layer_00.pt',v49/'formal/groups/layer_00_up_gate.pt',v49/'formal/groups/layer_00_down.pt',v48/'fast_accept.py',v48/'ACCEPTANCE_RULES.md',v48/'run_v48.py',v48/'COMPARISON.csv',BASE/'v47_integer_repair_cost_audit/instrumented_repair_v45.py',v46/'runtime_v46.py',v46/'repair_v45.py',v46/'precision_check.py',v46/'audit_sources.py',v18/'CALIBRATION_MANIFEST.json',v18/'M1/checkpoint/final/model.safetensors',ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/proto['model_revision']/'model.safetensors']
    audit=[dict(path=str(p),sha256=sha(p)) for p in files]
    assert sha(v49/'R1/checkpoint/final/model.safetensors')==lock['final_model_sha256']
    assert sha(v18/'M1/checkpoint/final/model.safetensors')==proto['initialization']['sha256']
    assert sha(v18/'CALIBRATION_MANIFEST.json')==proto['calibration']['manifest_sha256']
    assert sha(v48/'fast_accept.py')==proto['algorithm_source']['sha256']
    dump('SOURCE_AUDIT.json',audit)
    rows=list(csv.DictReader((v49/'STAGE_TIMINGS.csv').open()));cc=[r for r in rows if r['stage']=='C_construction'];assert len(cc)==97
    clean={}
    for r in cc:clean[(int(r['layer']),r['module'])]=r
    assert len(clean)==96
    wall=sum(float(r['seconds']) for r in csv.DictReader((v49/'BLOCK_TIMINGS.csv').open()) if int(r['new_groups'])>0)
    detail=[]
    for (layer,group),r in clean.items():
        d=4864 if group=='down' else 896
        detail.append(dict(layer=layer,group=group,N=262144,input_dim=d,gram_rows=d,gram_cols=d,sibling_count={'qkv':3,'o':1,'up_gate':2,'down':1}[group],seconds=float(r['seconds']),scope='last successful C event per committed group'))
    with (OUT/'V49_GRAM_CALLS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    local=list(csv.DictReader((v48/'COMPARISON.csv').open()))
    groups=[dict(input_dim=d,calls=sum(r['input_dim']==d for r in detail),seconds=sum(r['seconds'] for r in detail if r['input_dim']==d)) for d in [896,4864]]
    dump('HISTORICAL_COST_EVIDENCE.json',dict(v49_formal_block_wall_seconds=wall,v49_C_all_attempt_seconds=sum(float(r['seconds']) for r in cc),v49_C_all_attempt_calls=len(cc),v49_C_successful_seconds=sum(r['seconds'] for r in detail),v49_C_successful_calls=96,v49_C_failed_seconds=sum(float(r['seconds']) for r in cc)-sum(r['seconds'] for r in detail),shapes=groups,sibling_shared_already=True,v48_local_C_inclusive_speedup=sum(float(r['control_C_plus_repair_seconds']) for r in local)/sum(float(r['fast_C_plus_repair_seconds']) for r in local),v46_vs_v49_formal_speedup=5485.675196500008/wall))
    for n in ['inputs','reports','local_states']:(OUT/n).mkdir(exist_ok=True)
    probe=OUT/'write_probe.tmp';probe.write_text('v50');assert probe.read_text()=='v50';probe.unlink()
    dump('PROTOCOL.json',dict(version='v50',requested_model='Sol high; interface cannot verify/switch',frozen_before_GPU=True,paths=['baseline_v49_FP64_Gram','mixed_FP32_block_FP64_accumulation'],token_chunk=4096,N=262144,targets=[dict(module='mlp.up_proj',rows=4864,d=896),dict(module='mlp.down_proj',rows=896,d=4864)],shadow_rows=[0,224,448,672],initialization='original trusted formal M1 codes/scales/zero/g_idx',input_prefix='locked v49 final model; recapture block0 group inputs and match saved group hashes',search_implementation='unchanged v48 batch_repair plus v45 candidate helpers',max_accepts=32,candidate_cap='128*d',TF32=False,autocast=False,rules_sha256=sha(OUT/'NUMERICAL_RULES.md'),new_path_sha256=sha(OUT/'mixed_gram.py'),runner_sha256=sha(OUT/'run_v50.py'),baseline_sha256=sha(v48/'fast_accept.py'),source_audit_sha256=sha(OUT/'SOURCE_AUDIT.json'),GPU_cap_seconds=1200,planning_estimate_seconds=910,estimate_is_not_measurement=True,no_whole_model_repair=True,no_held_or_PPL=True,no_model_checkpoint=True,freeze_unix=time.time()))
    dump('PREPARATION.json',dict(pass_all=True,CPU_only=True,output_writable=True,source_files=len(audit),protocol_sha256=sha(OUT/'PROTOCOL.json')))
if __name__=='__main__':main()
