"""Freeze hashes for the immutable v36 inputs used by the v37 correctness audit."""
from __future__ import annotations
import hashlib, json, platform, subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
V36=ROOT/"experiments/range_search_w4/results/range_w4_awq_gptq_v36"
PREFIX=ROOT/"experiments/range_search_w4/results/range_w4_fullmodel_v18/M2/checkpoint/final"
BASE=ROOT/".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"

def hfile(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()

manifest_path=HERE/"manifests/SOURCE_ARTIFACT_HASHES.json"
previous_manifest=None
if manifest_path.exists():
 previous_manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
 backup=HERE/"manifests/SOURCE_ARTIFACT_HASHES_PRE_GPU_INITIAL.json"
 if not backup.exists():backup.write_text(json.dumps(previous_manifest,indent=2)+"\n",encoding="utf-8")
rel=[
 "run_v36.py","manifests/FROZEN_PROTOCOL.json","manifests/ARTICLE_MANIFEST.json","manifests/GROUP_TOKEN_CACHE_MANIFEST.json",
 "manifests/ENVIRONMENT.json","manifests/SOURCE_HASHES.json","JOB_TABLE.csv","JOB_TABLE_AUDIT.json",
 "raw/FINAL_RUN_REPORT.json","raw/RUN_STATE.json","raw/ACTUAL_EXECUTION_AUDIT.json","raw/COST_AUDIT.json",
 "raw/BUDGET_LEDGER.csv","raw/ATTEMPT_LOG.csv","raw/FAILED_ATTEMPTS.csv","correctness/STOP_DIAGNOSIS.json",
 "correctness/S0_PRE_HELD_CORRECTNESS.json","correctness/S0_POST_HELD_CORRECTNESS.json","correctness/S0_D_identity.json",
 "correctness/S0_selected_D_full_mlp_fold.json","raw/S0_P0_capture.json","raw/S0_P1_capture.json","raw/S1_P2_capture.json",
 "cache/P0_tokens.json","cache/P1_tokens.json","cache/P2_tokens.json",
 "cache/S0_P0_hidden_l12.f16","cache/S0_P0_prenorm.f16","cache/S0_P0_postnorm.f16",
 "cache/S0_P1_hidden_l12.f16","cache/S0_P1_prenorm.f16","cache/S0_P1_postnorm.f16",
 "cache/S1_P2_hidden_l12.f16","cache/S1_P2_prenorm.f16","cache/S1_P2_postnorm.f16",
 "cache/S0_G00_down_input.f16","cache/S0_G01_down_input.f16","cache/S0_G10_down_input.f16","cache/S0_G11_down_input.f16",
 "cache/S0_H_down_G00.pt","cache/S0_H_down_G01.pt","cache/S0_H_down_G10.pt","cache/S0_H_down_G11.pt",
 "cache/S0_H_gateup_original.pt","cache/S0_H_gateup_scaled.pt","cache/S0_D_ratio_bank.pt",
 "raw/bank_S0_original_gate_proj.pt","raw/bank_S0_original_up_proj.pt","raw/bank_S0_scaled_gate_proj.pt","raw/bank_S0_scaled_up_proj.pt",
 "raw/downbank_S0_G00.pt","raw/downbank_S0_G01.pt","raw/downbank_S0_G10.pt","raw/downbank_S0_G11.pt",
 "final/S0_selected_D.pt","final/S0_selected_tensors.pt","final/S0_SELECTION_LOCK.json","raw/S0_held_mlp_outputs.npz","raw/S0_held_result.json"
]
files={}
for r in rel:
 p=V36/r
 if not p.is_file():raise FileNotFoundError(p)
 files[str(p.relative_to(ROOT)).replace("\\","/")]=hfile(p)
external={}
for label,p in (("prefix_checkpoint",PREFIX/"model.safetensors"),("prefix_config",PREFIX/"config.json"),
                ("base_checkpoint",BASE/"model.safetensors"),("base_config",BASE/"config.json")):
 if not p.is_file():raise FileNotFoundError(p)
 external[label]={"path":str(p),"sha256":hfile(p),"size_bytes":p.stat().st_size}
head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
manifest={"protocol_id":"v37_v36_correctness_repair_and_s0_validity","git_head":head,
 "frozen_protocol_sha256":hfile(HERE/"FROZEN_DIAGNOSTIC_PROTOCOL.json"),
 "correctness_script_sha256":hfile(HERE/"run_v37_correctness.py"),
 "freeze_script_sha256":hfile(Path(__file__)),"python":platform.python_version(),
 "v36_sources":files,"model_files":external,
 "dependency_sources":{"experiments/range_search_w4/results/range_w4_m2_parity_v17_1/native_m2.py":
  hfile(ROOT/"experiments/range_search_w4/results/range_w4_m2_parity_v17_1/native_m2.py")},
 "pre_gpu_import_failure":None if previous_manifest is None else {
  "status":"IMPORT_ERROR_BEFORE_CUDA_INITIALIZATION","module":"native_m2",
  "cause":"The v37 preflight initially omitted the v17.1 Python source path used by v36 for native parity.",
  "gpu_seconds_charged":0.0,"superseded_audit_script_sha256":previous_manifest["correctness_script_sha256"]},
 "write_scope":"Only v37 directory. All v36 inputs listed here are read-only and hash-pinned."}
manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"status":"FROZEN","v36_files":len(files),"models":external,"protocol_sha256":manifest["frozen_protocol_sha256"],
 "audit_script_sha256":manifest["correctness_script_sha256"]},indent=2))
