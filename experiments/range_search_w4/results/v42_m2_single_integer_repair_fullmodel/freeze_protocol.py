"""Freeze choices after CPU audit, before the first GPU call."""
import ast,json
from audit_sources import OUT,V18,V41,sha,dump
for f in ['audit_sources.py','run_v42.py']:ast.parse((OUT/f).read_text(encoding='utf8'))
assert json.loads((OUT/'SOURCE_AUDIT.json').read_text())['pass']
assert not (OUT/'GPU_START.json').exists()
dump('PROTOCOL.json',{
 'status':'FROZEN_BEFORE_GPU','experiment':'v42 formal M2 initialized batched single-coordinate integer repair transfer',
 'model_setting':'User requested Sol high; current interface cannot confirm or change this setting.',
 'v18':json.loads((V18/'PROTOCOL.json').read_text()),
 'source_audit_sha256':sha(OUT/'SOURCE_AUDIT.json'),
 'v41_protocol_sha256':sha(V41/'PROTOCOL.json'),'v41_A_source_sha256':sha(V41/'run_pilot.py'),
 'initialization':'Exactly original formal v18 M2 dense FP16 checkpoint. Pilot discarded; formal repair starts fresh from original M2.',
 'grid':'Immutable trusted original FP32 scale, zero=8, native g_idx=column//128; codes unsigned4 [0,15]. Scored/deployed weight FP16(FP32(scale*(code-8))).',
 'packing':'Auxiliary row-major low nibble first unsigned4 storage roundtrip. Original v18 has no packed deployment kernel; original dense FP16 deployment structure and SDPA retained.',
 'code_provenance':'Unique exact bitwise enumeration of all16 known values from original saved native grid metadata, not approximate floating weight division or original GPTQ code-trajectory recovery.',
 'objective':'L_i=mean((X_R.float() @ (q_FP16.float()-w_pristine_FP16.float()))^2); same current R prefix input, no dual-stream residual target.',
 'C':'Full undamped FP64 X_R.T@X_R/N accumulated in fixed 4096-token chunks, symmetrized; FP64 E and p, FP32 direct output matmul + FP64 SSE; no damped GPTQ H.',
 'A':'Each complete neighborhood: all legal coordinate +/-1; first argmin coordinate ascending, -1 then+1. Best prediction below -max(1e-14,1e-8*initial direct MSE); direct selected proposal strictly decreases; discrepancy <=max(2e-6*current MSE,1e-3*abs(prediction)); otherwise stop numerical failure.',
 'budget_per_row':{'evaluations': '128*input_dim','accepted_updates':32},
 'actual_budget':'All singles evaluated each complete round; legal count recorded. 32 neighborhoods require at most64*dim; common-node pauses do not change A choices and all five historical states must match.',
 'batching':'Output rows scored in chunks256; one Python neighborhood loop, no row search loops. Full-row matrix p maintained, independent coordinate argmin per row; one coordinate per row committed per round.',
 'numerical':'Initial/final and every accepted proposal direct recomputation; drift at updates1,8,16,24,32 and final. Original v41 tolerances above; original v18 reload rtol1e-5 atol1e-6. No actual rise accepted. Four small sequential historical reference trajectories; all96 v41 A terminals and common-budget states match.',
 'dependency_order':'qkv -> o -> up/gate -> down -> next block; siblings frozen common input/C, all siblings commit before downstream capture. Current R prefix only.',
 'checkpoint':'Atomic per dependency-group code/grid/moves/weight/input-hash/C-hash/eval-budget/committed state; restoring committed groups copies checkpoint weight and skips repair.',
 'pilot':'All7 block0 Linears, all12672 output rows, input896/4864, true dependency refresh, code packing and commits; no held/PPL parameter selection.',
 'cost_gate':'Whole measured block0 runtime x24 x1.35+30sec setup; all24 audited shapes must match. Save/reload + WT2/C4 reserve360sec. No formal start if forecast+reserve exceeds remaining hard budget.',
 'gpu_hard_cap_seconds':5400,'time_scope':'Wall from first GPU warmup through pilot/failures/diagnostics/formal/save/reload/eval. CPU provenance preparation reported separately.',
 'evaluation':'One WT2 test and one fixed-budget C4 validation through unchanged benchmark.eval.ppl.evaluate_ppl, frozen token manifests/max2048/stride512/valid targets; only after final checkpoint/hash lock. No P3, no local held gate.',
 'claims':'Application validation of existing discrete repair ideas, not official ReQuant reproduction or proof of original contribution. Local reconstruction does not imply PPL; one run not cross-seed stability.',
 'stop':'After this frozen run; no new grids/update limits/model sizes/objectives/next version.'})
dump('IMPLEMENTATION_LOCK.json',{'protocol_sha256':sha(OUT/'PROTOCOL.json'),'implementation_sha256':sha(OUT/'run_v42.py'),'audit_sha256':sha(OUT/'audit_sources.py')})
print('FROZEN',sha(OUT/'PROTOCOL.json'))
