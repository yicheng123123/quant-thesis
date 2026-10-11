import json,time,hashlib
from pathlib import Path
t0=time.time()
out=Path(__file__).resolve().parent

def load(p): return json.loads((out/p).read_text(encoding='utf8'))
def save(p,d):
    q=out/p
    q.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')

reportp=out/'REPORT.md'
report=reportp.read_text(encoding='utf8')
old='- 从本次layer2至23的正式依赖组块wall合计：4240.102秒；新增接受更新 8801399。完整formal runner进程wall（含模型构建、恢复前缀、修复、传播与save/reload直到配置序列化断言）：4383.618秒。'
new='- 本轮有效正式修复增量成本（新修layer2至23的完整依赖组块wall）：4240.102秒；新接受更新 8801399。有效复用的block0/1组未在v55重复修复，其历史生成时间已经包含于继承账本。本次完整formal runner进程wall（含模型构建、恢复前缀、修复、传播与保存/reload至原配置序列化断言）：4383.618秒。'
assert old in report
report=report.replace(old,new)
reportp.write_text(report,encoding='utf8')

ledgerp=out/'ACTIVE_JOB_LEDGER.json'
ledger=load('ACTIVE_JOB_LEDGER.json')
actual=max(0,time.time()-t0)
ledger['segments'].append({
 'version':'v55','label':'v55 final status and report wording correction',
 'charged_seconds':actual+5.0,'cpu_only':True,'GPU_calls':0,
 'measured_script_seconds':actual,'conservative_closeout_pad_seconds':5.0,
 'basis':'CPU-only final metadata correction; model, checkpoint, and PPL outputs unchanged.'
})
ledger['finalized_unix']=time.time()
save('ACTIVE_JOB_LEDGER.json',ledger)
charged=[s.get('charged_seconds',max(0,time.time()-s.get('start_epoch',time.time()))) for s in ledger['segments']]
total=float(ledger['v51_charged_seconds'])+sum(charged)
v55=sum(v for s,v in zip(ledger['segments'],charged) if s.get('version')=='v55')
cpu=sum(v for s,v in zip(ledger['segments'],charged) if s.get('version')=='v55' and s.get('cpu_only'))
noncpu=v55-cpu
fb=load('FINAL_BUDGET_ACCOUNTING.json')
fb.update(final_cumulative_charged_seconds=total,remaining_seconds=12600-total,
          v55_added_charged_seconds=v55,v55_cpu_only_charged_seconds=cpu,
          v55_non_cpu_only_process_wall_seconds=noncpu,
          reserve_met=(total<=12600-360))
save('FINAL_BUDGET_ACCOUNTING.json',fb)
status=load('RUN_STATUS.json')
status.update(category='COMPLETE_MODEL_AND_PPL',shared_elapsed_seconds=total,
              shared_remaining_seconds=12600-total,finalized_unix=time.time())
save('RUN_STATUS.json',status)
complete=load('COMPLETE.json')
complete['final_budget_sha256']=hashlib.sha256((out/'FINAL_BUDGET_ACCOUNTING.json').read_bytes()).hexdigest()
save('COMPLETE.json',complete)
print(json.dumps({'status':status['status'],'category':status['category'],'total':total,'remaining':12600-total,'v55_added':v55,'cpu_only':cpu},indent=2))
