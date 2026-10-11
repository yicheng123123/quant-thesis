import hashlib, json, os, sys, time
from pathlib import Path
import torch
ROOT=Path(r'E:\quant-thesis'); OUT=Path(__file__).resolve().parent; V46=ROOT/'experiments/range_search_w4/results/v46_m1_single_integer_repair_fullmodel'
sys.path.insert(0,str(V46))
import runtime_v46 as rt

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
protocol_sha=sha(OUT/'PROTOCOL.json')
orig=torch.load(OUT/'original_codes/layer_00.pt',map_location='cpu',weights_only=True)
entries=[]
for group in ['qkv','o','up_gate','down']:
 p=OUT/'formal/groups'/f'layer_00_{group}.pt'
 cp=torch.load(p,map_location='cpu',weights_only=False)
 assert cp['committed'] and cp['protocol_sha256']==protocol_sha and cp['layer']==0 and cp['group']==group
 before=sha(p);mods=[]
 for name,r in cp['modules'].items():
  s=orig[name];codes=r['codes'].cpu().byte()
  assert codes.shape==s['codes'].shape and bool(((codes>=0)&(codes<=15)).all())
  for key in ['scales','zeros','g_idx']: assert torch.equal(r[key].cpu(),s[key].cpu()),(group,name,key)
  packed=rt.pack(codes).cpu();assert torch.equal(rt.unpack(packed).cpu(),codes)
  q=(s['scales'][:,s['g_idx'].long()]*(codes.float()-8)).half()
  assert torch.equal(q.view(torch.int16),r['weight'].cpu().half().view(torch.int16))
  assert 'packed_u4' not in r
  r['packed_u4']=packed
  mods.append({'module':name,'rows':codes.shape[0],'input_dim':codes.shape[1],'codes_sha256':hashlib.sha256(codes.contiguous().numpy().tobytes()).hexdigest(),'packed_bytes':packed.numel()})
 tmp=p.with_suffix('.pt.schema001.tmp');torch.save(cp,tmp);tmp.replace(p)
 check=torch.load(p,map_location='cpu',weights_only=False)
 for name,r in check['modules'].items(): assert torch.equal(rt.unpack(r['packed_u4']),r['codes'].cpu().byte())
 entries.append({'group':group,'checkpoint':str(p),'old_sha256':before,'new_sha256':sha(p),'input_sha256':cp['input_sha256'],'C_sha256':cp['C_sha256'],'modules':mods,'code_grid_and_dequantized_weight_unchanged':True})
status=OUT/'formal/COMMIT_STATUS.json';st=json.loads(status.read_text(encoding='utf8'));last=OUT/st['group_checkpoint'].replace('E:\\quant-thesis\\experiments\\range_search_w4\\results\\v49_fullmodel_fast_integer_repair\\','')
assert st['last_committed_layer']==0 and st['group']=='down' and last.exists()
st['sha256']=sha(last);st['checkpoint_schema_amendment']='001';status.write_text(json.dumps(st,indent=2,ensure_ascii=False)+chr(10),encoding='utf8')
record={'status':'PASS','amendment_id':'001','created_unix':time.time(),'reason':'The frozen v48 batch_repair retains its packed tensor locally but omits packed_u4 from the result dictionary; v49 group reload checks require that field.','change':'Added packed_u4=the frozen runtime pack(codes) representation to each already committed block0 group checkpoint.','algorithm_or_selection_changed':False,'codes_scales_zeros_g_idx_or_weights_changed':False,'groups_repaired_again':False,'protocol_sha256':protocol_sha,'updated_commit_status_sha256':sha(status),'groups':entries}
(OUT/'CHECKPOINT_SCHEMA_AMENDMENT_001.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+chr(10),encoding='utf8')
print('CHECKPOINT_SCHEMA_MIGRATION_PASS',[(e['group'],e['old_sha256'],e['new_sha256']) for e in entries])
