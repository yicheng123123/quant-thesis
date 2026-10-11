import ast,csv,json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parent
p=O/'CALLSITE_INVENTORY.csv';rows=list(csv.DictReader(p.open(encoding='utf8')));added=[]
for D in [B/'range_w4_fullmodel_v18',B/'range_w4_winner_union_v40/A',B/'range_w4_winner_union_v40/B',B/'range_w4_winner_union_v40/PILOT']:
 srcfile=D/'run_precheck.py'
 if not srcfile.exists():continue
 text=srcfile.read_text(encoding='utf-8-sig')
 for fn in ast.walk(ast.parse(text)):
  if not isinstance(fn,ast.FunctionDef):continue
  norms=set();sliced={};normal={}
  for n in ast.walk(fn):
   if isinstance(n,ast.Assign):
    if isinstance(n.value,ast.Attribute) and n.value.attr=='input_layernorm':norms.update(x.id for x in n.targets if isinstance(x,ast.Name))
    if isinstance(n.value,ast.Subscript) and isinstance(n.value.slice,ast.Constant) and n.value.slice.value==0 and isinstance(n.value.value,ast.Call) and ast.unparse(n.value.value.func)=='layer':
     for x in n.targets:
      if isinstance(x,ast.Name):sliced[x.id]=n.lineno
  for n in ast.walk(fn):
   if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and (('input_layernorm' in ast.unparse(n.value.func)) or (isinstance(n.value.func,ast.Name) and n.value.func.id in norms)):
    names=[a.id for a in n.value.args if isinstance(a,ast.Name) and a.id in sliced]
    if names:
     for x in n.targets:
      if isinstance(x,ast.Name):normal[x.id]=sliced[names[0]]
  for n in ast.walk(fn):
   if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id in normal and isinstance(n.slice,ast.Constant) and n.slice.value==0:
    row=dict(file=str(srcfile),function=fn.name,line=normal[n.value.id],kind='decoder_zero_then_norm_alias_then_zero_cache_slice',expression='decoder[0] -> next_norm -> normalized[0]',followup_zero_lines=str(n.lineno),source_sha256=hashlib.sha256(srcfile.read_bytes()).hexdigest(),role='project_legacy_precheck_boundary')
    if not any(x['file']==row['file'] and str(x['line'])==str(row['line']) and x['kind']==row['kind'] for x in rows):rows.append(row);added.append(row)
with p.open('w',newline='',encoding='utf8') as f:
 cols=list(dict.fromkeys(k for x in rows for k in x));w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
s=json.loads((O/'CPU_AUDIT_SUMMARY.json').read_text());s.update(project_callsite_risk_locations=len(rows),additional_norm_alias_sites=len(added));(O/'CPU_AUDIT_SUMMARY.json').write_text(json.dumps(s,indent=2)+'\n');print('NORM_ALIAS_COMPLETE',len(added),'added',len(rows),'sites')
