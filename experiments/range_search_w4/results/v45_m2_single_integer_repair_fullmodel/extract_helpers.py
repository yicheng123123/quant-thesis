"""Extract unchanged read-only v44 CPU audit functions, never invoke its main."""
import ast
from pathlib import Path
OUT=Path(__file__).resolve().parent
s=(OUT.parent/'v44_m2_single_integer_repair_fullmodel/finalize_v44.py').read_text(encoding='utf8')
wanted={'verify_delta','replay_legacy','replay_revision','ppl_audit'}
pieces=[ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
assert len(pieces)==4
head='''"""Unchanged extracted v44 CPU audit functions, not its main."""
import csv,json,math,torch
from pathlib import Path
from audit_sources import OUT,V18,sha
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\\n',encoding='utf8')
def csvread(n):return list(csv.DictReader((OUT/n).open(encoding='utf8')))
def csvwrite(n,rr):
 if not rr:return
 with (OUT/n).open('w',encoding='utf8',newline='') as fp:
  w=csv.DictWriter(fp,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
'''
(OUT/'audit_helpers.py').write_text(head+'\n\n'.join(pieces)+'\n',encoding='utf8')
print('HELPERS_EXTRACTED',len(pieces))
