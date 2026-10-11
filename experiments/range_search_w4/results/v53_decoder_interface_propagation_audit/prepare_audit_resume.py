from pathlib import Path
O=Path(__file__).resolve().parent;p=O/'audit_history.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace("res['nll']","res.get('nll',res.get('average_nll'))").replace("res['ppl']","res.get('ppl',res.get('perplexity'))")
p.write_text(s,encoding='utf8')
# A resumed CPU-only audit may reuse the saved AST inventory once the original scan reaches it; no model or GPU work repeats.
header=s[:s.index('# Actual expression inventory')]
tail=s[s.index('source_paths='):]
resume=header+"locations=list(csv.DictReader((O/'CALLSITE_INVENTORY.csv').open(encoding='utf8')));excluded=list(csv.DictReader((O/'NON_DECODER_DOUBLE_INDEX.csv').open(encoding='utf8')))\n"+tail
(O/'complete_history_audit.py').write_text(resume,encoding='utf8');compile(resume,'complete_history_audit.py','exec')
print('HISTORY_FORMAT_COMPATIBLE_RESUME_READY')

