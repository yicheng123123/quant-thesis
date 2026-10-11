from pathlib import Path
O=Path(__file__).resolve().parent;p=O/'audit_history.py';s=p.read_text(encoding='utf8')
s=s.replace("try:s=p.read_text(encoding='utf-8-sig');a=ast.parse(s)","try:\n  s=p.read_text(encoding='utf-8-sig')\n  if '[0][0]' not in s and not re.search(r'\\b\\w+\\s*=\\s*(?:layer|block|decoder_layer)\\(',s):continue\n  a=ast.parse(s)")
p.write_text(s,encoding='utf8');compile(s,str(p),'exec');print('CPU_STATIC_SCAN_NARROWED_WITHOUT_CHANGING_SEARCH_SCOPE')
