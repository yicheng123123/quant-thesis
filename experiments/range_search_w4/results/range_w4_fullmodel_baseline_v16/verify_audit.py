"""Validate v16 frozen tokens/counts/provenance; no inference, training, or candidate generation."""
from pathlib import Path
import json, hashlib, re, math, csv, subprocess, ast, os
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
os.environ['HF_HUB_OFFLINE']='1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def j(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def tsha(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()
def main():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from collect_audit import articles
    checks={}
    required=['PROTOCOL_DRAFT.md','OMNIQUANT_AUDIT.md','QWEN25_PORT_PLAN.md','FULLMODEL_SEARCH_PLAN.md',
              'COMPUTE_BUDGET.md','PPL_EVAL_PROTOCOL.md','BASELINE_MATRIX.csv','SMOKE_TEST_PLAN.md',
              'SMOKE_RESULTS.md','RUN_REPORT.md','PROTOCOL.json']
    checks['required_deliverables_present']=all((OUT/p).is_file() for p in required)
    before=j(OUT/'HISTORICAL_FINGERPRINTS.json')
    differences=[p for p,h in before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    directories={str(Path(p).parent) for p in before if re.search(r'/range_w4_.*_v(?:8|9|10|11|12|13|14|15)/',p)}
    all_hist={p.relative_to(ROOT).as_posix() for folder in OUT.parent.iterdir() if folder.is_dir() and re.search(r'_v(?:8|9|10|11|12|13|14|15)$',folder.name) for p in folder.rglob('*') if p.is_file()}
    old_hist={p for p in before if re.search(r'/range_w4_.*_v(?:8|9|10|11|12|13|14|15)/',p)}
    checks['historical_file_hashes_unchanged']=not differences
    checks['historical_file_sets_unchanged']=all_hist==old_hist
    checks['historical_dynamic_and_eval_sources_unchanged']=all(sha(ROOT/p)==h for p,h in before.items() if p not in old_hist)
    official=OUT/'official_source'
    commit=subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()
    origin=subprocess.check_output(['git','-C',str(official),'remote','get-url','origin'],text=True).strip()
    checks['official_origin_and_commit']=origin=='https://github.com/OpenGVLab/OmniQuant.git' and commit=='feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14'
    source_paths=subprocess.check_output(['git','-C',str(official),'ls-tree','-rz','--name-only','HEAD']).decode().split('\0')
    blob_mismatch=[];blob_hashes={}
    for name in source_paths:
        if not name or not (name.endswith(('.py','.sh','.toml')) or name in ['LICENCE','README.md']):continue
        original=subprocess.check_output(['git','-C',str(official),'show',f'HEAD:{name}']).replace(b'\r\n',b'\n')
        current=(official/name).read_bytes().replace(b'\r\n',b'\n')
        if original!=current:blob_mismatch.append(name)
        blob_hashes[name]=hashlib.sha256(original).hexdigest()
    checks['official_algorithm_source_git_blob_parity']=not blob_mismatch
    cal=j(OUT/'CALIBRATION_MANIFEST.json');e1=j(OUT/'WIKITEXT_TEST_MANIFEST.json');e2=j(OUT/'C4_EVAL_MANIFEST.json');proto=j(OUT/'PROTOCOL.json')
    cache=ROOT/'.hf_cache/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3'
    train=Dataset.from_file(str(cache/'wikitext-train.arrow'));test=Dataset.from_file(str(cache/'wikitext-test.arrow'))
    source={idx:(row,text) for idx,row,text in articles(train)}
    old=j(OUT.parent/'range_w4_calibration_coverage_v14/article_manifest.json')
    last=j(OUT.parent/'range_w4_quantized_prefix_v15/article_manifest.json')
    oldlist=old['historical_blacklist']+old['calibration']+old['heldout']+last['calibration']+last['heldout']
    oldids={a['source_article_index'] for a in oldlist};oldhash={a['article_sha256'] for a in oldlist}
    calids={a['source_article_index'] for a in cal['samples']};calhash={a['article_sha256'] for a in cal['samples']}
    tok=AutoTokenizer.from_pretrained(str(ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/proto['revision']),local_files_only=True)
    token_ok=True;article_ok=True
    for a in cal['samples']:
        row,text=source[a['source_article_index']]
        article_ok &= row==a['source_row_index'] and hashlib.sha256(text.encode()).hexdigest()==a['article_sha256']
        ids=tok(text,add_special_tokens=False)['input_ids'][:2048]
        token_ok &= ids==a['token_ids'] and tsha(ids)==a['token_sha256']
    checks['calibration128_unique_whole_articles']=len(calids)==len(calhash)==len(cal['samples'])==128
    checks['calibration262144_exact_token_budget']=all(len(a['token_ids'])==2048 for a in cal['samples']) and sum(len(a['token_ids']) for a in cal['samples'])==262144
    checks['calibration_disjoint_all_v8_v15_articles']=not (calids&oldids or calhash&oldhash)
    checks['historical_exclusion280_confirmed']=len(oldids)==cal['historical_articles_excluded']==280
    checks['calibration_tokens_and_article_hash_rebuilt']=bool(token_ok and article_ok)
    checks['source_arrow_hashes_fixed']=sha(cache/'wikitext-train.arrow')==cal['source_arrow_sha256'] and sha(cache/'wikitext-test.arrow')==e1['source_arrow_sha256']
    testids=tok('\n\n'.join(test['text']),add_special_tokens=False)['input_ids']
    checks['E1_exact_test_tokens_rebuilt']=testids==e1['token_ids'] and tsha(testids)==e1['token_sha256'] and len(testids)-1==e1['expected_evaluated_tokens']
    text=(OUT/'C4_EVAL_TEXT.txt').read_text(encoding='utf-8')
    c4ids=tok(text,add_special_tokens=False)['input_ids'][:131072]
    checks['E2_exact_bounded_tokens_rebuilt']=c4ids==e2['token_ids'] and tsha(c4ids)==e2['token_sha256'] and hashlib.sha256(text.encode()).hexdigest()==e2['joined_text_sha256']
    checks['E2_documents_and_fixed_budget']=len(e2['documents'])==1024 and [r['shard_document_index'] for r in e2['documents']]==list(range(32,1056)) and len(c4ids)==131072 and e2['expected_evaluated_tokens']==131071
    testhash={hashlib.sha256(text.encode()).hexdigest() for _,_,text in articles(test)}
    c4hash={a['text_sha256'] for a in e2['documents']}
    checks['cal_full_article_hashes_disjoint_final_eval_documents']=not (calhash&testhash or calhash&c4hash)
    inv=j(OUT/'MODEL_INVENTORY.json');mods=list(csv.DictReader((OUT/'MODULE_INVENTORY.csv').open(encoding='utf-8')))
    checks['exact_candidate_inventory']=len(mods)==168 and sum(int(m['output_rows']) for m in mods)==304128 and inv['row_alpha_candidates']==3041280
    checks['native_protocol_preserved']=proto['native']==dict(bits=4,activation_bits=16,sym=True,group_size=128,block_size=128,damp_percent=.1,desc_act=False,static_groups=False,true_sequential=True,mse=0.,math='FP32',damp_auto_increment=0)
    checks['grid_per_row_shared_and_no_gate']=proto['search']['grid']==[k/10 for k in range(1,11)] and proto['search']['no_gate'] and 'per output row' in proto['search']['unit']
    checks['official_weight_only_configuration_frozen']=proto['omni']['lwc'] and not proto['omni']['let'] and not proto['omni']['symmetric'] and proto['omni']['epochs']==20 and proto['omni']['batch_size']==1
    smoke=j(OUT/'BUDGET_SMOKE.json')
    checks['limited_budget_smoke_finite_and_state_isolated']=smoke['candidate_slots']==70 and all(r['finite'] for r in smoke['cases']) and all(r['batch_first3_half_exact'] and r['pristine_and_hinv_unchanged'] and r['quadratic_loss_allclose'] for r in smoke['cases'] if r['rows']==32)
    checks['python_deliverables_compile']=True
    for path in OUT.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'));compile(path.read_text(encoding='utf-8'),str(path),'exec')
    checks['markdown_no_control_characters']=all(not re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',p.read_text(encoding='utf-8')) for p in OUT.glob('*.md'))
    checks['run_commands_intact']='experiments\\range_search_w4\\results\\range_w4_fullmodel_baseline_v16\\collect_audit.py' in (OUT/'RUN_REPORT.md').read_text(encoding='utf-8')
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
        verdict='FULLMODEL_BASELINE_PROTOCOL_READY' if all(checks.values()) else 'FULLMODEL_BASELINE_PROTOCOL_NOT_READY',
        scope='protocol/manifests/audit validation, not Omni port compatibility or full-model quality',
        historical_files_checked=len(before),historical_mismatches=differences,official_blob_mismatches=blob_mismatch,
        official_code_blob_sha256=blob_hashes,calibration_historical_overlap_count=len(calids&oldids),
        Omni_Qwen_compatibility_smoke='NOT_RUN',full_model_quantization='NOT_RUN',PPL='NOT_RUN')
    (OUT/'VERIFY_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (OUT/'ARTIFACT_SHA256.json').write_text(json.dumps({p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='ARTIFACT_SHA256.json'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='official_code_blob_sha256'},ensure_ascii=False,indent=2))
    if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
