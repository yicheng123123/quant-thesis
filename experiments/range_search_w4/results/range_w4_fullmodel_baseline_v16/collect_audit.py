"""Read-only source/result audit and CPU token-manifest freeze; no model inference."""
from pathlib import Path
import hashlib, json, random, re, os, csv, importlib.metadata as md
from collections import defaultdict

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
REV = '7ae557604adf67be50417f59c2c2f167def9a775'
SNAP = ROOT / '.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots' / REV
os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HOME=str(ROOT/'.hf_cache'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def jread(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(name,obj):
    s=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    p=OUT/name
    if p.exists():
        if p.read_text(encoding='utf-8')!=s: raise RuntimeError('frozen artifact differs: '+name)
    else: p.write_text(s,encoding='utf-8')

def articles(ds):
    current=[]; start=None; idx=0
    for r,line in enumerate(ds['text']):
        if re.match(r'^= [^=].* =$',line.strip()) and ';' not in line:
            if current:
                yield idx,start,'\n'.join(current)
                idx+=1
            current=[line];start=r
        elif current: current.append(line)
    if current: yield idx,start,'\n'.join(current)

def tokensha(ids): return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

def main():
    from datasets import Dataset
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(SNAP),local_files_only=True)
    cache=ROOT/'.hf_cache/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3'
    train=Dataset.from_file(str(cache/'wikitext-train.arrow'))
    test=Dataset.from_file(str(cache/'wikitext-test.arrow'))
    v15=OUT.parent/'range_w4_quantized_prefix_v15'
    hist=jread(v15/'article_manifest.json')
    # v15 embeds the complete independently audited v8-v14 inventory.
    v14=jread(OUT.parent/'range_w4_calibration_coverage_v14/article_manifest.json')
    old=v14['historical_blacklist']+v14['calibration']+v14['heldout']+hist['calibration']+hist['heldout']
    oldids={a['source_article_index'] for a in old}
    oldhash={a['article_sha256'] for a in old}
    eligible=[]
    for idx,row,text in articles(train):
        h=hashlib.sha256(text.encode()).hexdigest()
        if idx in oldids or h in oldhash: continue
        ids=tok(text,add_special_tokens=False)['input_ids']
        if len(ids)>=2048:
            eligible.append(dict(source_article_index=idx,source_row_index=row,article_sha256=h,
                                 full_article_tokens=len(ids),token_ids=ids[:2048],token_sha256=tokensha(ids[:2048])))
    print('eligible unseen 2048-token articles:',len(eligible),flush=True)
    if len(eligible)<128: raise RuntimeError('not enough articles for frozen 128x2048 protocol')
    random.Random(20261005).shuffle(eligible)
    chosen=eligible[:128]
    assert len({a['article_sha256'] for a in chosen})==128
    write('CALIBRATION_MANIFEST.json',dict(dataset='Salesforce/wikitext',revision='b08601e04326c79dfdd32d625aee71d232d685c3',
        config='wikitext-2-raw-v1',split='train',source_arrow_sha256=sha(cache/'wikitext-train.arrow'),
        tokenizer_revision=REV,seed=20261005,nsamples=128,seqlen=2048,token_budget=262144,
        eligibility='v15 full-article parser; exclude all v8-v15 article IDs and full article hashes; length>=2048',
        selection='Random(20261005).shuffle eligible in source order, first128, first2048 tokens each, no special tokens',
        historical_manifest_sha256=sha(v15/'article_manifest.json'),historical_articles_excluded=len(oldids),
        eligible_count=len(eligible),same_inputs_M1_M2_M3=True,samples=chosen))
    testids=tok('\n\n'.join(test['text']),add_special_tokens=False)['input_ids']
    write('WIKITEXT_TEST_MANIFEST.json',dict(dataset='Salesforce/wikitext',revision='b08601e04326c79dfdd32d625aee71d232d685c3',
        config='wikitext-2-raw-v1',split='test',source_arrow_sha256=sha(cache/'wikitext-test.arrow'),
        tokenizer_revision=REV,join='two newlines including empty rows',add_special_tokens=False,
        token_ids=testids,num_input_tokens=len(testids),token_sha256=tokensha(testids),max_length=2048,stride=512,
        expected_evaluated_tokens=len(testids)-1,use_cache=False,apply_chat_template=False))
    cfg=jread(SNAP/'config.json'); h=cfg['hidden_size']; ff=cfg['intermediate_size']; kv=cfg['num_key_value_heads']*(h//cfg['num_attention_heads'])
    dims={'self_attn.q_proj':(h,h),'self_attn.k_proj':(kv,h),'self_attn.v_proj':(kv,h),
          'self_attn.o_proj':(h,h),'mlp.gate_proj':(ff,h),'mlp.up_proj':(ff,h),'mlp.down_proj':(h,ff)}
    mods=[dict(module=f'model.layers.{l}.{n}',output_rows=o,input_dim=i,groups=i//128,weight_elements=o*i)
          for l in range(24) for n,(o,i) in dims.items()]
    with (OUT/'MODULE_INVENTORY.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=mods[0]);w.writeheader();w.writerows(mods)
    write('MODEL_INVENTORY.json',dict(config_sha256=sha(SNAP/'config.json'),module_count=len(mods),
        output_rows=sum(m['output_rows'] for m in mods),row_alpha_candidates=10*sum(m['output_rows'] for m in mods),
        quantized_weight_elements=sum(m['weight_elements'] for m in mods),
        scale_zero_pairs=sum(m['groups']*m['output_rows'] for m in mods),excluded=['lm_head tied to embed_tokens','embeddings','RMSNorm','all biases'],dims=dims))
    timings=defaultdict(list)
    for p in sorted((v15/'raw').glob('*.json')):
        r=jread(p)
        # three rows repeat the same call timing; retain one row per module/regime.
        key=(r['layer'],r['projection'],r['input_regime'])
        if not any(x['key']==list(key) for x in timings[r['projection']]):
            timings[r['projection']].append(dict(key=list(key),seconds_full10_3rows=sum(r['candidate_module_alpha_seconds'].values())))
    write('HISTORICAL_TIMING_EVIDENCE.json',dict(v15_unique_timings=dict(timings),
        v15_candidate_timing=jread(v15/'candidate_run_timing.json'),
        v15_cost_csv_sha256=sha(v15/'cost_summary.csv'),v14_cost_csv_sha256=sha(OUT.parent/'range_w4_calibration_coverage_v14/cost_summary.csv'),
        warning='v15 prefix peak memory logged after a later reset; it is not native full-model quantization peak'))
    sources=[ROOT/'experiments/range_search_w4/dynamic_core.py',ROOT/'benchmark/eval/ppl.py',ROOT/'benchmark/eval/ppl_datasets.py',
        ROOT/'.venv-gptq/Lib/site-packages/gptqmodel/quantization/gptq.py',ROOT/'.venv-gptq/Lib/site-packages/gptqmodel/models/definitions/qwen2.py',
        ROOT/'.venv-gptq/Lib/site-packages/gptqmodel/nn_modules/qlinear/__init__.py',
        ROOT/'.venv-gptq/Lib/site-packages/transformers/models/qwen2/modeling_qwen2.py']
    for folder in OUT.parent.iterdir():
        if folder.is_dir() and re.search(r'_v(?:8|9|10|11|12|13|14|15)$',folder.name):
            sources.extend(p for p in folder.rglob('*') if p.is_file())
    write('HISTORICAL_FINGERPRINTS.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)})
    official=OUT/'official_source'
    write('OFFICIAL_SOURCE_SHA256.json',{p.relative_to(official).as_posix():sha(p) for p in sorted(official.rglob('*')) if p.is_file() and '.git' not in p.relative_to(official).parts})
    write('ENVIRONMENT_AUDIT.json',{n:md.version(n) for n in ['torch','transformers','tokenizers','datasets','gptqmodel','accelerate']})
    print('CPU freeze complete; no model or quantization run',flush=True)

if __name__=='__main__': main()
