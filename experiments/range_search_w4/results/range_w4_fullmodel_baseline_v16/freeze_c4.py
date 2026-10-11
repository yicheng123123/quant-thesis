"""Freeze a bounded C4 validation token stream, without inference or full shard download."""
from pathlib import Path
import gzip, urllib.request, json, hashlib, os, time
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
REV='1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
MODEL_REV='7ae557604adf67be50417f59c2c2f167def9a775'
URL=f'https://huggingface.co/datasets/allenai/c4/resolve/{REV}/en/c4-validation.00000-of-00008.json.gz'
os.environ['HF_HUB_OFFLINE']='1'
def main():
    if (OUT/'C4_EVAL_MANIFEST.json').exists(): raise RuntimeError('already frozen; reuse manifest')
    from transformers import AutoTokenizer
    started=time.perf_counter();docs=[];records=[]
    # Exclude the first32 documents already used by the repository's C4 smoke.
    with urllib.request.urlopen(URL,timeout=45) as response:
        with gzip.GzipFile(fileobj=response) as stream:
            for index in range(1056):
                obj=json.loads(stream.readline())
                if index>=32:
                    text=obj['text'];docs.append(text)
                    records.append(dict(shard_document_index=index,text_sha256=hashlib.sha256(text.encode()).hexdigest(),url=obj.get('url'),timestamp=obj.get('timestamp')))
    joined='\n\n'.join(docs)
    tok=AutoTokenizer.from_pretrained(str(ROOT/'.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots'/MODEL_REV),local_files_only=True)
    allids=tok(joined,add_special_tokens=False)['input_ids']
    if len(allids)<131072: raise RuntimeError('bounded document pool shorter than frozen token budget')
    ids=allids[:131072]
    (OUT/'C4_EVAL_TEXT.txt').write_text(joined,encoding='utf-8',newline='\n')
    manifest=dict(dataset='allenai/c4',revision=REV,config='en',split='validation',source_url=URL,
        shard='en/c4-validation.00000-of-00008.json.gz',document_start=32,document_limit=1024,
        selection='source order; documents32..1055 inclusive; join with two newlines; first131072 tokens',
        joined_text_sha256=hashlib.sha256(joined.encode()).hexdigest(),tokenizer_revision=MODEL_REV,
        add_special_tokens=False,apply_chat_template=False,token_ids=ids,
        token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),num_input_tokens=len(ids),
        num_pool_tokens=len(allids),expected_evaluated_tokens=131071,max_length=2048,stride=512,use_cache=False,
        label='C4 validation fixed token budget; not standard full C4 PPL',documents=records)
    (OUT/'C4_EVAL_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'C4_PREPARATION_COST.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-started,document_records_read=1056,
        full_shard_downloaded=False,model_inference=False),indent=2)+'\n',encoding='utf-8')
    print('C4 frozen:',len(ids),'tokens; no inference',flush=True)
if __name__=='__main__': main()
