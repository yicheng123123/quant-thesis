from pathlib import Path
import json,csv,hashlib,math
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
V18=OUT.parent/'range_w4_fullmodel_v18'
V45=OUT.parent/'v45_m2_single_integer_repair_fullmodel'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def aggregate(rawp,windowp,manifest_sha,tokens,windows,model_sha=None):
    r=json.loads(rawp.read_text());rows=list(csv.DictReader(windowp.open(encoding='utf8')))
    assert r['status']=='PASS' and r['max_length']==2048 and r['stride']==512
    assert (r.get('manifest_sha256') or r.get('evaluation_manifest_sha256'))==manifest_sha
    n=sum(int(x['valid_tokens']) for x in rows);nll=sum(float(x['window_nll']) for x in rows)/n
    assert n==tokens and len(rows)==windows
    saved=r.get('average_nll') or r.get('exact_window_weighted_average_nll')
    if saved is not None:assert abs(nll-float(saved))<1e-10
    ppl=float(r.get('perplexity') or r.get('ppl'));assert abs(ppl-math.exp(nll))<1e-8
    if model_sha is not None:
        wf=r.get('model_weight_files')
        if wf:assert wf[0]['sha256']==model_sha
        elif r.get('model_sha256'):assert r['model_sha256']==model_sha
    return {'status':'PASS','ppl':ppl,'average_nll_valid_tokens':nll,'valid_tokens':n,'windows':len(rows),'result_sha256':sha(rawp),'windows_sha256':sha(windowp)}
def main():
    p=json.loads((V18/'PROTOCOL.json').read_text());cal=V18/'CALIBRATION_MANIFEST.json'
    assert sha(cal)==p['source_protocol']['v16_calibration_manifest_sha256']
    assert p['model']['revision']=='7ae557604adf67be50417f59c2c2f167def9a775'
    assert p['quantization']['weight_bits']==4 and p['quantization']['activation_bits']==16 and p['quantization']['group_size']==128 and p['quantization']['symmetric']
    form=json.loads((V18/'FORMAL_RUNTIME.json').read_text());models={}
    for m in ['M1','M2']:
        msha=form[m.lower()]['final_model_sha256'];ck=V18/m/'checkpoint/final/model.safetensors'
        assert sha(ck)==msha
        rr=json.loads((V18/m/'RUN_REPORT.json').read_text())
        assert rr['status']=='COMPLETE' and rr['final_model_sha256']==msha and rr['linear_modules']==168
        assert json.loads((V18/m/'RELOAD_CORRECTNESS.json').read_text())['pass']
        models[m]={'checkpoint_sha256':msha,'checkpoint_bytes':ck.stat().st_size,'config_sha256':sha(V18/m/'checkpoint/final/config.json'),'run_report_sha256':sha(V18/m/'RUN_REPORT.json')}
    manifests={'wikitext':p['source_protocol']['v16_wikitext_manifest_sha256'],'c4_fixed_budget':p['source_protocol']['v16_c4_manifest_sha256']}
    counts={'wikitext':(299077,582),'c4_fixed_budget':(131071,253)};baseline={}
    for m in ['M1','M2']:
        for ds in manifests:baseline[f'{m}_{ds}']=aggregate(V18/f'evaluation/raw/{m}_{ds}.json',V18/f'evaluation/windows/{m}_{ds}.csv',manifests[ds],*counts[ds],model_sha=models[m]['checkpoint_sha256'])
    lock=json.loads((V45/'CHECKPOINT_LOCK.json').read_text());r2sha=lock['final_model_sha256']
    assert sha(V45/'R/checkpoint/final/model.safetensors')==r2sha
    assert json.loads((V45/'DEPLOYMENT_LOCK.json').read_text())['configuration_pass'] is True
    for ds in manifests:baseline[f'R2_{ds}']=aggregate(V45/f'R_{ds}.json',V45/f'R_{ds}_WINDOWS.csv',manifests[ds],*counts[ds],model_sha=r2sha)
    src=json.loads((OUT/'SOURCE_AUDIT.json').read_text());assert src['pass'] and src['modules']==168 and src['rows']==304128
    report={'pass':True,'comparison_protocol_shared':True,'model_revision':p['model']['revision'],'calibration_sha256':sha(cal),'quantization_protocol_sha256':sha(V18/'PROTOCOL.json'),'evaluation_manifests':manifests,'token_counts':counts,'models':models,'ppl_and_window_baselines':baseline,'M1_zero_reload_and_quantization_correctness':True,'M1_method':'native GPTQ; all rows selected alpha=1.0','M2_method':'scalar10 independent row-alpha selection','R2':'v45 final checkpoint and lock verified; windows reaggregated; no reevaluation','M1_codes':'v46 SOURCE_AUDIT: all16 legal code values bitwise enumerated and uniquely matched per stored FP16 coordinate','P3_read':False}
    (OUT/'BASELINE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('BASELINE_AUDIT_PASS')
if __name__=='__main__':main()
