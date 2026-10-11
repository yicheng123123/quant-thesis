"""CPU-only consistency/provenance checks; never execute model diagnostics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def reject(value):
        raise ValueError(f'nonfinite JSON literal: {value}')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


def main():
    parsed={p.name:read(p) for p in HERE.rglob('*.json')}
    for p in HERE.glob('*.py'):
        compile(p.read_text(encoding='utf-8'),str(p),'exec')
    required=['PROTOCOL.json','FAILURE_REPRODUCIBILITY.json','FORWARD_NUMERICS.json',
              'FIRST_NONFINITE_BACKWARD.json','AMP_STATE.json','AMP_DIAGNOSTIC_COMPARISON.json',
              'QUANTIZER_NUMERICS_AUDIT.json','OPTIMIZER_STATE_AUDIT.json','LAYER_SPECIFICITY.json',
              'SAMPLE_SPECIFICITY.json','correctness_checks.json','FINAL_STATUS.json']
    assert all(name in parsed for name in required)
    for name in ['REPORT.md','ROOT_CAUSE_ASSESSMENT.md','AMP_NUMERICAL_STABILITY_ASSESSMENT.md','REPRO_COMMANDS.md']:
        assert (HERE/name).stat().st_size>0
    with (HERE/'LWC_GRADIENT_FAILURE_MAP.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==126
    replicas=[parsed[f'REPLAY_{i}.json'] for i in (1,2,3)]
    for row in rows:
        replica=replicas[int(row['replay'])-1]
        values=(replica['gradients'] if row['phase']=='after_native_scaler_call'
                else replica['gradient_phases'][row['phase']])
        match=next(v for v in values if v['parameter_name']==row['parameter_name'])
        for key in ['finite_count','nan_count','posinf_count','neginf_count']:
            assert int(row[key])==match['gradient'][key]
    assert parsed['correctness_checks.json']['status']=='PASS'
    assert parsed['FINAL_STATUS.json']['verdict']=='M3_NUMERICAL_ROOT_CAUSE_IDENTIFIED'
    assert parsed['GEMM_OVERFLOW_VERIFICATION.json']['status']=='PASS'
    assert parsed['FIRST_NONFINITE_BACKWARD_D2_FULL_FP32.json']['all_observed_backward_gradients_finite']
    assert all(r['loss']==0.15326565504074097 and r['optimizer']['step_call_count']==0 for r in replicas)
    assert parsed['LAYER_SPECIFICITY.json']['runs'][0]['loss']==0.07404229044914246
    code=parsed['CODE_HASHES.json']
    assert set(code)=={p.name for p in HERE.glob('*.py')}
    assert all(digest(HERE/name)==sha for name,sha in code.items())
    fixtures=parsed['FIXTURE_MANIFEST.json']['files']
    assert all(digest(HERE/item['path'])==item['sha256'] for item in fixtures)
    protected=parsed['PROTECTED_HASHES.json']
    assert all(digest(Path(path))==sha for path,sha in protected.items())
    assert not list(HERE.glob('*.tmp'))
    result={'status':'PASS','JSON_files_checked':len(parsed),'CSV_rows_checked':len(rows),
            'code_files_checked':len(code),'fixtures_checked':len(fixtures),
            'protected_parent_files_checked':len(protected),
            'checks':['strict JSON without nonfinite literals','all diagnostic Python syntax',
                      'CSV equals original gradient counts','required delivery files',
                      'three exact historical-loss failure repeats','same-operand overflow verification',
                      'full FP32 observed backward finite','L20 historic first-loss match',
                      'code and fixture SHA256','protected historical SHA256'],
            'GPU_forward_backward_executed':False}
    (HERE/'ARTIFACT_INTEGRITY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (HERE/'PROTECTED_ARTIFACT_CHECK.json').write_text(json.dumps({
        'status':'PASS','protected_files':len(protected),'all_hashes_unchanged':True},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
