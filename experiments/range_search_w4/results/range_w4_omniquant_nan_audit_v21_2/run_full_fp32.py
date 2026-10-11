"""Allowed pure FP32 diagnostic, including unoptimized original weight leaves."""
import torch
from audit_common import HERE,context,read_json,write_json
from replay_basic import run_step
from trace_backward import TraceObserver


def main():
    context()
    if read_json(HERE/'FAILURE_REPRODUCIBILITY.json')['status']!='PASS':
        raise RuntimeError('repro gate failed')
    path=HERE/'DIAGNOSTIC_D2_FULL_FP32.json'
    if path.exists():
        raise RuntimeError('pure FP32 diagnostic already exists; preserve it')
    fixture=torch.load(HERE/'fixtures/actual_l21.pt',map_location='cpu',weights_only=False)
    result=run_step(fixture,mode='D2_FULL_FP32',native_call=False,observer=TraceObserver('D2_FULL_FP32'))
    write_json(path,result)
    print('FULL FP32',result['loss'],result['all_lwc_gradients_finite'],flush=True)


if __name__=='__main__':
    main()
