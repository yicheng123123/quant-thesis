"""Re-execute the fixed disposable failure; require a new result path."""
import argparse

import torch

from audit_common import HERE, context, write_json, sha_file
from replay_basic import run_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['D1','D2','D2_FULL_FP32','D3'], default='D1')
    parser.add_argument('--repeats', type=int, default=3, choices=[1,2,3])
    cli = parser.parse_args()
    path = __import__('pathlib').Path(cli.output).resolve()
    if path.exists() or not path.is_relative_to(HERE):
        raise RuntimeError('use a new output path inside the v21.2 directory; historical results are preserved')
    context()
    fixture_path = HERE / 'fixtures/actual_l21.pt'
    fixture = torch.load(fixture_path,map_location='cpu',weights_only=False)
    runs = [run_step(fixture,mode=cli.mode,native_call=cli.mode=='D1') for _ in range(cli.repeats)]
    write_json(path,{'diagnostic_only':True,'fixture_sha256':sha_file(fixture_path),'mode':cli.mode,'runs':runs})
    print([(r['loss'],r['all_lwc_gradients_finite']) for r in runs],flush=True)


if __name__ == '__main__':
    main()
