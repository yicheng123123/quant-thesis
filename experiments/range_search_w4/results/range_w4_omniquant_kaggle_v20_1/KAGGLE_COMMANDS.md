# v20.1 Kaggle commands and run record

The completed private preflight is kernel version 2: `yichenggaoooo/w4-omniquant-v20-1-2048-token-preflight`. Its dataset is `yichenggaoooo/range-w4-omniquant-v20-1-bundle` (private). The v2 bundle archive is `kaggle_bundle_v20_1_r2.zip`, SHA-256 `a39c94df22754e8f231e30cadbf579a091695e90bb52a069db42d2a8187de213`. The earlier failed attempt and its output remain preserved separately.

To reproduce the preflight in a Kaggle GPU notebook with the private dataset attached:

```bash
B=/kaggle/input/datasets/yichenggaoooo/range-w4-omniquant-v20-1-bundle/kaggle_bundle
python "$B/environment_check.py" --output /kaggle/working/KAGGLE_ENVIRONMENT.json --disk-path /kaggle/working
python "$B/verify_bundle.py" --bundle-root "$B" --output /kaggle/working/BUNDLE_VERIFY.json
python "$B/audit_step_count.py" --bundle-root "$B" --output /kaggle/working/FORMAL_STEP_COUNT.json
python "$B/audit_calibration.py" --manifest "$B/calibration/CALIBRATION_MANIFEST.json" --output /kaggle/working/KAGGLE_CALIBRATION_CHECK.json
python "$B/run_preflight.py" --bundle-root "$B" --output-dir /kaggle/working/range_w4_omniquant_kaggle_v20_1
```

Use the frozen notebook/device settings and do not run formal training or PPL from this preflight command set.

The output was retrieved with:

```bash
kaggle kernels output yichenggaoooo/w4-omniquant-v20-1-2048-token-preflight \
  --path <this-result-directory>/kaggle_output_v2
```

The archive and all 29 directly downloaded members were hash-compared; see `KAGGLE_OUTPUT_MANIFEST.json`.

Recompute the local step-count-independent correctness and runtime sensitivity audit with Python 3.10+:

```bash
python -B audit_cost_model.py
```

For package-only checks without another GPU run:

```bash
python kaggle_bundle/audit_step_count.py \
  --bundle-root kaggle_bundle --output FORMAL_STEP_COUNT.json
python kaggle_bundle/audit_calibration.py \
  --manifest kaggle_bundle/calibration/CALIBRATION_MANIFEST.json \
  --output KAGGLE_CALIBRATION_CHECK.json
python kaggle_bundle/verify_bundle.py \
  --bundle-root kaggle_bundle --output BUNDLE_VERIFY.json
```

The completed timing evidence came from Kaggle `cuda:0`; rerunning the local package checks does not replace that GPU measurement. Formal calibration and full PPL were not run.
