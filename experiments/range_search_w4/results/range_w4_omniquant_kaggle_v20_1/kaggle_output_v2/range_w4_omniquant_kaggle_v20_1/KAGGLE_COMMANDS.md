# Kaggle commands

Run these from a Kaggle notebook with one GPU enabled and the bundle mounted under `/kaggle/input`. The commands use only Linux paths and keep outputs in `/kaggle/working`.

```bash
python /kaggle/input/<bundle-dataset>/kaggle_bundle/environment_check.py \
  --output /kaggle/working/KAGGLE_ENVIRONMENT.json \
  --disk-path /kaggle/working

python /kaggle/input/<bundle-dataset>/kaggle_bundle/verify_bundle.py \
  --bundle-root /kaggle/input/<bundle-dataset>/kaggle_bundle \
  --output /kaggle/working/BUNDLE_VERIFY.json

python /kaggle/input/<bundle-dataset>/kaggle_bundle/audit_step_count.py \
  --bundle-root /kaggle/input/<bundle-dataset>/kaggle_bundle \
  --output /kaggle/working/FORMAL_STEP_COUNT.json

python /kaggle/input/<bundle-dataset>/kaggle_bundle/audit_calibration.py \
  --manifest /kaggle/input/<bundle-dataset>/kaggle_bundle/calibration/CALIBRATION_MANIFEST.json \
  --output /kaggle/working/KAGGLE_CALIBRATION_CHECK.json

python /kaggle/input/<bundle-dataset>/kaggle_bundle/run_preflight.py \
  --bundle-root /kaggle/input/<bundle-dataset>/kaggle_bundle \
  --output-dir /kaggle/working/range_w4_omniquant_kaggle_v20_1
```

If package versions differ from the frozen v20 port environment, inspect `requirements_kaggle.txt` and install only those pins before running. Do not replace the Kaggle CUDA-enabled torch build.

The preflight downloads only the exact model revision in the config if no local snapshot is supplied. A local snapshot must include a revision marker and pass every frozen model-file SHA-256 check. The code always selects `cuda:0` and records all visible GPUs.
