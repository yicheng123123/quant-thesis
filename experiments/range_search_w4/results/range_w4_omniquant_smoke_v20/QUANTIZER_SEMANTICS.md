# OmniQuant weight-only W4A16 semantics

The smoke uses the official `UniformAffineQuantizer` and `QuantLinear` without modification. For each output row and each consecutive group of 128 input weights, asymmetric bounds start from group min/max; LWC applies `sigmoid(lowbound_factor)` / `sigmoid(upbound_factor)` to the respective extrema. The official affine scale is `(xmax-xmin)/15`, clamped to `[1e-5,1e4]`; zero point is `round(clamp(-xmin/scale,-1e4,1e4))`. Integer codes are rounded and clamped to `[0,15]`, then dequantized. LWC factors initialize to 4.0. `round_ste` supplies the official STE in training. W16 activation quantizer returns its input unchanged. LET is disabled; no GPTQ Hessian or error compensation is involved.

This is asymmetric affine, unlike M1/M2 symmetric GPTQ (`zero=8`); any later M2/M3 quality comparison remains method-level.
