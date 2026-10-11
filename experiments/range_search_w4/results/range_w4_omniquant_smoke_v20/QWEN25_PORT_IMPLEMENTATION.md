# Qwen2.5 architecture adapter

The copied, frozen OmniQuant source is under `upstream/`. `models/int_qwen2_layer.py` deep-copies the installed native Qwen2DecoderLayer and replaces only its seven Qwen projection `nn.Linear` modules with the unchanged upstream `QuantLinear`. Its `forward` delegates to that copied native decoder layer. Qwen's attention class, GQA, RoPE, masks, cache API, residuals, MLP, norms and SDPA dispatch remain native Transformers code.

The copied `quantize/omniquant.py` adds model_type=`qwen2` device/discovery dispatch and records/replays the complete kwargs received by each calibration sample's native first decoder layer. The replay supplies those same per-sample kwargs, including position embeddings/cache position, to the native block forward. Official Qwen formal configuration uses batch size one; the adapter rejects another batch size instead of merging or dropping per-sample metadata.

Two optional smoke-only controls bound the driver to one decoder layer and two optimizer steps while the formal configuration remains 20 epochs. Optional observers record loss, gradients, updates and synchronized timings. They do not alter official quantizer, LWC, block MSE, teacher/student inputs, AMP, AdamW or GradScaler step math. Formal execution has no smoke controls and was not run.

The source file `quantize/quantizer.py`, `quantize/int_linear.py`, `quantize/utils.py`, and `utils.py` are byte-identical to the upstream freeze. `QuantLinear` weight quantization runs official asymmetric affine W4A16/G128 semantics; activation quantization is identity at 16 bits. The smoke is a Qwen architecture integration test, not a new OmniQuant objective or a quality result.
