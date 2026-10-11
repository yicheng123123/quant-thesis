# Gate decision after six preregistered rows

- A parity and checks: PASS.
- B C vs D held-out: 3/6 improve, median 11.90%, pooled SSE 15.08%; stable rule NOT MET.
- C rerank: top-2 16/60 GPTQ evaluations, median held regret 0.00%; top-3 21/60, median regret 0.00%; criterion PASS.
- D measured six-row wall time 133.1s. The 60 C GPTQ candidates took 12.8s; a linear 96-row C candidate-only estimate is 205.4s plus capture/Hessian. Repeating every diagnostic gives a much larger 2128.8s estimate. Neither is measured 96-row runtime.

No 96-row or full-model PPL run is authorized by this gate alone; the six-row result is local FP-input evidence.
