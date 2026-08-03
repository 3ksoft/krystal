# References and Evidence Notes

This file records the evidence basis used for architecture revision 0.3. It is not itself an architecture decision.

## v0.2 architecture pack

Revision 0.3 starts from the v0.2 architecture/ADA split and preserves its useful status vocabulary and separation between stable decisions and active experiments.

## Current Chomato implementation / experiments

The project snapshot and experiment notes establish the working baseline:

- LFM2.5-1.2B-Instruct runs end to end on the custom WebGPU runtime.
- The runtime maintains separate short-convolution and attention KV state.
- Decode can remain GPU-resident without a CPU decision between tokens.
- A shallow depth-2 cached representation exists before the first attention layer and uses bounded boundary repair.
- Deeper attention-crossing cached context is not generally exact and is treated separately.
- The current WQ4 converter uses 32-weight blocks with four packed 4-bit words plus one exponent word.
- Recent WQ4 matmul candidates are promising but do not settle whole-model memory/layout.

## Runtime/session clarification for v0.3

The intended daemon model is:

```text
one daemon owns one model/runtime/BlockStore
one live client session at a time
session work serialized
on disconnect: cancel current request + drop session state
```

Independent applications may own independent daemons rather than sharing one multi-client runtime.

Future reattachable/keep-alive sessions are extensions, not baseline semantics.

## Structured-decoding design discussion

The v0.3 structured-decoding direction is based on the project discussion that established:

- LFM2.5 vocabulary size 65,536 = `2^16`,
- one dense allowed-token bitmap is 8,192 bytes = 2,048 `u32`,
- a flat schema-derived `OutputPlan` is a better internal target than recursive JSON parsing,
- tokenizer token-byte metadata/tries/automata are required to map semantic/lexical state to token IDs,
- allowed-token sets are state-dependent (including numeric/string states),
- a dense mask can remain device-resident rather than being copied into workgroup memory each token,
- GPU structured decoding can choose between full masked LM-head execution and exact sparse allowed-row evaluation,
- top-K-before-validation is not semantically equivalent to exact constrained sampling,
- JSON can be a renderer/API format rather than the internal generation state machine.

## v0.2 architecture reviews

The supplied reviews contributed several useful falsification checks that are incorporated into v0.3:

- exact cache claims should be tested at state/logit level, not only greedy text output,
- attention-crossing block experiments must verify position/order handling,
- cancellation latency is coupled to queued GPU work/submission depth,
- reliable request output must remain distinct from droppable telemetry,
- tokenizer/template provenance is part of reusable-block correctness.

The reviews' multi-client assumptions are intentionally not adopted because v0.3 explicitly selects a single-owner, one-live-session daemon model.

## External model facts retained from v0.2

Official LiquidAI material for `LFM2.5-1.2B-Instruct` was used in v0.2 to establish:

- approximately 1.17B parameters,
- 16 layers,
- 10 double-gated LIV convolution blocks,
- 6 grouped-query attention blocks,
- vocabulary size 65,536,
- stated context length 32,768 tokens.

These facts justify the reference-model decision but do not define Chomato's practical runtime limits.

## Native portability evidence retained from v0.2

Current `wgpu` documentation was used in v0.2 to establish Rust + `wgpu` + WASM/WebGPU as a credible candidate shared-core architecture, while also noting that browser and native threading/I/O/runtime behavior differ.

The host-language decision remains DEFERRED in v0.3.
