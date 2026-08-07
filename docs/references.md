# References and Evidence Notes

This file records the evidence basis used for the current architecture pack. It is not itself an architecture decision.

## Historical architecture packs

Revision 0.4 builds on the v0.2/v0.3 architecture/ADA split and keeps the useful distinction between accepted runtime contracts and historical experiments.

`changes-v0.3.md` and the files under `docs/experiments/` remain historical records; they are not the current implementation specification.

## Current implementation evidence

The current Chomato real-engine tests and benchmark work establish the following baseline:

- `LFM2.5-1.2B-Instruct` executes end to end through WebGPU.
- Sandblaster links LFM2 WGSL/resources into a serialized build-time artifact that is deserialized/compiled at runtime.
- The runtime maintains separate rolling short-convolution state and attention KV state.
- Exact checkpoints are physical snapshots and do not re-prefill their source prefix.
- Checkpoints branch, chain, survive dropping source blocks, and materialize only populated KV prefix state.
- Typed structured generation is exposed through `engine.generate(schema, { checkpoint?, blocks? })`.
- Structured token selection executes on the GPU as `constraint_mask -> constraint_argmax + state commit`.
- CPU and Dawn paths have an equivalence oracle over the same packed constraint/tokenizer blobs.
- Public structured-generation tests run through `loadModel()` and the normal engine API on the real backend.
- WQ4 raw-weight bindings require runtime-sized storage arrays; fixed two-element placeholder lowering previously produced out-of-bounds weight reads and invalid logits.

## Structured-generation implementation evidence

The current implementation establishes:

- vocabulary size 65,536 = `2^16`,
- one dense allowed-token mask is 8,192 bytes = 2,048 `u32`,
- the constraint VM is byte-oriented and uses token ID -> raw bytes metadata,
- optional/enum branches are determinized into compact tries/switches,
- bounded dynamic arrays are compiled directly from JSON-schema semantics rather than fixed binary layout,
- candidate validation is transactional and only the selected token commits decoder state,
- accepted root completion transitions to EOS-only,
- sparse LM-head evaluation is not implemented,
- current mask cost is small relative to model forward, so full-vocabulary masked execution remains the accepted baseline.

The structured benchmark suite covers representative root strings/numbers, enums, objects, optional fields, nested objects, bounded arrays, a larger record, checkpoint composition, and deterministic greedy behavior.

## Checkpoint evidence

The real-engine checkpoint suite verifies:

```text
checkpoint continuation == uninterrupted continuation
branch reuse does not mutate checkpoint
checkpoint chaining works without replaying the base prefix
materialized checkpoint survives dropping source blocks
checkpoint prefix is not re-prefilled
KV/checkpoint bytes scale with populated prefix length
```

For the current LFM2 model, short-convolution checkpoint state is fixed-size rolling state (`convCacheLength = 3`), while attention KV storage grows with the populated prefix.

## Runtime/buffer-layout evidence

Several implementation bugs established the need to distinguish:

```text
logical ABI size
dynamic-offset stride
physical backing-buffer capacity
shader-visible runtime-array vs fixed-array type
```

Notable examples:

- `OpParams` is a 64-byte logical record even when dynamic offsets are aligned to 256 bytes.
- The physical `op` backing buffer must hold all dispatch parameter records in a submit.
- Raw WQ4 weight buffers must remain runtime-sized storage arrays; fixed-size placeholder lowering can silently change WGSL semantics.

These are now documented as explicit runtime contracts rather than incidental implementation details.

## External model facts retained from earlier revisions

Official LiquidAI material used in earlier architecture work established the reference-model facts:

- approximately 1.17B parameters,
- 16 layers,
- 10 double-gated LIV convolution blocks,
- 6 grouped-query attention blocks,
- vocabulary size 65,536,
- stated model context length 32,768 tokens.

These facts define the reference model, not Chomato's current runtime capacity. The current implementation allocates a 1,024-token context.

## Native portability evidence retained from earlier revisions

Rust + `wgpu` + WASM/WebGPU remains a credible future shared-core architecture. Browser and native threading/I/O/runtime behavior differ, and the final host-language choice remains deferred.
