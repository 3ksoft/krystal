# Chomato Architecture Specification

Status: Current implementation architecture  
Revision: 0.4  
Date: 2026-08-07

## 1. Purpose

Chomato is a focused local inference runtime for `LiquidAI/LFM2.5-1.2B-Instruct` built around WebGPU, exact reusable continuation state, and GPU-resident structured generation.

The project deliberately optimizes one real model/runtime path before introducing generic model abstractions. The current implementation is already end-to-end: model execution, checkpoints, typed structured generation, constraint masking, and constrained argmax run on the real WebGPU backend.

## 2. Primary goals

- Keep model weights and reusable model state resident across requests.
- Expose a small headless API independent of browser/CLI/UI concerns.
- Make exact continuation checkpoints a first-class semantic capability.
- Keep token-critical model execution and token selection on the GPU.
- Compile structured-output constraints before generation instead of validating after sampling.
- Keep model-specific implementation details behind the engine boundary.
- Preserve explicit correctness classes for checkpointing, quantization, and constrained decoding.

## 3. Current non-goals

- Generic support for arbitrary transformer/SSM/hybrid model families.
- Internal multi-client scheduling or a shared multi-tenant daemon.
- Paged attention or a generic KV memory manager.
- Sparse constrained LM-head execution before measurements justify it.
- Full JSON Schema coverage.
- Freezing a final native host language or transport protocol.
- Treating historical composable-block experiments as required runtime semantics.

## 4. Public runtime model

The fundamental typed operation is:

```ts
const result = await engine.generate(schema, {
  checkpoint,
  blocks,
});
```

Conceptually:

```ts
generate<T>(
  schema: Type<T>,
  context: {
    checkpoint?: ContextCheckpoint;
    blocks?: ContextBlock[];
  },
): Promise<T>
```

`schema` defines the value to generate. The runtime does not require an object root; strings, numbers, booleans, arrays and objects are all ordinary root values.

Examples:

```ts
await engine.generate(type("string < 512"), { blocks });
await engine.generate(type("0 <= number <= 10"), { checkpoint });
await engine.generate(type({ id: "number" }), { checkpoint, blocks });
```

Chomato does not perform an additional ArkType assertion after generation. A caller that wants runtime validation can do so explicitly:

```ts
const t = type({ id: "number" });
const result = await engine.generate(t, context);
const validated = t.assert(result);
```

A future `prompt(text)` helper is sugar over typed generation of a bounded string; it does not introduce chat-template semantics into the core engine.

## 5. Reference model

The current reference model is `LFM2.5-1.2B-Instruct`:

- 16 model layers,
- hybrid short-convolution + grouped-query attention architecture,
- 6 attention layers/slots,
- hidden size 2,048,
- vocabulary size 65,536 (`2^16`),
- convolution cache length 3 in the current model,
- model-advertised context larger than the current Chomato runtime allocation.

The **current Chomato runtime context capacity is 1,024 tokens**. This is an implementation capacity, not a statement about the model's trained context window.

The current decode budget allocation is also 1,024 tokens, subject to:

```text
prompt/checkpoint position + maxNewTokens - 1 <= contextCapacity
```

Therefore the usable response budget shrinks as the occupied prefix grows.

## 6. Execution stack

The current execution stack is:

```text
public Engine API
    ↓
engine-ts transport/protocol
    ↓
LFM2 runtime / forward orchestration
    ↓
Sandblaster program/resource graph
    ↓
serialized AOT artifact
    ↓
WebGPU / WGSL
```

Sandblaster is the build/link/dispatch layer for model programs. Runtime shader source generation is not part of the production path.

Build time:

```text
WGSL shader bodies + includes
        ↓
defineLfm2({ sources, includes })
        ↓
engine.link()
        ↓
engine.serialize(...)
        ↓
lfm2.artifact.generated.ts
```

Runtime:

```text
defineLfm2()
    ↓
engine.deserialize(lfm2Artifact)
    ↓
engine.compile({ device })
    ↓
pass.run(...)
```

See [runtime.md](runtime.md) for the current buffer/pipeline contracts.

## 7. Context model

A request context is the ordered combination of:

```text
optional exact checkpoint
+
zero or more appended ContextBlocks
```

A `ContextBlock` represents immutable token content/provenance. It is not itself an exact continuation state.

A `ContextCheckpoint` is the materialized model state after processing an exact prefix. It can be reused, branched, chained into a new checkpoint, and used after its source blocks have been dropped.

For LFM2.5 the checkpoint contains:

```text
attention KV state up to the populated prefix
+
fixed-size rolling short-convolution state
+
position/runtime metadata needed for continuation
```

Checkpoint continuation does **not** prefill the checkpoint prefix again.

See [checkpoints.md](checkpoints.md).

## 8. Attention layout

`attentionLayerSlots` is a compact fixed mapping from model layer index to KV-cache slot. It is **not paged attention**.

Conceptually:

```text
model layer 0   conv       -> no attention slot
model layer 1   attention  -> slot 0
model layer 2   conv       -> no attention slot
...
model layer N   attention  -> slot K
```

The current KV allocation is fixed for the configured `contextCapacity`. Checkpoint materialization copies only the populated prefix rather than the entire configured capacity.

`CacheBlockOptions.depth` belongs to composable cached-representation experiments; it is not a KV page size or page hierarchy.

## 9. Structured generation

Structured generation is an implemented GPU decode path.

The external schema is compiled on the host to a deterministic byte-level constraint program. The model still emits ordinary tokenizer token IDs; the constraint VM evaluates each token's raw byte representation against the current decoder state.

The production path is:

```text
Type<T>
  ↓
JSON Schema representation
  ↓
constraint compiler
  ↓
GPU constraint program
  ↓
model forward -> logits
  ↓
constraint_mask
  ↓
constraint_argmax + decoder-state commit
  ↓
next token
```

For the 65,536-entry vocabulary:

```text
1 bit/token mask = 65,536 bits
                 = 8,192 bytes
                 = 2,048 × u32
```

`constraint_mask` uses 2,048 invocations; each invocation evaluates 32 vocabulary tokens and writes one `u32` mask word. No atomic OR is required.

`constraint_argmax` selects the best allowed token and commits the corresponding decoder transition. Constraint state mutation happens only for the selected token; mask evaluation uses transactional local copies.

The current implementation scans the full vocabulary. Sparse LM-head row evaluation is not implemented.

The measured constraint mask cost is small relative to model forward: 0.07–0.30 ms for the mask and 0.14–0.16 ms for the masked argmax against ~8.5 ms/token, with the end-to-end constrained-vs-unconstrained difference inside run-to-run noise. Sparse execution is also capped from above — the LM head is only ~8% of a decode step against ~90% for the block stack — so even a perfect oracle could not save more than that. This does not justify a sparse execution path.

See [structured-generation.md](structured-generation.md).

## 10. Strict JSON wire value

Structured generation uses **strict canonical JSON as the internal emitted wire value** for the typed API.

This is deliberately simple:

```text
type("string")     -> model emits JSON string bytes -> JSON.parse -> string
type("number")     -> model emits JSON number       -> JSON.parse -> number
object schema       -> model emits JSON object       -> JSON.parse -> object
array schema        -> model emits JSON array        -> JSON.parse -> array
```

Once the root value is complete, only EOS is allowed. The v0 path does not intentionally generate trailing whitespace after the completed root.

This is different from the earlier v0.3 proposal in which JSON could have been rendered after a typed internal result. The implemented v0.4 path uses canonical JSON directly because it provides a compact, deterministic bridge from constrained token generation to JavaScript values.

## 11. Correctness boundaries

### Checkpoints

Exact checkpoints are physical model continuation snapshots. Tests cover:

- checkpoint vs uninterrupted continuation equivalence,
- branching without checkpoint mutation,
- checkpoint chaining,
- survival after source-block deletion,
- every tested prefix split,
- no checkpoint-prefix re-prefill,
- checkpoint bytes scaling with populated KV prefix.

### Structured generation

The constraint stack has a CPU reference implementation and a Dawn/WebGPU implementation. Tests compare packed masks and decoder transitions bit-for-bit on the same uploaded program/tokenizer blobs.

Public E2E tests run through:

```text
loadModel()
→ model.engine
→ engine.generate(schema, context)
```

rather than bypassing the runtime with test-only WebGPU pipelines.

### Numerical model execution

WQ4 remains a numerical approximation axis independent of checkpoint exactness and structured-decoding correctness.

## 12. Current structured-schema envelope

The v0 compiler covers the structures currently required by the dataset/E2E suite, including:

- root scalar values,
- objects,
- required and optional object fields,
- nested objects,
- enums,
- booleans,
- bounded strings,
- bounded numbers/integers,
- bounded arrays with `minItems`/`maxItems`,
- fixed arrays where applicable.

The compiler requires a finite output budget. Unbounded arrays/strings are therefore not a useful public v0 contract without an explicit bound.

Not currently part of the v0 feature envelope:

- unbounded arrays,
- tuple / `prefixItems` semantics,
- general recursive schemas,
- general unions beyond the implemented lowering cases,
- `multipleOf`/step semantics,
- full JSON Schema keyword coverage.

## 13. Quantized model execution

WQ4 is the current practical quantized model path. Runtime correctness requires raw weight bindings to remain runtime-sized storage arrays; fixed-size placeholder lowering must not silently turn weight buffers into small WGSL vectors/arrays.

This is guarded as an implementation contract because an earlier fixed `u32[] == 2` placeholder was lowered to `vec2`, making weight reads past word 1 out-of-bounds and producing invalid model output while superficially allowing some greedy tests to continue.

The model execution path and structured-decoding correctness are tested separately so failures in model numerics are not misdiagnosed as constraint failures.

## 14. Stability map

| Area | Status |
|---|---|
| LFM2.5-1.2B reference model | ACCEPTED |
| WebGPU + WGSL execution | ACCEPTED |
| Sandblaster AOT program artifact | ACCEPTED |
| Exact ContextCheckpoint semantics | ACCEPTED |
| Exact physical checkpoint implementation | ACCEPTED |
| Checkpoint branching/chaining | ACCEPTED |
| Composable cached representations | EXPERIMENTAL |
| Typed `generate(schema, context)` | ACCEPTED v0 |
| GPU constraint mask | ACCEPTED v0 |
| Constraint argmax + state commit | ACCEPTED v0 |
| Full-vocabulary masked execution | ACCEPTED baseline |
| Sparse constrained LM-head execution | DEFERRED / EXPERIMENTAL |
| Full JSON Schema coverage | NON-GOAL for v0 |
| Current WQ4 runtime | ACCEPTED implementation / numerical approximation |
| Final native host language | DEFERRED |

## 15. Guiding principles

1. **Generate values, not chat transcripts.** The core operation is `generate(Type<T>, Context) -> T`.
2. **Exact checkpoints are the semantic foundation of context reuse.** Composable block representations are optional acceleration experiments.
3. **Keep token-critical decisions on the GPU.** Host compilation and terminal readback are expected; per-token CPU grammar decisions are not.
4. **Use the same program/data path in tests and production.** Public E2E tests go through `loadModel()` and `engine.generate()`.
5. **Keep logical types separate from physical buffer capacity/alignment.** A 64-byte ABI record does not become a 256-byte type because dynamic offsets are 256-byte aligned.
6. **Bound output through the schema.** The typed API does not need a separate user-facing `maxTokens` for ordinary structured generation.
7. **Treat historical experiments as evidence, not as the current contract.**
