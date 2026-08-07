# Changes in Architecture Revision 0.4

Revision 0.4 records implementation results that were still provisional in v0.3.

## 1. Structured generation is now an implemented typed API

The core operation is now:

```ts
engine.generate(schema, { checkpoint?, blocks? }) -> Promise<T>
```

The schema defines the value to generate. Scalar roots are first-class; there is no object envelope requirement and no chat-template layer in the core API.

The generated wire value is strict canonical JSON and is parsed to `T` at request completion. Chomato does not automatically call `schema.assert(result)`.

## 2. Structured decoding moved from target architecture to real GPU path

The implemented token loop is:

```text
forward -> logits
→ constraint_mask
→ constraint_argmax + decoder-state commit
→ next token
```

The mask is the full exact 65,536-bit allowed-token set (`2,048 × u32`). The current implementation tests all vocabulary tokens; sparse LM-head execution remains deferred.

CPU and Dawn implementations are compared on the same packed program/tokenizer blobs, and public E2E tests run through the real model backend.

## 3. JSON is the implemented constrained wire representation

v0.3 deliberately left open whether JSON would be rendered after a typed internal result.

v0.4 records the actual implementation: the model emits one strict canonical JSON value, and the host parses it to the inferred TypeScript value at terminal completion.

This keeps the GPU constraint VM byte-oriented and tokenizer-correct while preserving a simple public typed API.

## 4. Exact checkpoint implementation is validated

Exact checkpoints are no longer only provisional implementation work.

The real-engine suite verifies:

- exact continuation vs uninterrupted context,
- branch reuse without mutation,
- checkpoint chaining,
- survival after source-block deletion,
- no re-prefill of checkpoint prefix,
- storage scaling with populated KV prefix.

For LFM2.5, checkpoint state is populated attention KV plus fixed-size rolling convolution state.

## 5. `attentionLayerSlots` is clarified

The attention-slot table is a compact fixed mapping from attention model layers to KV storage slots. It is not paged attention.

`CacheBlockOptions.depth` belongs to composable cached-representation experiments and does not describe KV paging.

## 6. Sandblaster AOT is the current shader/program path

The production path no longer relies on runtime shader-source generation.

WGSL shader bodies/includes are linked at build time through Sandblaster, serialized as an artifact, deserialized at runtime, and compiled for the selected `GPUDevice`.

## 7. Runtime ABI/capacity is separated explicitly

A logical schema type's size is not the same thing as a dynamic-offset stride or physical backing-buffer capacity.

The concrete example is `OpParams`:

```text
logical ABI size = 64 B
dynamic offset stride = device-aligned (typically 256 B)
physical backing buffer = sized for all dispatch records in the submit
```

v0.4 documents this explicitly because treating alignment padding as part of the logical type caused real buffer/binding failures.

## 8. Runtime-array weight bindings are a guarded contract

Raw WQ4 weight buffers must lower to runtime-sized storage arrays. A fixed two-element placeholder was previously lowered to `vec2`, making almost all weight reads out-of-bounds and producing invalid logits.

The current integration guards the runtime-array contract so this class of linker regression fails loudly.

## 9. Current runtime capacities increased

The current runtime allocates up to 1,024 decode tokens. The effective budget is still bounded by the 1,024-token context capacity:

```text
prefix + response - 1 <= contextCapacity
```

Increasing the practical response/context envelope is now a context/KV-capacity task rather than a structured-generation feature task.

## 10. Experiments remain historical

The `docs/experiments/` files remain useful records of how the architecture was discovered, but v0.4 does not expand them. New performance evidence should be reported as benchmark results instead.
