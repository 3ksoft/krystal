# Runtime and WebGPU Execution

Status: Current implementation notes  
Date: 2026-08-07

## 1. Layering

The current runtime has four practical layers:

```text
engine-ts
    public engine API + transport/protocol

lfm2
    model-specific forward/checkpoint/tokenizer orchestration

Sandblaster
    typed GPU resources, program definitions, linking/AOT artifact, dispatch

WebGPU
    buffers, command encoders, compute pipelines, Dawn/browser backends
```

The LFM2 package is intentionally model-specific. Sandblaster/schema-pop own GPU schema/layout/code-generation concerns that are not LFM2 semantics.

## 2. AOT program artifact

WGSL source is a build input, not a runtime dependency.

Build:

```text
packages/lfm2/src/shaders/*.wgsl
packages/lfm2/src/shaders/includes/*.wgsl
        ↓
defineLfm2({ sources, includes })
        ↓
engine.link()
        ↓
engine.serialize(linked)
        ↓
lfm2.artifact.generated.ts
```

Runtime:

```text
const definition = defineLfm2();
definition.engine.deserialize(lfm2Artifact);
await definition.engine.compile({ device });
```

Shaders in `src/shaders/` are body-oriented Sandblaster program sources. Shared WGSL helper functions belong in includes rather than being embedded as independent `@compute` modules.

## 3. `OpParams`

`OpParams` is a **64-byte logical ABI record**.

It is selected per dispatch by a dynamic buffer offset. Dynamic-offset alignment and logical type size are different concepts:

```text
OpParams logical size     = 64 B
dynamic offset alignment  = typically 256 B
per-record stride         = aligned to device limit
physical backing buffer   = large enough for all records in one submit
```

Do not pad the `OpParams` schema itself to 256 bytes. Doing so changes the shader ABI and binding range rather than merely satisfying offset alignment.

The current runtime accumulates `OpParams` records for a submit and uploads them in one `queue.writeBuffer`. The backing allocation is therefore sized for the whole submit rather than a small ring. With the current 1,024-token decode budget the implementation uses a substantially larger backing buffer than the early bring-up path.

The important invariant is:

```text
logical binding range = sizeof(OpParams)
physical allocation   >= highest dynamic offset + sizeof(OpParams)
```

## 4. Typed resources vs physical capacity

Sandblaster resource types describe the shader-visible value. Physical allocation capacity is a separate property.

For example, a typed `ConstraintDecoderState` buffer should remain a `ConstraintDecoderState` resource even if host uploads use a packed `Uint32Array` representation of its bytes.

Do not replace a typed resource with `u32[]` merely because its host codec happens to use words. That loses the resource type and can change generated WGSL/layout semantics.

## 5. Runtime-sized weight buffers

Raw model weight resources must be runtime-sized storage arrays.

A fixed placeholder such as `u32[] == 2` is not an acceptable substitute: fixed-length lowering may legally become a small vector/fixed array, after which weight indexing is out of bounds.

The current Chomato/Sandblaster integration uses a runtime-array resource contract for raw weights and keeps a regression guard so this fails loudly if linker semantics change.

This contract is particularly important for:

```text
weightRaw : runtime array<u32>
weight32  : runtime array<f32>
```

## 6. Forward/decode boundary

The model forward path computes the next-token logits. Token selection is then performed by either the normal sampler path or the structured path.

Unconstrained:

```text
forward
→ logits
→ argmax/sampler
→ current token
```

Structured:

```text
forward
→ logits
→ constraint_mask
→ constraint_argmax + constraint-state commit
→ current token
```

The structured path deliberately does not modify the model matmul/LM-head kernels. The constraint passes are separate because their measured cost is small compared with model forward and this keeps the model kernels independent of schema semantics.

## 7. Current capacities

Current bring-up/runtime capacities:

```text
contextCapacity = 1024
maxNewTokens allocation = 1024
vocabSize = 65536
```

The effective response budget must fit the occupied context:

```text
prefix position + response budget - 1 <= contextCapacity
```

Increasing response length beyond this point requires increasing context/KV capacity, not merely increasing `maxNewTokens`.

## 8. Browser/native portability

WebGPU limits are part of the runtime contract. In particular:

- large physical backing buffers must respect the binding class's hardware limits,
- dynamic offset alignment comes from device limits,
- storage-buffer capacity and uniform-buffer capacity are not interchangeable,
- tests that pass on one Dawn backend do not automatically prove browser/D3D12 portability.

The runtime should request only limits that the backend can actually expose and prefer storage bindings for large read-only dynamic data where uniform binding-size limits would otherwise become a portability cap.
