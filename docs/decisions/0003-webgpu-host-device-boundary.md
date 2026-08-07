# ADA-0003 — WebGPU Execution and Host/Device Boundary

Status: **ACCEPTED**  
Date: 2026-08-03  
Updated: 2026-08-07

## Context

Chomato performs real end-to-end LFM2.5 inference through WebGPU. Model execution, normal sampling, exact checkpoint state operations, and structured token selection are all implemented on the real backend.

The important boundary is not simply "WebGPU is used". Token-critical execution must not accidentally acquire a mandatory host grammar/sampling decision between generated tokens.

## Decision

WebGPU/WGSL is the primary GPU execution layer.

Sandblaster is the current typed resource/program/linking layer above WebGPU. WGSL source is linked at build time into a serialized LFM2 artifact and is not regenerated as runtime shader-source glue.

The architecture distinguishes host work from token-critical GPU work independently of the eventual native host language.

## Host responsibilities

Current host-side responsibilities include:

```text
public API / request lifecycle
model/container I/O
CPU tokenization
ContextBlock identity/provenance
GPU allocation and model upload
Sandblaster artifact deserialization/compile
checkpoint ownership/lifetime
schema -> constraint-program compilation
tokenizer byte-metadata construction/upload
terminal result/readback
telemetry/debug readback
```

Host work before/after generation is expected. Per-token CPU grammar advancement is not.

## GPU responsibilities

Current GPU-side responsibilities include:

```text
embedding/model forward
prefill
decode
attention KV updates
rolling convolution-state updates
LM head
normal argmax/sampling
constraint-mask evaluation
constrained argmax
constraint decoder-state commit
state copy/materialization used by checkpointing
```

## AOT program path

Build time:

```text
shader bodies + includes
→ defineLfm2({ sources, includes })
→ engine.link()
→ engine.serialize(...)
→ generated artifact
```

Runtime:

```text
defineLfm2()
→ deserialize artifact
→ compile({ device })
→ dispatch existing programs
```

Tests should not create ad-hoc compute programs after runtime compilation when the goal is to validate the production LFM2 path.

## Token-loop invariant

For normal generation:

```text
forward -> sampler -> next token
```

For structured generation:

```text
forward -> logits
→ constraint_mask
→ constraint_argmax + decoder-state commit
→ next token
```

The structured path therefore preserves constraint-before-selection semantics without a CPU full-vocabulary validation round-trip.

## Synchronization points

Expected host/device coordination occurs at:

### Initialization

```text
model/resource upload
artifact compile
model-global tokenizer constraint metadata upload
```

### Request start

```text
select checkpoint/blocks
upload request/constraint state
initialize decoder state
```

### Explicit state operations

```text
checkpoint materialization/restoration
resource lifetime operations
```

### Observation/completion

```text
terminal generation result
execution stats/telemetry
```

Telemetry remains observational and must not become a prerequisite for the next token.

## Buffer/layout rule

Logical ABI type size, dynamic-offset stride, and physical buffer capacity are independent.

For example:

```text
OpParams logical size     = 64 B
dynamic offset stride     = device-aligned (typically 256 B)
backing buffer capacity   = enough for all records in the submit
```

Alignment padding must not be encoded by artificially enlarging the logical schema type.

Large dynamic resources must use a WebGPU binding class that is portable for their physical/binding-size requirements; WebGPU uniform/storage limits are not interchangeable.

## Cancellation

Cancellation cannot revoke GPU work already submitted. Submission horizon therefore remains a throughput/cancellation-latency trade-off.

No fixed horizon is frozen by this decision.

## Native portability

A future native host remains possible. The stable requirement is the WebGPU/WGSL execution contract and logical runtime API, not TypeScript object identity.

Rust + `wgpu` remains a plausible future host but is not selected by this ADA.
