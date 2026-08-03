# ADA-0003 — WebGPU Execution and Host/Device Boundary

Status: **ACCEPTED**  
Date: 2026-08-03

## Context

The current Chomato vertical slice performs real end-to-end inference through WebGPU.

The useful baseline is not merely "WebGPU works". The decode path is structured so model execution and sampling can proceed without a mandatory CPU decision between generated tokens.

Future features — especially structured decoding, checkpointing, telemetry, cancellation, and a native host — need an explicit responsibility boundary so they do not accidentally insert synchronous host dependencies into the token loop.

## Decision

WebGPU is the primary GPU execution abstraction for Chomato.

WGSL remains the primary kernel representation unless a measured limitation justifies another path.

The architecture distinguishes **host responsibilities** from **per-token GPU execution**, independent of whether the host is TypeScript, Rust, C++, or another language.

## Host responsibilities

Expected host-side work includes:

```text
API/session lifecycle
model/container I/O
CPU tokenization
ContextBlock token/template provenance
GPU resource allocation
pipeline setup
BlockStore policy / residency decisions
checkpoint metadata / ownership
schema/constraint compilation to OutputPlan
construction/loading of tokenizer token-byte automata/metadata
telemetry consumption
```

This list is a responsibility map, not a ban on future GPU acceleration of individual items.

## GPU responsibilities

Expected GPU-side work includes:

```text
embedding/model forward
prefill
decode
Context State update
LM head
constraint application during structured generation
structured-decoder transition state required per token
sampling
state copy/materialization primitives
```

The exact representation of structured-decoder state is PROVISIONAL under ADA-0006.

## Unconstrained decode invariant

For unconstrained generation, the target baseline is:

> no mandatory host decision/readback between generated tokens.

Telemetry may observe asynchronously. Client rendering may lag. Neither is part of the dependency chain that produces the next token.

A future optimization/feature must not silently replace this baseline with a per-token submit/readback loop.

## Structured decode relationship

ADA-0006 selects a GPU-oriented direction specifically to preserve this execution model where practical.

Host-side schema compilation before generation is expected. Per-token CPU grammar advancement is not the target architecture.

A correctness/reference implementation may use host validation during experimentation, but that path is not automatically the production execution model.

## Legal synchronization points

Expected host/device coordination includes:

### Initialization

```text
load model
create buffers/pipelines
upload precomputed metadata
```

### Request start

```text
upload/request configuration
select context/checkpoints/blocks
initialize structured-decoder state
```

### Explicit state operations

```text
create/restore checkpoint
spill/restore residency tier
resource management
```

### Asynchronous observation

```text
token events
telemetry
progress
```

Observation should use small/staged data and must not become a prerequisite for the next token unless the selected execution mode explicitly requires it.

### Request completion

Terminal status/results may be read back normally.

## Cancellation and GPU submission horizon

Cancellation cannot retroactively cancel already-submitted GPU work.

Therefore:

```text
larger future-work submission window
→ lower host/submit overhead
→ potentially worse cancellation latency
```

and:

```text
smaller submission window
→ faster cancellation observation
→ potentially more host/queue overhead
```

The runtime may expose or internally tune a maximum speculative/submitted decode horizon.

No fixed value is frozen in this ADA. It must be benchmarked against throughput and acceptable cancellation latency.

## Telemetry

Telemetry is observational:

- use persistent/staged readback where practical,
- throttle/batch/drop diagnostic samples when a consumer is slow,
- do not synchronize the decode loop for status display.

Reliable token/result delivery is distinct from telemetry and must not inherit drop-if-busy semantics.

## Native portability

The exact native host remains deferred.

A Rust + `wgpu` implementation is a credible candidate because the same WebGPU/WGSL execution architecture can target native GPU APIs and browser WebGPU through WASM.

This does not imply that browser and native hosts have identical threading, file I/O, memory, or transport behavior.

## Consequences

### Positive

- The performance-critical boundary is explicit.
- Browser execution remains first-class.
- The same shader architecture can plausibly survive a native host migration.
- Structured decoding has a clear target: precompile on host, execute token-critical state on GPU.

### Negative

- Some convenient CPU-side algorithms may be unsuitable for the production token loop.
- Cancellation responsiveness and batching/queue efficiency must be traded explicitly.
- WebGPU limits remain part of the engineering envelope.
