# ADA-0007 — Host Runtime Language

Status: **DEFERRED**  
Date: 2026-08-03

## Context

The reference runtime currently discovers architecture quickly in TypeScript/WebGPU.

A future native daemon needs stable ownership, API/session, Context State/checkpoint, BlockStore, structured-decoder, and GPU orchestration semantics.

Rust + `wgpu` + WASM is a strong candidate because it could provide a shared core for native execution and browser WebGPU while keeping WGSL kernels common.

However, moving unstable runtime semantics into a systems language too early would fossilize discovery-era APIs and make experiments slower.

## Decision

Do **not** choose or port the final native core yet.

The current TypeScript/WebGPU runtime is a **reference/discovery implementation**, not a promise that the final native daemon will be TypeScript.

Likewise, Rust/WASM is a credible target, not a current requirement.

## Candidate A — TypeScript/JavaScript host

Potential advantages:

- current implementation already exists,
- very fast architecture iteration,
- natural browser integration,
- minimal port cost now.

Risks:

- native WebGPU bindings/runtime choices,
- host-side threading/control limitations,
- long-term systems/runtime ergonomics,
- possible duplication if a later native core is required.

## Candidate B — Rust core + wgpu

Potential architecture:

```text
chomato-core
├── native + wgpu
└── WASM + browser WebGPU
```

Potential shared responsibilities:

```text
runtime state machine
checkpoint/BlockStore logic
structured-decoder plan execution metadata
sampling/orchestration
GPU resource/pipeline orchestration
protocol codecs where useful
```

Host adapters would still differ for:

```text
filesystem/model streaming
browser fetch/storage
native sockets
browser MessagePort/WebSocket
WASM memory/threading integration
```

## Why the decision remains deferred

v0.3 now defines more of the stable runtime shape, but several implementation-sensitive boundaries still need evidence:

- physical exact checkpoint implementation,
- composable block economics,
- structured-decoder GPU IR/layout,
- final model-memory/WQ4 layout.

A port should encode stable semantics, not help discover them by making every experiment harder.

## Portability constraint on current architecture

Current logical types/boundaries should not require TypeScript-specific object identity or Rust-specific ownership concepts to express core semantics.

In particular, the following should remain language-neutral concepts:

```text
Session
Request/operation
ContextBlock
ContextCheckpoint
CachedRepresentation
GenerationContinuation
OutputPlan
structured decoder state
correctness/exactness class
```

## Decision trigger

Revisit this ADA when:

1. browser/UI interaction is fully behind the logical runtime API,
2. exact checkpoint semantics have a validated implementation,
3. composable representation boundaries are understood well enough not to churn core types weekly,
4. structured decoding has one validated GPU execution path,
5. native daemon work becomes a concrete near-term deliverable.

At that point compare:

- code duplication,
- browser performance,
- native performance,
- build/tooling complexity,
- WASM boundary/memory cost,
- WebGPU/wgpu portability,
- schema/codegen reuse.
