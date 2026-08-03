# Chomato Architecture Specification

Status: Draft  
Revision: 0.3  
Date: 2026-08-03

## 1. Purpose

Chomato is a focused local LLM inference engine built around WebGPU, persistent model/runtime ownership, reusable Context State, and GPU-oriented structured decoding.

The current reference implementation runs `LiquidAI/LFM2.5-1.2B-Instruct` and is used to discover the execution/runtime semantics before the final native host implementation is selected.

The architecture intentionally optimizes for one real model and one real GPU execution path before attempting generic model-runtime abstractions.

## 2. Primary goals

- Keep model and reusable context resident across individual requests.
- Expose the engine through a headless logical API usable by browser, CLI, LSP/editor integration, or another client.
- Keep daemon ownership and session semantics simple enough to reason about.
- Preserve an unconstrained GPU-resident decode path with no mandatory host decision between generated tokens.
- Make exact reusable continuation checkpoints a first-class capability.
- Treat composable context blocks as an acceleration layered on top of exact continuation semantics.
- Support schema/grammar-constrained generation without making JSON parsing or CPU round-trips the center of the token loop.
- Keep correctness classes explicit whenever optimization changes numeric execution or context semantics.
- Delay the final host-language decision until the stable core boundary is visible.

## 3. Non-goals for the current phase

- A shared multi-tenant or multi-client inference daemon.
- Concurrent scheduling of independent client workloads inside one daemon.
- A generic runtime for arbitrary transformer/SSM/hybrid architectures.
- Freezing a public socket framing or final wire schema.
- Freezing BlockStore eviction/paging before GPU-resident economics are measured.
- Treating approximate deep cached context as exact continuation state.
- Freezing the current WQ4 file/layout as the production model format.
- Choosing Rust, TypeScript, C++, or another language as the final native core before the runtime boundary stabilizes.

## 4. System model

### 4.1. Daemon ownership

The baseline native architecture is deliberately single-owner and single-session:

```text
Application A / LSP
        │
        ▼
  chomato daemon A
  ├── model A
  ├── runtime A
  └── BlockStore A

Application B / Web UI
        │
        ▼
  chomato daemon B
  ├── model B
  ├── runtime B
  └── BlockStore B
```

A daemon owns its model, GPU resources, Context State storage, checkpoints, and BlockStore.

It accepts at most one live client session at a time in the baseline protocol. Session work is serialized; the engine is not designed as an internal multi-client scheduler.

This is an architectural simplification, not a transport limitation.

### 4.2. Session lifetime

Baseline behavior:

```text
connect
  ↓
open live session
  ↓
serial commands / request streams
  ↓
disconnect
  ├── cancel current request
  ├── drop session-scoped state
  └── retain daemon-owned model + BlockStore
```

Future keep-alive/reattachable sessions may extend this model if a real use case requires them.

### 4.3. Browser host

The browser remains a first-class execution environment:

```text
Web UI
  ↓
ChomatoClient
  ↓
in-process/session adapter
  ↓
Chomato Runtime
  ↓
WebGPU
```

The Web UI does not own inference semantics and should use the same logical operations as an out-of-process client.

## 5. Reference model and Context State

`LFM2.5-1.2B-Instruct` is the reference optimization target.

For the current backend, continuation state contains at least:

```text
Context State
├── bounded short-convolution history/cache
└── attention KV state for GQA layers
```

These state classes have different scaling behavior:

- short-convolution history is bounded/fixed-size per active sequence,
- attention KV grows with processed sequence length and exists only for the model's attention layers.

The public architecture therefore uses **Context State**, not `KV cache`, as the generic concept.

## 6. Reusable context hierarchy

### 6.1. ContextBlock

A `ContextBlock` is an immutable sequence of token IDs with stable logical identity.

Correctness is defined over the exact token sequence consumed by the model, including any relevant special/template tokens. Raw source text is not sufficient identity.

### 6.2. ContextCheckpoint

A `ContextCheckpoint` is the exact model continuation state after a specific ordered token prefix has been fully applied.

Conceptually:

```text
prefix token IDs
      ↓
normal model execution
      ↓
exact Context State
      ↓
ContextCheckpoint
```

Checkpointing is a fundamental capability independent from composable block caching.

The same exact checkpoint may seed many independent future generations.

### 6.3. GenerationContinuation

A reusable model checkpoint is intentionally narrower than an in-progress generation.

```text
GenerationContinuation
├── ContextCheckpoint
├── sampler state
├── validator / structured-decoder state
└── request-specific generation state
```

Validator or sampler state does not contaminate reusable `ContextCheckpoint` identity unless the runtime explicitly persists a full generation continuation.

### 6.4. CachedRepresentation

A `CachedRepresentation` is a model/depth-specific acceleration derived from a `ContextBlock`.

It may be:

- exact under documented composition rules,
- prefix-dependent,
- approximate,
- reconstructable from tokens.

The current depth-2 LFM2.5 experiment is an example of a candidate independently composable representation before the first attention layer.

### 6.5. Live computation frontier

When cached/approximate state is materialized, the runtime tracks where normal causal computation resumes and the exactness class of the resulting frontier.

Approximation is monotonic: once an approximate representation contributes to a frontier, downstream continuation is not reported as exact.

## 7. Host/device architecture

The stable boundary is conceptual rather than language-specific.

### Host / CPU responsibilities

Expected host-side work:

```text
API + session lifecycle
model/container loading
CPU tokenizer and token-block provenance
resource allocation / pipeline setup
BlockStore policy and residency decisions
checkpoint metadata / ownership
schema → OutputPlan compilation
tokenizer token-byte metadata / automata construction
telemetry consumption and presentation
```

### GPU responsibilities

Expected device-side work:

```text
model forward / prefill / decode
Context State transforms
LM head
constraint application during structured generation
structured decoder state where needed per token
sampling
checkpoint/state copy or materialization primitives
```

Exact placement of every structured-decoder data structure remains provisional, but the target is to avoid a mandatory host dependency between tokens.

### Legal synchronization points

Normal host interaction may occur at:

```text
model/load initialization
request start
explicit resource/checkpoint operations
asynchronous token/telemetry observation
request completion
```

The unconstrained decode baseline does not require a host decision between tokens.

Cancellation is not instantaneous once work has already been submitted to the GPU. The amount of future decode work submitted at once therefore trades throughput/overhead against cancellation latency and remains a measurable runtime parameter.

## 8. Structured generation architecture

Structured output is treated as a decode execution problem, not primarily as post-hoc JSON validation.

Target pipeline:

```text
schema / semantic constraint
        ↓
host compilation
        ↓
Flat OutputPlan + tokenizer automata/metadata
        ↓
GPU structured decoder state
        ↓
exact allowed-token set
        ↓
LM-head execution mode
        ↓
GPU sampling
        ↓
typed output state / token stream
        ↓
optional renderer (JSON or another format)
```

For the current vocabulary:

```text
65,536 tokens = 2^16
full token mask = 65,536 bits
                = 8,192 bytes
                = 2,048 × u32
```

The execution layer may choose among:

### DENSE

Normal unconstrained LM head + sampler.

### MASKED_DENSE

Compute the normal vocabulary logits, apply the exact allowed-token set, then sample on GPU.

A dense 1-bit mask is small enough to be a practical representation; it does not need to be copied into workgroup memory every token.

### SPARSE

When the exact allowed-token set is sufficiently small, evaluate only the LM-head rows corresponding to allowed token IDs and sample over that exact set.

`SPARSE` is valid only when the allowed set is known before logits are discarded. It is not top-K-before-validation.

The threshold between `MASKED_DENSE` and `SPARSE` is a performance experiment, not a semantic choice.

## 9. Logical API classes

The physical protocol remains unfrozen.

### Session/control operations

Likely operations include:

```text
LoadModel
ConfigureRuntime
Generate
CancelCurrent
IndexBlock
UnloadBlock
CreateCheckpoint
RestoreCheckpoint
GarbageCollect
ConfigureTelemetry
```

The baseline session executes commands serially and allows at most one active long-running operation.

`Generate` and `IndexBlock` may both be long-running GPU operations and may expose progress/result events.

### Reliable request output

Examples:

```text
TokenEmitted
RequestFinished
CheckpointCreated
BlockIndexed
Error
```

Reliable request output is not droppable telemetry.

### Telemetry

Examples:

```text
InferenceStats
EngineStats
CacheStats
SamplingTrace
```

Telemetry is observational and may be throttled, batched, sampled, or dropped. It must not become a correctness dependency.

JSON remains acceptable for low-frequency semantic control. Binary schemas remain attractive for high-rate events/telemetry. The exact split and framing remain transport details.

## 10. WQ4/model-memory position

The custom WQ4 work proves useful quantized execution paths can be built for WebGPU, but the current whole-model resident footprint is not yet explained by the raw quantization ratio alone.

Therefore v0.3 does not treat the present WQ4 storage/repacking arrangement as a stable model-memory design.

The project separates:

```text
quantization math
matmul kernel
runtime weight layout/repacking
whole-model resident memory
on-disk container
```

Each may evolve independently until measurements justify freezing them.

## 11. Stability map

| Area | Status | Confidence |
|---|---|---:|
| Single-owner daemon | ACCEPTED | High |
| One live client session per daemon | ACCEPTED | High |
| Disconnect → cancel current + drop session state | ACCEPTED baseline | High |
| LFM2.5-1.2B reference model | ACCEPTED | High |
| WebGPU execution abstraction | ACCEPTED | High |
| Unconstrained GPU-resident decode baseline | ACCEPTED | High |
| Host/device responsibility boundary | ACCEPTED direction | Medium-High |
| Context State terminology | ACCEPTED | High |
| Exact ContextCheckpoint semantics | ACCEPTED | High |
| Exact checkpoint implementation/economics | PROVISIONAL | Medium |
| Immutable ContextBlock token identity | ACCEPTED | High |
| Depth-2 composable representation | PROVISIONAL | Medium-High |
| Attention-crossing/deep cached representations | EXPERIMENTAL | Medium-Low |
| Host spill/paging | OPEN | Low-Medium |
| Structured output requirement | ACCEPTED | High |
| OutputPlan + tokenizer-automata direction | PROVISIONAL | Medium-High |
| GPU structured-decoder execution | PROVISIONAL | Medium |
| DENSE / MASKED_DENSE / SPARSE policy | EXPERIMENTAL | Medium |
| Final WQ4 runtime/model layout | EXPERIMENTAL | Medium |
| Final native host language | DEFERRED | Medium |
| Rust + wgpu + WASM shared core | CANDIDATE | Medium-High |

## 12. Decision index

- [ADA-0001 — Single-Owner Headless Runtime and Live Session](decisions/0001-single-owner-runtime-session.md)
- [ADA-0002 — Reference Model: LFM2.5-1.2B-Instruct](decisions/0002-reference-model-lfm2.5.md)
- [ADA-0003 — WebGPU Execution and Host/Device Boundary](decisions/0003-webgpu-host-device-boundary.md)
- [ADA-0004 — Context State and Exact Checkpoints](decisions/0004-context-state-exact-checkpoints.md)
- [ADA-0005 — Composable Context Representations](decisions/0005-composable-context-representations.md)
- [ADA-0006 — Structured GPU Decoding](decisions/0006-structured-gpu-decoding.md)
- [ADA-0007 — Host Runtime Language](decisions/0007-host-runtime-language.md)
- [ADA-0008 — Correctness Classes and Optimization Boundaries](decisions/0008-correctness-classes.md)

## 13. Experiment index

- [Exact checkpoints](experiments/exact-checkpoints.md)
- [Composable context blocks](experiments/composable-blocks.md)
- [Structured decoding](experiments/structured-decoding.md)
- [WQ4 model memory/layout](experiments/wq4-model-layout.md)
- [WQ4 matmul](experiments/wq4-matmul.md)

## 14. Guiding principles

1. **One daemon owns one coherent runtime universe.** Scale isolation by running another daemon before designing an internal scheduler.
2. **Exact checkpoints are the semantic foundation of reuse.** Composable blocks are an acceleration.
3. **Optimization state must not silently become semantic state.** Exact and approximate execution are explicit.
4. **Keep the token loop on the GPU unless evidence justifies a synchronization boundary.**
5. **Compile structure before generation; do not parse correctness back into the output after the fact.**
6. **Freeze only boundaries that survive the current experiments.**
