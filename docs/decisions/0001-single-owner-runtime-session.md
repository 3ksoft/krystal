# ADA-0001 — Single-Owner Headless Runtime and Live Session

Status: **ACCEPTED**  
Date: 2026-08-03

## Context

Chomato needs to run from browser UI, CLI, LSP/editor integrations, and other applications while keeping model weights, GPU pipelines, exact checkpoints, and BlockStore data alive across individual requests.

A conventional shared inference daemon would introduce multi-client ownership, scheduling, cancellation, cache reference counting, and session-recovery problems before the project needs them.

The intended deployment model is simpler: an application that needs an independent model/cache universe can own an independent Chomato daemon.

## Decision

A Chomato daemon is a **single owner of one coherent runtime instance**.

It owns:

```text
Model
Tokenizer/backend metadata
GPU resources and pipelines
BlockStore
Context checkpoints
runtime configuration
telemetry state
```

The baseline daemon accepts **at most one live client session at a time**.

Commands in that session are serialized. At most one long-running inference/indexing operation is active at a time.

This is the baseline architectural contract, not an implementation accident.

## Deployment model

```text
LSP server
   │
   └── chomato daemon A
       ├── model A
       └── BlockStore A

Web application
   │
   └── chomato daemon B
       ├── model B
       └── BlockStore B
```

If two applications require independent runtime state, they use two daemon instances rather than competing inside one daemon.

## Session lifetime

Baseline session behavior:

```text
client connects
→ session becomes live
→ commands execute serially
→ client disconnects
→ cancel current request
→ discard session-scoped state
→ keep daemon-owned model / BlockStore / durable checkpoints
```

Session-scoped state includes any request/generation continuation that is not explicitly promoted to daemon-owned persistent state.

## Cancellation

The logical operation is `CancelCurrent` or an equivalent cancellation of the single active long-running request.

A request identifier may still be used for diagnostics/correlation, but the baseline design does not need request IDs to solve concurrent-client arbitration.

Cancellation cannot revoke GPU work already submitted. Cancellation latency therefore depends partly on how much future work is queued/submitted in advance; ADA-0003 owns this performance/responsiveness trade-off.

## Long-running operations

`Generate` is not the only operation that may be long-running.

For example, `IndexBlock`/block precomputation may execute model layers over substantial token sequences and may need:

- progress events,
- cancellation,
- terminal success/error events.

The control message can remain simple even when the operation produces a stream of reliable events.

## Logical communication classes

### Semantic/control messages

Low-frequency commands may use JSON or another human-readable representation.

Likely examples:

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

### Reliable request events

Examples:

```text
TokenEmitted
RequestFinished
BlockIndexed
CheckpointCreated
Error
```

These are part of request semantics and must not be silently dropped.

### Observational telemetry

Examples:

```text
InferenceStats
EngineStats
CacheStats
SamplingTrace
```

Telemetry may be throttled, batched, sampled, or dropped and must not block request control or completion.

## Disconnect policy

The initial policy is intentionally destructive for session-scoped work:

```text
disconnect
→ cancel active work
→ drop live generation/session state
```

The daemon keeps its own persistent model/BlockStore state.

Reattachable/keep-alive sessions are **not** part of the baseline contract. They may be added later if a concrete client needs them.

## Consequences

### Positive

- No internal multi-client scheduler is required.
- BlockStore ownership is unambiguous.
- Session disconnect behavior is simple.
- Concurrent `UnloadBlock`/generation ownership races are avoided in the baseline design.
- Independent applications remain isolated by process/runtime instance.
- Persistent model/cache state still survives ordinary CLI/client invocations.

### Negative

- Sharing one large resident model/cache across unrelated clients is not a baseline feature.
- An application that needs independent state may pay the memory cost of an independent daemon.
- Future multi-session support would be a deliberate protocol/runtime extension.

## Non-decisions

This ADA does not decide:

- Unix socket vs TCP vs WebSocket vs another transport,
- whether control/events share a physical connection,
- exact message framing,
- exact JSON/binary schema,
- final native host language,
- keep-alive/reattach semantics for future sessions.
