# ADA-0008 — Correctness Classes and Optimization Boundaries

Status: **ACCEPTED**  
Date: 2026-08-03

## Context

Chomato uses aggressive optimizations across several independent axes:

- quantized numeric execution,
- cached intermediate representations,
- exact state checkpointing,
- approximate deep context,
- structured-decoding acceleration,
- GPU/host residency changes.

These optimizations can produce fluent output even when the internal state is subtly wrong.

The architecture therefore needs correctness classes that do not conflate "looks good", "same greedy token", "numerically close", and "semantically approximate".

## Decision

Every optimization that changes model execution, continuation state, or token-selection semantics must declare its correctness class and reference/oracle.

## 1. Exact continuation

An exact `ContextCheckpoint` means:

> restoring it and continuing with the same subsequent token sequence is equivalent to uninterrupted execution from the original prefix.

Evidence should be state/logit-level, not only final text or greedy-token equality.

## 2. Exact cached representation

A `CachedRepresentation` may be called exact only for documented composition rules that pass differential testing.

Exactness belongs to the representation **as used**, not just the stored artifact.

## 3. Approximate context

Approximate cached context is a separate semantic class.

Rules:

- it is never silently returned as exact,
- once incorporated, downstream frontier/checkpoint/generation remains approximate unless exact state is recomputed,
- request/result telemetry should make use of approximate context observable,
- an exact request may fall back to exact recompute or fail explicitly, but never silently switch to approximation.

## 4. Numerical approximation

Quantization and optimized matmul may change floating-point/model numerics without changing the context-reuse class.

For example:

```text
exact checkpoint semantics
+
WQ4 numerical approximation
```

is different from:

```text
approximate cached context
+
F16 execution
```

These axes must be measured/reported separately.

## 5. Structured-decoding equivalence

Optimized constrained-decoding paths must be compared against a correctness-first oracle.

`MASKED_DENSE` and `SPARSE` are acceptable alternatives only when they represent the same exact allowed-token semantics for the same decoder state.

A performance optimization may not discard candidate probability mass before applying the constraint unless the public sampling semantics explicitly define that different behavior.

## 6. Fallbacks

Preferred correctness-preserving patterns:

### Checkpoint/cache

```text
exact optimized state unavailable
→ ordinary exact recompute/prefill
```

A caller may explicitly request cache-only behavior and receive a failure, but the runtime does not silently substitute approximation.

### Structured decode

```text
optimized GPU path not trusted/available
→ correctness oracle/reference path for testing
```

The slow oracle need not be a production fallback if it violates required runtime characteristics, but it remains a differential reference during development.

### Numerical kernels

```text
optimized kernel
↔ reference tensor/kernel tests
```

## 7. Observability

Telemetry is not a correctness dependency, but results/debug output should be able to report relevant execution class, for example:

```text
checkpoint/restored vs recomputed
exact cached representation used
approximate context used
structured decode mode
quantization/kernel variant
```

The exact telemetry schema is not frozen.

## Consequences

### Positive

- "exact" becomes a testable promise rather than subjective output quality.
- Approximate BlockStore work can continue without contaminating exact checkpoints.
- WQ4 quality work remains separate from cache semantics.
- Dense/sparse structured decoding can be optimized without redefining constraints.

### Negative

- Reference implementations and differential tests must be retained.
- Some promising optimizations may remain PROVISIONAL longer.
- Benchmark reports need to state which correctness axis they exercise.
