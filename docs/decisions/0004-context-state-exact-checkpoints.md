# ADA-0004 — Context State and Exact Checkpoints

Status: **ACCEPTED semantics / PROVISIONAL implementation**  
Date: 2026-08-03

## Context

The most robust reusable-context capability is not an independently composable BlockStore representation. It is the ability to capture the **exact continuation state after an expensive prefix** and continue from that state later.

This remains valuable even if every deeper block-composition experiment fails.

For example:

```text
large stable prefix
      ↓
full normal prefill
      ↓
exact checkpoint
      ├── query A
      ├── query B
      └── query C
```

The architecture therefore promotes exact checkpoint semantics above optional cached-block acceleration.

## Decision

A `ContextCheckpoint` is the exact model continuation state produced after a specific ordered token prefix has been processed normally.

For LFM2.5 it contains the model-dependent Context State required to continue:

```text
bounded convolution history
+
attention KV state
```

The checkpoint is prefix- and position-dependent by definition.

## Checkpoint identity

Checkpoint compatibility includes at least:

```text
model identity / compatible weights
model execution semantics affecting Context State
tokenizer identity
exact ordered token prefix (or collision-resistant identity of it)
position implied by that prefix
```

Source text alone is insufficient.

A checkpoint is not interchangeable with a standalone cached representation of one block.

## Model checkpoint vs generation continuation

A reusable model checkpoint intentionally excludes request-specific generation machinery.

```text
ContextCheckpoint
    = reusable model continuation state

GenerationContinuation
    = ContextCheckpoint
    + sampler/RNG state
    + structured-decoder/validator state
    + request-specific generation state
```

This allows the same model checkpoint to seed:

- unconstrained generation,
- several different structured-output schemas,
- different sampling configurations,
- multiple branches.

If the runtime later persists/resumes an exact in-progress generation, it persists a `GenerationContinuation`, not merely a `ContextCheckpoint`.

## Fundamental invariant

Restoring an exact checkpoint and feeding the same subsequent token sequence must be equivalent to uninterrupted execution from the original prefix.

Greedy output equality is not sufficient evidence.

## Exactness acceptance

The validation hierarchy is:

### Pure state-copy restore

If checkpoint creation/restoration only copies the exact stored model state, prefer bitwise equality of restored state and subsequent logits where the WebGPU path is deterministic.

### Reconstructed materialization

If restoration necessarily recomputes part of the state, define a numerical equivalence tolerance against ordinary execution and verify it at logits/state level.

In both cases, matching top-1 tokens is secondary evidence only.

## Branching

An exact checkpoint may be used as a branch point:

```text
checkpoint P
├── continuation A
├── continuation B
└── continuation C
```

Branch execution must not mutate the durable checkpoint in a way that changes later branches.

The physical implementation may use copies, copy-on-write, replay, or another mechanism; the semantic result is the same.

## Ownership and lifetime

The daemon owns persistent checkpoints.

Session-scoped temporary state may be discarded on disconnect. A checkpoint becomes durable only through an explicit runtime operation/policy.

With the single-session baseline, cross-client reference-counting is not required.

## Residency

Checkpoint identity is separate from where its state is stored.

Possible future residency forms:

```text
GPU-resident exact state
host-resident exact state
recomputable prefix metadata/tokens
```

If an exact representation is unavailable, ordinary prefill/recompute remains the semantic ground truth unless a caller explicitly requests cache-only behavior and accepts a `CacheMiss`-style failure.

The runtime must never substitute approximate context silently for an exact checkpoint request.

## Why checkpoint-first

Even without independently composable blocks, exact checkpoints provide:

- repeated continuation from large stable prompts,
- cheap branching,
- reusable tool/system/project state when the prefix is stable,
- a correctness anchor for BlockStore experiments,
- a natural future unit for persistence/spill economics.

Composable context blocks remain an optimization described separately in ADA-0005.

## Open implementation questions

- Physical checkpoint representation and copy cost.
- Best checkpoint granularity/frequency.
- GPU memory cost for many branch points.
- Copy-on-write vs explicit copies vs replay.
- Host spill/restore vs recompute economics.
- Context-length/position-limit behavior.
- Whether some exact checkpoints can share immutable state physically.

## Decision gate for implementation

Treat checkpoint implementation as production-ready only after:

1. state/logit differential tests demonstrate exact continuation semantics,
2. branching from one checkpoint is verified,
3. restoring after unrelated runtime work does not corrupt the checkpoint,
4. memory/time costs are characterized,
5. fallback/recompute remains simple and testable.
