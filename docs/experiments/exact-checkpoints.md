# Experiment — Exact Context Checkpoints

Status: **HIGH PRIORITY / REQUIRED FOR CHECKPOINT ACCEPTANCE**

## Goal

Prove that Chomato can capture and restore a model continuation state that behaves exactly like uninterrupted execution from the same prefix.

This is the fundamental context-reuse experiment in v0.3.

## Reference model

`LFM2.5-1.2B-Instruct`

Checkpoint state includes at least:

```text
short-convolution history/cache
attention KV state for GQA layers
position/prefix length implied by the checkpoint
```

## Core test

For a prefix `P` and continuation token sequence `S`:

```text
Path A:
P → normal execution → S

Path B:
P → normal execution → checkpoint
  → restore checkpoint → S
```

Compare Path A and Path B after every continuation step.

## Required comparisons

### 1. Stored state

If checkpoint creation/restoration is a pure GPU/host buffer copy, verify bitwise restoration of checkpointed state where possible.

### 2. Logits

Compare full logits (or a deterministic sufficiently complete projection if full readback is impractical during every debug run) after each continuation token.

Do not use only top-1 token agreement.

### 3. Subsequent Context State

After applying identical continuation tokens, compare the resulting conv/KV state at selected checkpoints.

### 4. Position-sensitive continuation

Run checkpoints at multiple prefix lengths and continue with the same suffix pattern to expose position/offset bugs.

## Branching test

```text
checkpoint P
├── branch A tokens
├── branch B tokens
└── branch C tokens
```

Requirements:

- branch A must not mutate the durable checkpoint used by B/C,
- restoring P after unrelated branch execution must reproduce the same continuation logits,
- branch order must not matter.

## Restore-after-unrelated-work test

1. Create checkpoint P.
2. Run another request/branch that mutates live runtime state.
3. Restore P.
4. Continue with the known token sequence.
5. Compare against uninterrupted reference.

This catches accidental aliasing between durable checkpoint storage and live frontier buffers.

## GenerationContinuation test

Model-checkpoint correctness should be tested independently from sampler state.

For deterministic generation-resume tests:

- either feed the same chosen token IDs explicitly, or
- persist sampler/RNG state separately as a `GenerationContinuation` and verify exact resume.

Do not require validator/sampler state to become part of reusable `ContextCheckpoint` identity.

## Scale sweep

Test checkpoints at:

```text
short prefix
medium prefix
near current practical context limit
```

Measure:

- checkpoint bytes,
- checkpoint creation time,
- restore time,
- branch cost,
- VRAM growth with checkpoint count.

## Pass criteria

The exact checkpoint capability is accepted when:

1. pure state-copy restoration is bitwise exact where expected,
2. continuation logits/state match uninterrupted execution under the defined exactness criterion,
3. multiple branches are independent,
4. restore remains correct after unrelated runtime work,
5. position/prefix-length tests pass,
6. memory/time costs are known,
7. ordinary prefill remains a working correctness fallback.
