# ADA-0004 — Context State and Exact Checkpoints

Status: **ACCEPTED**  
Date: 2026-08-03  
Updated: 2026-08-07

## Context

Exact checkpointing is now implemented and validated on the real LFM2.5 WebGPU runtime.

The capability is intentionally independent of deeper composable-context experiments: even if no independently composable cached representation survives, an exact physical continuation checkpoint remains useful for repeated continuation and branching.

## Decision

A `ContextCheckpoint` is the exact model continuation state after a specific ordered token prefix has been processed normally.

For LFM2.5 it materializes:

```text
populated attention KV prefix
+
fixed-size rolling short-convolution state
+
position/layout metadata required for continuation
```

A checkpoint is prefix- and position-dependent.

## Attention state

`attentionLayerSlots` is a fixed compact mapping from attention model layers to KV storage slots. It is not paged attention.

The runtime KV allocation is sized to `contextCapacity`, but checkpoint materialization copies only the populated prefix. Checkpoint KV bytes therefore grow with prefix length rather than with the configured maximum capacity.

## Rolling convolution state

The current LFM2 short-convolution path retains a fixed-size rolling cache (`convCacheLength = 3`).

Checkpointing copies this rolling state in full. There is no additional sequence-length-dependent convolution history buffer.

Its storage cost is therefore approximately:

```text
O(convLayers × hiddenSize × convCacheLength)
```

rather than `O(sequenceLength)`.

## No replay invariant

Restoring a checkpoint does not re-prefill its source prefix.

Given:

```text
checkpoint(P) + blocks(T)
```

only `T` is processed as appended prefill before decode. The state for `P` comes from the checkpoint.

## Branching

A durable checkpoint can seed multiple branches and branch execution must not mutate it:

```text
checkpoint P
├── continuation A
├── continuation B
└── continuation A again
```

All branches start from the same exact state.

## Chaining

A checkpoint can be extended into another checkpoint:

```text
checkpoint AB + block C -> checkpoint ABC
```

The base checkpoint prefix is not replayed.

## Source-block lifetime

A materialized checkpoint remains usable after the `ContextBlock`s that created it are dropped.

Blocks identify reusable token content. Checkpoints own physical continuation state.

## Model checkpoint vs generation continuation

The conceptual distinction remains:

```text
ContextCheckpoint
    exact reusable model continuation state

GenerationContinuation
    ContextCheckpoint
    + sampler/request state
    + structured-decoder state
```

The same `ContextCheckpoint` can seed different structured schemas or ordinary generation.

## Exactness evidence

The current real-engine suite verifies:

- checkpoint continuation equals uninterrupted continuation,
- branching does not mutate the checkpoint,
- checkpoint chaining is equivalent to direct context,
- source blocks may be dropped after materialization,
- checkpointing works across tested prefix split positions,
- restored prefixes are not re-prefilled,
- checkpoint/KV byte counts scale with the populated prefix.

These tests are the implementation acceptance gate for the current exact checkpoint path.

## Relationship to composable cached representations

`CacheBlockOptions.depth` and independently composable intermediate representations are governed by ADA-0005.

They are not part of the exact checkpoint representation and must not be confused with attention paging or checkpoint hierarchy.
