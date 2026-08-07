# Context State and Exact Checkpoints

Status: Implemented and validated  
Date: 2026-08-07

## 1. Definition

A `ContextCheckpoint` is the exact model continuation state after an ordered token prefix has been processed normally.

It is not a cached prompt embedding and it is not an independently composable block representation.

For the current LFM2.5 backend:

```text
ContextCheckpoint
├── attention KV prefix
├── rolling short-convolution state
└── continuation metadata (position/layout identity)
```

## 2. Attention checkpoint state

Only the attention layers have KV state. `attentionLayerSlots` maps those model layers onto compact KV slots.

This is a fixed mapping, not paged attention:

```text
layer -> attention slot or no slot
```

The runtime allocates KV capacity for the configured context window, but checkpoint materialization copies only the **populated prefix**.

Therefore checkpoint KV bytes grow with prefix length:

```text
O(attentionLayers × populatedSequenceLength × KVWidth)
```

They do not automatically snapshot the entire configured `contextCapacity`.

## 3. Short-convolution / SSM-style state

The LFM2 short-convolution path uses a fixed-size rolling state with `convCacheLength = 3` in the current model.

Checkpointing copies that rolling state in full. There is no additional conv-history buffer whose size grows with sequence length.

Its checkpoint cost is therefore approximately:

```text
O(convLayers × hiddenSize × convCacheLength)
```

rather than `O(sequenceLength)`.

This is the model's recurrent continuation state for those layers.

## 4. No re-prefill on restore

Restoring a checkpoint does not replay its source prefix.

If a checkpoint represents prefix `P` and the caller appends tail `T`:

```text
checkpoint(P) + blocks(T)
```

then only `T` is prefetched before decode. The prefix state is restored from the physical checkpoint.

The runtime exposes execution statistics used by tests to verify this invariant.

## 5. Branching

One checkpoint can seed independent branches:

```text
checkpoint P
├── tail A -> generation A
└── tail B -> generation B
```

Running branch A must not mutate the durable checkpoint in a way that changes branch B. The same checkpoint can be reused again after either branch.

## 6. Chaining

A checkpoint can be extended into a new checkpoint:

```text
checkpoint AB
+
block C
↓
checkpoint ABC
```

The base checkpoint prefix is not replayed to build the child checkpoint.

## 7. Source-block lifetime

A materialized checkpoint owns the continuation state it needs. Dropping the source `ContextBlock`s after materialization does not invalidate the checkpoint.

This distinguishes:

```text
ContextBlock       token/content identity
ContextCheckpoint  physical continuation state
```

## 8. `depth` is not paging

`CacheBlockOptions.depth` belongs to composable cached-representation experiments. It describes how far through the model a reusable representation was computed.

It does **not** describe:

- a KV page size,
- a page-table depth,
- a dynamically allocated attention page,
- checkpoint nesting.

Exact checkpoints do not depend on composable-block depth semantics.

## 9. Current test invariants

The real-engine checkpoint suite verifies:

- checkpoint + continuation equals ordinary full-context continuation,
- one checkpoint branches without mutation,
- checkpoints can be extended into new checkpoints,
- checkpoint survives dropping source blocks,
- checkpoint at every tested prefix split matches uninterrupted inference,
- checkpoint prefix is not prefetched again,
- checkpoint bytes and KV bytes grow with populated prefix length.

These tests execute through the public engine API and real LFM2 WebGPU runtime.
