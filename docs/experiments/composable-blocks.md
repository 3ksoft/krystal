# Experiment — Composable Context Blocks

Status: **ACTIVE EXPERIMENT**

## Goal

Determine which independently precomputed `CachedRepresentation`s can be safely and profitably composed, without making them the semantic foundation of context reuse.

Exact checkpoints remain the ground truth.

## Reference model

`LFM2.5-1.2B-Instruct`

Current parsed layer plan:

```text
CCACCACCACACACAC
```

The first global attention layer is layer 2.

## Depth-2 candidate

Current pipeline:

```text
block tokens
→ embedding
→ conv layer 0
→ conv layer 1
→ cached representation
```

Because the first two convolution layers have finite causal receptive field, composition can repair a bounded boundary region and rebuild the required convolution tail.

Current evidence includes 1:1 greedy output on tested prompts and a measurable prefill speedup in the existing experiment.

### Required upgrade to exactness evidence

Before labeling the representation `EXACT_INDEPENDENT`:

1. compare logits after cached composition vs ordinary full prefill,
2. compare relevant intermediate/Context State where practical,
3. test multiple block sizes,
4. test multiple block counts,
5. test sampled continuations after identical externally supplied continuation tokens,
6. test boundary token patterns designed to maximize convolution-edge effects.

Greedy-token equality alone is not enough.

## Tokenizer/template provenance

A `ContextBlock` contains the exact immutable token sequence.

Tests must include cases where text concatenation would tokenize differently from independent fragments.

The expected behavior is defined by the token IDs supplied to the block system, not by re-tokenizing concatenated source strings behind the cache's back.

## Attention-crossing position check — prerequisite

Before interpreting any depth ≥ first-attention experiment, verify the LFM2.5 implementation's positional handling:

- where position indices enter attention,
- whether rotation/position is baked into cached K/Q/hidden representations,
- whether a block precomputed at one offset can be relocated at another offset,
- whether any relocation transform is possible/required.

Do not run/tune a deeper depth sweep under the assumption that all observed degradation is only "attention data dependence" until position semantics are understood.

## Approximate deep representations

After the positional check, test the current useful framing:

```text
selected persistent cached context
→ ordered materialization
→ approximate frontier
→ live query
→ decode
```

Measure quality separately from exact checkpoint/cache tests.

## Depth sweep

Candidate depths:

```text
5 → 8 → 10 → 12 ...
```

For each depth measure:

- precompute cost,
- materialization cost,
- VRAM,
- live-query quality,
- sensitivity to order/position,
- compute actually saved.

## Block-count stress

Sweep:

```text
4 → 16 → 32 → 100 → ...
```

Log WebGPU adapter/device limits alongside requested allocation sizes, including relevant `maxBufferSize` and `maxStorageBufferBindingSize`, so an API binding-size ceiling is not mistaken for a fundamental architectural scaling limit.

Measure:

- total VRAM,
- materialization latency,
- live-query quality,
- ordering effects,
- boundary repair cost for exact shallow representations.

## Semantic stress tests

### Permutations

Same blocks, different order.

### Contradictory facts

Semantically similar blocks with conflicting values.

### Multi-hop

Require an answer to combine facts from 2–3 independent blocks.

### Irrelevant block load

Add many unrelated blocks and observe retrieval/attention quality.

## Economics test

Compare:

```text
ordinary full prefill
vs
cached shallow representation + boundary repair
vs
checkpoint restore when prefix is stable
```

The best method may differ by workload.

## Host spill

Do not design a complex pager until GPU-resident behavior is characterized.

Then compare:

```text
GPU keep
vs host spill/restore
vs exact recompute
```

## Output of this experiment

For each representation depth/kind, produce:

```text
exactness class
composition rules
position rules
memory cost
materialization cost
recommended workload
fallback
```
