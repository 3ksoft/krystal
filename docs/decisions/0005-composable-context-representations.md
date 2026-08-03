# ADA-0005 — Composable Context Representations

Status: **PROVISIONAL**  
Date: 2026-08-03

## Context

Exact checkpoints reuse a previously processed **ordered prefix**.

Chomato also investigates a more ambitious capability: independently precompute reusable context blocks and compose selected blocks into a later request without re-running their full computation from zero.

The current LFM2.5 experiments show two importantly different regimes:

1. a shallow representation before the first attention layer that appears independently composable with bounded local repair,
2. deeper representations that cross global attention and are not generally equivalent to normal joint-prefix execution.

This capability is useful, but it is an optimization layered on top of the exact checkpoint model, not the semantic foundation of reuse.

## Decision

The runtime distinguishes:

### ContextBlock

An immutable exact token sequence:

```text
ContextBlock {
  id
  tokenIds[]
  provenance / metadata
}
```

Token/template provenance matters. The block must represent the exact token sequence intended for the model, including relevant special/chat-template tokens.

The runtime does not assume that independently tokenized raw text fragments can always be concatenated to reproduce tokenization of concatenated source text.

### CachedRepresentation

A model/depth-specific precomputation derived from a `ContextBlock`.

Metadata includes at least:

```text
source block identity
model/tokenizer compatibility
representation depth/kind
composition rules
exactness class
```

### Materialized frontier

After selected representations are composed/materialized, the runtime tracks:

```text
ordered token/context identity
model Context State
where live causal computation resumes
exactness class of the resulting frontier
```

## Exactness classes

A representation/use is classified explicitly.

Examples:

```text
EXACT_INDEPENDENT
EXACT_PARENT_DEPENDENT
APPROXIMATE
```

Names are not frozen, but the distinction is.

Exactness is a property of the **representation plus its permitted use/composition rules**, not merely of the bytes stored in a cache entry.

## Exactness monotonicity

If an approximate representation contributes to a materialized frontier:

```text
frontier = APPROXIMATE
```

All downstream generations/checkpoints derived from that frontier are at best approximate unless the runtime recomputes an exact prefix through ordinary model execution.

Approximate context must never be reported as an exact checkpoint.

## Current depth-2 candidate

The current model begins with two local convolution layers before its first global attention layer.

Current experimental pipeline:

```text
block tokens
→ embedding
→ conv layer 0
→ conv layer 1
→ cached hidden representation
```

The local causal receptive field is bounded, so block concatenation can repair only a small boundary region and rebuild the final convolution tail required for continuation.

Current tests show 1:1 greedy output against ordinary prefill.

This is promising evidence for an independently composable shallow representation, but v0.3 does **not** promote it to proven logit-exactness until the stronger differential tests in `experiments/composable-blocks.md` pass.

## Attention-crossing representations

Once a cached representation crosses an attention layer, standalone block computation is not generally equivalent to computing the same block inside an arbitrary combined prefix.

Two separate reasons must be treated explicitly:

1. **data dependence** — attention depends on preceding content,
2. **position/order dependence** — position encoding may already be baked into Q/K/hidden state at the point a representation is cached.

The exact LFM2.5 positional handling must be verified from the implementation/model path before interpreting deeper block-composition experiments.

Until then, attention-crossing cached representations remain **EXPERIMENTAL / approximate**.

## Current useful approximate framing

The experimentally useful deep path is:

```text
persistent cached context representations
        ↓
ordered materialization
        ↓
approximate frontier
        ↓
live query evaluated causally
        ↓
decode
```

This may be valuable even when it is not equivalent to ordinary full prefill.

Its quality contract remains experimental and separate from exact checkpoint semantics.

## Residency

Logical block/representation identity is separate from residency.

Possible states may include:

```text
GPU resident
host resident
tokens/metadata only
evicted/recomputable
```

Eviction must preserve either:

- the ability to rematerialize with the same documented exactness class, or
- a fallback to ordinary exact recomputation from tokens/prefix.

## Ownership

The daemon owns the BlockStore.

Because the baseline daemon has one live session and serialized long-running work, the initial design does not require cross-client pinning/reference-counting semantics.

Internal pinning may still be needed to ensure the active request cannot evict its own required representation while materializing it.

## Open questions

- Logit-level exactness of the current depth-2 path.
- Exact boundary-repair economics vs ordinary prefill.
- Block-size sweet spot.
- Block-count scaling.
- Position handling at attention-crossing cache depths.
- Useful approximate depth(s).
- Ordering/permutation sensitivity.
- Contradictory-context behavior.
- Multi-hop composition quality.
- GPU residency policy.
- Host spill vs restore vs recompute economics.
- Whether deep representations should be grouped or share precompute.

## Decision gate

Promote a specific composable representation to ACCEPTED only when:

1. its composition rules are explicit,
2. exactness is demonstrated at state/logit level if claimed exact,
3. positional/order semantics are understood,
4. memory and materialization cost are characterized,
5. fallback behavior remains simple,
6. approximate representations, if exposed, carry an explicit semantic/telemetry marker.
