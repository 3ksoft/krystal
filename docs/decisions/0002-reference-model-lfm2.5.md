# ADA-0002 — Reference Model: LFM2.5-1.2B-Instruct

Status: **ACCEPTED**  
Date: 2026-08-03  
Updated: 2026-08-07

## Context

Chomato is currently a focused inference-engine project, not a generic model-runtime framework.

The working model is `LiquidAI/LFM2.5-1.2B-Instruct`.

The reference architecture has:

- approximately 1.17B parameters,
- 16 layers,
- 10 double-gated LIV convolution blocks,
- 6 grouped-query attention blocks,
- vocabulary size 65,536,
- stated context length 32,768 tokens.

The current backend already runs complete autoregressive inference through WebGPU and stores both short-convolution history and attention KV state.

## Decision

`LFM2.5-1.2B-Instruct` is the **reference architecture and optimization target** for the current Chomato phase.

Supporting arbitrary transformer, recurrent, SSM, or hybrid architectures is not a current goal.

General abstractions are introduced only when they:

- express a real model-independent invariant, or
- enable another backend without complicating/regressing the LFM2.5 path.

## Why this model

The model aligns with the project goals:

- small enough for local/on-device inference,
- appropriate for rapid kernel/cache experiments on commodity GPUs,
- hybrid architecture makes Context State reuse materially different from classical all-attention models,
- only 6 of 16 layers carry attention KV,
- the current runtime already executes it end to end.

## Context State schema

The generic continuation concept is **Context State**.

For the current LFM2.5 backend:

```text
Context State
├── short-convolution history/cache
└── attention KV state for GQA layers
```

These components are not symmetric:

### Short-convolution state

- bounded/fixed-size per active continuation,
- independent of total prefix length once the required local history is retained.

### Attention KV state

- grows with processed prefix length,
- exists only for attention layers,
- dominates long-context state growth and is the primary future target for paging/spill economics.

A model backend must declare the state needed to continue inference; the public architecture does not assume this state is always KV.

## Current implementation assumptions

Current kernels/runtime may contain model-specific assumptions such as:

- 16-layer hybrid plan,
- hidden size 2048,
- head dimension 64,
- convolution cache length 3,
- current KV layouts derived from model metadata.

These are implementation assumptions, not public architectural requirements.

The model advertises a larger trained context window, but the current Chomato runtime allocation is intentionally smaller:

```text
contextCapacity = 1024
maxNewTokens allocation = 1024
```

The effective response budget must fit the remaining context capacity.

## Vocabulary consequence

The current vocabulary size is exactly:

```text
65,536 = 2^16
```

This has useful engineering consequences for structured decoding, especially dense token masks and compact token-ID indexing. Those consequences are specified in ADA-0006; they do not change the model-selection decision itself.

## Compatibility strategy

If another model family is added later, first identify which concepts remain valid:

```text
Model
Tokenizer
Context State
Prefill / Forward
Decode
Checkpoint semantics
Sampler interface
Structured-decoder interface
```

Model-specific state layout and reusable-representation semantics remain backend-owned.

## Consequences

### Positive

- Optimization work stays focused.
- Model-specific fast paths are acceptable.
- Context/checkpoint design matches the actual hybrid architecture.
- Documentation does not falsely promise generic compatibility.

### Negative

- Some internals may require refactoring before supporting a second model family.
- Public API types must avoid exposing accidental LFM2.5-only buffer/layout details.
