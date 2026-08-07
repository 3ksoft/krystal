# ADA-0006 — Structured GPU Decoding

Status: **ACCEPTED v0 BASELINE**  
Date: 2026-08-03  
Updated: 2026-08-07

## Context

Structured generation is now implemented end to end on the real LFM2.5 WebGPU backend.

The earlier v0.3 design left several points open: whether JSON would merely be a renderer, whether sparse LM-head execution would be part of the first implementation, and what exact GPU IR would be used. v0.4 records the implementation that survived testing.

## Decision

Structured generation constrains token selection **before** sampling using a host-compiled deterministic byte-level program executed on the GPU.

Public API:

```ts
engine.generate(Type<T>, {
  checkpoint?,
  blocks?,
}) -> Promise<T>
```

The root may be a scalar, array or object.

## Implemented pipeline

```text
Type<T>
    ↓ host
JSON Schema semantics
    ↓
constraint IR
    ↓
deterministic packed GPU program
    + model-global tokenId -> raw bytes table
    ↓ GPU per token
model forward -> logits
    ↓
constraint_mask
    ↓
constraint_argmax + decoder-state commit
    ↓
next token
    ↓ terminal host readback
strict JSON bytes
    ↓
JSON.parse
    ↓
T
```

The runtime does not perform an implicit `schema.assert(result)` after generation.

## Strict JSON representation

v0.4 uses strict canonical JSON directly as the emitted constrained value.

Examples:

```text
string schema  -> "text" -> JSON.parse -> string
number schema  -> 42.5   -> JSON.parse -> number
object schema  -> {...}  -> JSON.parse -> object
```

Once the root reaches the accept state, only EOS is allowed. Trailing whitespace is not intentionally generated in the v0 path.

This supersedes the v0.3 possibility of generating a separate typed result buffer and rendering JSON afterward.

## Constraint compiler

The runtime compiles JSON-schema semantics directly into the constraint program rather than routing variable-length JSON semantics through a binary memory-layout plan.

This matters for constructs such as bounded dynamic arrays (`minItems`/`maxItems`), whose semantic cardinality is not a fixed binary layout.

The deterministic linker uses compact byte tries/switches and explicit bounded-repeat/jump structure. Optional fields and enums do not require a runtime NFA set of active branches.

## Token byte metadata

Constraint validity is byte-based.

For each vocabulary token the GPU has metadata describing its raw byte sequence and special-token status. The same JSON fragment may be formed by different token segmentations; correctness therefore cannot depend on one canonical tokenization.

Special/empty tokens are rejected unless explicitly handled by generation semantics.

## Vocabulary mask

For LFM2.5:

```text
vocab = 65,536
mask  = 65,536 bits
      = 8,192 bytes
      = 2,048 × u32
```

Current `constraint_mask` execution uses 2,048 invocations, each evaluating 32 token IDs and writing one mask word. This avoids atomic OR/races.

Candidate evaluation is transactional: each candidate uses a local decoder-state copy and does not mutate durable state.

## Constrained argmax and commit

After mask creation, `constraint_argmax`:

- applies the exact mask to logits,
- preserves reserved/sentinel token rules,
- selects the best allowed token for the current greedy path,
- advances the constraint VM with the selected token bytes,
- commits the new decoder state,
- updates normal generation runtime/token state.

A state with no legal continuation is a hard constraint dead-end. The runtime must not silently relax the schema.

## Dense vs sparse

The current accepted baseline is **masked dense**:

```text
full logits -> exact full-vocabulary mask -> constrained argmax
```

Sparse LM-head row execution is **not implemented**.

Sparse edges/tries inside the constraint VM reduce the cost of validating one token, but the mask kernel still considers all vocabulary IDs.

Current measurements show the mask kernel at roughly 0.08–0.48 ms versus ~31 ms/token model forward in the present benchmark setup, with total structured overhead around 1%. This is not enough evidence to justify sparse LM-head complexity yet.

## Output bounds

The structured typed API derives a conservative token budget from the schema rather than requiring an ordinary public `maxTokens` parameter.

Schemas therefore need a finite representational bound where necessary, e.g. bounded strings and arrays.

The resulting budget must still fit the runtime context capacity.

## Current coverage

The v0 path covers the currently validated envelope:

- bounded strings,
- numbers/integers with current range semantics,
- booleans,
- literals/enums,
- objects,
- required/optional fields,
- nested objects,
- bounded arrays (`minItems`/`maxItems`),
- fixed arrays where applicable.

Deferred/non-goal coverage includes:

- unbounded arrays,
- tuples/`prefixItems`,
- general recursive schemas,
- full arbitrary union semantics,
- `multipleOf`/step,
- complete JSON Schema keyword compatibility.

## Correctness evidence

The implementation acceptance chain is:

```text
linker tests
→ CPU reference VM
→ CPU packed-mask oracle
→ Dawn AOT mask equivalence
→ constrained argmax/state-transition tests
→ real-model public Engine.generate E2E
```

The CPU oracle consumes the same packed program/tokenizer blobs as the GPU path.

## Sampling invariant

Constraints apply before token selection.

This path is not accepted:

```text
unconstrained top-K
→ remove invalid tokens
→ sample remainder
```

The current greedy implementation masks the complete vocabulary distribution before argmax. Any future sampler must preserve the same ordering of semantics.
