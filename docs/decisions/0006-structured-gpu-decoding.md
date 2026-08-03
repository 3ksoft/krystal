# ADA-0006 — Structured GPU Decoding

Status: **PROVISIONAL**  
Date: 2026-08-03

## Context

Chomato needs structured/constrained generation for typed tool arguments, machine-consumed data, local protocols, and schema-driven output.

The earlier design treated this broadly as a validator-placement problem. Further analysis narrows the target architecture substantially.

The reference model has exactly 65,536 vocabulary entries (`2^16`), making dense token-set representations unusually convenient for WebGPU.

The project also already has schema/layout/code-generation infrastructure through schema-pop, making a flat typed output plan more natural than treating JSON punctuation as the core semantic target.

## Decision

Structured generation is implemented conceptually as a **compiled decoder program that constrains token selection before sampling**.

Target pipeline:

```text
semantic schema / constraint
        ↓
HOST: compile to flat OutputPlan
        + tokenizer token-byte metadata / automata
        ↓
GPU: structured decoder state
        ↓
exact allowed-token set for current state
        ↓
LM-head execution mode
        ↓
mask/restrict before sampling
        ↓
GPU sampler
        ↓
advance decoder state
        ↓
typed output state / emitted tokens
```

The exact IR and kernels remain experimental, but this direction is selected over a design centered on per-token CPU validation/readback.

## JSON is an output format, not the internal semantic model

A request may ask for JSON, JSON Schema compatibility, or another familiar external format.

Internally, the engine may represent the target as typed operations such as:

```text
BEGIN_OBJECT
FIELD
STRING
U8
ENUM
BOOL
END_OBJECT
END
```

The exact opcode set is not frozen.

The structured decoder may write a typed result representation and render JSON afterward, or emit syntax tokens directly when the requested wire format requires it.

The architecture must not require the model to generate redundant punctuation solely because JSON was used to describe the constraint at the API boundary.

## OutputPlan

The host compiles the high-level constraint before generation into a flat plan suitable for bounded incremental execution.

Expected state may include concepts such as:

```text
program counter
field/variant state
lexer/scalar state
array counters
optional/variant jumps
termination state
```

Complex recursive host object graphs are not the target runtime representation.

## Tokenizer automata / token metadata

Constraints are defined over semantic values/bytes/lexical states while the model selects token IDs.

The runtime therefore needs tokenizer-aware metadata that can answer, for a decoder state:

```text
which token IDs are valid?
what decoder/lexer state follows each valid token?
```

Potential representations include:

- token ID → byte sequence metadata,
- token tries,
- enum-specific tries,
- precompiled `(type/state) → allowed token IDs / transitions`,
- GPU-friendly transition tables.

A global rule such as "all numeric-looking tokens are valid for numbers" is insufficient. Validity depends on the current lexical/semantic state.

UTF-8/multi-byte token boundaries and special tokens are explicit correctness cases.

## Vocabulary mask geometry

For LFM2.5:

```text
vocab = 65,536 tokens
      = 2^16

1-bit dense allowed-token mask
      = 65,536 bits
      = 8,192 bytes
      = 2,048 × u32
```

This is small enough to be a practical device-resident data structure.

The mask does not need to be copied wholesale into workgroup memory per token. It can remain in storage/device-visible memory and be queried by the relevant LM-head/sampling kernels.

## Execution modes

The execution mode changes how logits are computed/restricted, not the constraint semantics.

### DENSE

Unconstrained generation:

```text
compute full LM head
→ sample
```

### MASKED_DENSE

For a large allowed set:

```text
compute full LM head
→ apply exact allowed-token mask
→ sample over allowed tokens
```

This is the natural baseline when the constraint allows a substantial fraction of the vocabulary, e.g. many free-form string states.

### SPARSE

For a sufficiently small exact allowed set:

```text
allowed token IDs known first
→ evaluate only corresponding LM-head rows
→ sample over exactly those logits
```

This may be attractive for states such as small enums, booleans, delimiters, or tightly constrained numeric/structural positions.

However, allowed-set size is **state-dependent**, not simply type-dependent. A numeric field may allow many tokens in one lexical state and few in another.

The dense/sparse threshold is determined empirically.

## Semantic invariant: constrain before sampling

The intended constrained distribution is defined by applying the constraint to the model distribution **before** token selection.

Therefore a path like:

```text
unconstrained top-K
→ discard invalid tokens
→ sample remaining
```

is not accepted as an equivalent default because it can remove valid probability mass before the constraint is applied.

`SPARSE` is semantically valid only because the exact allowed set is known first and only those required LM-head rows are evaluated.

## GPU residency target

Host responsibilities occur before generation:

```text
compile schema/constraint
build/select OutputPlan
automata/token metadata upload
```

The target per-token loop is device-resident:

```text
model
→ allowed-token computation/state
→ masked/sparse logits
→ sample
→ advance decoder state
→ next token
```

A CPU full-vocabulary validator remains valuable as a correctness oracle during development, but is not the target production critical path.

## Dead ends

The runtime must detect a decoder state with no valid continuation.

Exact public error semantics remain to be selected, but the engine must not silently relax the constraint to continue generation.

## Termination

The decoder must distinguish:

- valid terminal completion,
- model EOS where allowed,
- invalid/dead-end state,
- max-token/request cancellation.

## Typed result and rendering

The internal result may be represented as:

- the emitted constrained token sequence,
- a typed result buffer driven by OutputPlan,
- both.

JSON rendering, if requested, may occur after typed generation and does not define the internal correctness model.

## Decision gate

Move the execution design toward ACCEPTED only after:

1. a correctness oracle exists for small plans,
2. token-byte/UTF-8 semantics are tested,
3. GPU decoder transitions match the oracle,
4. `MASKED_DENSE` and `SPARSE` produce the same allowed-set semantics,
5. the sparse path never performs top-K-before-validation,
6. per-token host synchronization is not required in the production path,
7. memory cost for plan/automata state is bounded,
8. dead-end and terminal behavior are deterministic,
9. real decode TPS impact is measured.

## Non-decisions

This ADA does not freeze:

- the final public schema language,
- exact OutputPlan opcodes,
- exact GPU automata layout,
- sparse/dense threshold,
- sampling algorithm beyond constraint-before-sample semantics,
- final typed-result encoding.
