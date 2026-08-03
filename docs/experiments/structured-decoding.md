# Experiment — Structured GPU Decoding

Status: **HIGH PRIORITY / ARCHITECTURE VALIDATION**

## Goal

Validate the v0.3 structured-decoding direction:

```text
schema/constraint
→ OutputPlan
→ tokenizer automata/metadata
→ GPU decoder state
→ exact allowed token set
→ DENSE / MASKED_DENSE / SPARSE
→ GPU sampling
```

The key unknown is no longer whether structured validation is useful. It is whether the plan/tokenizer state can be compiled into a GPU-executable form with low enough per-token cost.

## Fixed reference geometry

For LFM2.5:

```text
vocab = 65,536 = 2^16
mask  = 65,536 bits
      = 8,192 bytes
      = 2,048 × u32
```

## Phase 0 — tokenizer metadata

Export/construct:

```text
token ID → token byte sequence
special-token classification
```

Verify byte semantics independently from displayed Unicode strings.

Required cases:

- ASCII tokens,
- UTF-8 multi-byte characters split across token boundaries,
- tokens containing multiple semantic characters,
- special tokens,
- whitespace-prefixed tokens,
- empty/edge tokenizer cases if present.

## Phase 1 — correctness oracle

Implement a deliberately slow host reference that can answer:

```text
given OutputPlan/decoder state S
and token T
is T valid?
what next state results?
```

For small plans, enumerate the full vocabulary and construct the exact allowed-token set.

This oracle defines correctness for GPU paths.

## Initial plan subset

Start with deliberately small structures:

```text
bool
small enum
small integer/range
fixed object with scalar fields
short constrained string/enum
```

Do not start by implementing every JSON Schema feature.

## Phase 2 — flat OutputPlan

Compile schema/semantic constraints into a compact flat plan.

Prototype state may include:

```text
program counter
field/variant state
scalar lexer state
array count
jumps
terminal/dead state
```

Measure plan size and update cost.

The exact opcodes are experimental.

## Phase 3 — MASKED_DENSE

Implement a GPU path where the exact allowed-token set is represented as a dense 8 KiB bitmap.

Pipeline:

```text
full LM head
→ mask invalid logits
→ GPU sample/argmax
→ GPU decoder-state advance
```

The bitmap may be generated/updated by GPU logic or selected from precomputed transition data. Avoid a required CPU round-trip per token.

Measure:

- mask lookup/update time,
- masking kernel cost,
- sampler cost,
- additional storage bandwidth,
- total token TPS vs unconstrained DENSE.

## Phase 4 — GPU transition execution

Move enough tokenizer/plan transition logic onto GPU that the selected token advances decoder state without host intervention.

Compare every transition/allowed set against the CPU oracle.

## Phase 5 — SPARSE

When the **exact** allowed set is small, evaluate only the corresponding LM-head rows.

Pipeline:

```text
GPU decoder state
→ exact allowed IDs
→ gather/evaluate only allowed output rows
→ sample over exact allowed logits
```

This is not top-K candidate reduction.

Measure the crossover against `MASKED_DENSE`.

Sweep allowed-set cardinality, e.g.:

```text
1
4
8
16
32
64
128
256
512
1k
2k
4k
8k
16k
32k
```

The actual crossover is hardware/kernel dependent.

## Semantic-equivalence test

For the same decoder state:

```text
oracle allowed set
== MASKED_DENSE allowed set
== SPARSE allowed set
```

For deterministic argmax, selected token must match when logits are identical.

For probabilistic sampling, compare the normalized distribution over the allowed set; sparse execution must not remove valid probability mass.

## Numeric/string state tests

Do not preclassify types globally.

For example, "numeric" tokens may or may not be legal depending on whether the decoder currently expects:

- sign,
- first digit,
- subsequent digit,
- decimal point,
- exponent,
- delimiter/end.

Likewise free-form string states may admit a very large portion of the vocabulary while enum/trie states may admit very few tokens.

Mode selection is therefore based on the **current exact allowed set**, not only the schema type.

## Dead-end tests

Create intentionally unsatisfiable states and verify:

- no silent constraint relaxation,
- deterministic terminal/error signal,
- no invalid token is sampled as an escape hatch.

## Host-sync microbenchmark

Even though the target is GPU-resident, retain a measurement of minimal host round-trip cost:

```text
GPU writes selected token ID
→ 4-byte readback / mapAsync
→ host observes
→ next GPU submit
```

This quantifies the cost of any future design that proposes a CPU decision per token.

It is evidence, not the chosen architecture.

## Decision criteria

The structured GPU decoder is viable when:

1. GPU allowed-set/transition results match the oracle,
2. UTF-8/token-boundary cases pass,
3. no production per-token CPU dependency is required,
4. `MASKED_DENSE` overhead is acceptable for broad states,
5. `SPARSE` has a measured crossover for small allowed sets,
6. sparse and masked modes preserve identical constraint semantics,
7. plan/automata memory is bounded,
8. dead-end/termination behavior is deterministic.
