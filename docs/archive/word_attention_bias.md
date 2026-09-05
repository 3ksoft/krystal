# Word-group attention bias experiment

## Goal

Test whether an additional structural relation:

```text
Record → Word → Token
```

can bind modifiers to properties without adding another encoder, pooling stage, output head or trainable parameter set.

The experiment must run on the current Krystal model. It is an isolated architecture assay, not a continuation of S1–S9 and not a new curriculum milestone.

The primary question is:

> Can a fixed additive same-word attention bias distinguish records containing exactly the same tokens but different token-to-word assignments?

Example:

```text
Record A:
  [some, red]
  [much, yellow]

Record B:
  [much, red]
  [some, yellow]
```

Both records contain the identical token multiset:

```text
some, much, red, yellow
```

Only word membership specifies which quantifier modifies which color.

## Architectural constraint

Do not add:

* a word encoder;
* materialized `WordState`;
* word pooling;
* another attention head;
* another selector;
* another prediction loss;
* learned word-index embeddings.

Add only parser/lowering-provided word membership and an additive attention-logit bias, using the existing same-record structural-bias mechanism as the precedent.

For tokens `i` and `j`:

```text
sameWord(i, j) =
  sameRecord(i, j)
  && wordId[i] != INVALID
  && wordId[i] == wordId[j]
```

The local attention score becomes:

```text
score(i, j) =
  qᵢ · kⱼ / sqrt(H)
  + existingStructuralBias(i, j)
  + wordAlpha * sameWord(i, j)
```

If the current encoder already restricts attention to one record, retain that restriction unchanged. `sameRecord` remains part of the semantic definition so local word IDs may safely be reused in different records.

`KEY`, `VALUE`, pooling, query and other synthetic control slots must use `wordId = INVALID`. They receive no word bias and continue to aggregate the complete record normally.

Word IDs are arbitrary local labels. The model must never receive:

* the numeric word ID as an embedding;
* the ordinal position of the word;
* a stable correspondence between word ID and semantic role.

Renumbering words must not change the result.

## Experimental tokens

Use existing suitable experimental tokens if available. Otherwise reserve the minimum experimental vocabulary:

```text
APPLE
SOME
MUCH
RED
YELLOW
```

Do not let vocabulary extension perturb unrelated weight initialization. All profiles for a given model seed must start from an identical parameter checkpoint. Dataset, layout and noise RNG streams must remain independent from model initialization.

## Phase W0 — intent discrimination

Create balanced paired frames with one visible, near, edible Apple.

All non-property state is identical:

```text
same comfort
same distance
same capabilities
same candidate masks
same record count
same word count
same word sizes
same token multiset
```

Only token-to-word membership changes.

### Variant A

```text
[apple]
[much, red]
[some, yellow]

gold intent: EAT
gold argument: Apple ref
```

### Variant B

```text
[apple]
[some, red]
[much, yellow]

gold intent: LOOK
gold argument: Apple ref
```

Both `EAT.target` and `LOOK.target` must legally admit the same Apple record. Therefore intent selection cannot be inferred from argument feasibility.

The semantic rule for this isolated assay is:

```text
MUCH RED → EAT
SOME RED → LOOK
```

This rule is intentionally minimal. The purpose is to test modifier binding, not to establish the final meaning of ripeness.

Report:

* intent accuracy;
* pointer accuracy;
* joint exact match;
* invalid-pointer rate;
* mean `P(EAT)` and `P(LOOK)` for both variants;
* per-seed confusion matrices.

## Phase W1 — exact pointer selection

After W0 passes, create frames containing two visible, near, edible Apples.

```text
Apple A:
  [apple]
  [much, red]
  [some, yellow]

Apple B:
  [apple]
  [some, red]
  [much, yellow]
```

The gold output is:

```text
EAT(Apple A)
```

Both Apple records:

* contain the same flat token multiset;
* are admitted by the EAT argument mask;
* differ only in word membership;
* receive independently randomized runtime refs, record positions and physical layouts.

This phase tests whether the existing selector can use word-bound token representations to choose the exact record.

Report:

* intent accuracy;
* pointer accuracy;
* joint exact match;
* invalid-pointer rate;
* full 2×2 pointer confusion;
* probability assigned to the correct and incorrect Apple.

## Anti-shortcut construction

The dataset must make the flat representation non-predictive.

For every semantic frame:

* randomly permute physical token positions;
* preserve word membership through the sidecar;
* allow tokens belonging to one word to occupy non-contiguous physical positions;
* randomly permute local word IDs;
* randomly permute record order;
* randomize runtime refs independently;
* keep group sizes identical between labels;
* keep the flat token multiset identical between labels.

The serialized token order, word ID number, record index, ref value, padding and layout must be statistically independent of the target.

Add a dataset audit proving this independence.

If the current packed format requires contiguous words, add an experimental word-membership sidecar that supports arbitrary token membership. Do not rely on adjacency as a substitute for grouping.

## Profiles

Run the exact same initialized checkpoint and exact same dataset through all profiles.

### P0 — no word structure

```text
wordAlpha = 0
```

Word IDs may be present in metadata but have no effect.

Expected result:

```text
approximately chance
```

If P0 learns reliably, the dataset leaks the answer and the experiment is invalid.

### P1 — correct word bias

```text
wordAlpha = 4.0
```

Use `4.0` as the primary value because it is the established successful same-record bias scale. Do not tune it before running the frozen comparison.

Expected result:

```text
near-perfect W0 and W1
```

### P2 — randomized word-membership control

Use `wordAlpha = 4.0`, but independently shuffle token-to-word assignments while preserving:

* word count;
* word sizes;
* token multiset;
* all other frame data.

Expected result:

```text
approximately chance
```

This proves that improvement comes from correct binding structure, not merely from adding stronger local attention.

### P3 — word-ID renumbering conformance

Evaluate a trained P1 checkpoint twice on the same frames:

```text
evaluation A: original local word IDs
evaluation B: randomly bijected local word IDs
```

Outputs and probabilities must match within normal numeric tolerance, and exact predictions must be identical.

## Protocol

Use a small standalone balanced dataset:

```text
train semantic pairs: 256
held-out semantic pairs: 64
```

Each semantic pair must contain both counterfactual variants before nuisance randomization.

Use the current optimizer and model dimensions unchanged. Begin with the established learning rate and epoch count used by the smallest reliable current-model overfit assays.

Run at least these model initialization seeds:

```text
42
7
1337
```

Every profile within one seed must begin from an identical parameter checkpoint. Print and compare initial parameter checksums.

Do not mix S1–S9 replay into this experiment. The assay must measure only whether the present architecture can exploit word membership.

## Required parity and regression coverage

Before interpreting training:

1. Add CPU/GPU forward parity for the word-biased attention probabilities and outputs.
2. Add backward parity through the affected attention operation.
3. Verify `wordAlpha = 0` is numerically equivalent to the unchanged current implementation.
4. Verify `wordId = INVALID` produces no word bias.
5. Verify equal local word IDs in different records do not create cross-record word binding.
6. Verify word-ID bijection leaves outputs unchanged.
7. Keep all existing record-binding, selector and production-emission tests green.

Because `wordAlpha` is fixed, no gradient or optimizer state is required for it.

## Pass conditions

The experiment supports word grouping only if:

```text
P0: near chance
P1: ≥ 0.99 joint exact match on W0 and W1 across all three seeds
P2: near chance
P3: exact predictions invariant under word-ID renumbering
CPU/GPU parity: within existing tolerances
invalid/fabricated pointers: zero
```

Additionally, P1 must remain correct under:

```text
fresh runtime refs
record permutation
token permutation
non-contiguous word members
fresh physical layouts
```

Do not accept a result that works only for one initialization seed.

## Interpretation

### P0 fails and P1 passes

The same-word relation provides information that the current flat record representation cannot recover. Accept word membership as part of the structural model contract.

### P0 and P1 both pass

The assay leaks grouping through token order, adjacency, IDs or another nuisance variable. Audit and rebuild the dataset before drawing architectural conclusions.

### P0 and P1 both remain at chance

Check first:

* whether the bias reaches the intended record-local attention operation;
* whether record pooling can aggregate across multiple words;
* whether `wordAlpha` is numerically active;
* whether gradients reach the relevant token representations.

Do not add a word encoder until the fixed-bias path has been verified.

### P1 and randomized P2 both pass

The model is exploiting an unintended structural shortcut. The experiment does not demonstrate semantic word binding.

### W0 passes but W1 fails

Word grouping supports intent discrimination, but the existing selector cannot reliably expose the bound property at record-pointer selection. Diagnose selector access to the resulting token/field states before changing the architecture.

## Optional follow-up

Only after the primary experiment passes, run a small fixed sweep:

```text
wordAlpha = 2.0, 4.0, 8.0
```

The purpose is to check that the effect is not isolated to one exact constant. Do not use this sweep to select a value based only on the held-out test set.

A later compositional follow-up may introduce additional colors and quantifiers, with held-out combinations, but this is outside the primary architecture assay.

## Deliverable

Create a short result document containing:

1. the exact representation and bias formula;
2. the dataset anti-shortcut audit;
3. P0/P1/P2/P3 results per seed;
4. W0 intent and W1 pointer metrics;
5. CPU/GPU parity results;
6. accepted or rejected hypothesis;
7. whether word grouping should enter the next Krystal ABI/model revision.

Do not modify the main curriculum or declare the new structure production-ready until this isolated experiment passes.
