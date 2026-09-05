# M-A (S2–S4) milestone — status and findings

Date: 2026-08-17. Working notes for resuming the S2–S10 grounded-policy curriculum
(`docs/S2_S10_CURRICULUM_TASK.md`), Milestone M-A (S2–S4: intent selection plus
exact visible-resource pointer).

## Resolution: production mixer scratch overflow (2026-08-17)

The apparent intent-learning collapse was primarily an inference-path memory
layout bug, not an architectural inability to discriminate resource presence.

- Training runs the forward graph with saved activations. Its mixer K/V slices
  correctly have `[maxRecords, H]` capacity.
- Production evaluation runs the scratch path. `mixerK` and `mixerV` were
  allocated and bounds-checked as `[maxQueries, H]`, but the mixer matmuls wrote
  `[R, H]` into them.
- The small canonical fixture (`R=7`) fit by accident inside
  `maxQueries * H = 8 * 128`. Policy frames (`R≈100–111`) overflowed the two
  scratch regions and overwrote later forward-arena state, so training saw the
  correct graph while production evaluation saw corrupted state.

After allocating and validating both scratch regions as `[maxRecords, H]`:

- the large policy forward CPU/GPU parity test passes;
- balanced EAT-vs-CRY reaches 100% train/eval on init seeds 42, 7 and 2026;
- mean intent probabilities are decisive (`P(EAT|EAT)≈0.995–0.996`,
  `P(CRY|CRY)≈0.988–0.990`);
- pointer accuracy is 100% and invalid-pointer rate is 0%;
- the real GPU query separation is strong at initialization (relative
  `0.695–0.926`) and remains strong after training (`0.770–0.985`). The older
  "trained qSep" diagnostic was invalid because it re-read the unchanged host
  initialization arrays after GPU-resident SGD.

The corrected 48-seed M-A run improved from 65.1% to 83.7%, exposing a second,
ordinary data-coverage issue: only one S2 EAT example was present. A balanced
variant hash removed the 1/3/5 train vs 7/3/1 eval skew. The final M-A proof
uses 256 train seeds so every S2 counterfactual has the already-validated
minimum coverage (16 EAT / 16 CRY / 17 LAUGH), still with two epochs and
`lr=0.01`.

Final held-out result (seeds 256..287):

```text
intent accuracy          1.0  (43/43)
pointer accuracy         1.0  (25/25 reference-bearing frames)
pointer | correct intent 1.0  (25/25)
joint exact-match        1.0  (43/43)
invalid-pointer rate     0.0  (0/43)
CRY -> CRY:14
EAT -> EAT:25
LAUGH -> LAUGH:4
```

The accepted root cause is therefore: production scratch-buffer overflow,
followed by insufficient S2 counterfactual coverage once inference was fixed.
The dilution/extra-mixer and decision-head/argGather hypotheses are rejected
for M-A.

## Backward performance follow-up

Two semantics-preserving changes shorten the expensive training step:

- record-local encoder attention now iterates only the compact token range of
  the current record (at most 8), instead of all ~840 tokens hidden behind a
  block-diagonal mask;
- composed training updates only embedding rows referenced by the current
  frame, using a fused sparse backward+SGD pass instead of scanning all 8,469
  rows against every active token.

The local benchmark fell from roughly 26.5 ms/step to 11–14 ms/step depending
on warm-up/driver variance (about 1.9–2.3x faster). A four-columns-per-thread
matmul experiment regressed and was not retained.

## What is done and verified

### Three-part supervision contract (S2+)

Implemented and covered by tests:

- `intentMask` — structural legality only (ActionIntent catalog records).
- `argMaskFor(intentId, argumentIndex)` — selected-intent-conditional arg mask,
  capability-aware (EAT.target → "edible" covers Apple/Berry/Bread; Stone/Feces/
  Mother excluded; LOOK/MOVE_TOWARDS.target → "observable"). Arity-0 intents
  produce an all-blocked row.
- `argumentTargets[q][argument]` — per-argument pointer-loss targets in the
  runner (`KrystalTrainStepOptions`), replacing the old shared `argGold`;
  `INVALID_U32` for arity-0/unlabelled rows. `intentGold` trains the catalog
  selection directly.
- `emitIntentSet` resolves the selected record through the packed runtime-ref
  sidecar (never synthesizes a handle) and drops proposals whose required
  argument cannot resolve (no executable proposal, no fabricated reference).

Coverage: `tests/krystal-policy-lowering.test.ts` (lowering contract, mask
negatives, sidecar-after-shuffle, no-fabrication, transition oracle, curriculum
60/30/10, held-out layout/id disjointness).

### The 0.155 dFieldStates parity bug — root cause and fix

The backward parity test initially failed with `dFieldStates` off by 0.155
(magnitude 0.22). Diagnosis:

- The failure reproduced on the **small canonical fixture frame** too, so it was
  not f32 accumulation on the 840-token frames.
- The GPU `trainStep` was given `intentGold` (intent-slot pointer loss) but the
  CPU `brainBackwardOracle` had **no intent-pointer-loss path** — its intent
  slot always used `noTargets`. The GPU computed extra gradients the oracle did
  not model.
- Removing `intentGold` from the GPU made every region match to ~1e-7 across
  all frame sizes (fixture-small, S1-comfort, S2-policy).

Fix: added `intentTargets` to `BrainBackwardOracleInput` (backward.ts), used for
the intent selector slot exactly like the GPU's `selectorGold`. Updated
`tests/krystal-policy-parity.test.ts` to pass it. Now all 3 parity tests pass
(forward conditional-arg-mask parity, backward pointer-loss parity, arity-0
no-pointer-loss parity). This completes the CPU/GPU parity requirement for the
new intent pointer loss.

## M-A end-to-end training test (`tests/krystal-policy-train.test.ts`)

Design:

- Curriculum: stages S2+S3+S4, replay S1, train seeds 0–47, eval seeds 48–63
  (60/30/10 mixture, logged; deterministic).
- Training: per-frame `trainStep` with teacher-forced gold-intent arg mask,
  CE route-kind gold, `intentGold` (catalog record), `argumentTargets[0]`
  (exact ref bank index / INVALID). lr 0.01, 2 epochs (current settings).
- Eval: production path only — intent head under structural mask, arg head
  re-run under the **selected-intent** mask, `emitIntentSet` emission; metrics
  reported: intent accuracy, pointer accuracy, pointer | correct intent,
  joint exact-match, invalid-pointer rate, per-gold confusion matrix.

Current result (lr 0.01, 2 epochs):

```
M-A eval: 21 held-out frames (seeds 48..63)
  intent accuracy          0.762   (16/21)
  pointer accuracy         1.0     (16/16 ref-bearing frames)
  pointer | correct intent 1.0     (16/16)
  joint exact-match        0.762
  invalid-pointer rate     0.0     (0/21)
  EAT -> EAT:16
  CRY -> (none):5
```

The pointer contract is perfect end-to-end (16/16 exact refs, zero fabricated
handles — the 5 CRY failures were EAT selections correctly dropped for lack of
an edible argument). The single failure mode: the intent selector never emits
CRY on bad-comfort-no-apple frames.

## Debugging the CRY failure (all experiments, results)

Experiment scripts were throwaway (`dbg-ma.ts`, `dbg-s3.ts`, `dbg-q.ts` at repo
root); they have been removed. The test itself is kept.

1. **CRY-only isolation**: train only on CRY frames (S3-f1 + S1 replay, ~19/epoch)
   → 19/19 train, 5/5 eval CRY correct. The signal is learnable when all frames
   are CRY (but note: this is the "bad → CRY" shortcut, no discrimination needed).

2. **Mixed training, CE-by-gold** (lr 0.01): decision-head CE on CRY frames
   improves (1.00 → 0.75 by epoch 2) while the intent selector stays at EAT.
   The two heads diverge: the decision head partially learns CRY, the intent
   selector does not.

3. **Arg-slot fully disabled** (temporary runner edit, reverted): CRY still
   0/19. The shared-intent/arg selector weights are **not** the blocker — the
   "arg head's uniform gradient conflicts with the intent head" hypothesis is
   REJECTED.

4. **Intent gold-P on CRY frames** (after 6 epochs, lr 0.01): 0.42 (uniform
   would be 0.167) — the selector partially learns CRY but EAT still wins the
   argmax.

5. **Longer runs are unstable**:
   - lr 0.01: CRY gold-P ~0.27→0.31 through epoch 6, jumps to 0.55 at epoch 8
     (eval briefly 5/21), then **NaN at epoch 10** (weight explosion).
   - lr 0.003, 16 epochs: gold-P flat ~0.28 through epoch 14, jumps to 0.89 at
     epoch 16 — but the model **flips to the CRY pole**: train EAT 0/34
     (→ CRY:34), eval 5/21.
   - The training is **bistable** between "always EAT on bad frames" and
     "always CRY on bad frames"; the apple-presence discrimination is never
     reliably learned.

6. **Query-state apple-presence** (the key measurement): with the trained
   weights, the EAT-vs-CRY query-output difference is **1.46e-2 (relative
   3.3%) — identical to the random-init value**. Training never amplifies the
   apple-presence signal into the query state; the "EAT 34/34" fit is
   majority-class bias, not discrimination.

## Root-cause hypothesis

The intent selector must detect apple-presence through the mixed query state,
but:

- the mixer query attends over ~105 bank records (≈100 noise + few real), so
  each record's contribution to the query state is heavily diluted;
- nothing forces the mixer to amplify the edible-presence signal: the decision
  head short-circuits the need via the **argGather** (mask-peaked at the apple
  on EAT frames), so the CE gradient does not push apple info into the query
  state, and the intent pointer loss alone is too weak against the class-bias
  gradients;
- the result is a comfort-only bias (EAT when bad) that sways between poles
  instead of learning "no edible ⇒ CRY".

Also noted: **curriculum coverage gap** — in the mixture, the S2 stage is only
ever drawn for `seed % 3 == 0`, and S2's internal variant is also `seed % 3`
(variant 0 = EAT). So S2 variant 1 (bad, no apple → CRY) and variant 2
(good → LAUGH) never appear in training; LAUGH comes only from S1 replay
(~6/epoch). The required S2 behaviors "bad without Apple → CRY" and
"good → LAUGH even if Apple is visible" have no direct training coverage.

## Next steps (untested)

- **Balanced S3-only experiment** (CRY/EAT 50/50, frames differ only by apple
  presence): decisive for whether rebalancing the curriculum fixes the
  discrimination or whether it is structural. Was interrupted before results.
- Fix the S2-variant coverage gap in the curriculum sampler so all three S2
  variants are trained.
- If balanced training still cannot discriminate: the intent decision needs a
  stronger bank→query path (mixer capacity is maxed at `KRYSTAL_MAX_BLOCKS=2`
  in the current arena) or a different supervision mechanism for the intent
  selector (e.g., training it against the decision head's choice).

## Current repo state

- Uncommitted M-A work: three-part contract (masks/intentset/backward/
  fixtures/capabilities), parity tests, `tests/krystal-policy-train.test.ts`.
- Full suite at last check: **107 pass / 0 fail**, root typecheck clean
  (only the 8 pre-existing `validate-krystal-shaders` Deno errors).
- The M-A training test currently asserts 100% joint and FAILS at 0.762 —
  it should stay red until the discrimination is actually learned, or be
  re-scoped with honest reported metrics.
