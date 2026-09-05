# S2–S10 grounded-policy curriculum

## Objective

Extend the completed S1 Comfort proof into a grounded, reference-selecting
policy.  The proof is not merely that the model emits `EAT`; it must select
both the right intent and the exact `ResourceRef` that makes the intent valid.

S1 remains a regression test:

```text
bad  -> CRY()
good -> LAUGH()
```

S1 is the one place where `argMask === intentMask` is harmless: both actions
have arity zero.  That alias must not survive into S2 or later.

## Ownership and input boundary

Pira owns a complete, unbounded raw sensory snapshot: concrete resource
identity, exact schema-valid values, modality availability and world facts.
Krystal compiles it into a bounded `BrainFrame`: vocabulary encoding, numeric
projection, ranking/shuffle/truncation and runtime-reference sidecars.

Do not use Village's legacy `buildSpatialSenseRecords`/`maxRecords` output as
the data contract.  It already aggregates and tokenizes records on the Pira
side.  The bridge contract is documented in
`/home/kr/Projects/pirapitinga/docs/krystal-sensory-bridge.md`.

For this task, build deterministic synthetic raw-snapshot fixtures in Krystal
first.  Pira does not yet export S2–S10 scenario streams.  Keep the fixture
adapter narrow so it can later consume those streams unchanged.

## Required selection contract

From S2 onward, model supervision and runtime resolution must carry three
distinct things:

| Name | Meaning | Required behaviour |
| --- | --- | --- |
| `intentMask` | legal ActionIntent-catalog records | Structural legality only; it must not encode the desired action. |
| `argMask` | legal bank records for one selected intent argument | Conditioned on `(selectedIntent, argumentIndex)` and the catalog argument descriptor.  It must exclude `Mother`, distractors and incompatible records. |
| `argumentTarget` | gold selected bank record / exact runtime-ref sidecar | Exact pointer label for a required reference argument; `INVALID_U32` only for arity-0 or explicitly unlabelled rows. |

The temporary implementation may still have one pointer head while every
training action has at most one reference argument.  Its mask and loss target
must nevertheless be selected-intent conditional.  Do not retain one shared
`argGold` as a global query-row label.  Design the public shape for future
arity as either `argMask[q][intent][argument][record]` or an equivalent
`argMaskFor(intentId, argumentIndex)` lowering, plus
`argumentTarget[q][argument]`.

`emitIntentSet` must resolve the selected record through the packed
runtime-reference sidecar, never synthesize an `Apple#12` handle from a token.
For zero-arity actions the argument selector has no target and contributes no
pointer loss.

Add explicit tests that an all-masked or incompatible argument row produces no
fabricated reference and no executable proposal.

## Curriculum and acceptance cases

Every stage uses held-out seeds, resource ids and layouts.  "Apple" below
means the exact resource reference in that episode, never a fixed vocabulary
position.

| Stage | Active evidence | Required decision / test |
| --- | --- | --- |
| S1 | Comfort | bad -> `CRY()`, good -> `LAUGH()`; preserve existing proof. |
| S2 | Comfort + Vision | bad + visible Apple -> `EAT(AppleRef)`; bad without Apple -> `CRY()`; good -> `LAUGH()` even if Apple is visible. |
| S3 | Multi-frame Mother | bad/no Apple -> `CRY()`; Mother delivers Apple on a later frame -> `EAT(the delivered ref)`; improved Comfort -> `LAUGH()`. |
| S4 | Reference choice | Apple + Mother + arbitrary distractors -> `EAT(AppleRef)` and never a non-Apple reference.  Permute record order and ids. |
| S5 | Spatial availability | reachable Apple -> `EAT(ref)`; far Apple -> `MOVE_TOWARDS(ref)`; after transition to reachable -> `EAT(same ref)`. |
| S6 | Active perception | an incomplete/unknown record -> `LOOK(ref)`; a deterministic zoom reveals properties, then choose `EAT(ref)` or reject it. |
| S7 | Capability | Apple, Berry and Bread are candidates through `TARGET_OF(EAT)`, not an `Apple` schema check.  Stone and Feces are negative candidates. |
| S8 | Consequence | ordinary food improves Comfort.  PoisonedApple and Feces worsen it.  Evaluation changes names/variants so consequence, not a string label, determines the policy. |
| S9 | Working memory | an observed target can leave Vision while its exact ref remains in Memory/Focus; continue the pending `MOVE_TOWARDS(ref)` or `EAT(ref)`. |
| S10 | Full policy | randomized layouts, distance, distractors, deterministic inactive-band noise and generated resource variants; no tutorial-specific prompt or fixed record order. |

The following records are mandatory direct lowering and end-to-end assertions:

```text
Satiation=-1, Vision=[]                  => CRY()
Satiation=-1, Vision=[Apple#12]          => EAT(Apple#12)
Satiation=+1, Vision=[Apple#12]          => LAUGH()

Satiation=-1, Vision=[Mother#3,Apple#12] => EAT(Apple#12)
Satiation=-1, Vision=[Apple#12 far]      => MOVE_TOWARDS(Apple#12)
Satiation=-1, Vision=[Unknown#12]        => LOOK(Unknown#12)
```

## Milestones and mixture

Do not gate on a perfect score after each individual stage.  Train and report
three milestones:

1. **M-A: S2–S4** — intent selection plus exact visible-resource pointer.
2. **M-B: S5–S6** — temporal transition, distance and deliberate `LOOK`.
3. **M-C: S7–S10** — affordance, learned consequences, memory and randomized
   full-policy generalization.

Each training run mixes episodes deterministically:

```text
60% current milestone stage(s)
30% replay from completed stages
10% adversarial / negative episodes
```

The adversarial tenth must include at least: no target, duplicate-looking
resources, reordered records, a salient Mother/distractor, inaccessible target,
incompatible target and poisoned/negative consumables.  Log the mixture and
root seeds so failures replay exactly.

## Implementation work

1. Factor the S1-only comfort bridge into a stage-capable fixture/episode
   compiler.  Keep `comfort-episodes@1` readable and its test unchanged.
2. Extend the fixture catalog with `EAT`, `MOVE_TOWARDS` and `LOOK` reference
   arguments.  Make arity derive from the action descriptor; do not hardcode
   an Apple-only `EAT` contract.
3. Change forward/backward selector supervision, CPU oracle, WebGPU path and
   host `emitIntentSet` to implement the three-part contract above.  Preserve
   CPU/GPU parity for every new mask/loss path.
4. Add capability predicates/traits to catalog lowering for S7 (for example
   `TARGET_OF(EAT)`), separate from resource class/schema identity.
5. Add a small deterministic transition simulator for fixture episodes:
   Mother delivery, movement reachability, `LOOK` revelation, consumption
   consequence and memory/focus retention.  It is a test oracle, not a second
   Pira world implementation.
6. Add a curriculum runner that samples the stated 60/30/10 mixture and emits
   train/eval splits with disjoint seeds/layouts/resource ids.

Likely touch points are `packages/krystal/src/bridge/comfort.ts`,
`fixtures/action-intents.ts`, `fixtures/record-schemas.ts`,
`forward/masks.ts`, `forward/oracle.ts`, `forward/backward.ts`,
`forward/intentset.ts`, and the matching WebGPU forward/backward selectors.
Prefer a new policy-curriculum fixture/test module over mutating the canonical
S1 fixture into an implicit multi-stage test.

## Test requirements

Add focused tests before the training proof:

- lowering: `intentMask`, intent-conditioned `argMask`, and
  `argumentTarget` are independent; S1 zero-arity is the only no-target case;
- mask negatives: `EAT` cannot point to Mother/Stone/Feces, while `LOOK` can
  point to an unknown observable record;
- sidecars: `EAT(Apple#12)` and `MOVE_TOWARDS(Apple#12)` resolve the exact
  runtime handle after shuffled packing;
- transition oracle: S3, S5, S6, S8 and S9 advance only when the previous
  selected intent/ref is correct;
- CPU/GPU parity: new conditional argument-mask, pointer-loss and no-argument
  rows match numerically;
- end-to-end generalization: each milestone is evaluated on unseen seeds,
  layouts, record order and resource ids, reporting intent accuracy,
  pointer accuracy conditional on intent, and joint exact-match accuracy.

The success metric is the joint result.  A prediction with `EAT` and the wrong
resource is wrong, as is the right resource with the wrong intent.  Report
intent accuracy, pointer accuracy (given correct pointer-bearing intent),
joint exact-match, invalid-pointer rate and per-stage confusion matrix.

## Non-goals for this task

- Do not reintroduce Pira-side spatial banding, record budgets or tokenization.
- Do not make fixed resource names (`Apple`) the capability mechanism.
- Do not treat `CRY`/`LAUGH` as effect-free; their Pira transition eventually
  creates observable Sound records, although sound-based policy is outside
  this curriculum.
- Do not add arbitrary-handle generation, reinforcement learning or
  multi-intent planning.  The scope is one grounded proposal per frame with a
  deterministic supervised transition oracle.
