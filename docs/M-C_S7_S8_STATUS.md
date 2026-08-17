# M-C slice S7–S8 — completed

Date: 2026-08-17. Isolated capability/consequence slice before adding S9
working memory and the randomized S10 policy.

## Curriculum

- current stages: S7 + S8;
- replay: S1–S6;
- deterministic mixture: 60% current, 30% replay, 10% adversarial;
- train seeds: `[0, 256)`;
- held-out eval seeds: `[320, 384)` with disjoint runtime refs/layouts;
- three epochs, SGD learning rate `0.01`.

The held-out set covers Apple, Berry and Bread for:

- S7 capability-grounded `EAT(exactRef)` with Stone/Feces distractors;
- S8 pre-consequence `EAT(exactRef)`;
- S8 safe consequence -> `LAUGH()`;
- S8 known `POISONED` consequence -> `CRY()`.

## Result

```text
intent accuracy            1
pointer accuracy           1 (64/64)
pointer | correct intent   1 (64/64)
joint exact-match          1
invalid-pointer rate       0 (0/96)

EAT   -> EAT:64
CRY   -> CRY:16
LAUGH -> LAUGH:16
```

Every individual `stage × action × food × consequence` eval bucket passed.

## Curriculum defects fixed

1. S8 used `seed % 2` for its consequence while S7/S8 stage selection also
   used seed parity. This collapsed sampled S8 episodes to one branch. The
   consequence now comes from an independently salted hash.
2. A generated S8 frame labelled a visibly `POISONED` Apple as `EAT`, while
   the adversarial generator labelled the same evidence as `CRY`. The
   contradictory labels made the task ill-defined.
3. S8 now models the sequence explicitly: an initially ordinary-looking
   edible is explored with `EAT(ref)`; the next frame preserves the exact ref
   and exposes either a safe positive consequence or the `POISONED` negative
   consequence. Known poison is never supervised as EAT.
4. Food identity is varied across Apple/Berry/Bread in both outcomes, so the
   consequence boundary cannot be reduced to one resource schema name.

This completes the S7–S8 slice, not all of M-C. S9 memory retention and S10
full-policy randomization remain separate work.
