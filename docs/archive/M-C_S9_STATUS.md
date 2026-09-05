# M-C slice S9 — completed

Date: 2026-08-17. Isolated working-memory slice after the completed S7–S8
capability/consequence proof and before S10 full-policy randomization.

## Curriculum and contract

- current stage: S9;
- replay: S1–S8;
- deterministic mixture: 60% current, 30% replay, 10% adversarial;
- train seeds: `[0, 256)`;
- held-out S9 eval seeds: `[384, 448)`;
- three epochs, SGD learning rate `0.01`.

Each S9 episode has two frames. The target begins in Vision and then leaves
Vision completely while the same exact runtime ref is retained in Memory.
FAR targets must remain `MOVE_TOWARDS(ref)` and NEAR targets must remain
`EAT(ref)` across the transition.

## S9 result

```text
intent accuracy            1
pointer accuracy           1 (128/128)
pointer | correct intent   1 (128/128)
joint exact-match          1
invalid-pointer rate       0 (0/128)

Vision + EAT               32/32
Memory + EAT               32/32
Vision + MOVE_TOWARDS      32/32
Memory + MOVE_TOWARDS      32/32
```

The lowering regression also verifies that the Vision record is absent in the
second frame, the Memory record carries `RECORD_FLAGS.remembered`, its packed
runtime sidecar resolves the original ref, and the selected action's argument
mask admits that exact memory-bank record.

## Replay audit

The same S9-trained model was evaluated on 96 additional held-out S1–S8
frames, rather than relying only on separately initialized milestone tests:

```text
overall  93/96
S1        8/8
S2        8/8
S3       16/16
S4        8/8
S5       16/16
S6       16/16
S7        7/8
S8       14/16
```

This audit exposed another stage/variant coupling: S9 replay selected S1 only
on even raw seeds, while S1 also used parity for CRY/LAUGH, removing LAUGH
from replay. The S9 training slice now derives its S1 replay counterfactual
from an independently salted hash while preserving the already-closed M-A/M-B generator behaviour;
retention improved from 89/96 to 93/96. A global salted-S1 experiment was
rejected because it made the established M-B trajectory diverge to NaN.

The remaining three replay misses are retained as visible evidence for the
combined M-C run: one S7 edible is rejected and two S8 poisoned outcomes are
treated as edible. The S9 slice is gated at 100% on its own memory contract;
the task specification explicitly says not to gate each individual stage on a
perfect global score. The eventual S7–S10 M-C proof must close these replay
misses rather than inheriting the slice threshold.
