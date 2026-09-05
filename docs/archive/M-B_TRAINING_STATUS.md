# M-B (S5–S6) milestone — completed

Date: 2026-08-17. Continuation of the grounded-policy curriculum from
`S2_S10_CURRICULUM_TASK.md` after the completed M-A proof.

## Result

The composed WebGPU model was trained for three epochs on the deterministic
60/30/10 curriculum:

- current stages: S5 + S6;
- replay: S1 + S2 + S3 + S4;
- adversarial/negative episodes: the shared seven-kind pool;
- train seeds: `[0, 256)`;
- held-out eval seeds: `[256, 320)` with disjoint runtime-ref bands.

Held-out evaluation contains 128 frames:

```text
intent accuracy            1
pointer accuracy           1 (112/112)
pointer | correct intent   1 (112/112)
joint exact-match          1
invalid-pointer rate       0 (0/128)
S5 joint                   1 (64/64)
S6 joint                   1 (64/64)
```

Confusion matrix:

```text
MOVE_TOWARDS -> MOVE_TOWARDS:32
EAT          -> EAT:48
LOOK         -> LOOK:32
CRY          -> CRY:16
```

## Bugs exposed by the milestone

1. The shared policy test harness mapped `MOVE_TOWARDS` to token `0x602`,
   which is `WAIT`; the catalog token is `0x607`. Action ids now come from
   `fixtureTokenId(...)` rather than duplicated numeric literals.
2. S6 used seed parity both indirectly for stage selection and directly for
   its reveal outcome. Every sampled S6 train/eval episode therefore took the
   same negative branch. The outcome now uses an independently salted hash.
3. The negative post-LOOK frame was observationally identical to its initial
   unknown frame but had a different gold action (`LOOK` then `CRY`). The zoom
   now reveals the same exact runtime ref as an inedible Stone, providing the
   new evidence required by the stage contract.

With two epochs the corrected task reached 92.97% joint accuracy: all S6
frames passed, while 9/32 far S5 frames still emitted EAT. A third pass learned
the FAR/NEAR intent boundary and reached the complete milestone contract.
