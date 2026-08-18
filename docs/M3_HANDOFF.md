# M3 Hand-off Report

Status: **M3 (Krystal backward operators) fully implemented and closed — composed trainStep runner + overfit proof green.**  
Date: 2026-08-16  (refresh: composed runner + overfit landed as `9962144`)
Branch: `master`, tree clean, nothing pushed.

## 1. Test results (fresh full run)

```
 81 pass
 15 fail
 14 errors
Ran 96 tests across 31 files.
```

### What passes (81)

- **Composed trainStep** (`tests/krystal-train.test.ts`, 2 tests, NEW): GPU gradients of one full forward→CE→backward→SGD trainStep match the composed CPU oracle **exactly** (all 11 gradient regions ≤ 2e-6 maxAbsDiff at the real learning rate), and the §19 **overfit proof** — 40 trainSteps lower the route-kind CE loss on the canonical fixture from ~1.39 to < 0.6, monotonically.
- **M3 Krystal backward** (`tests/krystal-backward.test.ts`, 11 tests): relu, attention (cross, masked), pool, selector (soft gather + pointer loss), decision head — each with per-op GPU-vs-CPU parity **and** a finite-difference gradient check against a forward-only loss.
- **M2b forward** (`tests/krystal-forward.test.ts`, 9 tests): per-op (relu, attention, pool, selector, field-embed, **decision head**) + **composed CPU/GPU parity on the canonical fixture frame** (fieldStates, bank keys/values, query output, P, gather, argmax indices, decision logits, IntentSet emission).
- **M2b IntentSet** (`tests/krystal-intentset.test.ts`, 9 tests): EAT→Apple exact-handle resolution, LOOK+Apple masking, WAIT no-args, masked→count 0, sidecar round-trip, entropy/candidates, emptyProposal envelope.
- **M2a packer** (`tests/krystal-packer.test.ts`, 13 tests): SoA BinaryLayoutPlan geometry, runtime-ref sidecars, validation rejections.
- **M1 training** (training-ops, training-attention, training-encoder, training-gradcheck, training-overfit, training-attention-gradcheck, attention tests): the tiny GPU-resident vertical slice and its gradient checks.
- Plus the surviving core suite (bandwidth, matmul variants, structured-generation, sampling, checkpoint, etc.).

### What fails (1 real failure)

| Test | Why |
|---|---|
| `guide-empty-token` (`tests/guide-empty-token.test.ts`) | Reads `packages/finetune/tokenizer.json`, which was **deleted** in the M4-preview cleanup (`64e7391`). Pre-existing; unrelated to M3. |

### The other 14 fails / 13 errors — suite **load** errors, not test failures

All are `Cannot find module`/`ENOENT` for packages removed in `64e7391` (Chomato/LFM2 cleanup):

- `src/index.ts` → `packages/backend/src/local-model.ts`
- `checkpoint.test.ts` → `packages/engine-ts/src/gpu-constraint.ts`
- `structured-benchmark.test.ts` → `packages/engine-ts/src/index.ts`
- `reserved-tokens.test.ts` → `packages/lfm2/src/tokenizer.ts`
- `smoke_test.ts` → `packages/quant/src/gguf/source-web.ts`
- `vision.test.ts` / `vision-gpu.test.ts` → `packages/quant/src/gguf/reader.ts`
- `vision-vl.test.ts` → `packages/quant/src/gguf/source-node.ts`
- `packages/webgpu/test/checkpoint-transport.test.ts` → `@chomato/engine-ts` (missing node_modules link)
- `packages/finetune/tokenizer.json` (guide-empty-token, listed above)

These are exactly the M4 cleanup targets (per `docs/concerns_response.md` answers 33–35: keep old code until the replacement is green, delete in M4). **Zero M3/M2 regressions.**

### Typecheck (`tsgo --noEmit`)

- All `packages/krystal/src/**`, `packages/webgpu/src/krystal-forward.ts`, `krystal-backward.ts`, `lfm2-layout/definition/artifact.ts`, `tests/krystal-*`, `tests/training-*`: **clean**.
- Remaining errors are pre-existing imports of the deleted legacy packages (same list as above).
- Also fixed a stale pre-existing `TS2739` (`Uint32Array` vs `readonly number[]`) in `tests/training-ops.test.ts` (commit `5166f71`).

## 2. CPU/GPU parity status

All parity checks run on the real Dawn/WebGPU backend through the shared training harness (Sandblaster artifact, one submit per forward, per-op standalone dispatches for backward), compared against plain-TS CPU oracles in `packages/krystal/src/forward/oracle.ts` + `backward.ts`.

| Layer | Parity evidence | Tolerance |
|---|---|---|
| Encoder forward (field embed, attention, relu FFN, pool) | per-op GPU-vs-CPU + composed parity | 1e-2 activations, 1e-4 distributions |
| Mixer forward (cross-attention + FFN) | composed parity | 1e-2 |
| Catalog selection (P, gather, argmax) | per-op + composed parity | 1e-4 P/gather, exact indices |
| Decision head forward (route-kind logits) | per-op + composed parity | 1e-2 |
| IntentSet emission | GPU heads → same host resolver as CPU heads; compared on count, lifecycle, intentId, confidence, argument kinds/statuses/handles | exact/1e-4 |
| **Composed trainStep gradients** (fieldStates, queryValues, bank keys/values, pool, selector Wq/Wk, decision Wh, intent/arg gather) | one-submit GPU trainStep vs `brainBackwardOracle` | **exact** (≤ 2e-6, bit-level at f32 precision) |
| Backward: relu / attention / pool / selector / field-embed / decision head | per-op GPU-vs-CPU | 1e-4 (1e-6 relu) |
| Gradient checks | central differences, all tensors | 1e-3 + 5e-2·\|g\| bound |
| Overfit (§19) | 40 trainSteps on the canonical fixture, route-kind CE decreases | final < 0.6, monotone |

## 3. The composed runner (M3 close, §17 item 10)

`packages/webgpu/src/krystal-backward.ts` — one GPU submit per trainStep:

```
forward (with per-block saved activations: In/FfnIn/Q/K/V/P/H1)
→ cross_entropy over route kinds (+ loss_reduce telemetry)
→ decision head backward (dCtx parts + dWh) + SGD on Wh
→ both selector slots (soft gather + pointer loss) → dQueryValues/dBankKeys/dBankValues
→ mixer blocks reverse (attention + FFN), per-block weight grads
→ pool backward ×2 (bank + query) + dPool reduce + SGD on pool queries
→ encoder blocks reverse
→ field embed backward + SGD on embeddings
```

Design notes:

- **Gradient exactness vs SGD ordering.** The first implementation interleaved `sgd_step` between the weight-gradient and input-gradient passes of each block; `matmul_backward_input` therefore read weight pages that SGD had already moved, corrupting all upstream gradients (0.23 maxAbsDiff on `dFieldStates` at lr=0.1). The fix: each block's full backward runs on the pre-update weights, then SGD lands at the end of the block (dW regions still consumed before the next block reuses them). After the fix all 11 gradient regions match the CPU oracle **exactly** at the real learning rate.
- **No atomics anywhere**: every gradient element has exactly one owner (row-owned score passes, gid-linear qkv splits, per-record dPool partials reduced by `krystal_pool_dpool`).
- **dQueryValues aliases the `dDecisionQuery` region**: seeded by the decision head, then residual-added by the selector routing and mixer loop; the standalone `dQueryValues` region stays reserved for parity reads.
- Per-block dW regions are shared across blocks (block-local weights), so interleaved SGD is exact for one step.
- The only readback in the normal path is the scalar loss telemetry (`telemetry: true`); everything else stays resident.

## 4. Known shortcuts / TODOs (deliberate, per plan §17–19)

1. ~~**Decision head forward shader does not exist yet**~~ — **DONE**: `krystal_decision_head.wgsl` + weight page + `decisionHeadOracle`, wired into the composed forward parity.
2. ~~**No composed `KrystalBackward` runner yet**~~ — **DONE**: `krystal-backward.ts` with one-submit trainStep; overfit test green.
3. **Selector projections are shared** across intent/argument slots (`selector.wq/wk` one pair, per-slot projections = "later ablation" — noted in `model.ts`/`oracle.ts`).
4. ~~**No optimizer wiring on Krystal weights yet**~~ — **DONE for plain SGD**: `sgd_step` applied to embeddings, encoder/mixer blocks, pool queries, selector Wq/Wk and decision Wh inside trainStep. No momentum/Adam; no weight decay; no LR schedule.
5. **Route-kind class count is capacity, not ABI** — `KRYSTAL_MAX_ROUTE_KINDS = 8`; the fixture has no frozen route-kind enum yet (`routeKind` per architecture v2 §9 remains open).
6. **Masked-pointer loss uses `0xffffffff` sentinel** for "no target" rows — documented in shader/oracle.
7. **Arena regions are fixed-capacity** (1536 active tokens, 288 records, 8 queries, 8 route kinds); shaders read actual dims from OpParams, so growth is a layout change only. Token capacity is a PROCESSING budget, not the frame's slot geometry: encoder attention preallocates `[heads, T, T]`, so sizing it from `frameTokens` blows past the default 256 MB `maxBufferSize` and corrupts the arena silently.
8. **Frozen/fixture vocabularies are provisional** (`docs/KRYSTAL_ABI_V0.md`, `fixtures/vocabulary.ts`, `record-schemas.ts`, `action-intents.ts`) — test manifests, not the production ABI.
9. **Training harness requests the adapter's max storage-binding size** (arena outgrew the default 128 MiB in M3a) — fine on this adapter, worth re-checking on others.
10. **`docs/KRYSTAL_BRAIN_ARCHITECTURE_V2.md` §12 pipeline is only partially wired**: steps 1–8 (packed frame → encoder → mixer → selectors → soft gather) and step 9's decision head + backward are green; step 9's **TypedPlan writeback** (materializing the typed decision result from the logits) and step 10's runtime validation/execution (the IntentSet host resolver) remain.
11. **Training is single-fixture overfit only** — no generalization signal, batching, or data pipeline yet; the overfit test is the plan's correctness floor, not a training system.

## 5. Git history (M3 scope)

```
9962144 Close M3: composed KrystalBackward runner + training on synthetic fixture
5166f71 Fix Uint32Array vs readonly number[] in training-ops oracle calls
d177598 Add typed decision head backward (dCtx parts + dWh, CE route-kind loss)
628cb37 Add krystal selector backward (soft gather gradients + pointer loss)
f43ceac M3b: learned-query pooling backward (record mixer training path)
3714f34 M3a: Krystal backward operators — relu, cross attention, field embed
18e66d9 M2b: IntentSet output
61c2764 M2b: catalog selection + soft gather heads
6d2790a M2b: record/query encoder and mixer forward
f6fc9c7 M2a: SoA BinaryLayoutPlan, fixture catalogs and frame packer
```

## 6. Next step when you're back

**M3 is closed.** The remaining work per the plan:

- **M4 cleanup** (the M4-preview deletion targets): remove the replaced Chomato/LFM2 code (backend, engine-ts, quant, lfm2, finetune legacy paths, gui), package rename, README, checkpoint guards, and re-enable/port the load-erroring tests (`guide-empty-token` needs a tokenizer or a fixture replacement).
- After that: TypedPlan writeback + runtime validation (architecture v2 §12 steps 9–10), then the training-data pipeline and multi-fixture generalization checks (§19 beyond overfit).
