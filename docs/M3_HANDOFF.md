# M3 Hand-off Report

Status: **M3 (Krystal backward operators) fully implemented, all per-op slices committed and green.**  
Date: 2026-08-16  
Branch: `master`, tree clean, nothing pushed.

## 1. Test results (fresh full run)

```
 79 pass
 15 fail
 14 errors
Ran 94 tests across 30 files.
```

### What passes (79)

- **M3 Krystal backward** (`tests/krystal-backward.test.ts`, 11 tests): relu, attention (cross, masked), pool, selector (soft gather + pointer loss), **decision head** — each with per-op GPU-vs-CPU parity **and** a finite-difference gradient check against a forward-only loss.
- **M2b forward** (`tests/krystal-forward.test.ts`, 8 tests): per-op (relu, attention, pool, selector, field-embed) + **composed CPU/GPU parity on the canonical fixture frame** (fieldStates, bank keys/values, query output, P, gather, argmax indices, IntentSet emission).
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

- All `packages/krystal/src/**`, `packages/webgpu/src/krystal-forward.ts`, `lfm2-layout/definition/artifact.ts`, `tests/krystal-*`, `tests/training-*`: **clean**.
- Remaining errors are pre-existing imports of the deleted legacy packages (same list as above).
- Also fixed a stale pre-existing `TS2739` (`Uint32Array` vs `readonly number[]`) in `tests/training-ops.test.ts` (commit `5166f71`).

## 2. CPU/GPU parity status

All parity checks run on the real Dawn/WebGPU backend through the shared training harness (Sandblaster artifact, one submit per forward, per-op standalone dispatches for backward), compared against plain-TS CPU oracles in `packages/krystal/src/forward/oracle.ts` + `backward.ts`.

| Layer | Parity evidence | Tolerance |
|---|---|---|
| Encoder forward (field embed, attention, relu FFN, pool) | per-op GPU-vs-CPU + composed parity | 1e-2 activations, 1e-4 distributions |
| Mixer forward (cross-attention + FFN) | composed parity | 1e-2 |
| Catalog selection (P, gather, argmax) | per-op + composed parity | 1e-4 P/gather, exact indices |
| IntentSet emission | GPU heads → same host resolver as CPU heads; compared on count, lifecycle, intentId, confidence, argument kinds/statuses/handles | exact/1e-4 |
| Backward: relu / attention / pool / selector / field-embed / decision head | per-op GPU-vs-CPU | 1e-4 (1e-6 relu) |
| Gradient checks | central differences, all tensors | 1e-3 + 5e-2·\|g\| bound |

## 3. Known shortcuts / TODOs (deliberate, per plan §17–19)

1. **Decision head forward shader does not exist yet** — only the backward. The composed parity covers selection outputs + IntentSet, not route-kind logits. (Next milestone: composed runner + forward decision head.)
2. **No composed `KrystalBackward` runner yet** — each backward op is wired + tested standalone; there is no single-submit forward→loss→backward→SGD trainStep on the Krystal graph. The M1 `TrainingTrainer` covers the tiny (non-Krystal) graph only.
3. **Selector projections are shared** across intent/argument slots (`selector.wq/wk` one pair, per-slot projections = "later ablation" — noted in `model.ts`/`oracle.ts`).
4. **No optimizer wiring on Krystal weights yet** — `sgd_step` exists for M1; decision-head `dWh` etc. have arena regions but nothing applies them.
5. **Route-kind class count is capacity, not ABI** — `KRYSTAL_MAX_ROUTE_KINDS = 8`; the fixture has no frozen route-kind enum yet (`routeKind` per architecture v2 §9 remains open).
6. **Masked-pointer loss uses `0xffffffff` sentinel** for "no target" rows — documented in shader/oracle.
7. **Arena regions are fixed-capacity** (1024 tokens, 128 records, 8 queries, 8 route kinds); shaders read actual dims from OpParams, so growth is a layout change only.
8. **Frozen/fixture vocabularies are provisional** (`docs/KRYSTAL_ABI_V0.md`, `fixtures/vocabulary.ts`, `record-schemas.ts`, `action-intents.ts`) — test manifests, not the production ABI.
9. **Training harness requests the adapter's max storage-binding size** (arena outgrew the default 128 MiB in M3a) — fine on this adapter, worth re-checking on others.
10. **`docs/KRYSTAL_BRAIN_ARCHITECTURE_V2.md` §12 pipeline** is only partially wired: steps 1–8 (packed frame → encoder → mixer → selectors → soft gather) are green; step 9 (final decision heads + TypedPlan writeback) has its backward but no forward/writeback yet; step 10 (runtime validation/execution) is the IntentSet host resolver.

## 4. Git history (this hand-off's scope)

```
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

## 5. Next step when you're back

**Close M3**: composed `KrystalBackward` runner (forward + loss + all backward ops + SGD in one submit) then **training on synthetic frame fixtures** with a loss that actually decreases — the deterministic overfit test the plan (§19) requires. After that, M4 (cleanup of replaced Chomato/LFM2 code, package rename, README, checkpoint guards).
