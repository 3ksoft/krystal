# Concerns & questions before starting

Status: I read `docs/task.md`, `docs/KRYSTAL_BRAIN_ARCHITECTURE_V2.md`, `docs/WEBGPU_BACKWARD_PLAN.md`, `docs/TRAINING_DESIGN.md`, `packages/schema/src/krystal-engine-schema.ts`, and explored the whole repo plus the sibling `schema-pop` and `sandblaster-v2` checkouts. Below is everything I need answered before (or while) implementing. Please answer by number; short answers are fine.

---

## A. Infrastructure — Step 1 status and fixes

**Verified working right now (from the repo root, inside the flake shell):**

- `bun test tests/smoke_test.ts` — passes.
- `bun test tests/matmul-variants.test.ts` — passes on the real NVIDIA adapter through the `webgpu` (Dawn) bindings, including real GPU readback via Sandblaster's `readback()`.
- `bun run build:schema` — works (regenerates webgpu types/codec/abi.cpp/schema.wgsl).
- `bun run build:webgpu` (Sandblaster AOT artifact build) — works.
- `packages/schema/src/krystal-engine-schema.ts` analyzes cleanly through schema-pop (`fromModule` → 78 types, no analyzer errors). It is currently **not** wired into `build.ts` or `index.ts` — nothing imports it yet.

So the CLI WebGPU pipeline effectively already works with Bun + Dawn. Remaining Step-1 questions:

1. **Root `package.json` `"test"` script says `deno test tests`, but every test is `bun:test` and Deno cannot run them** (verified: deno fails on the `bun:test` import). README says `bun test`. Should I change the root test script to `bun test tests` and drop/ignore `deno.json`? (The task says Deno is a dead end; I agree from what I've seen.)

2. `packages/webgpu/package.json` has `build:shaders` / `build:shaders:full` scripts that call `deno run scripts/validate-lfm2-shaders.ts`. Should these be converted to Bun, or deleted with the LFM2 code (they validate the LFM2 artifact)?

3. **flake.nix** exists and the shell works (`nodejs`, `deno`, `bun`, CUDA/vulkan, etc.). Is the current flake sufficient, or do you expect additions? I will not start real work outside it.

4. Minor observation, not a blocker: a hand-rolled raw `mapAsync` probe returned `AbortError`, but Sandblaster's `readback()` (same underlying path) works — the passing `matmul-variants` test does real GPU readbacks. So I'll use the Sandblaster readback path and not worry about raw mapping. Flagging only so you're aware; I don't think it's a schema-pop/sandblaster bug.

---

## B. Scope and milestones — what exactly is "done" in the first pass

The three documents describe three different-sized targets and I need to know the intended order.

5. **Ordering.** My reading:
   1. Step 1: infra (mostly done, see A).
   2. Milestone 1 of `WEBGPU_BACKWARD_PLAN.md`: tiny graph `embedding → linear → CE → backward → SGD` on a toy dataset (M=4, V=8, H=6), with CPU oracle + finite differences + overfit test. This is a *generic* training vertical slice — it is **not** the Krystal brain architecture (no records/mixer/soft gather/typed heads).
   3. Real Krystal forward per `KRYSTAL_BRAIN_ARCHITECTURE_V2.md`.
   4. Krystal backward operators in the order from `WEBGPU_BACKWARD_PLAN.md` §17.
   
   Is that the intended sequence, and is milestone 1 (the tiny slice) the target for this first working session? Or do you want the tiny slice skipped and the real Krystal forward+backward attacked directly?

6. **"This will surely take more than one session"** — do you want me to propose a concrete milestone breakdown (M0 infra → M1 tiny training slice → M2 Krystal forward → M3 Krystal backward…) and then work through it session by session, checking in between? Or should I just go until blocked?

7. **Bug-report protocol**: the task says to report schema-pop/sandblaster bugs *before doing anything else*. I have not found a confirmed bug yet (see A4). When I do, is "report" = tell you in the conversation and stop, or open an issue in the sibling repos, or write it into `docs/concerns.md`/a file? I won't fix them myself without being asked.

---

## C. Model profile — numbers I need confirmed

The architecture (baseline) and the schema disagree on some numbers, and several are left open. Please settle these for the *first* implementation.

8. **Vocabulary size.** `krystal-engine-schema.ts` `KRYSTAL_ABI` says `tokenBits: 12`, `vocabSize: 0x1000` (4096), and `KRYSTAL_TOKEN_RANGES` cover 13 classes over 0x000–0xFFF. But `KRYSTAL_BRAIN_ARCHITECTURE_V2.md` baseline says **Physical Vocab 256**, ~183–200 active semantic concepts, and the schema's own `BrainModelConfig` default is `vocabSize = KRYSTAL_ABI.vocabSize`. **Which vocab size does the model actually use for the first implementation: 4096 (schema/12-bit ABI) or 256 (architecture baseline)?** This changes embedding tables, logits, masks, and everything else.

9. **Hidden size / FFN / block count.** Architecture baseline: H=128, FFN=384, and "matches the depth of the legacy 4-block model". Semcore's `FWD_ARCHITECTURE_PLAN.md` records the baseline as **128 / 4 / 384**. Confirm: first profile = hidden 128, 4 blocks total, FFN 384?

10. **Block split.** The 4 blocks are "partitioned functionally between local encoding and global selection" (open decision). What split do you want for the first pass — e.g. 2 record-encoder blocks + 1 query-encoder block + 1 mixer? Or all record-encoder blocks + a separate cross-attention mixer on top? The mixer is described as a cross-attention layer, so I need to know whether the "4 blocks" includes the mixer or not.

11. **Attention heads.** Not specified anywhere. For H=128, propose e.g. 4 heads × 32 dims (GQA with shared KV heads like LFM2)? Or do you have a number in mind? The schema's `BrainModelConfig` has `attentionHeads` and a `recordSize` field — what should `recordSize` be (the key/value dim of a record state)? Equal to H?

12. **Query encoder weight sharing** with the record encoder is listed as open. First implementation: shared or separate? (Shared is cheaper and simpler; separate is more faithful to "distinct streamRole".)

13. **Record encoder block type.** Architecture allows "LFM2 ShortConv blocks, local attention, or a combination". The repo already has LFM2 ShortConv + GQA shaders. Do you want the local encoder to reuse LFM2-style ShortConv blocks (same `shortconv`/`attention`/`rms_norm`/`silu`/`matmul` building blocks) — which also matches the backward-plan operator list (ShortConv, RMSNorm, SiLU, attention backward)? Or a plain local attention block without ShortConv?

14. **Positional encoding.** Architecture rejects Grouped RoPE; allows "an optional local position within the record". LFM2 uses RoPE with absolute positions. For Krystal:
    - Does the local record encoder use RoPE with **record-local positions** (0..7 reset per record), no RoPE at all, or learned position embeddings?
    - Does the mixer (cross-attention over record keys) use any positional signal (band/recency)? Architecture says keys should be permutation-invariant for unordered records, so I'd guess no position in the mixer — confirm.

15. **Attention masking semantics.** Architecture: local encoder masks prevent cross-record attention; mixer is explicit query→record-bank cross-attention; selectors mask invalid candidates with −∞. Are candidate/record masks computed **on the host** from ABI metadata and uploaded per frame, or derived **in-shader** from record metadata? (Host-computed is simpler for v0 — confirm that's acceptable.)

16. **Active-token execution.** "Runs on `activeTokenCount` only; padding ignored." Confirm the first forward executes only active records/tokens (dynamic workgroup counts), not the full 128-slot/1024-token frame.

---

## D. Schema / ABI questions

17. **Wire `krystal-engine-schema.ts` into the schema build?** Currently `packages/schema/src/build.ts` builds the chomato constraint ABI (`./schema`) into `webgpu/src/types.ts`, `backend/src/abi.cpp`, generated codec, `schema.wgsl`. Should I:
    - (a) add the krystal schema as a second schema-pop build target (generating krystal TS types / codec / WGSL structs), keeping the chomato ABI until it's deleted, or
    - (b) replace the chomato schema entirely now?
    I'd lean (a) for a smooth transition. Confirm.

18. **BinaryLayoutPlan.** The schema file explicitly says the SoA GPU layout (`tokenIds[recordSlot][localToken]`, `schemaIds`, `bandIds`, `fieldRoles`, `runtimeRefs`, `recordFlags`) is "intentionally not frozen yet", and the AoS `BrainFrame` is canonical. The forward pass needs concrete GPU buffers. Should I **freeze a SoA layout** as part of this task (documenting it in the schema/build), or hand-write the WGSL storage buffers without formalizing the plan first?

19. **Where does the actual vocabulary manifest come from?** `VocabManifestHeader/Entry` exist in the schema, and the architecture mentions ~183–200 semantic concepts, but there is no vocab manifest file in this repo. Semcore's `KRYSTAL_ABI_V0.md` (which lives in `/home/kr/Projects/semcore/docs/`, **not** in this repo) says "vocabulary remains provisional, ~200–400 tokens". For tests and the forward pass:
    - Should I copy/vendor `KRYSTAL_ABI_V0.md` (and any other authoritative ABI docs) into this repo's `docs/`?
    - Do you have a concrete token manifest (symbol → 12-bit id) somewhere, or should I define a minimal test vocabulary myself?

20. **Record schemas / ActionIntent catalog.** The schema defines `RecordSchemaEntry`, `ActionIntentDescriptor`, etc., but no concrete manifest. For the first forward/training passes, should I hand-author a tiny set of record schemas + a few ActionIntents as fixtures (e.g., the architecture's `APPLE`/`VisionObject`, `HomeostasisQuery` examples), or is there a compiled manifest elsewhere?

21. **Checkpoint/compatibility hashes.** The architecture lists hash guards (vocab/ABI/graph/head-layout). In scope for the first implementation, or later?

---

## E. Forward pass — concrete questions

22. **CPU reference.** Architecture DoD says "output parity between CPU reference and WebGPU within defined tolerances". So I'll write a plain TS f32 CPU reference for the whole Krystal forward, not just for training ops. Confirm that's expected (it's a lot of extra code, but it's the only way to verify the forward without a pretrained model).

23. **Weights.** There are no Krystal weights anywhere (the debil-chomato models are LFM2.5). First implementation = **random initialization (seeded) + train from scratch on WebGPU**? Any preferred init scheme/distribution for f32 (e.g., small uniform / normal, fixed seed)?

24. **Embedding.** Architecture formula: `x_t = E_token[id] + E_field[role] + E_schema[schema] + E_band[band] + E_stream[stream]`, no `E_recordIndex`. Five additive tables. Confirm all five in the first implementation (they're small), and that `fieldRole`/`schema`/`band`/`stream` come from the sidecar metadata (host-packed) rather than being learned from tokens.

25. **RecordState.** Encoder outputs `key[H]`, `value[H]`, optional `fieldStates`. Default pooling is "role-aware or dedicated learned KEY/VALUE slots; simple token averaging not part of the contract". For v0, do you accept **learned KEY/VALUE pooling slots** (extra learned vectors per record) or role-aware pooling? And do we implement `optionalFieldStates` at all in the first pass, or only for specific schemas later?

26. **Selector/soft-gather.** Each slot computes `score(q_s, key_i) + mask_i`, softmax over valid records, `g_s = Σ p_i·value_i`, runtime resolves `handles[argmax(p)]`. Confirm: dot-product scoring for v0? Number of selector slots in the first forward (route + controller + how many typed argument slots)? The schema allows up to 4 arguments (`maxActionArguments`).

27. **TypedPlan output.** The forward emits `routeKind`, `controllerHandle?`, `argumentHandles[N]`, `scalarClasses?`, `confidence?`, `diagnostics?`. Confirm the first implementation writes this exact schema (`IntentSet`/`TypedPlan`-shaped GPU output) and reads it back on the host. Are `confidence`/`diagnostics` required in v0 or optional?

---

## F. Backward pass — milestone-1 specifics

28. Confirm milestone 1 follows `WEBGPU_BACKWARD_PLAN.md` literally: the seven shaders (or minimal subset), f32 everywhere, SGD no momentum, `trainStep` host API, GPU-resident between step boundaries, deterministic tiny dataset `0→3, 1→4, 2→5, 3→6`, tolerances `atol 1e-5 / rtol 1e-4`.

29. The tiny milestone graph is **not** the Krystal architecture. That's fine as a pipeline proof — but confirm you want the `trainStep` API shaped for later Krystal ops (static backward plan, `TrainingOpSpec` registry) from the start, or hard-coded first as the plan allows.

30. **Where does the Krystal code live?** The backward plan scopes `packages/webgpu`. Options:
    - (a) add Krystal forward/backward into `packages/webgpu` alongside/replacing the LFM2 code, or
    - (b) a new package (e.g. `packages/krystal` / `packages/krystal-engine`).
    The schema already sits in `packages/schema`. Which do you prefer? (LFM2-specific subdirs like `vision/`, `constraint.ts`, checkpoint code would then be deleted.)

31. **Reuse of existing shaders.** Plan says reuse `embedding.wgsl` and `matmul_f32.wgsl`. Note `embedding.wgsl` is currently **f16/WQ4-only** (`load_f16`), so a small f32 embedding shader (or an f32 path) is needed. Confirm reusing the existing `matmul_f32.wgsl` + `rms_norm.wgsl` + `silu_mul.wgsl` etc. where layout matches, and writing new Krystal-specific shaders otherwise.

32. **Arena/OpParams conventions.** Reuse the LFM2 conventions (OpParams uniform dynamic-offset records, one big f32 arena, `pass.run("shader", params, weights)`) for Krystal training? I believe yes; confirm.

---

## G. Repo cleanup — what may I delete, and when

The working tree already has deletions (old docs, `packages/gui/`, `scripts/deploy-pages.sh`, `Chomato.png`) plus the new untracked krystal docs/schema. The task says delete redundant files immediately if sure. My proposed keep/delete plan — please confirm or correct:

33. **Delete now (chomato-specific, not needed by Krystal):**
    - `packages/lfm2` (BPE tokenizer — Krystal uses 12-bit semantic IDs)
    - `packages/quant` (GGUF/WQ4 reader — no quantized Krystal weights)
    - `packages/engine-ts` (structured-JSON constraint compiler + transport — Krystal emits TypedPlan, not JSON)
    - `packages/bridge` (stdio frame protocol — native-exe host concept)
    - `packages/backend` (scriptc native exe + `local-model.ts` mock fallback)
    - `packages/finetune` (Python/Unsloth — task says bypass Python; **or keep as a verification reference?**)
    - `packages/webgpu/vision/`, `constraint.*`, checkpoint code, LFM2 definition/model/forward (after Krystal forward replaces them)
    - `tests/*` LFM2 tests, `tests/fixtures/cat.jpg`, `misc/*`, `src/index.ts`, old `docs/`, `deno.json`, `scripts/deploy-pages.sh`, `packages/webgpu/test/browser` (vite browser pages)
    - `README.md` (still describes chomato; needs rewriting for Krystal)

34. **Keep:** `packages/schema` (krystal ABI + build pipeline), `packages/webgpu` (as the engine home — pending B30), Sandblaster/schema-pop wiring, `flake.nix`, `.envrc`, new krystal docs, `tests/` as the home for new krystal tests, the `webgpu` (Dawn) bindings.

35. **Timing.** Should I delete the chomato-specific packages at the *start* (task says "delete immediately if sure"), or keep them until the Krystal forward/backward actually replaces them so I can copy conventions from the working LFM2 code? I'd prefer to keep LFM2 code until the Krystal equivalent compiles, then delete. Confirm.

36. **Package names.** Packages are still named `@chomato/*` and the repo README/package.json say "chomato". Rename to `@krystal/*` as part of cleanup, or leave names until the end?

37. **Commit policy.** Everything is git-tracked and I'm told I can delete freely. Do you want me to commit milestone-by-milestone (and if so, commit directly to `master`), or leave all changes uncommitted for your review? I won't push anywhere.

---

## H. Testing / verification

38. New tests live in root `tests/` with `bun:test` + the `tests/dawn.ts` singleton pattern, like today — confirm. Per-op CPU-oracle tests, finite-difference checks, and one deterministic overfit integration test are in the plan; I'll follow it.

39. For the Krystal forward, the parity test needs synthetic frames (records/queries/gold plans) since there's no simulation. I'll hand-build tiny deterministic fixtures (a few records + one query + gold selections). Confirm that's the expected data source for now, per TRAINING_DESIGN's "samples derive from compiled frames" being out of scope here.

---

## I. Misc / process

40. Anything else in `/home/kr/Projects/semcore` (K0/K1 experiments, `FWD_ARCHITECTURE_PLAN.md`, curriculum docs) I should treat as authoritative for the forward design, or is `KRYSTAL_BRAIN_ARCHITECTURE_V2.md` the sole contract? In particular the G3 result (same-record attention bias, normal RoPE) vs the v2 "no grouped RoPE" rule.

41. The `@sandblaster/core` link resolves to `/home/kr/Projects/sandblaster-v2` (v0.2.3) and `@schema-pop/*` to `/home/kr/Projects/schema-pop` (v0.2.0) via bun global links. Confirm those two checkouts are the intended dependency sources and I should not expect npm-published versions.

42. Is there anything you already know is broken or missing that I should be aware of before starting (e.g., a schema-pop exporter limitation for the krystal types, a sandblaster pass-runner limitation for dynamic workgroup counts, the `u32[] == 2` runtime-array caveat, etc.)?

---

## Proposed first-session plan (for your approval)

```text
M0  Infra: switch test script to bun, verify Dawn CLI path, wire krystal schema
    into the schema-pop build (A17), freeze a SoA BinaryLayoutPlan (A18).
M1  Tiny training vertical slice per WEBGPU_BACKWARD_PLAN §1–20: seven shaders,
    CPU oracle, finite differences, overfit test, trainStep host API.
M2  Krystal forward per architecture v2: frame packing, record encoder,
    query encoder, mixer, selectors + soft gather, typed heads/plan writeback,
    CPU reference parity tests.
M3  Krystal backward operators in plan §17 order, each with CPU oracle +
    gradient checks.
M4  Cleanup: delete all chomota-specific code, rewrite README, final pass.
```

If you'd rather reorder (e.g. M1 skipped, or M2 before M1), say so and I'll adjust.
