**# Concerns & questions before starting**

Status: I read \`docs/task.md\`, \`docs/KRYSTAL\_BRAIN\_ARCHITECTURE\_V2.md\`, \`docs/WEBGPU\_BACKWARD\_PLAN.md\`, \`docs/TRAINING\_DESIGN.md\`, \`packages/schema/src/krystal-engine-schema.ts\`, and explored the whole repo plus the sibling \`schema-pop\` and \`sandblaster-v2\` checkouts. Below is everything I need answered before (or while) implementing. Please answer by number; short answers are fine.

**---**

**## A. Infrastructure — Step 1 status and fixes**

**\*\*Verified working right now (from the repo root, inside the flake shell):\*\***

\- \`bun test tests/smoke\_test.ts\` — passes.
\- \`bun test tests/matmul-variants.test.ts\` — passes on the real NVIDIA adapter through the \`webgpu\` (Dawn) bindings, including real GPU readback via Sandblaster's \`readback()\`.
\- \`bun run build\:schema\` — works (regenerates webgpu types/codec/abi.cpp/schema.wgsl).
\- \`bun run build\:webgpu\` (Sandblaster AOT artifact build) — works.
\- \`packages/schema/src/krystal-engine-schema.ts\` analyzes cleanly through schema-pop (\`fromModule\` → 78 types, no analyzer errors). It is currently **\*\*not\*\*** wired into \`build.ts\` or \`index.ts\` — nothing imports it yet.

So the CLI WebGPU pipeline effectively already works with Bun + Dawn. Remaining Step-1 questions:

1\. **\*\*Root \`package.json\` \`"test"\` script says \`deno test tests\`, but every test is \`bun\:test\` and Deno cannot run them\*\*** (verified: deno fails on the \`bun\:test\` import). README says \`bun test\`. Should I change the root test script to \`bun test tests\` and drop/ignore \`deno.json\`? (The task says Deno is a dead end; I agree from what I've seen.)

2\. \`packages/webgpu/package.json\` has \`build\:shaders\` / \`build\:shaders\:full\` scripts that call \`deno run scripts/validate-lfm2-shaders.ts\`. Should these be converted to Bun, or deleted with the LFM2 code (they validate the LFM2 artifact)?

3\. **\*\*flake.nix\*\*** exists and the shell works (\`nodejs\`, \`deno\`, \`bun\`, CUDA/vulkan, etc.). Is the current flake sufficient, or do you expect additions? I will not start real work outside it.

4\. Minor observation, not a blocker: a hand-rolled raw \`mapAsync\` probe returned \`AbortError\`, but Sandblaster's \`readback()\` (same underlying path) works — the passing \`matmul-variants\` test does real GPU readbacks. So I'll use the Sandblaster readback path and not worry about raw mapping. Flagging only so you're aware; I don't think it's a schema-pop/sandblaster bug.

**---**

**## B. Scope and milestones — what exactly is "done" in the first pass**

The three documents describe three different-sized targets and I need to know the intended order.

5\. **\*\*Ordering.\*\*** My reading:
   1\. Step 1: infra (mostly done, see A).
   2\. Milestone 1 of \`WEBGPU\_BACKWARD\_PLAN.md\`: tiny graph \`embedding → linear → CE → backward → SGD\` on a toy dataset (M=4, V=8, H=6), with CPU oracle + finite differences + overfit test. This is a *\*generic\** training vertical slice — it is **\*\*not\*\*** the Krystal brain architecture (no records/mixer/soft gather/typed heads).
   3\. Real Krystal forward per \`KRYSTAL\_BRAIN\_ARCHITECTURE\_V2.md\`.
   4\. Krystal backward operators in the order from \`WEBGPU\_BACKWARD\_PLAN.md\` §17.
   Is that the intended sequence, and is milestone 1 (the tiny slice) the target for this first working session? Or do you want the tiny slice skipped and the real Krystal forward+backward attacked directly?

6\. **\*\*"This will surely take more than one session"\*\*** — do you want me to propose a concrete milestone breakdown (M0 infra → M1 tiny training slice → M2 Krystal forward → M3 Krystal backward…) and then work through it session by session, checking in between? Or should I just go until blocked?

7\. **\*\*Bug-report protocol\*\***: the task says to report schema-pop/sandblaster bugs *\*before doing anything else\**. I have not found a confirmed bug yet (see A4). When I do, is "report" = tell you in the conversation and stop, or open an issue in the sibling repos, or write it into \`docs/concerns.md\`/a file? I won't fix them myself without being asked.

**---**

**## C. Model profile — numbers I need confirmed**

The architecture (baseline) and the schema disagree on some numbers, and several are left open. Please settle these for the *\*first\** implementation.

8\. **\*\*Vocabulary size.\*\*** \`krystal-engine-schema.ts\` \`KRYSTAL\_ABI\` says \`tokenBits: 12\`, \`vocabSize: 0x1000\` (4096), and \`KRYSTAL\_TOKEN\_RANGES\` cover 13 classes over 0x000–0xFFF. But \`KRYSTAL\_BRAIN\_ARCHITECTURE\_V2.md\` baseline says **\*\*Physical Vocab 256\*\***, \~183–200 active semantic concepts, and the schema's own \`BrainModelConfig\` default is \`vocabSize = KRYSTAL\_ABI.vocabSize\`. **\*\*Which vocab size does the model actually use for the first implementation: 4096 (schema/12-bit ABI) or 256 (architecture baseline)?\*\*** This changes embedding tables, logits, masks, and everything else.

9\. **\*\*Hidden size / FFN / block count.\*\*** Architecture baseline: H=128, FFN=384, and "matches the depth of the legacy 4-block model". Semcore's \`FWD\_ARCHITECTURE\_PLAN.md\` records the baseline as **\*\*128 / 4 / 384\*\***. Confirm: first profile = hidden 128, 4 blocks total, FFN 384?

10\. **\*\*Block split.\*\*** The 4 blocks are "partitioned functionally between local encoding and global selection" (open decision). What split do you want for the first pass — e.g. 2 record-encoder blocks + 1 query-encoder block + 1 mixer? Or all record-encoder blocks + a separate cross-attention mixer on top? The mixer is described as a cross-attention layer, so I need to know whether the "4 blocks" includes the mixer or not.

11\. **\*\*Attention heads.\*\*** Not specified anywhere. For H=128, propose e.g. 4 heads × 32 dims (GQA with shared KV heads like LFM2)? Or do you have a number in mind? The schema's \`BrainModelConfig\` has \`attentionHeads\` and a \`recordSize\` field — what should \`recordSize\` be (the key/value dim of a record state)? Equal to H?

12\. **\*\*Query encoder weight sharing\*\*** with the record encoder is listed as open. First implementation: shared or separate? (Shared is cheaper and simpler; separate is more faithful to "distinct streamRole".)

13\. **\*\*Record encoder block type.\*\*** Architecture allows "LFM2 ShortConv blocks, local attention, or a combination". The repo already has LFM2 ShortConv + GQA shaders. Do you want the local encoder to reuse LFM2-style ShortConv blocks (same \`shortconv\`/\`attention\`/\`rms\_norm\`/\`silu\`/\`matmul\` building blocks) — which also matches the backward-plan operator list (ShortConv, RMSNorm, SiLU, attention backward)? Or a plain local attention block without ShortConv?

14\. **\*\*Positional encoding.\*\*** Architecture rejects Grouped RoPE; allows "an optional local position within the record". LFM2 uses RoPE with absolute positions. For Krystal:
    \- Does the local record encoder use RoPE with **\*\*record-local positions\*\*** (0..7 reset per record), no RoPE at all, or learned position embeddings?
    \- Does the mixer (cross-attention over record keys) use any positional signal (band/recency)? Architecture says keys should be permutation-invariant for unordered records, so I'd guess no position in the mixer — confirm.

15\. **\*\*Attention masking semantics.\*\*** Architecture: local encoder masks prevent cross-record attention; mixer is explicit query→record-bank cross-attention; selectors mask invalid candidates with −∞. Are candidate/record masks computed **\*\*on the host\*\*** from ABI metadata and uploaded per frame, or derived **\*\*in-shader\*\*** from record metadata? (Host-computed is simpler for v0 — confirm that's acceptable.)

16\. **\*\*Active-token execution.\*\*** "Runs on \`activeTokenCount\` only; padding ignored." Confirm the first forward executes only active records/tokens (dynamic workgroup counts), not the full 128-slot/1024-token frame.

**---**

**## D. Schema / ABI questions**

17\. **\*\*Wire \`krystal-engine-schema.ts\` into the schema build?\*\*** Currently \`packages/schema/src/build.ts\` builds the chomato constraint ABI (\`./schema\`) into \`webgpu/src/types.ts\`, \`backend/src/abi.cpp\`, generated codec, \`schema.wgsl\`. Should I:
    \- (a) add the krystal schema as a second schema-pop build target (generating krystal TS types / codec / WGSL structs), keeping the chomato ABI until it's deleted, or
    \- (b) replace the chomato schema entirely now?
    I'd lean (a) for a smooth transition. Confirm.

18\. **\*\*BinaryLayoutPlan.\*\*** The schema file explicitly says the SoA GPU layout (\`tokenIds[recordSlot][localToken]\`, \`schemaIds\`, \`bandIds\`, \`fieldRoles\`, \`runtimeRefs\`, \`recordFlags\`) is "intentionally not frozen yet", and the AoS \`BrainFrame\` is canonical. The forward pass needs concrete GPU buffers. Should I **\*\*freeze a SoA layout\*\*** as part of this task (documenting it in the schema/build), or hand-write the WGSL storage buffers without formalizing the plan first?

19\. **\*\*Where does the actual vocabulary manifest come from?\*\*** \`VocabManifestHeader/Entry\` exist in the schema, and the architecture mentions \~183–200 semantic concepts, but there is no vocab manifest file in this repo. Semcore's \`KRYSTAL\_ABI\_V0.md\` (which lives in \`/home/kr/Projects/semcore/docs/\`, **\*\*not\*\*** in this repo) says "vocabulary remains provisional, \~200–400 tokens". For tests and the forward pass:
    \- Should I copy/vendor \`KRYSTAL\_ABI\_V0.md\` (and any other authoritative ABI docs) into this repo's \`docs/\`?
    \- Do you have a concrete token manifest (symbol → 12-bit id) somewhere, or should I define a minimal test vocabulary myself?

20\. **\*\*Record schemas / ActionIntent catalog.\*\*** The schema defines \`RecordSchemaEntry\`, \`ActionIntentDescriptor\`, etc., but no concrete manifest. For the first forward/training passes, should I hand-author a tiny set of record schemas + a few ActionIntents as fixtures (e.g., the architecture's \`APPLE\`/\`VisionObject\`, \`HomeostasisQuery\` examples), or is there a compiled manifest elsewhere?

21\. **\*\*Checkpoint/compatibility hashes.\*\*** The architecture lists hash guards (vocab/ABI/graph/head-layout). In scope for the first implementation, or later?

**---**

**## E. Forward pass — concrete questions**

22\. **\*\*CPU reference.\*\*** Architecture DoD says "output parity between CPU reference and WebGPU within defined tolerances". So I'll write a plain TS f32 CPU reference for the whole Krystal forward, not just for training ops. Confirm that's expected (it's a lot of extra code, but it's the only way to verify the forward without a pretrained model).

23\. **\*\*Weights.\*\*** There are no Krystal weights anywhere (the debil-chomato models are LFM2.5). First implementation = **\*\*random initialization (seeded) + train from scratch on WebGPU\*\***? Any preferred init scheme/distribution for f32 (e.g., small uniform / normal, fixed seed)?

24\. **\*\*Embedding.\*\*** Architecture formula: \`x\_t = E\_token[id] + E\_field[role] + E\_schema[schema] + E\_band[band] + E\_stream[stream]\`, no \`E\_recordIndex\`. Five additive tables. Confirm all five in the first implementation (they're small), and that \`fieldRole\`/\`schema\`/\`band\`/\`stream\` come from the sidecar metadata (host-packed) rather than being learned from tokens.

25\. **\*\*RecordState.\*\*** Encoder outputs \`key[H]\`, \`value[H]\`, optional \`fieldStates\`. Default pooling is "role-aware or dedicated learned KEY/VALUE slots; simple token averaging not part of the contract". For v0, do you accept **\*\*learned KEY/VALUE pooling slots\*\*** (extra learned vectors per record) or role-aware pooling? And do we implement \`optionalFieldStates\` at all in the first pass, or only for specific schemas later?

26\. **\*\*Selector/soft-gather.\*\*** Each slot computes \`score(q\_s, key\_i) + mask\_i\`, softmax over valid records, \`g\_s = Σ p\_i·value\_i\`, runtime resolves \`handles[argmax(p)]\`. Confirm: dot-product scoring for v0? Number of selector slots in the first forward (route + controller + how many typed argument slots)? The schema allows up to 4 arguments (\`maxActionArguments\`).

27\. **\*\*TypedPlan output.\*\*** The forward emits \`routeKind\`, \`controllerHandle?\`, \`argumentHandles[N]\`, \`scalarClasses?\`, \`confidence?\`, \`diagnostics?\`. Confirm the first implementation writes this exact schema (\`IntentSet\`/\`TypedPlan\`-shaped GPU output) and reads it back on the host. Are \`confidence\`/\`diagnostics\` required in v0 or optional?

**---**

**## F. Backward pass — milestone-1 specifics**

28\. Confirm milestone 1 follows \`WEBGPU\_BACKWARD\_PLAN.md\` literally: the seven shaders (or minimal subset), f32 everywhere, SGD no momentum, \`trainStep\` host API, GPU-resident between step boundaries, deterministic tiny dataset \`0→3, 1→4, 2→5, 3→6\`, tolerances \`atol 1e-5 / rtol 1e-4\`.

29\. The tiny milestone graph is **\*\*not\*\*** the Krystal architecture. That's fine as a pipeline proof — but confirm you want the \`trainStep\` API shaped for later Krystal ops (static backward plan, \`TrainingOpSpec\` registry) from the start, or hard-coded first as the plan allows.

30\. **\*\*Where does the Krystal code live?\*\*** The backward plan scopes \`packages/webgpu\`. Options:
    \- (a) add Krystal forward/backward into \`packages/webgpu\` alongside/replacing the LFM2 code, or
    \- (b) a new package (e.g. \`packages/krystal\` / \`packages/krystal-engine\`).
    The schema already sits in \`packages/schema\`. Which do you prefer? (LFM2-specific subdirs like \`vision/\`, \`constraint.ts\`, checkpoint code would then be deleted.)

31\. **\*\*Reuse of existing shaders.\*\*** Plan says reuse \`embedding.wgsl\` and \`matmul\_f32.wgsl\`. Note \`embedding.wgsl\` is currently **\*\*f16/WQ4-only\*\*** (\`load\_f16\`), so a small f32 embedding shader (or an f32 path) is needed. Confirm reusing the existing \`matmul\_f32.wgsl\` + \`rms\_norm.wgsl\` + \`silu\_mul.wgsl\` etc. where layout matches, and writing new Krystal-specific shaders otherwise.

32\. **\*\*Arena/OpParams conventions.\*\*** Reuse the LFM2 conventions (OpParams uniform dynamic-offset records, one big f32 arena, \`pass.run("shader", params, weights)\`) for Krystal training? I believe yes; confirm.

**---**

**## G. Repo cleanup — what may I delete, and when**

The working tree already has deletions (old docs, \`packages/gui/\`, \`scripts/deploy-pages.sh\`, \`Chomato.png\`) plus the new untracked krystal docs/schema. The task says delete redundant files immediately if sure. My proposed keep/delete plan — please confirm or correct:

33\. **\*\*Delete now (chomato-specific, not needed by Krystal):\*\***
    \- \`packages/lfm2\` (BPE tokenizer — Krystal uses 12-bit semantic IDs)
    \- \`packages/quant\` (GGUF/WQ4 reader — no quantized Krystal weights)
    \- \`packages/engine-ts\` (structured-JSON constraint compiler + transport — Krystal emits TypedPlan, not JSON)
    \- \`packages/bridge\` (stdio frame protocol — native-exe host concept)
    \- \`packages/backend\` (scriptc native exe + \`local-model.ts\` mock fallback)
    \- \`packages/finetune\` (Python/Unsloth — task says bypass Python; **\*\*or keep as a verification reference?\*\***)
    \- \`packages/webgpu/vision/\`, \`constraint.\*\`, checkpoint code, LFM2 definition/model/forward (after Krystal forward replaces them)
    \- \`tests/\*\` LFM2 tests, \`tests/fixtures/cat.jpg\`, \`misc/\*\`, \`src/index.ts\`, old \`docs/\`, \`deno.json\`, \`scripts/deploy-pages.sh\`, \`packages/webgpu/test/browser\` (vite browser pages)
    \- \`README.md\` (still describes chomato; needs rewriting for Krystal)

34\. **\*\*Keep:\*\*** \`packages/schema\` (krystal ABI + build pipeline), \`packages/webgpu\` (as the engine home — pending B30), Sandblaster/schema-pop wiring, \`flake.nix\`, \`.envrc\`, new krystal docs, \`tests/\` as the home for new krystal tests, the \`webgpu\` (Dawn) bindings.

35\. **\*\*Timing.\*\*** Should I delete the chomato-specific packages at the *\*start\** (task says "delete immediately if sure"), or keep them until the Krystal forward/backward actually replaces them so I can copy conventions from the working LFM2 code? I'd prefer to keep LFM2 code until the Krystal equivalent compiles, then delete. Confirm.

36\. **\*\*Package names.\*\*** Packages are still named \`@chomato/\*\` and the repo README/package.json say "chomato". Rename to \`@krystal/\*\` as part of cleanup, or leave names until the end?

37\. **\*\*Commit policy.\*\*** Everything is git-tracked and I'm told I can delete freely. Do you want me to commit milestone-by-milestone (and if so, commit directly to \`master\`), or leave all changes uncommitted for your review? I won't push anywhere.

**---**

**## H. Testing / verification**

38\. New tests live in root \`tests/\` with \`bun\:test\` + the \`tests/dawn.ts\` singleton pattern, like today — confirm. Per-op CPU-oracle tests, finite-difference checks, and one deterministic overfit integration test are in the plan; I'll follow it.

39\. For the Krystal forward, the parity test needs synthetic frames (records/queries/gold plans) since there's no simulation. I'll hand-build tiny deterministic fixtures (a few records + one query + gold selections). Confirm that's the expected data source for now, per TRAINING\_DESIGN's "samples derive from compiled frames" being out of scope here.

**---**

**## I. Misc / process**

40\. Anything else in \`/home/kr/Projects/semcore\` (K0/K1 experiments, \`FWD\_ARCHITECTURE\_PLAN.md\`, curriculum docs) I should treat as authoritative for the forward design, or is \`KRYSTAL\_BRAIN\_ARCHITECTURE\_V2.md\` the sole contract? In particular the G3 result (same-record attention bias, normal RoPE) vs the v2 "no grouped RoPE" rule.

41\. The \`@sandblaster/core\` link resolves to \`/home/kr/Projects/sandblaster-v2\` (v0.2.3) and \`@schema-pop/\*\` to \`/home/kr/Projects/schema-pop\` (v0.2.0) via bun global links. Confirm those two checkouts are the intended dependency sources and I should not expect npm-published versions.

42\. Is there anything you already know is broken or missing that I should be aware of before starting (e.g., a schema-pop exporter limitation for the krystal types, a sandblaster pass-runner limitation for dynamic workgroup counts, the \`u32[] == 2\` runtime-array caveat, etc.)?

**---**

**## Proposed first-session plan (for your approval)**

\`\`\`text
M0  Infra: switch test script to bun, verify Dawn CLI path, wire krystal schema
    into the schema-pop build (A17), freeze a SoA BinaryLayoutPlan (A18).
M1  Tiny training vertical slice per WEBGPU\_BACKWARD\_PLAN §1–20: seven shaders,
    CPU oracle, finite differences, overfit test, trainStep host API.
M2  Krystal forward per architecture v2: frame packing, record encoder,
    query encoder, mixer, selectors + soft gather, typed heads/plan writeback,
    CPU reference parity tests.
M3  Krystal backward operators in plan §17 order, each with CPU oracle +
    gradient checks.
M4  Cleanup: delete all chomota-specific code, rewrite README, final pass.
\`\`\`

If you'd rather reorder (e.g. M1 skipped, or M2 before M1), say so and I'll adjust.

---

# Answers and working decisions

These answers are the working implementation contract for the first passes. Items explicitly marked as provisional are experiment profiles, not frozen ABI decisions.

## A. Infrastructure

1. **Yes.** Change the root `test` script to `bun test tests`. Remove `deno.json` only during cleanup, after confirming that no surviving build step reads it.

2. Do not spend time porting dead LFM2-only validators to Bun. Keep them with the LFM2 implementation for now and delete them when that implementation is removed. Any generic validator that survives cleanup should then use Bun.

3. The current `flake.nix` is sufficient. Add nothing without a concrete missing dependency or driver/tool requirement.

4. Use Sandblaster's tested `readback()` path. The raw `mapAsync` result is not currently a blocker or a confirmed upstream bug.

## B. Scope and milestones

5. **Yes, with one refinement.** The intended sequence is:

   1. minimal infrastructure correction;
   2. tiny generic training vertical slice;
   3. Krystal forward;
   4. Krystal backward;
   5. cleanup.

   The tiny slice is the target for the first working session. Do not skip it and do not attack the complete Krystal forward/backward simultaneously.

6. Work milestone by milestone. Within the current milestone, continue until its tests are green or a real blocker is found. Check in at milestone boundaries rather than after every shader.

7. Report a confirmed schema-pop/Sandblaster bug in the conversation with:

   - a minimal reproduction;
   - expected and actual behavior;
   - affected checkout/version;
   - whether it blocks the current milestone.

   Do not open an issue, modify the sibling repository or introduce a contract-changing workaround without approval. A non-blocking bug may be reported and work may continue.

## C. First model profile

8. The logical token ABI and input ID space is **4096 / 12 bits**. The initial active semantic vocabulary is approximately **256 or fewer concepts**. Output heads mask inactive tokens.

   `0xExx` remains the runtime-reference range. The architecture document's “Physical Vocab 256” should be interpreted as the initial active semantic bank, not the ABI address-space size.

   The simplest first implementation may use a table addressable by all 4096 IDs, but `0xExx` rows must not acquire permanent slot semantics. Use a shared/tied context-reference representation and carry exact identity through sidecar metadata/bias. A compact manifest-to-row indirection may be added later without changing token IDs.

9. Confirmed first profile:

   ```text
   hidden size = 128
   block count = 4
   FFN size = 384
   ```

10. **Provisional first split:** two shared local record/query encoder blocks followed by two query-to-record-bank mixer blocks. The four-block count includes the mixer.

11. **Provisional first profile:** four full attention heads × 32 dimensions, four KV heads, `recordSize = hiddenSize = 128`. Do not add GQA to the first tiny Krystal profile.

12. Share record/query encoder weights initially. `E_stream` distinguishes their roles. A separate query encoder remains a later ablation.

13. Use a plain local bidirectional self-attention + FFN encoder first. Do not use ShortConv in the first Krystal profile. An eight-token record makes full local attention cheap and structurally direct. Add ShortConv only if an experiment justifies it.

14. Use learned record-local position embeddings `0..7`, reset for every record. Use no RoPE or global record-index position in the mixer. Band, stream and recency are explicit metadata/features. Mixer access to unordered records is position-independent.

15. In v0, candidate and record masks are compiled on the host from ABI metadata and uploaded with the frame. Do not independently reconstruct them in WGSL.

16. Dispatch over active records/tokens only. Fixed bands may produce non-contiguous active slots, so use an `activeRecordIndices` list rather than assuming the first `activeRecordCount` slots are active. Record-local width remains eight, with a field/token activity mask.

## D. Schema and ABI

17. Choose **(a)**: add Krystal as a second schema-pop build target. Keep the current Chomato target until its consumers are removed.

18. Do not freeze the production SoA layout in M0 and do not block the generic tiny training slice on it. Define a versioned minimal `BinaryLayoutPlan` before Krystal forward, as M2a. Do not maintain an undocumented hand-written WGSL layout beside the schema.

19. No production vocabulary manifest is currently known. For the first forward tests:

   - copy only the normative `KRYSTAL_ABI_V0.md` into this repository if it is absent;
   - do not vendor the entire semcore experiment archive;
   - define a small fixture vocabulary using legal token ranges;
   - label it explicitly as a test/provisional manifest.

20. No concrete production record-schema or `ActionIntent` manifest is currently known. Hand-author a small fixture catalog for M2, for example `Self`, `VisionObject`, `HomeostasisQuery`, `Apple`, `LOOK(ref)`, `EAT(ref)` and `WAIT`. Do not present it as the final world/domain catalog.

21. Full checkpoint hash guards are later work. Preserve version/hash fields now. A simple schema/catalog version check is enough for M2; complete checkpoint compatibility belongs to the cleanup/compatibility milestone.

## E. Krystal forward

22. Write CPU references per operator and one small composed CPU forward for parity. Do not create a second production-grade CPU runtime for the entire model.

23. Use seeded random initialization and train the new tiny brain from scratch. Initial convention:

   - matrices: Xavier/Glorot uniform;
   - token and metadata embeddings: normal, `std = 0.02`;
   - normalization scales: `1`;
   - biases: `0`;
   - deterministic project PRNG, not `Math.random()`.

   Generate once on the host and upload identical weights to CPU/GPU tests.

24. Include all five additive embeddings in the first Krystal forward:

   ```text
   E_token + E_field + E_schema + E_band + E_stream
   ```

   Field/schema/band/stream IDs come from host-packed sidecar metadata. There is no `E_recordIndex`.

25. Use two learned pooling queries to derive record key and value from encoded field states. Preserve local `fieldStates [record, 8, H]` at least until typed field/reference selection completes. Do not add physical KEY/VALUE tokens to the frame.

26. Initial selector scoring:

   ```text
   score = dot(Wq * query, Wk * key) / sqrt(H)
   ```

   The first fixture needs one catalog intent selection and one required reference argument. Infrastructure should respect catalog arity up to four, but must not always run four argument selectors. A controller selector exists only when the descriptor requires it.

27. The canonical output is `IntentSet`, not a separate legacy `TypedPlan`. The first forward may emit `count = 0/1`, `start`, `intentId` and typed arguments. Confidence, entropy and detailed diagnostics are optional/debug outputs.

## F. Backward milestone

28. Follow `WEBGPU_BACKWARD_PLAN.md` milestone 1: f32, SGD without momentum, GPU-resident step, CPU oracle, finite differences and deterministic toy overfit. The listed tolerances are targets and may change only for a documented numerical reason. Implement only the shader subset actually required by existing buffer semantics.

29. The first static backward plan may be hard-coded. Keep tensor ownership, lifetimes and operation boundaries explicit, but do not build a general autograd/`TrainingOpSpec` framework before the first loss curve works.

30. Preferred package boundary:

   ```text
   packages/webgpu  generic kernels, arena, dispatch and backward primitives
   packages/krystal BrainFrame packing, model graph, catalog and trainStep orchestration
   packages/schema  contracts and generated ABI
   ```

   M1 can remain entirely in `packages/webgpu`; introduce `packages/krystal` with M2. Shaders do not require a separate package.

31. Add a small f32 embedding shader/path. Reuse existing `matmul_f32`, normalization, activation and attention code only when physical layout, precision and semantics match. Do not distort the Krystal design to reuse an incompatible LFM2 shader.

32. Reuse the current `OpParams` dynamic-offset convention, one f32 arena and `pass.run(...)` orchestration for the first training path. Keep the whole step GPU-resident.

## G. Cleanup

33. Do **not** delete the entire proposed list at the beginning.

   In particular:

   - keep LFM2 code/shaders as working implementation references until replacements pass;
   - keep `packages/finetune` temporarily as a verification reference;
   - do not remove tests before removing their tested implementation;
   - keep browser WebGPU tests unless the final runtime target explicitly drops browsers;
   - rewrite `README.md` rather than deleting it;
   - keep `packages/quant` until the frozen-WQ4-backbone decision is final.

   `engine-ts`, bridge, backend, structured constraints and legacy checkpoints are likely cleanup candidates, but remove them in M4 after their dependencies are understood.

34. Confirmed keep set: schema, WebGPU backend, schema-pop/Sandblaster build wiring, flake/env, new Krystal docs, root test home and Dawn bindings.

35. Keep old implementations until the corresponding replacement compiles and has green tests. Delete replaced code in a separate, reviewable cleanup change.

36. New packages may use `@krystal/*`. Do not perform a mechanical rename of the surviving old tree during M0/M1. Rename or remove old packages in M4.

37. Commit each green milestone on the current branch; do not push. Keep pre-existing unrelated deletions/changes out of milestone commits. Do not perform branch surgery.

## H. Testing

38. Confirmed: root `tests/`, `bun:test`, existing Dawn singleton, per-op CPU oracles, finite-difference checks and one deterministic overfit integration test.

39. Use small synthetic latent fixtures. Prefer deriving the packed frame and gold targets from an exact fixture description instead of hand-writing expected final tensors. The real simulation/data compiler is outside this first forward milestone.

## I. Sources and process

40. `KRYSTAL_BRAIN_ARCHITECTURE_V2.md` is normative. Semcore documents are experimental evidence and test inspiration; they do not silently override v2. Same-record/exact-reference bias may return as an optional ablation. Grouped/global RoPE is not part of the first v2 forward.

41. The locally linked schema-pop and Sandblaster checkouts are the intended dependency sources for this work. Do not wait for npm-published versions.

42. There is no confirmed upstream bug at this point. Known missing or unverified pieces are:

   - no production vocabulary manifest;
   - no production record-schema or `ActionIntent` catalog;
   - no frozen SoA `BinaryLayoutPlan`;
   - dynamic dispatch over non-contiguous active records is unverified;
   - the existing attention kernel may assume LFM2/causal semantics;
   - embedding lacks the required f32 path;
   - CE/backward/optimizer shaders do not exist yet;
   - compatibility of `matmul_f32` with both gradient layouts is unverified;
   - the Krystal schema analyzes cleanly but has not passed a complete independent export/build target;
   - there is no simulation-produced dataset or trained Krystal checkpoint.

## Approved milestone order

```text
M0
  - root test script -> Bun
  - verify existing Dawn/Sandblaster path
  - add a second Krystal schema build target
  - do not freeze the full SoA yet

M1
  - tiny f32 training vertical slice
  - CE + backward + SGD
  - CPU oracle and finite differences
  - deterministic overfit

M2a
  - minimal vocab/record/action fixtures
  - versioned SoA BinaryLayoutPlan
  - frame packer + activeRecordIndices

M2b
  - shared local record/query encoder
  - query-to-record mixer
  - catalog selection + soft gather
  - IntentSet output
  - CPU/GPU forward parity

M3
  - Krystal backward operator by operator
  - gradient checks
  - training on synthetic frame fixtures

M4
  - remove replaced Chomato/LFM2 code
  - package rename/cleanup
  - rewrite README
  - complete checkpoint/compatibility guards
```

## Still deliberately unknown

The following are not blockers for M1 and should be resolved through later manifests or experiments:

- the final semantic vocabulary;
- the final record-schema and `ActionIntent` catalogs;
- whether the provisional 2-local + 2-mixer split is optimal;
- whether local attention remains better than ShortConv after measurement;
- whether any frozen WQ4 backbone remains in the final system;
- final loss weights, curriculum thresholds and AdamW/mixed-precision hyperparameters;
- the final production SoA geometry beyond the minimal M2 contract.
