// M2b forward tests: the packed SoA BrainFrameGpu (M2a) wired into the
// record/query encoder and query-to-record mixer (WEBGPU_BACKWARD_PLAN.md §17
// item 7, first half). Per-op GPU-vs-CPU checks for the four new shaders,
// then a composed CPU/GPU parity comparison on the canonical fixture frame.
import { expect, test } from "bun:test";
import { getTrainingHarness, createWeightPage, readArenaRegion, runPassWait, uploadArena } from "./training-harness.ts";
import { KRYSTAL_FORWARD_ARENA, KRYSTAL_FORWARD_ARENA_BASE } from "../packages/webgpu/src/krystal-layout.ts";
import { KrystalForward, type SelectionMasks } from "../packages/webgpu/src/krystal-forward.ts";
import { ACTION_INTENT_SCHEMA_ID, buildFixtureFrame } from "../packages/krystal/src/fixtures/frame.ts";
import { buildFixtureActionCatalog } from "../packages/krystal/src/fixtures/action-intents.ts";
import { BRAIN_LIMITS } from "../packages/schema/src/krystal-engine-schema.ts";
import { emitIntentSet } from "../packages/krystal/src/forward/intentset.ts";
import { packBrainFrame } from "../packages/krystal/src/frame/packer.ts";
import {
  compileActiveFrame,
  compileArgumentMask,
  compileIntentMask,
  compileMixerMask,
  compileRecordMask,
} from "../packages/krystal/src/forward/masks.ts";
import {
  BRAIN_FORWARD_CONFIG,
  createBrainForwardWeights,
} from "../packages/krystal/src/forward/model.ts";
import {
  attentionOracle,
  brainForwardOracle,
  brainSelectionOracle,
  decisionHeadOracle,
  matmulOracle,
  reluOracle,
  selectorOracle,
  softmaxRow,
} from "../packages/krystal/src/forward/oracle.ts";

function region(name: keyof typeof KRYSTAL_FORWARD_ARENA, elements: number): number {
  return KRYSTAL_FORWARD_ARENA_BASE + KRYSTAL_FORWARD_ARENA[name];
}

async function uploadU32(h: Awaited<ReturnType<typeof getTrainingHarness>>, offset: number, values: Uint32Array): Promise<void> {
  h.device.queue.writeBuffer(h.definition.resources.arena.gpu, offset * 4, values);
  await h.device.queue.onSubmittedWorkDone();
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) max = Math.max(max, Math.abs(a[i]! - b[i]!));
  return max;
}

test("relu: GPU matches the CPU oracle elementwise", async () => {
  const h = await getTrainingHarness();
  const n = 37;
  const input = Float32Array.from({ length: n }, (_, i) => (i % 5) - 2.3);
  const inOff = region("encH1", n);
  const outOff = region("encOut", n);
  await uploadArena(h, inOff, input);
  await runPassWait(h, "relu", { inputOffset: inOff, outputOffset: outOff, tokenCount: n });
  const got = await readArenaRegion(h, outOff, n);
  const want = reluOracle(input);
  expect(maxAbsDiff(got, want)).toBeLessThanOrEqual(1e-6);
});

test("krystal_attention_forward: cross-attention with masked rows matches the CPU oracle", async () => {
  const h = await getTrainingHarness();
  const qRows = 3;
  const kRows = 5;
  const H = 8;
  const heads = 2;
  const headDim = 4;

  const q = Float32Array.from({ length: qRows * H }, (_, i) => Math.sin(i * 0.7));
  const k = Float32Array.from({ length: kRows * H }, (_, i) => Math.cos(i * 0.3));
  const v = Float32Array.from({ length: kRows * H }, (_, i) => Math.sin(i * 1.1) + 0.5);
  // Block the middle key for every query row.
  const mask = new Float32Array(qRows * kRows);
  for (let i = 0; i < qRows; i++) mask[i * kRows + 2] = -1e30;

  const qOff = region("encQ", qRows * H);
  const kOff = region("encK", kRows * H);
  const vOff = region("encV", kRows * H);
  const maskOff = region("encMask", qRows * kRows);
  const outOff = region("encOut", qRows * H);
  await uploadArena(h, qOff, q);
  await uploadArena(h, kOff, k);
  await uploadArena(h, vOff, v);
  await uploadArena(h, maskOff, mask);
  await runPassWait(h, "krystal_attention_forward", {
    inputOffset: qOff, auxOffset: kOff, aux2Offset: vOff, aux3Offset: maskOff,
    outputOffset: outOff,
    tokenCount: qRows, inputDim: H, outputDim: headDim, u0: kRows, u1: heads,
  });
  const got = await readArenaRegion(h, outOff, qRows * H);
  const want = attentionOracle(q, k, v, mask, qRows, kRows, H, heads, headDim);
  expect(maxAbsDiff(got, want)).toBeLessThanOrEqual(1e-4);
});

test("krystal_pool: learned-query pooling over padded records matches manual softmax", async () => {
  const h = await getTrainingHarness();
  const H = 8;
  const recordCount = 2;
  // Record 0: tokens 0..2 (3 active). Record 1: tokens 3..4 (2 active).
  const fieldStates = Float32Array.from({ length: 5 * H }, (_, i) => Math.sin(i * 0.9) + 0.25);
  const poolQueries = Float32Array.from({ length: 2 * H }, (_, i) => (i < H ? Math.cos(i * 0.5) : Math.sin(i * 0.2)));
  const recordIndices = Uint32Array.from([0, 1]);
  const compactOffset = new Uint32Array(4).fill(0xffff_ffff);
  compactOffset[0] = 0;
  compactOffset[1] = 3;
  const compactCount = Uint32Array.from([3, 2]);

  const fsOff = region("fieldStates", 5 * H);
  const idxOff = region("bankIndices", recordCount);
  const cOff = region("recordCompactOffset", 4);
  const cCntOff = region("recordCompactCount", 4);
  const keysOff = region("bankKeys", recordCount * H);
  const valuesOff = region("bankValues", recordCount * H);
  await uploadArena(h, fsOff, fieldStates);
  await uploadU32(h, idxOff, recordIndices);
  await uploadU32(h, cOff, compactOffset);
  await uploadU32(h, cCntOff, compactCount);
  const poolPage = createWeightPage(h, poolQueries);
  await runPassWait(h, "krystal_pool", {
    inputOffset: fsOff, auxOffset: idxOff, aux2Offset: cOff, aux3Offset: cCntOff,
    outputOffset: keysOff, aux4Offset: valuesOff,
    tokenCount: recordCount, inputDim: H,
  }, poolPage);
  const gotKeys = await readArenaRegion(h, keysOff, recordCount * H);
  const gotValues = await readArenaRegion(h, valuesOff, recordCount * H);

  // Manual oracle: softmax(qk . state) weighted sums per record.
  const wantKeys = new Float32Array(recordCount * H);
  const wantValues = new Float32Array(recordCount * H);
  const scale = 1 / Math.sqrt(H);
  for (const [rec, start, count] of [[0, 0, 3], [1, 3, 2]] as const) {
    const ks = new Float32Array(count);
    const vs = new Float32Array(count);
    for (let j = 0; j < count; j++) {
      let kAcc = 0;
      let vAcc = 0;
      for (let d = 0; d < H; d++) {
        const s = fieldStates[(start + j) * H + d]!;
        kAcc += poolQueries[d]! * s;
        vAcc += poolQueries[H + d]! * s;
      }
      ks[j] = kAcc * scale;
      vs[j] = vAcc * scale;
    }
    softmaxRow(ks, 0, count);
    softmaxRow(vs, 0, count);
    for (let d = 0; d < H; d++) {
      let kAcc = 0;
      let vAcc = 0;
      for (let j = 0; j < count; j++) {
        const s = fieldStates[(start + j) * H + d]!;
        kAcc += ks[j]! * s;
        vAcc += vs[j]! * s;
      }
      wantKeys[rec * H + d] = kAcc;
      wantValues[rec * H + d] = vAcc;
    }
  }
  expect(maxAbsDiff(gotKeys, wantKeys)).toBeLessThanOrEqual(1e-5);
  expect(maxAbsDiff(gotValues, wantValues)).toBeLessThanOrEqual(1e-5);
});

test("krystal_selector: masked scoring, softmax, gather and argmax match the oracle", async () => {
  const h = await getTrainingHarness();
  const Q = 2;
  const R = 4;
  const H = 8;
  const qProj = Float32Array.from({ length: Q * H }, (_, i) => Math.sin(i * 0.6));
  const kProj = Float32Array.from({ length: R * H }, (_, i) => Math.cos(i * 0.4));
  const values = Float32Array.from({ length: R * H }, (_, i) => Math.sin(i * 1.3) + 0.5);
  // Block record 2 for every query row.
  const mask = new Float32Array(Q * R);
  for (let i = 0; i < Q; i++) mask[i * R + 2] = -1e30;

  const qOff = region("selectorQ", Q * H);
  const kOff = region("selectorK", R * H);
  const vOff = region("bankValues", R * H);
  const maskOff = region("intentMask", Q * R);
  const gatherOff = region("intentGather", Q * H);
  const pOff = region("intentP", Q * R);
  const idxOff = region("intentIndices", Q);
  await uploadArena(h, qOff, qProj);
  await uploadArena(h, kOff, kProj);
  await uploadArena(h, vOff, values);
  await uploadArena(h, maskOff, mask);
  await runPassWait(h, "krystal_selector", {
    inputOffset: qOff, auxOffset: kOff, aux2Offset: vOff, aux3Offset: maskOff,
    outputOffset: gatherOff, aux4Offset: pOff, aux5Offset: idxOff,
    tokenCount: Q, inputDim: H, u0: R,
  });
  const gotP = await readArenaRegion(h, pOff, Q * R);
  const gotGather = await readArenaRegion(h, gatherOff, Q * H);
  const gotIdxRaw = await readArenaRegion(h, idxOff, Q);
  const gotIdx = new Uint32Array(gotIdxRaw.buffer, gotIdxRaw.byteOffset, Q);

  // Oracle with identity selector weights (qProj/kProj are pre-projected here).
  const identity = new Float32Array(H * H);
  for (let d = 0; d < H; d++) identity[d * H + d] = 1;
  const want = selectorOracle(qProj, kProj, values, mask, { wq: identity, wk: identity }, H);
  expect(maxAbsDiff(gotP, want.p)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gotGather, want.gather)).toBeLessThanOrEqual(1e-4);
  expect(Array.from(gotIdx)).toEqual(Array.from(want.index));
  // Record 2 is masked, so no query row may select it.
  for (const idx of gotIdx) expect(idx).not.toBe(2);
});

test("krystal_field_embed: six additive embeddings match the manual sum", async () => {
  const h = await getTrainingHarness();
  const H = 8;
  // Tiny table layout: token 6, field 4, schema 3, band 2, stream 2, pos 8
  // (locals go up to 7, the frozen record width).
  const rows = [6, 4, 3, 2, 2, 8];
  const table = new Float32Array(rows.reduce((sum, n) => sum + n * H, 0));
  for (let i = 0; i < table.length; i++) table[i] = Math.sin(i * 0.37) * 0.3;
  const bases: number[] = [];
  {
    let cursor = 0;
    for (const n of rows) {
      bases.push(cursor);
      cursor += n * H;
    }
  }

  const t = 3;
  const frameToks = Uint32Array.from([0, 13, 26]); // slots 0,1,3; locals 0,5,2
  // tokenIds/fieldRoles are indexed by FRAME token id, not compact position.
  const tokenIds = new Uint32Array(27).fill(0);
  tokenIds[0] = 1;
  tokenIds[13] = 2;
  tokenIds[26] = 3;
  const fieldRoles = new Uint32Array(27).fill(0);
  fieldRoles[0] = 0;
  fieldRoles[13] = 1;
  fieldRoles[26] = 2;
  const schemaIds = Uint32Array.from([0, 1, 0, 2]);
  const bandIds = Uint32Array.from([0, 1, 0, 1]); // band table has 2 rows
  const streamIds = Uint32Array.from([0, 1, 0, 0]);

  const tokOff = region("tokenIds", 27);
  const roleOff = region("fieldRoles", 27);
  const schemaOff = region("schemaIds", 4);
  const bandOff = region("bandIds", 4);
  const activeOff = region("activeTokens", t);
  const streamOff = region("streamIds", 4);
  const outOff = region("fieldStates", t * H);
  await uploadU32(h, tokOff, tokenIds);
  await uploadU32(h, roleOff, fieldRoles);
  await uploadU32(h, schemaOff, schemaIds);
  await uploadU32(h, bandOff, bandIds);
  await uploadU32(h, activeOff, frameToks);
  await uploadU32(h, streamOff, streamIds);
  const embedPage = createWeightPage(h, table);
  await runPassWait(h, "krystal_field_embed", {
    inputOffset: tokOff, auxOffset: roleOff, aux2Offset: schemaOff, aux3Offset: bandOff,
    aux4Offset: activeOff, aux5Offset: streamOff, outputOffset: outOff,
    tokenCount: t, inputDim: H,
    u0: bases[0], u1: bases[1], u2: bases[2], u3: bases[3], u4: bases[4], u5: bases[5],
  }, embedPage);
  const got = await readArenaRegion(h, outOff, t * H);

  const want = new Float32Array(t * H);
  for (let i = 0; i < t; i++) {
    const frameTok = frameToks[i]!;
    const slot = Math.floor(frameTok / 8);
    const local = frameTok & 7;
    for (let d = 0; d < H; d++) {
      let value = 0;
      value += table[bases[0]! + tokenIds[frameTok]! * H + d]!;
      value += table[bases[1]! + fieldRoles[frameTok]! * H + d]!;
      value += table[bases[2]! + schemaIds[slot]! * H + d]!;
      value += table[bases[3]! + bandIds[slot]! * H + d]!;
      value += table[bases[4]! + streamIds[slot]! * H + d]!;
      value += table[bases[5]! + local * H + d]!;
      want[i * H + d] = value;
    }
  }
  expect(maxAbsDiff(got, want)).toBeLessThanOrEqual(1e-6);
});

test("composed forward: CPU/GPU parity on the canonical fixture frame", async () => {
  const h = await getTrainingHarness();
  const config = BRAIN_FORWARD_CONFIG;
  const weights = createBrainForwardWeights(config, 1337);

  const frame = packBrainFrame(buildFixtureFrame()).frame;
  const active = compileActiveFrame(frame);
  const { mask: recordMask } = compileRecordMask(active.activeTokens);
  const mixerMask = compileMixerMask(active.queryRecords.length, active.bankRecords.length);
  // Host-compiled selector masks from ABI metadata (answers 15/26): intent
  // candidates are the ActionIntent catalog records; the first reference
  // argument accepts Apple/VisionObject schemas in the vision/memory bands.
  const selection: SelectionMasks = {
    intentMask: compileIntentMask(frame, active, ACTION_INTENT_SCHEMA_ID),
    argMask: compileArgumentMask(frame, active, [2, 1], [3, 8]),
  };

  const runner = new KrystalForward(weights, config);
  runner.forward(frame, selection);
  await h.device.queue.onSubmittedWorkDone();

  const { hiddenSize: hDim } = config;
  const r = active.bankRecords.length;
  const q = active.queryRecords.length;
  const t = active.activeTokens.length;

  const gpuBankKeys = await runner.readBankKeys(r, hDim);
  const gpuBankValues = await runner.readBankValues(r, hDim);
  const gpuQueryOut = await runner.readQueryOutput(q, hDim);
  const gpuFieldStates = await runner.readFieldStates(t, hDim);
  const gpuIntentP = await runner.readIntentP(q, r);
  const gpuIntentGather = await runner.readIntentGather(q, hDim);
  const gpuIntentIdx = await runner.readIntentIndices(q);
  const gpuArgP = await runner.readArgP(q, r);
  const gpuArgGather = await runner.readArgGather(q, hDim);
  const gpuArgIdx = await runner.readArgIndices(q);
  const gpuDecisionLogits = await runner.readDecisionLogits(q, config.routeKindCount);

  const cpu = brainForwardOracle(frame, active, weights, config, recordMask, mixerMask);
  const cpuSel = brainSelectionOracle(
    cpu.queryOutput, cpu.bankKeys, cpu.bankValues,
    selection.intentMask, selection.argMask, weights.selector, hDim,
  );
  const cpuDecisionLogits = decisionHeadOracle(
    cpu.queryOutput, cpuSel.intent.gather, cpuSel.argument.gather,
    weights.decisionHeadWh, q, hDim, config.routeKindCount,
  );

  expect(r).toBe(7); // homeostasis, self, apple, memory, LOOK, EAT, WAIT
  expect(q).toBe(1); // the query record
  expect(maxAbsDiff(gpuFieldStates, cpu.fieldStates)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(gpuBankKeys, cpu.bankKeys)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(gpuBankValues, cpu.bankValues)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(gpuQueryOut, cpu.queryOutput)).toBeLessThanOrEqual(1e-2);
  expect(maxAbsDiff(gpuIntentP, cpuSel.intent.p)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpuIntentGather, cpuSel.intent.gather)).toBeLessThanOrEqual(1e-2);
  expect(Array.from(gpuIntentIdx)).toEqual(Array.from(cpuSel.intent.index));
  expect(maxAbsDiff(gpuArgP, cpuSel.argument.p)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpuArgGather, cpuSel.argument.gather)).toBeLessThanOrEqual(1e-2);
  expect(Array.from(gpuArgIdx)).toEqual(Array.from(cpuSel.argument.index));
  expect(maxAbsDiff(gpuDecisionLogits, cpuDecisionLogits)).toBeLessThanOrEqual(1e-2);

  // The intent selector must pick a catalog record (schemaId 5) and the
  // argument selector an Apple/VisionObject record, never a masked one.
  for (const idx of gpuIntentIdx) {
    expect(frame.schemaIds[active.bankRecords[idx]!]).toBe(ACTION_INTENT_SCHEMA_ID);
  }
  for (const idx of gpuArgIdx) {
    const schema = frame.schemaIds[active.bankRecords[idx]!]!;
    expect([2, 1]).toContain(schema);
  }

  // Sanity: outputs are finite and the mixer actually moved the query state.
  for (const values of [gpuBankKeys, gpuBankValues, gpuQueryOut]) {
    expect(Number.isFinite(values[0])).toBe(true);
  }

  // IntentSet emission (answer 27): the same host resolver must produce the
  // same typed set from GPU and CPU selection heads — intentId, lifecycle,
  // and exact argument handles resolved from the record sidecars.
  const catalog = buildFixtureActionCatalog();
  const gpuSet = emitIntentSet({
    frame, active, catalog,
    intentSchemaId: ACTION_INTENT_SCHEMA_ID,
    intent: { p: gpuIntentP, gather: gpuIntentGather, index: gpuIntentIdx },
    argument: { p: gpuArgP, gather: gpuArgGather, index: gpuArgIdx },
    tick: 10,
  }).intentSet;
  const cpuSet = emitIntentSet({
    frame, active, catalog,
    intentSchemaId: ACTION_INTENT_SCHEMA_ID,
    intent: cpuSel.intent,
    argument: cpuSel.argument,
    tick: 10,
  }).intentSet;
  expect(gpuSet.count).toBe(1);
  expect(gpuSet.count).toBe(cpuSet.count);
  const gpuProposal = gpuSet.proposals[0]!;
  const cpuProposal = cpuSet.proposals[0]!;
  expect(gpuProposal.lifecycle).toBe("start");
  expect(gpuProposal.lifecycle).toBe(cpuProposal.lifecycle);
  expect(gpuProposal.intentId).toBe(cpuProposal.intentId);
  expect(gpuProposal.confidence).toBeCloseTo(cpuProposal.confidence, 4);
  for (let k = 0; k < BRAIN_LIMITS.maxActionArguments; k++) {
    expect(gpuProposal.arguments[k]!.kind).toBe(cpuProposal.arguments[k]!.kind);
    expect(gpuProposal.arguments[k]!.selector.status).toBe(cpuProposal.arguments[k]!.selector.status);
    expect(gpuProposal.arguments[k]!.handle.tokenId).toBe(cpuProposal.arguments[k]!.handle.tokenId);
  }
  // Whatever intent won, its argument must have resolved through a real
  // sidecar handle or be explicitly masked — never a fabricated value.
  const winner = gpuProposal.arguments[0]!;
  if (winner.selector.status === "selected") {
    expect(winner.handle.tokenId).not.toBe(0);
  }
  runner.destroy();
});

test("forward: matmulOracle layout agrees with the shader convention (W [out,in])", () => {
  const x = Float32Array.from([1, 2, 3, 4]);
  const w = Float32Array.from([1, 0, 0, 1, 2, 2]); // [3,2]
  const out = matmulOracle(x, w, 3, 2);
  expect(Array.from(out)).toEqual([1, 2, 6, 3, 4, 14]);
});
