// M3 backward tests (WEBGPU_BACKWARD_PLAN.md §17 order): the Krystal backward
// operators for the M2b forward graph — relu backward, cross-capable attention
// backward, and the field-embedding scatter-add — each compared against the
// CPU oracle and (for the attention path) finite-difference gradient checks.
//
// The attention backward needs the persisted probabilities P; the forward
// (krystal_attention_forward) now writes P through aux4Offset exactly like the
// M1 attention_forward contract.
import { expect, test } from "bun:test";
import { createWeightPage, getTrainingHarness, readArenaRegion, runPassWait, uploadArena } from "./training-harness.ts";
import {
  KRYSTAL_BACKWARD_ARENA,
  KRYSTAL_BACKWARD_ARENA_BASE,
  KRYSTAL_FORWARD_ARENA,
  KRYSTAL_FORWARD_ARENA_BASE,
} from "../packages/webgpu/src/lfm2-layout.ts";
import {
  attentionBackwardQkv,
  attentionBackwardScores,
  fieldEmbedBackward,
  poolBackward,
  reluBackward,
} from "../packages/krystal/src/forward/backward.ts";
import { attentionOracle, softmaxRow } from "../packages/krystal/src/forward/oracle.ts";
import { buildFixtureFrame } from "../packages/krystal/src/fixtures/frame.ts";
import { compileActiveFrame } from "../packages/krystal/src/forward/masks.ts";
import { BRAIN_FORWARD_CONFIG, type BrainForwardConfig } from "../packages/krystal/src/forward/model.ts";
import { packBrainFrame } from "../packages/krystal/src/frame/packer.ts";

function fwdRegion(name: keyof typeof KRYSTAL_FORWARD_ARENA, elements: number): number {
  return KRYSTAL_FORWARD_ARENA_BASE + KRYSTAL_FORWARD_ARENA[name];
}

function bwdRegion(name: keyof typeof KRYSTAL_BACKWARD_ARENA, elements: number): number {
  return KRYSTAL_BACKWARD_ARENA_BASE + KRYSTAL_BACKWARD_ARENA[name];
}

async function uploadU32(
  h: Awaited<ReturnType<typeof getTrainingHarness>>,
  offset: number,
  values: Uint32Array,
): Promise<void> {
  h.device.queue.writeBuffer(h.definition.resources.arena.gpu, offset * 4, values);
  await h.device.queue.onSubmittedWorkDone();
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) max = Math.max(max, Math.abs(a[i]! - b[i]!));
  return max;
}

// ---------------------------------------------------------------------------
// relu backward
// ---------------------------------------------------------------------------

test("relu_backward: GPU matches the CPU oracle", async () => {
  const h = await getTrainingHarness();
  const n = 41;
  // Post-activation values (the runner applies relu in place, so the saved
  // tensor is the relu OUTPUT).
  const out = Float32Array.from({ length: n }, (_, i) => Math.max(0, (i % 7) - 2.6));
  const dOut = Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.4) + 1);
  const outOff = bwdRegion("dFieldStates", n);
  const dOutOff = bwdRegion("dEncQ", n);
  const dInOff = bwdRegion("dEncK", n);
  await uploadArena(h, outOff, out);
  await uploadArena(h, dOutOff, dOut);
  await runPassWait(h, "relu_backward", {
    inputOffset: outOff, auxOffset: dOutOff, outputOffset: dInOff, tokenCount: n,
  });
  const got = await readArenaRegion(h, dInOff, n);
  const want = reluBackward(out, dOut);
  expect(maxAbsDiff(got, want)).toBeLessThanOrEqual(1e-6);
});

test("relu_backward: zero gradient through non-positive activations", async () => {
  const h = await getTrainingHarness();
  const n = 16;
  const out = Float32Array.from({ length: n }, (_, i) => (i % 3 === 0 ? -1 : i % 3 === 1 ? 0 : i));
  const dOut = new Float32Array(n).fill(2.5);
  const outOff = bwdRegion("dFieldStates", n);
  const dOutOff = bwdRegion("dEncQ", n);
  const dInOff = bwdRegion("dEncK", n);
  await uploadArena(h, outOff, out);
  await uploadArena(h, dOutOff, dOut);
  await runPassWait(h, "relu_backward", {
    inputOffset: outOff, auxOffset: dOutOff, outputOffset: dInOff, tokenCount: n,
  });
  const got = await readArenaRegion(h, dInOff, n);
  for (let i = 0; i < n; i++) {
    expect(got[i]!).toBe(out[i]! > 0 ? 2.5 : 0);
  }
});

// ---------------------------------------------------------------------------
// krystal attention backward (cross-capable, multi-head, masked)
// ---------------------------------------------------------------------------

const ATTN = {
  qRows: 3,
  kRows: 5,
  heads: 2,
  headDim: 4,
};

function attnOffsets(): { q: number; k: number; v: number; mask: number; out: number; p: number } {
  const H = ATTN.heads * ATTN.headDim;
  const { qRows, kRows } = ATTN;
  return {
    q: fwdRegion("encQ", qRows * H),
    k: fwdRegion("encK", kRows * H),
    v: fwdRegion("encV", kRows * H),
    mask: fwdRegion("encMask", qRows * kRows),
    out: fwdRegion("encOut", qRows * H),
    p: fwdRegion("encP", ATTN.heads * qRows * kRows),
  };
}

function attnOffsetsBackward(): {
  dOut: number; dScores: number; dQ: number; dK: number; dV: number;
} {
  const H = ATTN.heads * ATTN.headDim;
  const { qRows, kRows } = ATTN;
  return {
    dOut: bwdRegion("dFieldStates", qRows * H),
    dScores: bwdRegion("dScoresEnc", ATTN.heads * qRows * kRows),
    dQ: bwdRegion("dEncQ", qRows * H),
    dK: bwdRegion("dEncK", kRows * H),
    dV: bwdRegion("dEncV", kRows * H),
  };
}

async function runAttnForward(
  h: Awaited<ReturnType<typeof getTrainingHarness>>,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  mask: Float32Array,
): Promise<void> {
  const { qRows, kRows, heads, headDim } = ATTN;
  const H = heads * headDim;
  const off = attnOffsets();
  await uploadArena(h, off.q, q);
  await uploadArena(h, off.k, k);
  await uploadArena(h, off.v, v);
  await uploadArena(h, off.mask, mask);
  await runPassWait(h, "krystal_attention_forward", {
    inputOffset: off.q, auxOffset: off.k, aux2Offset: off.v, aux3Offset: off.mask,
    outputOffset: off.out, aux4Offset: off.p,
    tokenCount: qRows, inputDim: H, outputDim: headDim, u0: kRows, u1: heads,
  });
}

async function runAttnBackward(
  h: Awaited<ReturnType<typeof getTrainingHarness>>,
  dOut: Float32Array,
): Promise<{ dScores: Float32Array; dQ: Float32Array; dK: Float32Array; dV: Float32Array }> {
  const { qRows, kRows, heads, headDim } = ATTN;
  const H = heads * headDim;
  const off = attnOffsets();
  const b = attnOffsetsBackward();
  await uploadArena(h, b.dOut, dOut);
  await runPassWait(h, "krystal_attention_backward_scores", {
    inputOffset: b.dOut, auxOffset: off.v, aux2Offset: off.p, outputOffset: b.dScores,
    tokenCount: qRows, inputDim: H, outputDim: headDim, u0: kRows, u1: heads,
  });
  await runPassWait(h, "krystal_attention_backward_qkv", {
    inputOffset: b.dScores, auxOffset: off.q, aux2Offset: off.k,
    aux3Offset: off.p, aux4Offset: b.dOut,
    outputOffset: b.dQ, aux5Offset: b.dK, aux6Offset: b.dV,
    tokenCount: qRows, inputDim: H, outputDim: headDim, u0: kRows, u1: heads,
  });
  const dScores = await readArenaRegion(h, b.dScores, ATTN.heads * qRows * kRows);
  const dQ = await readArenaRegion(h, b.dQ, qRows * H);
  const dK = await readArenaRegion(h, b.dK, kRows * H);
  const dV = await readArenaRegion(h, b.dV, kRows * H);
  return { dScores, dQ, dK, dV };
}

function seededF32(count: number, seed: number): Float32Array {
  let a = seed >>> 0;
  const rand = () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = rand() * 2 - 1;
  return out;
}

test("attention backward: GPU dScores/dQ/dK/dV match the CPU oracle (cross, masked)", async () => {
  const h = await getTrainingHarness();
  const { qRows, kRows, heads, headDim } = ATTN;
  const H = heads * headDim;
  const q = seededF32(qRows * H, 311);
  const k = seededF32(kRows * H, 313);
  const v = seededF32(kRows * H, 317);
  // Block key 2 for every query row (masked position must have dScores 0).
  const mask = new Float32Array(qRows * kRows);
  for (let i = 0; i < qRows; i++) mask[i * kRows + 2] = -1e30;
  const dOut = seededF32(qRows * H, 331);

  await runAttnForward(h, q, k, v, mask);
  const gpu = await runAttnBackward(h, dOut);

  // CPU oracle: recompute P from the same forward math.
  attentionOracle(q, k, v, mask, qRows, kRows, H, heads, headDim);
  const pProbs = new Float32Array(heads * qRows * kRows);
  const scale = 1 / Math.sqrt(headDim);
  const scores = new Float32Array(kRows);
  for (let head = 0; head < heads; head++) {
    const hb = head * headDim;
    for (let i = 0; i < qRows; i++) {
      for (let j = 0; j < kRows; j++) {
        let s = 0;
        for (let d = 0; d < headDim; d++) {
          s += q[i * H + hb + d]! * k[j * H + hb + d]!;
        }
        scores[j] = s * scale + mask[i * kRows + j]!;
      }
      softmaxRow(scores, 0, kRows);
      for (let j = 0; j < kRows; j++) pProbs[(head * qRows + i) * kRows + j] = scores[j]!;
    }
  }
  const wantScores = attentionBackwardScores(dOut, v, pProbs, qRows, kRows, heads, headDim);
  const want = attentionBackwardQkv(wantScores, q, k, pProbs, dOut, qRows, kRows, heads, headDim);

  expect(maxAbsDiff(gpu.dScores, wantScores)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dQ, want.dQ)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dK, want.dK)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dV, want.dV)).toBeLessThanOrEqual(1e-4);

  // Masked key 2 must carry exactly zero dScores for every row/head.
  for (let head = 0; head < heads; head++) {
    for (let i = 0; i < qRows; i++) {
      expect(gpu.dScores[(head * qRows + i) * kRows + 2]!).toBeCloseTo(0, 12);
    }
  }
});

test("attention backward: dQ/dK/dV match central differences of a forward-only loss", async () => {
  const h = await getTrainingHarness();
  const { qRows, kRows, heads, headDim } = ATTN;
  const H = heads * headDim;
  const q = seededF32(qRows * H, 337);
  const k = seededF32(kRows * H, 347);
  const v = seededF32(kRows * H, 349);
  const mask = new Float32Array(qRows * kRows);
  for (let i = 0; i < qRows; i++) mask[i * kRows + 2] = -1e30;
  const g = seededF32(qRows * H, 353); // fixed downstream gradient; dOut = G

  await runAttnForward(h, q, k, v, mask);
  const gpu = await runAttnBackward(h, g);

  // Forward-only scalar loss L = sum(out . G). Runs the dispatch on the
  // CURRENT arena contents (the perturbation is already uploaded), so it must
  // not re-upload the original q/k/v/mask.
  const forwardLoss = async (): Promise<number> => {
    const { qRows: qr, kRows: kr, heads: hd, headDim: hdd } = ATTN;
    const Hd = hd * hdd;
    const off = attnOffsets();
    await runPassWait(h, "krystal_attention_forward", {
      inputOffset: off.q, auxOffset: off.k, aux2Offset: off.v, aux3Offset: off.mask,
      outputOffset: off.out, aux4Offset: off.p,
      tokenCount: qr, inputDim: Hd, outputDim: hdd, u0: kr, u1: hd,
    });
    const out = await readArenaRegion(h, off.out, qr * Hd);
    let loss = 0;
    for (let i = 0; i < out.length; i++) loss += out[i]! * g[i]!;
    return loss;
  };

  const EPS = 1e-3;
  const check = async (params: Float32Array, offset: number, gpuGrad: Float32Array, label: string) => {
    const perturbed = params.slice();
    const indices = [0, 3, 7, 11, 15]; // spread across the tensor
    for (const index of indices) {
      perturbed[index] = params[index]! + EPS;
      await uploadArena(h, offset, perturbed);
      await h.device.queue.onSubmittedWorkDone();
      const lossPlus = await forwardLoss();
      perturbed[index] = params[index]! - EPS;
      await uploadArena(h, offset, perturbed);
      await h.device.queue.onSubmittedWorkDone();
      const lossMinus = await forwardLoss();
      const numeric = (lossPlus - lossMinus) / (2 * EPS);
      const analytical = gpuGrad[index]!;
      const bound = 1e-3 + 5e-2 * Math.abs(analytical);
      expect(
        Math.abs(numeric - analytical) <= bound,
        `${label}[${index}]: numeric ${numeric}, analytical ${analytical}`,
      ).toBe(true);
      perturbed[index] = params[index]!;
    }
    await uploadArena(h, offset, params);
  };

  const off = attnOffsets();
  await check(q, off.q, gpu.dQ, "dQ");
  await check(k, off.k, gpu.dK, "dK");
  await check(v, off.v, gpu.dV, "dV");
});

// ---------------------------------------------------------------------------
// krystal pool backward (learned-query pooling gradients, §17 item 7)
// ---------------------------------------------------------------------------

const POOL = {
  H: 8,
  recordCount: 2,
  // Record 0: tokens 0..2 (3 active). Record 1: tokens 3..4 (2 active).
  starts: [0, 3],
  counts: [3, 2],
} as const;

function poolInputs(seedBase: number): {
  fieldStates: Float32Array;
  pool: Float32Array;
  dKeys: Float32Array;
  dValues: Float32Array;
} {
  const { H, recordCount } = POOL;
  const total = POOL.starts[1]! + POOL.counts[1]!; // 5 tokens
  return {
    fieldStates: Float32Array.from({ length: total * H }, (_, i) => Math.sin(i * 0.9 + seedBase) + 0.25),
    pool: Float32Array.from({ length: 2 * H }, (_, i) => (i < H ? Math.cos(i * 0.5) : Math.sin(i * 0.2))),
    dKeys: Float32Array.from({ length: recordCount * H }, (_, i) => Math.cos(i * 1.3) * 0.7),
    dValues: Float32Array.from({ length: recordCount * H }, (_, i) => Math.sin(i * 0.8) * 0.9),
  };
}

async function runPoolBackward(
  h: Awaited<ReturnType<typeof getTrainingHarness>>,
  inputs: ReturnType<typeof poolInputs>,
): Promise<{ dFieldStates: Float32Array; dPool: Float32Array }> {
  const { H, recordCount, starts, counts } = POOL;
  const total = starts[1]! + counts[1]!;
  const fsOff = bwdRegion("dFieldStates", total * H);
  const dKeysOff = bwdRegion("dBankKeys", recordCount * H);
  const dValuesOff = bwdRegion("dBankValues", recordCount * H);
  const idxOff = bwdRegion("dEncQ", recordCount);
  const cOff = bwdRegion("dEncK", 4);
  const cCntOff = bwdRegion("dEncV", 4);
  const dFsOutOff = bwdRegion("dH1", total * H);
  const dPoolPartialOff = bwdRegion("dPoolPartial", recordCount * 2 * H);
  const dPoolOff = bwdRegion("dPool", 2 * H);

  await uploadArena(h, fsOff, inputs.fieldStates);
  await uploadArena(h, dKeysOff, inputs.dKeys);
  await uploadArena(h, dValuesOff, inputs.dValues);
  await uploadU32(h, idxOff, Uint32Array.from([0, 1]));
  const compactOffset = new Uint32Array(4).fill(0xffff_ffff);
  compactOffset[0] = starts[0];
  compactOffset[1] = starts[1];
  const compactCount = Uint32Array.from(counts);
  await uploadU32(h, cOff, compactOffset);
  await uploadU32(h, cCntOff, compactCount);
  // dFieldStates is accumulated (+=), so zero the output region first.
  await uploadArena(h, dFsOutOff, new Float32Array(total * H));

  const poolPage = createWeightPage(h, inputs.pool);
  await runPassWait(h, "krystal_pool_backward", {
    inputOffset: fsOff, auxOffset: idxOff, aux2Offset: cOff, aux3Offset: cCntOff,
    aux4Offset: dKeysOff, aux5Offset: dValuesOff,
    outputOffset: dFsOutOff, aux6Offset: dPoolPartialOff,
    tokenCount: recordCount, inputDim: H,
  }, poolPage);
  await runPassWait(h, "krystal_pool_dpool", {
    inputOffset: dPoolPartialOff, outputOffset: dPoolOff,
    tokenCount: recordCount, inputDim: H,
  });
  const dFieldStates = await readArenaRegion(h, dFsOutOff, total * H);
  const dPool = await readArenaRegion(h, dPoolOff, 2 * H);
  return { dFieldStates, dPool };
}

test("pool_backward: GPU dFieldStates/dPool match the CPU oracle", async () => {
  const h = await getTrainingHarness();
  const { H, recordCount, starts, counts } = POOL;
  const inputs = poolInputs(0.4);
  const gpu = await runPoolBackward(h, inputs);

  const want = poolBackward(
    inputs.fieldStates, [0, 1], starts, counts, inputs.pool, inputs.dKeys, inputs.dValues, H,
  );
  const total = starts[1]! + counts[1]!;
  expect(maxAbsDiff(gpu.dFieldStates, want.dFieldStates)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dPool, want.dPool)).toBeLessThanOrEqual(1e-4);
  expect(gpu.dFieldStates.length).toBe(total * H);
});

test("pool_backward: dFieldStates/dPool match central differences of a forward-only loss", async () => {
  const h = await getTrainingHarness();
  const { H, recordCount, starts, counts } = POOL;
  const inputs = poolInputs(0.9);
  const total = starts[1]! + counts[1]!;
  const scale = 1 / Math.sqrt(H);
  const Gk = Float32Array.from({ length: recordCount * H }, (_, i) => Math.sin(i * 0.6) * 0.5);
  const Gv = Float32Array.from({ length: recordCount * H }, (_, i) => Math.cos(i * 0.4) * 0.6);

  // Forward-only scalar loss L = sum(keys . Gk) + sum(values . Gv), where
  // keys/values come from the same pooling math as krystal_pool forward.
  const forwardLoss = (fieldStates: Float32Array, pool: Float32Array): number => {
    const keyScores = new Float32Array(8);
    const valueScores = new Float32Array(8);
    let loss = 0;
    for (let rec = 0; rec < recordCount; rec++) {
      const start = starts[rec]!;
      const count = counts[rec]!;
      for (let j = 0; j < count; j++) {
        let ks = 0;
        let vs = 0;
        for (let d = 0; d < H; d++) {
          const s = fieldStates[(start + j) * H + d]!;
          ks += pool[d]! * s;
          vs += pool[H + d]! * s;
        }
        keyScores[j] = ks * scale;
        valueScores[j] = vs * scale;
      }
      softmaxRow(keyScores, 0, count);
      softmaxRow(valueScores, 0, count);
      for (let d = 0; d < H; d++) {
        let kAcc = 0;
        let vAcc = 0;
        for (let j = 0; j < count; j++) {
          const s = fieldStates[(start + j) * H + d]!;
          kAcc += keyScores[j]! * s;
          vAcc += valueScores[j]! * s;
        }
        loss += kAcc * Gk[rec * H + d]! + vAcc * Gv[rec * H + d]!;
      }
    }
    return loss;
  };

  const gpu = await runPoolBackward(h, { ...inputs, dKeys: Gk, dValues: Gv });
  const EPS = 1e-3;
  const check = (
    perturbed: Float32Array,
    gpuGrad: Float32Array,
    label: string,
    loss: (fs: Float32Array, pool: Float32Array) => number,
  ) => {
    const indices = [0, 1, 5, 8, 13];
    for (const index of indices) {
      const plus = perturbed.slice();
      plus[index] = perturbed[index]! + EPS;
      const minus = perturbed.slice();
      minus[index] = perturbed[index]! - EPS;
      const numeric = (loss(plus, plus) - loss(minus, minus)) / (2 * EPS);
      const analytical = gpuGrad[index]!;
      const bound = 1e-3 + 5e-2 * Math.abs(analytical);
      expect(
        Math.abs(numeric - analytical) <= bound,
        `${label}[${index}]: numeric ${numeric}, analytical ${analytical}`,
      ).toBe(true);
    }
  };
  // For the fieldStates check the perturbed tensor is the fieldStates slot;
  // for the pool check it is the pool-query slot — the other slot stays fixed.
  check(inputs.fieldStates, gpu.dFieldStates, "dFieldStates", (fs) => forwardLoss(fs, inputs.pool));
  check(inputs.pool, gpu.dPool, "dPool", (pool) => forwardLoss(inputs.fieldStates, pool));
});

// ---------------------------------------------------------------------------
// field embed backward (scatter-add into the six concatenated tables)
// ---------------------------------------------------------------------------

test("field_embed_backward: GPU scatter-add matches the CPU oracle", async () => {
  const h = await getTrainingHarness();
  const config: BrainForwardConfig = BRAIN_FORWARD_CONFIG;
  const { hiddenSize: H } = config;

  const frame = packBrainFrame(buildFixtureFrame()).frame;
  const active = compileActiveFrame(frame);
  const t = active.activeTokens.length;
  const dFieldStates = Float32Array.from({ length: t * H }, (_, i) => Math.sin(i * 0.37) + 0.5);

  const want = fieldEmbedBackward(frame, active, dFieldStates, config);

  // Arena payloads (same inputs the forward consumes).
  const dFsOff = bwdRegion("dFieldStates", t * H);
  const tokOff = fwdRegion("tokenIds", frame.tokenIds.length);
  const roleOff = fwdRegion("fieldRoles", frame.fieldRoles.length);
  const schemaOff = fwdRegion("schemaIds", frame.schemaIds.length);
  const bandOff = fwdRegion("bandIds", frame.bandIds.length);
  const streamOff = fwdRegion("streamIds", active.streamIds.length);
  const activeOff = fwdRegion("activeTokens", t);
  await uploadArena(h, dFsOff, dFieldStates);
  await uploadU32(h, tokOff, Uint32Array.from(frame.tokenIds));
  await uploadU32(h, roleOff, Uint32Array.from(frame.fieldRoles));
  await uploadU32(h, schemaOff, Uint32Array.from(frame.schemaIds));
  await uploadU32(h, bandOff, Uint32Array.from(frame.bandIds));
  await uploadU32(h, streamOff, active.streamIds);
  await uploadU32(h, activeOff, active.activeTokens);

  const dEmbedOff = bwdRegion("dEmbedding", want.length);
  const rows = [
    config.tokenSpace, config.fieldSpace, config.schemaSpace,
    config.bandSpace, config.streamSpace, config.posSpace,
  ];
  // Cumulative row counts (u0..u5) exactly as the shader expects.
  const cum = rows.map((_, i) => rows.slice(0, i + 1).reduce((a, b) => a + b, 0));
  await runPassWait(h, "krystal_field_embed_backward", {
    inputOffset: dFsOff, outputOffset: dEmbedOff,
    auxOffset: tokOff, aux2Offset: roleOff, aux3Offset: schemaOff,
    aux4Offset: bandOff, aux5Offset: streamOff, aux6Offset: activeOff,
    tokenCount: cum[cum.length - 1]!, inputDim: H, outputDim: t,
    u0: cum[0], u1: cum[1], u2: cum[2], u3: cum[3], u4: cum[4], u5: cum[5],
  });

  // The whole dEmbedding page is 8469*H words, larger than the staging
  // buffer; copy only the touched rows into staging in one packed pass and
  // compare those (untouched rows are written exactly 0 by the owner scan).
  const touched: { base: number; rows: Set<number> }[] = [];
  const tableBases = [0, config.tokenSpace, config.tokenSpace + config.fieldSpace];
  for (let tableId = 0; tableId < 3; tableId++) {
    const rowsSet = new Set<number>();
    for (let i = 0; i < t; i++) {
      const frameTok = active.activeTokens[i]!;
      if (tableId === 0) rowsSet.add(frame.tokenIds[frameTok]!);
      else if (tableId === 1) rowsSet.add(frame.fieldRoles[frameTok]!);
      else rowsSet.add(frame.schemaIds[frameTok! >> 3]!);
    }
    touched.push({ base: tableBases[tableId]!, rows: rowsSet });
  }
  const arena = h.definition.resources.arena;
  const staging = h.definition.resources.trainingReadback;
  const encoder = h.device.createCommandEncoder();
  let stagingOffset = 0;
  const slices: { offset: number; wantBase: number; rows: Set<number> }[] = [];
  for (const { base, rows } of touched) {
    for (const row of rows) {
      encoder.copyBufferToBuffer(arena.gpu, (dEmbedOff + base + row * H) * 4, staging.gpu, stagingOffset * 4, H * 4);
      slices.push({ offset: stagingOffset, wantBase: base + row * H, rows: new Set([row]) });
      stagingOffset += H;
    }
  }
  h.device.queue.submit([encoder.finish()]);
  await h.device.queue.onSubmittedWorkDone();
  const raw = (await staging.readback()) as unknown as ArrayLike<number>;
  const got = Float32Array.from(raw as ArrayLike<number>).slice(0, stagingOffset);
  for (const slice of slices) {
    for (let d = 0; d < H; d++) {
      const g = got[slice.offset + d]!;
      const w = want[slice.wantBase + d]!;
      expect(Math.abs(g - w)).toBeLessThanOrEqual(1e-6);
    }
  }
  expect(slices.length).toBeGreaterThan(0);
});
