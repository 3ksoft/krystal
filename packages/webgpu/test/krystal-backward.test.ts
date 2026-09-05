// M3 backward tests (docs/archive/WEBGPU_BACKWARD_PLAN.md §17 order): the Krystal backward
// operators for the M2b forward graph — relu backward, cross-capable attention
// backward, and the field-embedding scatter-add — each compared against the
// CPU oracle and (for the attention path) finite-difference gradient checks.
//
// The attention backward needs the persisted probabilities P; the forward
// (krystal_attention_forward) now writes P through aux4Offset exactly like the
// M1 attention_forward contract.
import { expect, test } from "bun:test";
import { createWeightPage, getTrainingHarness, readArenaRegion, runPassWait, uploadArena } from "./harness.ts";
import {
  KRYSTAL_BACKWARD_ARENA,
  KRYSTAL_BACKWARD_ARENA_BASE,
  KRYSTAL_FORWARD_ARENA,
  KRYSTAL_FORWARD_ARENA_BASE,
} from "../src/krystal-layout.ts";
import {
  attentionBackwardQkv,
  attentionBackwardScores,
  decisionHeadBackward,
  fieldEmbedBackward,
  poolBackward,
  reluBackward,
  selectorBackwardQkv,
  selectorBackwardScores,
} from "../../krystal/src/forward/backward.ts";
import { attentionOracle, softmaxRow } from "../../krystal/src/forward/oracle.ts";
import type { BrainForwardConfig } from "../../krystal/src/forward/model.ts";
import { TEST_CONFIG, TEST_TOKEN_ROWS, testFrame, toEmbeddingRows } from "./frame.ts";

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
// krystal selector backward (soft gather + pointer loss, §17 item 8)
// ---------------------------------------------------------------------------

const SEL = { q: 2, r: 5, h: 8 } as const;

function selectorInputs(seedBase: number): {
  qProj: Float32Array;
  kProj: Float32Array;
  value: Float32Array;
  mask: Float32Array;
  p: Float32Array;
  dGather: Float32Array;
} {
  const { q, r, h } = SEL;
  const qProj = Float32Array.from({ length: q * h }, (_, i) => Math.sin(i * 0.7 + seedBase));
  const kProj = Float32Array.from({ length: r * h }, (_, i) => Math.cos(i * 0.3 + seedBase));
  const value = Float32Array.from({ length: r * h }, (_, i) => Math.sin(i * 1.1) + 0.5);
  // Block the middle record for every query row (masked position, p == 0).
  const mask = new Float32Array(q * r);
  for (let i = 0; i < q; i++) mask[i * r + 2] = -1e30;
  // p from the forward selector oracle; recompute here for the oracle inputs.
  const scale = 1 / Math.sqrt(h);
  const p = new Float32Array(q * r);
  const scores = new Float32Array(r);
  for (let i = 0; i < q; i++) {
    for (let j = 0; j < r; j++) {
      let s = 0;
      for (let d = 0; d < h; d++) s += qProj[i * h + d]! * kProj[j * h + d]!;
      scores[j] = s * scale + mask[i * r + j]!;
    }
    softmaxRow(scores, 0, r);
    for (let j = 0; j < r; j++) p[i * r + j] = scores[j]!;
  }
  const dGather = Float32Array.from({ length: q * h }, (_, i) => Math.cos(i * 0.5 + seedBase));
  return { qProj, kProj, value, mask, p, dGather };
}

async function runSelectorBackward(
  h: Awaited<ReturnType<typeof getTrainingHarness>>,
  inputs: ReturnType<typeof selectorInputs>,
  gold: Uint32Array,
): Promise<{ dScore: Float32Array; dQProj: Float32Array; dKProj: Float32Array; dValue: Float32Array }> {
  const { q, r, h: hd } = SEL;
  const qProjOff = bwdRegion("dSelectorQProj", q * hd);
  const kProjOff = bwdRegion("dSelectorKProj", r * hd);
  const valueOff = bwdRegion("dSelectorValue", r * hd);
  const pOff = fwdRegion("intentP", q * r);
  const dGatherOff = bwdRegion("dFieldStates", q * hd);
  const goldOff = bwdRegion("selectorGold", q);
  const dScoreOff = bwdRegion("dSelectorScores", q * r);
  const dQProjOff = bwdRegion("dEncQ", q * hd);
  const dKProjOff = bwdRegion("dEncK", r * hd);
  const dValueOff = bwdRegion("dEncV", r * hd);
  await uploadArena(h, qProjOff, inputs.qProj);
  await uploadArena(h, kProjOff, inputs.kProj);
  await uploadArena(h, valueOff, inputs.value);
  await uploadArena(h, pOff, inputs.p);
  await uploadArena(h, dGatherOff, inputs.dGather);
  await uploadU32(h, goldOff, gold);
  await runPassWait(h, "krystal_selector_backward_scores", {
    inputOffset: dGatherOff, auxOffset: valueOff, aux2Offset: pOff,
    aux3Offset: goldOff, outputOffset: dScoreOff,
    tokenCount: q, inputDim: hd, u0: r,
  });
  await runPassWait(h, "krystal_selector_backward_qkv", {
    inputOffset: dScoreOff, auxOffset: qProjOff, aux2Offset: kProjOff,
    aux3Offset: pOff, aux4Offset: dGatherOff,
    outputOffset: dQProjOff, aux5Offset: dKProjOff, aux6Offset: dValueOff,
    tokenCount: q, inputDim: hd, u0: r,
  });
  return {
    dScore: await readArenaRegion(h, dScoreOff, q * r),
    dQProj: await readArenaRegion(h, dQProjOff, q * hd),
    dKProj: await readArenaRegion(h, dKProjOff, r * hd),
    dValue: await readArenaRegion(h, dValueOff, r * hd),
  };
}

test("selector_backward: GPU dScore/dQProj/dKProj/dValue match the CPU oracle (with pointer loss)", async () => {
  const h = await getTrainingHarness();
  const { q, r, h: hd } = SEL;
  const inputs = selectorInputs(0.2);
  const gold = Uint32Array.from([1, 4]); // record 2 is masked, so never gold
  const gpu = await runSelectorBackward(h, inputs, gold);

  const wantScores = selectorBackwardScores(inputs.dGather, inputs.value, inputs.p, Array.from(gold), q, r, hd);
  const want = selectorBackwardQkv(wantScores, inputs.qProj, inputs.kProj, inputs.p, inputs.dGather, q, r, hd);
  expect(maxAbsDiff(gpu.dScore, wantScores)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dQProj, want.dQProj)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dKProj, want.dKProj)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dValue, want.dValue)).toBeLessThanOrEqual(1e-4);
  // Masked record 2 must have pointerLossGrad exactly p (p == 0 at masked
  // positions, so dScore at the masked slot is 0 for the softmax term).
  for (let i = 0; i < q; i++) {
    expect(gpu.dScore[i * r + 2]!).toBeCloseTo(0, 12);
  }
});

test("selector_backward: dScore/dQProj/dKProj/dValue match central differences (gather + pointer loss)", async () => {
  const h = await getTrainingHarness();
  const { q, r, h: hd } = SEL;
  const inputs = selectorInputs(0.7);
  const gold = Uint32Array.from([0xffff_ffff, 3]); // row 1 has a pointer target
  const gpu = await runSelectorBackward(h, inputs, gold);

  const scale = 1 / Math.sqrt(hd);
  // Forward-only loss: L = sum(gather . G) - sum_i log(p[i, gold[i]]) over
  // rows with a valid gold. gather and p come from the selector forward.
  const forwardLoss = (qProj: Float32Array, kProj: Float32Array, value: Float32Array): number => {
    const p = new Float32Array(q * r);
    const gather = new Float32Array(q * hd);
    const scores = new Float32Array(r);
    for (let i = 0; i < q; i++) {
      for (let j = 0; j < r; j++) {
        let s = 0;
        for (let d = 0; d < hd; d++) s += qProj[i * hd + d]! * kProj[j * hd + d]!;
        scores[j] = s * scale + inputs.mask[i * r + j]!;
      }
      softmaxRow(scores, 0, r);
      for (let j = 0; j < r; j++) p[i * r + j] = scores[j]!;
      for (let d = 0; d < hd; d++) {
        let g = 0;
        for (let j = 0; j < r; j++) g += scores[j]! * value[j * hd + d]!;
        gather[i * hd + d] = g;
      }
    }
    let loss = 0;
    for (let i = 0; i < q; i++) {
      for (let d = 0; d < hd; d++) loss += gather[i * hd + d]! * inputs.dGather[i * hd + d]!;
      if (gold[i] !== 0xffff_ffff) loss -= Math.log(Math.max(p[i * r + gold[i]!]!, 1e-20));
    }
    return loss;
  };

  const EPS = 1e-3;
  // Perturb exactly one tensor per evaluation; the other two stay at their
  // reference values. Perturbing all three at once (holder-mutator style)
  // conflates the directional derivatives and NaNs the check.
  const check = (
    gpuGrad: Float32Array,
    label: string,
    at: (delta: number, index: number) => {
      qProj: Float32Array;
      kProj: Float32Array;
      value: Float32Array;
    },
  ) => {
    const indices = [0, 2, 5, 8, 13];
    for (const index of indices) {
      const plus = at(EPS, index);
      const minus = at(-EPS, index);
      const numeric = (forwardLoss(plus.qProj, plus.kProj, plus.value) -
        forwardLoss(minus.qProj, minus.kProj, minus.value)) / (2 * EPS);
      const analytical = gpuGrad[index]!;
      const bound = 1e-3 + 5e-2 * Math.abs(analytical);
      expect(
        Math.abs(numeric - analytical) <= bound,
        `${label}[${index}]: numeric ${numeric}, analytical ${analytical}`,
      ).toBe(true);
    }
  };
  const at = (delta: number, index: number, target: "qProj" | "kProj" | "value") => {
    const qProj = inputs.qProj.slice();
    const kProj = inputs.kProj.slice();
    const value = inputs.value.slice();
    if (target === "qProj") qProj[index] = qProj[index]! + delta;
    else if (target === "kProj") kProj[index] = kProj[index]! + delta;
    else value[index] = value[index]! + delta;
    return { qProj, kProj, value };
  };
  check(gpu.dQProj, "dQProj", (delta, i) => at(delta, i, "qProj"));
  check(gpu.dKProj, "dKProj", (delta, i) => at(delta, i, "kProj"));
  check(gpu.dValue, "dValue", (delta, i) => at(delta, i, "value"));
});

// ---------------------------------------------------------------------------
// typed decision head backward (§17 item 9): dCtx parts + dWh from dLogits
// ---------------------------------------------------------------------------

const DH = { q: 3, h: 8, c: 4 } as const;

function decisionHeadInputs(seedBase: number): {
  queryOutput: Float32Array;
  intentGather: Float32Array;
  argGather: Float32Array;
  wh: Float32Array;
  dLogits: Float32Array;
} {
  const { q, h, c } = DH;
  const hin = 3 * h;
  const queryOutput = Float32Array.from({ length: q * h }, (_, i) => Math.sin(i * 0.4 + seedBase));
  const intentGather = Float32Array.from({ length: q * h }, (_, i) => Math.cos(i * 0.7 + seedBase));
  const argGather = Float32Array.from({ length: q * h }, (_, i) => Math.sin(i * 1.3) + 0.25);
  const wh = Float32Array.from({ length: c * hin }, (_, i) => Math.cos(i * 0.9 + seedBase) * 0.4);
  const dLogits = Float32Array.from({ length: q * c }, (_, i) => Math.sin(i * 0.6 + seedBase + 1));
  return { queryOutput, intentGather, argGather, wh, dLogits };
}

async function runDecisionHeadBackward(
  h: Awaited<ReturnType<typeof getTrainingHarness>>,
  inputs: ReturnType<typeof decisionHeadInputs>,
): Promise<{
  dQueryOutput: Float32Array;
  dIntentGather: Float32Array;
  dArgGather: Float32Array;
  dWh: Float32Array;
}> {
  const { q, h: hd, c } = DH;
  const hin = 3 * hd;
  const qOff = fwdRegion("queryValues", q * hd);
  const iOff = fwdRegion("intentGather", q * hd);
  const aOff = fwdRegion("argGather", q * hd);
  const dLogitsOff = bwdRegion("dDecisionLogits", q * c);
  const dQOff = bwdRegion("dDecisionQuery", q * hd);
  const dIOff = bwdRegion("dDecisionIntent", q * hd);
  const dAOff = bwdRegion("dDecisionArg", q * hd);
  const dWhOff = bwdRegion("dDecisionWh", c * hin);
  await uploadArena(h, qOff, inputs.queryOutput);
  await uploadArena(h, iOff, inputs.intentGather);
  await uploadArena(h, aOff, inputs.argGather);
  await uploadArena(h, dLogitsOff, inputs.dLogits);
  const weight = createWeightPage(h, inputs.wh);
  await runPassWait(h, "krystal_decision_head_backward", {
    inputOffset: dLogitsOff,
    auxOffset: qOff, aux2Offset: iOff, aux3Offset: aOff,
    outputOffset: dQOff, aux4Offset: dIOff, aux5Offset: dAOff, aux6Offset: dWhOff,
    tokenCount: q, inputDim: hd, outputDim: c,
  }, weight);
  return {
    dQueryOutput: await readArenaRegion(h, dQOff, q * hd),
    dIntentGather: await readArenaRegion(h, dIOff, q * hd),
    dArgGather: await readArenaRegion(h, dAOff, q * hd),
    dWh: await readArenaRegion(h, dWhOff, c * hin),
  };
}

test("decision_head_backward: GPU dQueryOutput/dIntentGather/dArgGather/dWh match the CPU oracle", async () => {
  const h = await getTrainingHarness();
  const { q, h: hd, c } = DH;
  const inputs = decisionHeadInputs(0.3);
  const gpu = await runDecisionHeadBackward(h, inputs);

  const want = decisionHeadBackward(
    inputs.dLogits, inputs.queryOutput, inputs.intentGather, inputs.argGather, inputs.wh, q, hd, c,
  );
  expect(maxAbsDiff(gpu.dQueryOutput, want.dQueryOutput)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dIntentGather, want.dIntentGather)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dArgGather, want.dArgGather)).toBeLessThanOrEqual(1e-4);
  expect(maxAbsDiff(gpu.dWh, want.dWh)).toBeLessThanOrEqual(1e-4);
});

test("decision_head_backward: dQueryOutput/dIntentGather/dArgGather/dWh match central differences (CE over route kinds)", async () => {
  const h = await getTrainingHarness();
  const { q, h: hd, c } = DH;
  const hin = 3 * hd;
  const inputs = decisionHeadInputs(0.7);
  // Gold route kind per query row (all valid class ids).
  const gold = Uint32Array.from([0, 2, 3]);

  // Forward-only loss: mean cross-entropy of the decision-head logits over
  // route kinds. logits = ctx @ Wh^T, ctx = concat(query, intent, arg).
  const forwardLoss = (
    queryOutput: Float32Array,
    intentGather: Float32Array,
    argGather: Float32Array,
    wh: Float32Array,
  ): number => {
    const logits = new Float32Array(q * c);
    for (let i = 0; i < q; i++) {
      for (let cl = 0; cl < c; cl++) {
        let s = 0;
        for (let d = 0; d < hin; d++) {
          const ctx = d < hd
            ? queryOutput[i * hd + d]!
            : d < 2 * hd
              ? intentGather[i * hd + (d - hd)]!
              : argGather[i * hd + (d - 2 * hd)]!;
          s += ctx * wh[cl * hin + d]!;
        }
        logits[i * c + cl] = s;
      }
      // In-place softmax leaves the row's probabilities in logits[i*c..i*c+c).
      softmaxRow(logits, i * c, c);
    }
    let loss = 0;
    for (let i = 0; i < q; i++) {
      loss -= Math.log(Math.max(logits[i * c + gold[i]!]!, 1e-20));
    }
    return loss / q;
  };

  // dLogits for the CE loss (mean-reduced), the input the shader consumes.
  const dLogits = new Float32Array(q * c);
  {
    const logits = new Float32Array(q * c);
    for (let i = 0; i < q; i++) {
      for (let cl = 0; cl < c; cl++) {
        let s = 0;
        for (let d = 0; d < hin; d++) {
          const ctx = d < hd
            ? inputs.queryOutput[i * hd + d]!
            : d < 2 * hd
              ? inputs.intentGather[i * hd + (d - hd)]!
              : inputs.argGather[i * hd + (d - 2 * hd)]!;
          s += ctx * inputs.wh[cl * hin + d]!;
        }
        logits[i * c + cl] = s;
      }
      softmaxRow(logits, i * c, c);
      for (let cl = 0; cl < c; cl++) {
        dLogits[i * c + cl] = (logits[i * c + cl]! - (cl === gold[i] ? 1 : 0)) / q;
      }
    }
  }
  const gpuInputs = { ...inputs, dLogits };
  const gpu = await runDecisionHeadBackward(h, gpuInputs);
  const want = decisionHeadBackward(
    dLogits, inputs.queryOutput, inputs.intentGather, inputs.argGather, inputs.wh, q, hd, c,
  );

  // GPU vs oracle on the CE dLogits first.
  expect(maxAbsDiff(gpu.dWh, want.dWh)).toBeLessThanOrEqual(1e-4);

  // Then central differences against the scalar loss, one tensor at a time.
  const EPS = 1e-3;
  const check = (
    label: string,
    gpuGrad: Float32Array,
    length: number,
    at: (delta: number, index: number) => {
      queryOutput: Float32Array;
      intentGather: Float32Array;
      argGather: Float32Array;
      wh: Float32Array;
    },
  ) => {
    const indices = [0, 3, 7, 11];
    for (const index of indices) {
      if (index >= length) continue;
      const plus = at(EPS, index);
      const minus = at(-EPS, index);
      const numeric = (forwardLoss(plus.queryOutput, plus.intentGather, plus.argGather, plus.wh) -
        forwardLoss(minus.queryOutput, minus.intentGather, minus.argGather, minus.wh)) / (2 * EPS);
      const analytical = gpuGrad[index]!;
      const bound = 1e-3 + 5e-2 * Math.abs(analytical);
      expect(
        Math.abs(numeric - analytical) <= bound,
        `${label}[${index}]: numeric ${numeric}, analytical ${analytical}`,
      ).toBe(true);
    }
  };
  const at = (
    delta: number,
    index: number,
    target: "query" | "intent" | "arg" | "wh",
  ) => {
    const queryOutput = inputs.queryOutput.slice();
    const intentGather = inputs.intentGather.slice();
    const argGather = inputs.argGather.slice();
    const wh = inputs.wh.slice();
    if (target === "query") queryOutput[index] = queryOutput[index]! + delta;
    else if (target === "intent") intentGather[index] = intentGather[index]! + delta;
    else if (target === "arg") argGather[index] = argGather[index]! + delta;
    else wh[index] = wh[index]! + delta;
    return { queryOutput, intentGather, argGather, wh };
  };
  check("dQueryOutput", gpu.dQueryOutput, q * hd, (d, i) => at(d, i, "query"));
  check("dIntentGather", gpu.dIntentGather, q * hd, (d, i) => at(d, i, "intent"));
  check("dArgGather", gpu.dArgGather, q * hd, (d, i) => at(d, i, "arg"));
  check("dWh", gpu.dWh, c * hin, (d, i) => at(d, i, "wh"));
});

// ---------------------------------------------------------------------------
// field embed backward (scatter-add into the six concatenated tables)
// ---------------------------------------------------------------------------

test("field_embed_backward: GPU scatter-add matches the CPU oracle", async () => {
  const h = await getTrainingHarness();
  const config: BrainForwardConfig = TEST_CONFIG;
  const { hiddenSize: H } = config;

  const { frame, active } = testFrame();
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
  // The kernel indexes the embedding tables directly, so it takes rows, not
  // token ids — the same projection the forward runner applies on upload.
  await uploadU32(h, tokOff, toEmbeddingRows(frame.tokenIds));
  await uploadU32(h, roleOff, toEmbeddingRows(frame.fieldRoles));
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
      if (tableId === 0) rowsSet.add(TEST_TOKEN_ROWS[frame.tokenIds[frameTok]!]!);
      else if (tableId === 1) rowsSet.add(TEST_TOKEN_ROWS[frame.fieldRoles[frameTok]!]!);
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

test("field_embed_sgd: sparse fused update matches dense gradient + SGD", async () => {
  const h = await getTrainingHarness();
  const config: BrainForwardConfig = TEST_CONFIG;
  const { hiddenSize: H } = config;
  const { frame, active } = testFrame();
  const t = active.activeTokens.length;
  const dFieldStates = Float32Array.from({ length: t * H }, (_, i) => Math.sin(i * 0.37) + 0.5);
  const denseGradient = fieldEmbedBackward(frame, active, dFieldStates, config);
  const initial = Float32Array.from({ length: denseGradient.length }, (_, i) => Math.cos(i * 0.013) * 0.1);
  const learningRate = 0.025;

  const rows = [
    config.tokenSpace, config.fieldSpace, config.schemaSpace,
    config.bandSpace, config.streamSpace, config.posSpace,
  ];
  const cum = rows.map((_, i) => rows.slice(0, i + 1).reduce((a, b) => a + b, 0));
  const bases = [0, cum[0]!, cum[1]!, cum[2]!, cum[3]!, cum[4]!];
  const touched = new Set<number>();
  for (const frameTok of active.activeTokens) {
    const slot = frameTok >> 3;
    const indices = [
      // Rows, not ids — matching what is uploaded to the kernel below.
      TEST_TOKEN_ROWS[frame.tokenIds[frameTok]!]!,
      TEST_TOKEN_ROWS[frame.fieldRoles[frameTok]!]!,
      frame.schemaIds[slot]!,
      frame.bandIds[slot]!, active.streamIds[slot]!, frameTok & 7,
    ];
    for (let table = 0; table < indices.length; table++) touched.add(bases[table]! + indices[table]!);
  }
  const sparseRows = Uint32Array.from(touched);

  const dFsOff = bwdRegion("dFieldStates", t * H);
  const sparseOff = bwdRegion("dEmbedding", sparseRows.length);
  const tokOff = fwdRegion("tokenIds", frame.tokenIds.length);
  const roleOff = fwdRegion("fieldRoles", frame.fieldRoles.length);
  const schemaOff = fwdRegion("schemaIds", frame.schemaIds.length);
  const bandOff = fwdRegion("bandIds", frame.bandIds.length);
  const streamOff = fwdRegion("streamIds", active.streamIds.length);
  const activeOff = fwdRegion("activeTokens", t);
  await uploadArena(h, dFsOff, dFieldStates);
  await uploadU32(h, sparseOff, sparseRows);
  // The kernel indexes the embedding tables directly, so it takes rows, not
  // token ids — the same projection the forward runner applies on upload.
  await uploadU32(h, tokOff, toEmbeddingRows(frame.tokenIds));
  await uploadU32(h, roleOff, toEmbeddingRows(frame.fieldRoles));
  await uploadU32(h, schemaOff, Uint32Array.from(frame.schemaIds));
  await uploadU32(h, bandOff, Uint32Array.from(frame.bandIds));
  await uploadU32(h, streamOff, active.streamIds);
  await uploadU32(h, activeOff, active.activeTokens);

  const weights = createWeightPage(h, initial);
  await runPassWait(h, "krystal_field_embed_sgd", {
    inputOffset: dFsOff, outputOffset: sparseOff,
    auxOffset: tokOff, aux2Offset: roleOff, aux3Offset: schemaOff,
    aux4Offset: bandOff, aux5Offset: streamOff, aux6Offset: activeOff,
    tokenCount: sparseRows.length, inputDim: H, outputDim: t, f0: learningRate,
    u0: cum[0], u1: cum[1], u2: cum[2], u3: cum[3], u4: cum[4], u5: cum[5],
  }, weights);

  const staging = h.definition.resources.trainingReadback;
  const encoder = h.device.createCommandEncoder();
  let packedOffset = 0;
  for (const row of sparseRows) {
    encoder.copyBufferToBuffer(weights, row * H * 4, staging.gpu, packedOffset * 4, H * 4);
    packedOffset += H;
  }
  h.device.queue.submit([encoder.finish()]);
  await h.device.queue.onSubmittedWorkDone();
  const raw = (await staging.readback()) as unknown as ArrayLike<number>;
  const got = Float32Array.from(raw).slice(0, packedOffset);
  let packedRow = 0;
  for (const row of sparseRows) {
    for (let d = 0; d < H; d++) {
      const index = row * H + d;
      const want = initial[index]! - learningRate * denseGradient[index]!;
      expect(Math.abs(got[packedRow * H + d]! - want)).toBeLessThanOrEqual(1e-6);
    }
    packedRow++;
  }
  weights.destroy();
});
