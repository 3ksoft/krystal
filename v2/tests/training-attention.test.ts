// Per-operator GPU-vs-CPU tests for the attention training path
// (WEBGPU_BACKWARD_PLAN.md §17 item 6). Each shader runs standalone on the
// real Dawn backend, read back, and is compared against the plain-TS oracle.
//
// Krystal encoder semantics: bidirectional, host-masked, multi-head, no KV
// cache, no GQA. The mask is a host-compiled [M,M] f32 buffer where 0.0 =
// allowed and -1e30 = blocked; rows must contain at least one allowed entry.
import { expect, test } from "bun:test";
import {
  getTrainingHarness,
  uploadArena,
  readArenaRegion,
  runPassWait,
  type TrainingHarness,
} from "./training-harness.ts";
import {
  attentionForward,
  attentionBackwardScores,
  attentionBackwardQkv,
} from "./training-oracle.ts";

const ATOL = 1e-4;
const RTOL = 1e-3;

function expectClose(got: Float32Array, expected: Float32Array, label: string): void {
  expect(got.length).toBe(expected.length);
  let worst = 0;
  for (let i = 0; i < got.length; i++) {
    const error = Math.abs(got[i]! - expected[i]!);
    const bound = ATOL + RTOL * Math.abs(expected[i]!);
    if (error > bound) {
      throw new Error(`${label}[${i}]: got ${got[i]}, expected ${expected[i]} (err ${error} > bound ${bound})`);
    }
    worst = Math.max(worst, error);
  }
  expect(worst).toBeLessThanOrEqual(ATOL + RTOL);
}

function seededF32(count: number, seed: number): Float32Array {
  const out = new Float32Array(count);
  let s = seed >>> 0;
  for (let i = 0; i < count; i++) {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    out[i] = ((s % 2000) / 1000) - 1; // [-1, 1)
  }
  return out;
}

/** Local-band mask: each row allows itself plus the two immediate neighbors. */
function bandMask(m: number): Float32Array {
  const mask = new Float32Array(m * m).fill(-1e30);
  for (let i = 0; i < m; i++) {
    for (let j = Math.max(0, i - 1); j <= Math.min(m - 1, i + 1); j++) {
      mask[i * m + j] = 0.0;
    }
  }
  return mask;
}

// Arena offsets for the attention regions (all inside the shared arena).
const OFFSETS = {
  q: 0,
  k: 16384,
  v: 32768,
  mask: 49152,
  out: 53248,
  p: 57344,
  dOut: 73728,
  dScores: 77824,
  dQ: 81920,
  dK: 86016,
  dV: 90112,
} as const;

async function runForward(
  h: TrainingHarness,
  m: number,
  headCount: number,
  headDim: number,
): Promise<void> {
  await runPassWait(h, "attention_forward", {
    inputOffset: OFFSETS.q,
    auxOffset: OFFSETS.k,
    aux2Offset: OFFSETS.v,
    aux3Offset: OFFSETS.mask,
    outputOffset: OFFSETS.out,
    aux4Offset: OFFSETS.p,
    tokenCount: m,
    inputDim: headCount * headDim,
    outputDim: headDim,
    u0: headCount,
  });
}

test("attention_forward matches oracle (multi-head, local-band mask)", async () => {
  const h = await getTrainingHarness();
  const m = 5, headCount = 2, headDim = 4;
  const hDim = headCount * headDim;
  const q = seededF32(m * hDim, 71);
  const k = seededF32(m * hDim, 73);
  const v = seededF32(m * hDim, 79);
  const mask = bandMask(m);
  uploadArena(h, OFFSETS.q, q);
  uploadArena(h, OFFSETS.k, k);
  uploadArena(h, OFFSETS.v, v);
  uploadArena(h, OFFSETS.mask, mask);
  await runForward(h, m, headCount, headDim);
  const out = await readArenaRegion(h, OFFSETS.out, m * hDim);
  const p = await readArenaRegion(h, OFFSETS.p, headCount * m * m);
  const ref = attentionForward(q, k, v, mask, headCount, headDim);
  expectClose(out, ref.out, "attention.out");
  expectClose(p, ref.P, "attention.P");
});

test("attention_forward with single head matches oracle", async () => {
  const h = await getTrainingHarness();
  const m = 4, headCount = 1, headDim = 6;
  const hDim = headCount * headDim;
  const q = seededF32(m * hDim, 83);
  const k = seededF32(m * hDim, 89);
  const v = seededF32(m * hDim, 97);
  const mask = bandMask(m);
  uploadArena(h, OFFSETS.q, q);
  uploadArena(h, OFFSETS.k, k);
  uploadArena(h, OFFSETS.v, v);
  uploadArena(h, OFFSETS.mask, mask);
  await runForward(h, m, headCount, headDim);
  const out = await readArenaRegion(h, OFFSETS.out, m * hDim);
  const ref = attentionForward(q, k, v, mask, headCount, headDim);
  expectClose(out, ref.out, "attention.out.single");
});

test("attention_backward_scores matches dScores oracle", async () => {
  const h = await getTrainingHarness();
  const m = 5, headCount = 2, headDim = 4;
  const hDim = headCount * headDim;
  const dOut = seededF32(m * hDim, 101);
  const v = seededF32(m * hDim, 103);
  const p = seededF32(headCount * m * m, 107); // arbitrary probs in (0,1]
  for (let i = 0; i < p.length; i++) p[i] = Math.abs(p[i]!) + 0.01;
  uploadArena(h, OFFSETS.dOut, dOut);
  uploadArena(h, OFFSETS.v, v);
  uploadArena(h, OFFSETS.p, p);
  await runPassWait(h, "attention_backward_scores", {
    inputOffset: OFFSETS.dOut,
    auxOffset: OFFSETS.v,
    aux2Offset: OFFSETS.p,
    outputOffset: OFFSETS.dScores,
    tokenCount: m,
    inputDim: hDim,
    outputDim: headDim,
    u0: headCount,
  });
  const dScores = await readArenaRegion(h, OFFSETS.dScores, headCount * m * m);
  expectClose(dScores, attentionBackwardScores(dOut, v, p, headCount, headDim), "dScores");
});

test("attention_backward_qkv matches dQ/dK/dV oracle", async () => {
  const h = await getTrainingHarness();
  const m = 5, headCount = 2, headDim = 4;
  const hDim = headCount * headDim;
  const dScores = seededF32(headCount * m * m, 109);
  const q = seededF32(m * hDim, 113);
  const k = seededF32(m * hDim, 127);
  const p = seededF32(headCount * m * m, 131);
  for (let i = 0; i < p.length; i++) p[i] = Math.abs(p[i]!) + 0.01;
  const dOut = seededF32(m * hDim, 137);
  uploadArena(h, OFFSETS.dScores, dScores);
  uploadArena(h, OFFSETS.q, q);
  uploadArena(h, OFFSETS.k, k);
  uploadArena(h, OFFSETS.p, p);
  uploadArena(h, OFFSETS.dOut, dOut);
  await runPassWait(h, "attention_backward_qkv", {
    inputOffset: OFFSETS.dScores,
    auxOffset: OFFSETS.q,
    aux2Offset: OFFSETS.k,
    aux3Offset: OFFSETS.p,
    aux4Offset: OFFSETS.dOut,
    outputOffset: OFFSETS.dQ,
    aux5Offset: OFFSETS.dK,
    aux6Offset: OFFSETS.dV,
    tokenCount: m,
    inputDim: hDim,
    outputDim: headDim,
    u0: headCount,
  });
  const dQ = await readArenaRegion(h, OFFSETS.dQ, m * hDim);
  const dK = await readArenaRegion(h, OFFSETS.dK, m * hDim);
  const dV = await readArenaRegion(h, OFFSETS.dV, m * hDim);
  const ref = attentionBackwardQkv(dScores, q, k, p, dOut, headCount, headDim);
  expectClose(dQ, ref.dQ, "dQ");
  expectClose(dK, ref.dK, "dK");
  expectClose(dV, ref.dV, "dV");
});

test("masked positions contribute zero gradient", async () => {
  const h = await getTrainingHarness();
  const m = 3, headCount = 1, headDim = 2;
  const hDim = headCount * headDim;
  // Row 0 blocks every key except itself; a fully blocked row is not allowed.
  const mask = new Float32Array(m * m).fill(-1e30);
  mask[0] = 0.0; mask[4] = 0.0; mask[5] = 0.0; // row1 allows 1,2; row2 allows 1,2
  const q = seededF32(m * hDim, 139);
  const k = seededF32(m * hDim, 149);
  const v = seededF32(m * hDim, 151);
  uploadArena(h, OFFSETS.q, q);
  uploadArena(h, OFFSETS.k, k);
  uploadArena(h, OFFSETS.v, v);
  uploadArena(h, OFFSETS.mask, mask);
  await runForward(h, m, headCount, headDim);
  const p = await readArenaRegion(h, OFFSETS.p, headCount * m * m);
  // Masked P entries must be exactly zero (exp(-1e30) underflows to 0).
  expect(p[1]).toBe(0); // row0->key1 blocked
  expect(p[2]).toBe(0); // row0->key2 blocked
  // The rows that allow everything must still sum to ~1.
  for (let row = 1; row < m; row++) {
    let sum = 0;
    for (let j = 0; j < m; j++) sum += p[row * m + j]!;
    expect(sum).toBeCloseTo(1, 4);
  }
});
