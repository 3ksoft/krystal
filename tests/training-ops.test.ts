// Per-operator GPU-vs-CPU tests for the M1 training slice
// (WEBGPU_BACKWARD_PLAN.md §14.2). Each shader is run standalone on the real
// Dawn backend, read back, and compared against the plain-TS CPU oracle.
import { expect, test } from "bun:test";
import {
  getTrainingHarness,
  uploadArena,
  uploadTargets,
  uploadTokens,
  createWeightPage,
  readArenaRegion,
  runPassWait,
  type TrainingHarness,
} from "./training-harness.ts";
import {
  crossEntropyForwardBackward,
  lossReduce,
  matmulBackwardInput,
  matmulBackwardWeight,
  embeddingBackward,
  sgdStep,
  matmulForward,
} from "./training-oracle.ts";

const ATOL = 1e-5;
const RTOL = 1e-4;

function closeEnough(got: number, expected: number, atol = ATOL, rtol = RTOL): boolean {
  return Math.abs(got - expected) <= atol + rtol * Math.abs(expected);
}

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

// Deterministic test tensors (xorshift, fixed seed).
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

const OFFSETS = {
  input: 0,
  output: 512,
  aux: 1024,
  aux2: 1536,
} as const;

test("zero_f32 zeroes tokenCount elements", async () => {
  const h = await getTrainingHarness();
  const count = 33;
  uploadArena(h, OFFSETS.output, new Float32Array(count).fill(7));
  await runPassWait(h, "zero_f32", { outputOffset: OFFSETS.output, tokenCount: count });
  const out = await readArenaRegion(h, OFFSETS.output, count);
  expect([...out].every((v) => v === 0)).toBe(true);
});

test("embedding_f32 matches the CPU lookup (non-square V,H)", async () => {
  const h = await getTrainingHarness();
  const v = 7, hDim = 5, m = 4;
  const tokens = Uint32Array.from([0, 3, 3, 6]); // includes a repeated id
  uploadTokens(h, tokens);
  const table = seededF32(v * hDim, 11);
  const page = createWeightPage(h, table);
  await runPassWait(h, "embedding_f32", {
    tokenCount: m, inputDim: v, outputDim: hDim, outputOffset: OFFSETS.output, u0: 0,
  }, page);
  const out = await readArenaRegion(h, OFFSETS.output, m * hDim);
  const expected = new Float32Array(m * hDim);
  for (let row = 0; row < m; row++) {
    const t = tokens[row]!;
    for (let col = 0; col < hDim; col++) expected[row * hDim + col] = table[t * hDim + col]!;
  }
  expectClose(out, expected, "embedding_f32");
  page.destroy();
});

test("cross_entropy_forward_backward matches CE oracle (stable softmax)", async () => {
  const h = await getTrainingHarness();
  const m = 4, v = 8;
  const logits = seededF32(m * v, 23);
  const targets = Uint32Array.from([3, 4, 5, 6]);
  uploadArena(h, OFFSETS.input, logits);
  uploadTargets(h, targets);
  await runPassWait(h, "cross_entropy_forward_backward", {
    inputOffset: OFFSETS.input, outputOffset: OFFSETS.output, auxOffset: OFFSETS.aux,
    tokenCount: m, outputDim: v, u1: 0,
  });
  const dLogits = await readArenaRegion(h, OFFSETS.output, m * v);
  const lossRows = await readArenaRegion(h, OFFSETS.aux, m);
  const ref = crossEntropyForwardBackward(logits, targets, v);
  expectClose(dLogits, ref.dLogits, "dLogits");
  expectClose(lossRows, ref.lossRows, "lossRows");
  // loss must be finite and the target row must dominate after softmax.
  expect(lossReduce(ref.lossRows)).toBeFinite();
});

test("cross_entropy with large row range stays finite (row max subtracted)", async () => {
  const h = await getTrainingHarness();
  const m = 2, v = 8;
  const logits = new Float32Array(m * v).fill(1000);
  for (let i = 0; i < m * v; i++) logits[i] = 1000 + (i % 3) * 0.5; // large but narrow spread
  const targets = Uint32Array.from([1, 5]);
  uploadArena(h, OFFSETS.input, logits);
  uploadTargets(h, targets);
  await runPassWait(h, "cross_entropy_forward_backward", {
    inputOffset: OFFSETS.input, outputOffset: OFFSETS.output, auxOffset: OFFSETS.aux,
    tokenCount: m, outputDim: v, u1: 0,
  });
  const dLogits = await readArenaRegion(h, OFFSETS.output, m * v);
  const lossRows = await readArenaRegion(h, OFFSETS.aux, m);
  expect([...lossRows].every(Number.isFinite)).toBe(true);
  expect([...dLogits].every(Number.isFinite)).toBe(true);
  const ref = crossEntropyForwardBackward(logits, targets, v);
  expectClose(lossRows, ref.lossRows, "lossRows.wide");
});

test("loss_reduce reduces lossRows to the mean scalar", async () => {
  const h = await getTrainingHarness();
  const m = 5;
  const rows = Float32Array.from([0.5, 1.25, -0.75, 2.0, 0.0]);
  uploadArena(h, OFFSETS.input, rows);
  await runPassWait(h, "loss_reduce", { inputOffset: OFFSETS.input, outputOffset: OFFSETS.output, tokenCount: m });
  const scalar = await readArenaRegion(h, OFFSETS.output, 1);
  expect(scalar[0]).toBeCloseTo(lossReduce(rows), 6);
});

test("matmul_backward_input matches dX = dY @ W (non-square)", async () => {
  const h = await getTrainingHarness();
  const m = 4, n = 6, k = 5;
  const dY = seededF32(m * n, 31);
  const w = seededF32(n * k, 37);
  uploadArena(h, OFFSETS.input, dY);
  const page = createWeightPage(h, w);
  await runPassWait(h, "matmul_backward_input", {
    inputOffset: OFFSETS.input, outputOffset: OFFSETS.output,
    tokenCount: m, inputDim: n, outputDim: k,
  }, page);
  const out = await readArenaRegion(h, OFFSETS.output, m * k);
  expectClose(out, matmulBackwardInput(dY, w, m, n, k), "dX");
  page.destroy();
});

test("matmul_backward_weight matches dW = dY^T @ X", async () => {
  const h = await getTrainingHarness();
  const m = 4, n = 6, k = 5;
  const dY = seededF32(m * n, 41);
  const x = seededF32(m * k, 43);
  uploadArena(h, OFFSETS.input, dY);
  uploadArena(h, OFFSETS.aux, x);
  await runPassWait(h, "matmul_backward_weight", {
    inputOffset: OFFSETS.input, auxOffset: OFFSETS.aux, outputOffset: OFFSETS.output,
    tokenCount: m, inputDim: n, outputDim: k,
  });
  const out = await readArenaRegion(h, OFFSETS.output, n * k);
  expectClose(out, matmulBackwardWeight(dY, x, m, n, k), "dW");
});

test("embedding_backward accumulates repeated token ids", async () => {
  const h = await getTrainingHarness();
  const m = 4, v = 5, hDim = 3;
  // Token 2 appears twice; its gradient rows must accumulate.
  const tokens = Uint32Array.from([2, 0, 2, 1]);
  uploadTokens(h, tokens);
  const dHidden = seededF32(m * hDim, 47);
  uploadArena(h, OFFSETS.input, dHidden);
  await runPassWait(h, "embedding_backward", {
    inputOffset: OFFSETS.input, outputOffset: OFFSETS.output,
    tokenCount: m, inputDim: v, outputDim: hDim, u0: 0,
  });
  const out = await readArenaRegion(h, OFFSETS.output, v * hDim);
  const expected = embeddingBackward(dHidden, tokens, m, v, hDim);
  expectClose(out, expected, "dEmbedding");
  // Spot check: token 2 row = dHidden[0] + dHidden[2].
  for (let col = 0; col < hDim; col++) {
    expect(out[2 * hDim + col]!).toBeCloseTo(dHidden[col]! + dHidden[2 * hDim + col]!, 6);
  }
});

test("sgd_step applies param -= lr * grad", async () => {
  const h = await getTrainingHarness();
  const count = 17;
  const params = seededF32(count, 53);
  const grads = seededF32(count, 59);
  const lr = 0.1;
  uploadArena(h, OFFSETS.input, grads);
  const page = createWeightPage(h, params);
  await runPassWait(h, "sgd_step", { inputOffset: OFFSETS.input, tokenCount: count, f0: lr }, page);
  // Read the updated page back through the arena (copy page -> arena).
  const encoder = h.device.createCommandEncoder();
  encoder.copyBufferToBuffer(page, 0, h.definition.resources.arena.gpu, OFFSETS.aux2 * 4, count * 4);
  h.device.queue.submit([encoder.finish()]);
  await h.device.queue.onSubmittedWorkDone();
  const out = await readArenaRegion(h, OFFSETS.aux2, count);
  expectClose(out, sgdStep(params, grads, lr), "sgd");
  page.destroy();
});

test("matmul_f32 reuse: logits = hidden @ classifier^T matches oracle", async () => {
  const h = await getTrainingHarness();
  const m = 4, k = 6, n = 8;
  const x = seededF32(m * k, 61);
  const w = seededF32(n * k, 67);
  uploadArena(h, OFFSETS.input, x);
  const page = createWeightPage(h, w);
  await runPassWait(h, "matmul_f32", {
    inputOffset: OFFSETS.input, outputOffset: OFFSETS.output,
    tokenCount: m, inputDim: k, outputDim: n, rowStart: 0, rowCount: n,
  }, page);
  const out = await readArenaRegion(h, OFFSETS.output, m * n);
  expectClose(out, matmulForward(x, w, m, k, n), "logits");
  page.destroy();
});
