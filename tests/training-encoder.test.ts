// Attention encoder block end-to-end tests (WEBGPU_BACKWARD_PLAN.md §17 item 6
// wiring). The TrainingTrainer encoder path runs:
//
//   embedding -> QKV projections -> attention -> classifier -> CE
//
// with full backward (dScores -> dQ/dK/dV -> dHidden -> dEmbedding and
// dWq/dWk/dWv) and SGD on all trainable pages, all in one GPU submit.
import { expect, test } from "bun:test";
import { getTrainingHarness, type TrainingHarness } from "./training-harness.ts";
import { LFM2_TRAINING_ARENA, TRAINING_ARENA_BASE } from "../packages/webgpu/src/lfm2-layout.ts";
import { TrainingTrainer } from "../packages/webgpu/src/training.ts";
import { encoderBlockForwardBackward } from "./training-oracle.ts";

const V = 8;
const H = 12;
const HEADS = 2;
const HEAD_DIM = 6;
const M = 4;

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function xavier(rows: number, cols: number, rand: () => number): Float32Array {
  const limit = Math.sqrt(6 / (rows + cols));
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < out.length; i++) out[i] = (rand() * 2 - 1) * limit;
  return out;
}

function smallNormal(count: number, rand: () => number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const u = Math.max(1e-12, (rand() * 2) - 1);
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
    out[i] = z * 0.5;
  }
  return out;
}

function bandMask(m: number): Float32Array {
  const mask = new Float32Array(m * m).fill(-1e30);
  for (let i = 0; i < m; i++) {
    for (let j = Math.max(0, i - 1); j <= Math.min(m - 1, i + 1); j++) mask[i * m + j] = 0.0;
  }
  return mask;
}

interface Params {
  embedding: Float32Array;
  classifier: Float32Array;
  wq: Float32Array;
  wk: Float32Array;
  wv: Float32Array;
}

function initParams(): Params {
  const rand = mulberry32(0x0decaf);
  return {
    embedding: smallNormal(V * H, rand),
    classifier: xavier(V, H, rand),
    wq: xavier(H, H, rand),
    wk: xavier(H, H, rand),
    wv: xavier(H, H, rand),
  };
}

function region(h: TrainingHarness, name: keyof typeof LFM2_TRAINING_ARENA): number {
  return TRAINING_ARENA_BASE + LFM2_TRAINING_ARENA[name];
}

/** Read an arena region after the last trainStep (post-SGD). */
async function readRegion(h: TrainingHarness, name: keyof typeof LFM2_TRAINING_ARENA, elements: number): Promise<Float32Array> {
  const arena = h.definition.resources.arena;
  const staging = h.definition.resources.trainingReadback;
  const encoder = h.device.createCommandEncoder();
  encoder.copyBufferToBuffer(arena.gpu, region(h, name) * 4, staging.gpu, 0, elements * 4);
  h.device.queue.submit([encoder.finish()]);
  await h.device.queue.onSubmittedWorkDone();
  const raw = (await staging.readback()) as unknown as Float32Array;
  return raw.slice(0, elements);
}

test("encoder block overfits a toy mapping (loss down, 100% acc)", async () => {
  const h = await getTrainingHarness();
  const params = initParams();
  // Fixed toy mapping: token t -> target (t+1) % V, repeated over M.
  const tokens = Uint32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]);
  const targets = Uint32Array.from([1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4]);
  const mask = bandMask(tokens.length);

  const trainer = new TrainingTrainer({
    vocabSize: V,
    hiddenSize: H,
    embedding: params.embedding,
    classifier: params.classifier,
    encoder: { headCount: HEADS, headDim: HEAD_DIM, wq: params.wq, wk: params.wk, wv: params.wv },
  });

  const first = await trainer.trainStep({ tokens, targets, learningRate: 0.1, mask, telemetry: true });
  expect(first.loss).toBeDefined();
  const initial = first.loss!;

  let last = initial;
  for (let i = 0; i < 400; i++) {
    const res = await trainer.trainStep({ tokens, targets, learningRate: 0.1, mask, telemetry: true });
    last = res.loss!;
    if (last < 0.05) break;
  }
  expect(last).toBeLessThan(initial);
  expect(last).toBeLessThan(0.05);

  // 100% accuracy: argmax logits == target for every row.
  const logits = await trainer.readLogits(tokens.length, V);
  let correct = 0;
  for (let row = 0; row < tokens.length; row++) {
    let best = -Infinity;
    let bestIdx = -1;
    for (let col = 0; col < V; col++) {
      if (logits[row * V + col]! > best) { best = logits[row * V + col]!; bestIdx = col; }
    }
    if (bestIdx === targets[row]) correct++;
  }
  expect(correct).toBe(tokens.length);

  // All parameter pages moved from their initial values.
  const embAfter = await trainer.readEmbedding();
  const clsAfter = await trainer.readClassifier();
  const wqAfter = await trainer.readProjection("wq");
  const wkAfter = await trainer.readProjection("wk");
  const wvAfter = await trainer.readProjection("wv");
  expect(embAfter.some((x, i) => x !== params.embedding[i]!)).toBe(true);
  expect(clsAfter.some((x, i) => x !== params.classifier[i]!)).toBe(true);
  expect(wqAfter.some((x, i) => x !== params.wq[i]!)).toBe(true);
  expect(wkAfter.some((x, i) => x !== params.wk[i]!)).toBe(true);
  expect(wvAfter.some((x, i) => x !== params.wv[i]!)).toBe(true);

  trainer.destroy();
});

test("encoder block gradients match the composed CPU oracle", async () => {
  const h = await getTrainingHarness();
  const params = initParams();
  const tokens = Uint32Array.from([0, 1, 2, 3]);
  const targets = Uint32Array.from([3, 4, 5, 6]);
  const mask = bandMask(M);

  const trainer = new TrainingTrainer({
    vocabSize: V,
    hiddenSize: H,
    embedding: params.embedding,
    classifier: params.classifier,
    encoder: { headCount: HEADS, headDim: HEAD_DIM, wq: params.wq, wk: params.wk, wv: params.wv },
  });
  await trainer.trainStep({ tokens, targets, learningRate: 0.1, mask }); // telemetry off: no readback

  const ref = encoderBlockForwardBackward(
    params.embedding, params.classifier, params.wq, params.wk, params.wv,
    [...tokens], [...targets], mask, HEADS, HEAD_DIM, V, H,
  );

  // Gradient regions from the GPU (pre-SGD buffers are overwritten in place by
  // sgd_step, but the grads we compare live in the arena, not the pages).
  const dQ = await readRegion(h, "dQ", M * H);
  const dK = await readRegion(h, "dK", M * H);
  const dV = await readRegion(h, "dV", M * H);
  const dHidden = await readRegion(h, "dHidden", M * H);
  const dWq = await readRegion(h, "dWq", H * H);
  const dWk = await readRegion(h, "dWk", H * H);
  const dWv = await readRegion(h, "dWv", H * H);
  const dClassifier = await readRegion(h, "dClassifier", V * H);
  const dEmbedding = await readRegion(h, "dEmbedding", V * H);

  const ATOL = 2e-4;
  const RTOL = 1e-3;
  const expectClose = (got: Float32Array, expected: Float32Array, label: string) => {
    let worst = 0;
    for (let i = 0; i < got.length; i++) {
      const err = Math.abs(got[i]! - expected[i]!);
      const bound = ATOL + RTOL * Math.abs(expected[i]!);
      if (err > bound) throw new Error(`${label}[${i}]: got ${got[i]}, expected ${expected[i]} (err ${err} > ${bound})`);
      worst = Math.max(worst, err);
    }
    expect(worst).toBeLessThanOrEqual(ATOL + RTOL);
  };

  expectClose(dQ, ref.dQ, "dQ");
  expectClose(dK, ref.dK, "dK");
  expectClose(dV, ref.dV, "dV");
  expectClose(dHidden, ref.dHidden, "dHidden");
  expectClose(dWq, ref.dWq, "dWq");
  expectClose(dWk, ref.dWk, "dWk");
  expectClose(dWv, ref.dWv, "dWv");
  expectClose(dClassifier, ref.dClassifier, "dClassifier");
  expectClose(dEmbedding, ref.dEmbedding, "dEmbedding");

  trainer.destroy();
});

test("frozen encoder pages stay byte-identical while others train", async () => {
  const h = await getTrainingHarness();
  const params = initParams();
  const tokens = Uint32Array.from([0, 1, 2, 3]);
  const targets = Uint32Array.from([3, 4, 5, 6]);
  const mask = bandMask(M);

  const trainer = new TrainingTrainer({
    vocabSize: V,
    hiddenSize: H,
    embedding: params.embedding,
    classifier: params.classifier,
    encoder: { headCount: HEADS, headDim: HEAD_DIM, wq: params.wq, wk: params.wk, wv: params.wv },
    frozen: ["wq", "wk", "wv"],
  });
  await trainer.trainStep({ tokens, targets, learningRate: 0.1, mask });

  const wqAfter = await trainer.readProjection("wq");
  const wkAfter = await trainer.readProjection("wk");
  const wvAfter = await trainer.readProjection("wv");
  const embAfter = await trainer.readEmbedding();
  expect(wqAfter).toEqual(params.wq);
  expect(wkAfter).toEqual(params.wk);
  expect(wvAfter).toEqual(params.wv);
  expect(embAfter.some((x, i) => x !== params.embedding[i]!)).toBe(true);

  trainer.destroy();
});
