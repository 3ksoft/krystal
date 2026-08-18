// Finite-difference gradient check for the full attention encoder block
// (TrainingTrainer encoder path). The analytical gradients from one GPU
// trainStep (dWq/dWk/dWv/dClassifier/dEmbedding, all in the training arena)
// are compared against central differences of the scalar loss, recomputed
// through the GPU forward chain (embedding -> QKV -> attention -> classifier
// -> CE) with exactly one parameter page perturbed at a time.
import { expect, test } from "bun:test";
import { getTrainingHarness, type TrainingHarness } from "./training-harness.ts";
import { KRYSTAL_TRAINING_ARENA, TRAINING_ARENA_BASE } from "../../packages/webgpu/src/krystal-layout.ts";
import { TrainingTrainer } from "../../packages/webgpu/src/training.ts";
import { encoderBlockForwardBackward } from "./training-oracle.ts";

const V = 8;
const H = 10;
const HEADS = 2;
const HEAD_DIM = 5;
const M = 4;
const EPS = 1e-3;

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

function region(h: TrainingHarness, name: keyof typeof KRYSTAL_TRAINING_ARENA): number {
  return TRAINING_ARENA_BASE + KRYSTAL_TRAINING_ARENA[name];
}

async function readRegion(h: TrainingHarness, name: keyof typeof KRYSTAL_TRAINING_ARENA, elements: number): Promise<Float32Array> {
  const arena = h.definition.resources.arena;
  const staging = h.definition.resources.trainingReadback;
  const encoder = h.device.createCommandEncoder();
  encoder.copyBufferToBuffer(arena.gpu, region(h, name) * 4, staging.gpu, 0, elements * 4);
  h.device.queue.submit([encoder.finish()]);
  await h.device.queue.onSubmittedWorkDone();
  const raw = (await staging.readback()) as unknown as ArrayLike<number>;
  return Float32Array.from(raw).slice(0, elements);
}

test("encoder block dWq/dWk/dWv/dClassifier match central differences", async () => {
  const h = await getTrainingHarness();
  const rand = mulberry32(0xbeef);
  const init = {
    embedding: smallNormal(V * H, rand),
    classifier: xavier(V, H, rand),
    wq: xavier(H, H, rand),
    wk: xavier(H, H, rand),
    wv: xavier(H, H, rand),
  };
  const tokens = Uint32Array.from([0, 1, 2, 3]);
  const targets = Uint32Array.from([3, 4, 5, 6]);
  const mask = bandMask(M);

  // One GPU trainStep computes all gradients in the arena (pages then get
  // overwritten by SGD, but the arena gradient regions are left intact).
  const trainer = new TrainingTrainer({
    vocabSize: V, hiddenSize: H, embedding: init.embedding, classifier: init.classifier,
    encoder: { headCount: HEADS, headDim: HEAD_DIM, wq: init.wq, wk: init.wk, wv: init.wv },
  });
  await trainer.trainStep({ tokens, targets, learningRate: 0.01, mask });

  const gpuD = {
    wq: await readRegion(h, "dWq", H * H),
    wk: await readRegion(h, "dWk", H * H),
    wv: await readRegion(h, "dWv", H * H),
    classifier: await readRegion(h, "dClassifier", V * H),
  };

  // CPU oracle for the same step, to double-check the FD target itself.
  const ref = encoderBlockForwardBackward(
    init.embedding, init.classifier, init.wq, init.wk, init.wv,
    [...tokens], [...targets], mask, HEADS, HEAD_DIM, V, H,
  );

  const ATOL = 1e-3;
  const RTOL = 5e-2;
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
  expectClose(gpuD.wq, ref.dWq, "dWq");
  expectClose(gpuD.wk, ref.dWk, "dWk");
  expectClose(gpuD.wv, ref.dWv, "dWv");
  expectClose(gpuD.classifier, ref.dClassifier, "dClassifier");

  // Forward-only scalar loss through the GPU chain with the given pages.
  // The trainer pages are mutated by SGD, so FD uses fresh trainer instances
  // per page, each holding that page at its initial values and SGD updates
  // that are recomputed identically from the unperturbed forward.
  const forwardLoss = async (page: "wq" | "wk" | "wv" | "classifier", perturbed: Float32Array): Promise<number> => {
    const params = {
      embedding: init.embedding.slice(),
      classifier: init.classifier.slice(),
      wq: init.wq.slice(),
      wk: init.wk.slice(),
      wv: init.wv.slice(),
    };
    params[page] = perturbed.slice();
    const t = new TrainingTrainer({
      vocabSize: V, hiddenSize: H,
      embedding: params.embedding, classifier: params.classifier,
      encoder: { headCount: HEADS, headDim: HEAD_DIM, wq: params.wq, wk: params.wk, wv: params.wv },
    });
    const res = await t.trainStep({ tokens, targets, learningRate: 0.01, mask, telemetry: true });
    t.destroy();
    return res.loss!;
  };

  const check = async (page: "wq" | "wk" | "wv" | "classifier", gpuGrad: Float32Array) => {
    const base = init[page].slice();
    const indices = [0, 1, 7, 15, 31];
    for (const index of indices) {
      if (index >= base.length) continue;
      const plus = base.slice();
      const minus = base.slice();
      plus[index] = base[index]! + EPS;
      minus[index] = base[index]! - EPS;
      const lossPlus = await forwardLoss(page, plus);
      const lossMinus = await forwardLoss(page, minus);
      const numeric = (lossPlus - lossMinus) / (2 * EPS);
      const analytical = gpuGrad[index]!;
      const bound = 2e-3 + 8e-2 * Math.abs(analytical);
      expect(
        Math.abs(numeric - analytical) <= bound,
        `${page}[${index}]: numeric ${numeric}, analytical ${analytical}`,
      ).toBe(true);
    }
  };

  await check("wq", gpuD.wq);
  await check("wk", gpuD.wk);
  await check("wv", gpuD.wv);
  await check("classifier", gpuD.classifier);

  trainer.destroy();
});
