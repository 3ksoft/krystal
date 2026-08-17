// Finite-difference gradient checks for the complete tiny training graph
// (WEBGPU_BACKWARD_PLAN.md §14.3). The analytical gradients produced by the
// GPU backward pass are compared against central differences of the scalar
// loss, on a small selected subset of embedding and classifier parameters.
import { expect, test } from "bun:test";
import { getTrainingHarness, readArenaRegion, runPassWait, uploadTokens, uploadTargets, createWeightPage, type TrainingHarness } from "./training-harness.ts";
import { KRYSTAL_TRAINING_ARENA, TRAINING_ARENA_BASE } from "../packages/webgpu/src/krystal-layout.ts";
import { TrainingTrainer } from "../packages/webgpu/src/training.ts";

const V = 8;
const H = 6;
const M = 4;
const EPS = 1e-3;

// Deterministic init: same family as the documented convention (Xavier for
// matrices, small normal for embeddings), fixed seed so CPU/GPU tests agree.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(rand: () => number): number {
  // Box-Muller; two uniforms in (-1, 1].
  const u = Math.max(1e-12, (rand() * 2) - 1);
  const v = Math.max(1e-12, (rand() * 2) - 1);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function initParams(v: number, h: number): { embedding: Float32Array; classifier: Float32Array } {
  const rand = mulberry32(0xfeedbeef);
  const embedding = new Float32Array(v * h);
  for (let i = 0; i < embedding.length; i++) embedding[i] = normal(rand) * 0.5;
  const limit = Math.sqrt(6 / (h + v));
  const classifier = new Float32Array(v * h);
  for (let i = 0; i < classifier.length; i++) classifier[i] = (rand() * 2 - 1) * limit;
  return { embedding, classifier };
}

function region(h: TrainingHarness, name: keyof typeof KRYSTAL_TRAINING_ARENA): number {
  return TRAINING_ARENA_BASE + KRYSTAL_TRAINING_ARENA[name];
}

/** Forward-only scalar loss for the toy graph with the given parameter pages. */
async function forwardLoss(
  h: TrainingHarness,
  embeddingPage: GPUBuffer,
  classifierPage: GPUBuffer,
  tokens: Uint32Array,
  targets: Uint32Array,
): Promise<number> {
  const hidden = region(h, "hidden");
  const logits = region(h, "logits");
  const lossRows = region(h, "lossRows");
  const scalarLoss = region(h, "scalarLoss");
  await runPassWait(h, "embedding_f32", { tokenCount: M, inputDim: V, outputDim: H, outputOffset: hidden, u0: 0 }, embeddingPage);
  await runPassWait(h, "matmul_f32", {
    inputOffset: hidden, outputOffset: logits, tokenCount: M, inputDim: H, outputDim: V, rowStart: 0, rowCount: V,
  }, classifierPage);
  await runPassWait(h, "cross_entropy_forward_backward", {
    inputOffset: logits, outputOffset: region(h, "dLogits"), auxOffset: lossRows, tokenCount: M, outputDim: V, u1: 0,
  });
  await runPassWait(h, "loss_reduce", { inputOffset: lossRows, outputOffset: scalarLoss, tokenCount: M });
  const scalar = await readArenaRegion(h, scalarLoss, 1);
  return scalar[0]!;
}

test("embedding and classifier analytical gradients match central differences", async () => {
  const h = await getTrainingHarness();
  const { embedding: embInit, classifier: clsInit } = initParams(V, H);
  const tokens = Uint32Array.from([0, 1, 2, 3]);
  const targets = Uint32Array.from([3, 4, 5, 6]);
  await uploadTokens(h, tokens);
  await uploadTargets(h, targets);

  // One GPU trainStep produces dEmbedding/dClassifier in the arena (the step
  // also applies SGD, but the gradient buffers are left intact afterwards).
  const trainer = new TrainingTrainer({ vocabSize: V, hiddenSize: H, embedding: embInit, classifier: clsInit });
  await trainer.trainStep({ tokens, targets, learningRate: 0.01 });

  const gpuDEmbedding = await readArenaRegion(h, region(h, "dEmbedding"), V * H);
  const gpuDClassifier = await readArenaRegion(h, region(h, "dClassifier"), V * H);

  // Numeric gradient via central differences on a small selected subset of
  // parameters. `embPage`/`clsPage` are the pages used by forwardLoss; exactly
  // one of them varies while the other stays at its initial values.
  const embPage = createWeightPage(h, embInit);
  const clsPage = createWeightPage(h, clsInit);

  const check = async (
    params: Float32Array,
    perturbPage: GPUBuffer,
    fixedPage: GPUBuffer,
    gpuGrad: Float32Array,
    label: string,
  ) => {
    const perturbed = params.slice();
    const indices = [0, 1, 5, 13, 29, 40]; // spread over the page
    for (const index of indices) {
      perturbed[index] = params[index]! + EPS;
      h.device.queue.writeBuffer(perturbPage, 0, perturbed);
      await h.device.queue.onSubmittedWorkDone();
      const lossPlus = await forwardLoss(h, embPage, clsPage, tokens, targets);
      perturbed[index] = params[index]! - EPS;
      h.device.queue.writeBuffer(perturbPage, 0, perturbed);
      await h.device.queue.onSubmittedWorkDone();
      const lossMinus = await forwardLoss(h, embPage, clsPage, tokens, targets);
      const numeric = (lossPlus - lossMinus) / (2 * EPS);
      const analytical = gpuGrad[index]!;
      const bound = 1e-3 + 2e-2 * Math.abs(analytical);
      expect(
        Math.abs(numeric - analytical) <= bound,
        `${label}[${index}]: numeric ${numeric}, analytical ${analytical}`,
      ).toBe(true);
      perturbed[index] = params[index]!;
    }
    h.device.queue.writeBuffer(perturbPage, 0, params);
  };

  await check(embInit, embPage, clsPage, gpuDEmbedding, "dEmbedding");
  await check(clsInit, clsPage, embPage, gpuDClassifier, "dClassifier");

  embPage.destroy();
  clsPage.destroy();

  trainer.destroy();
});
