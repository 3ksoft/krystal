// Isolated M-C slice for S7-S8: capability-grounded food selection and
// consequence-conditioned avoidance, before adding S9 memory and S10 noise.
import { expect, test } from "bun:test";
import { buildFixtureActionCatalog } from "../../packages/krystal/src/fixtures/action-intents.ts";
import { buildCurriculum } from "../../packages/krystal/src/training/curriculum.ts";
import { lowerPolicyFrame, type PolicyEpisode } from "../../packages/krystal/src/training/policy.ts";
import { KrystalBackward } from "../../packages/webgpu/src/krystal-backward.ts";
import { KrystalForward } from "../../packages/webgpu/src/krystal-forward.ts";
import {
  POLICY_CONFIG,
  createBrainForwardWeights,
  emitPrediction,
  getTrainingHarness,
  packBrainFrame,
  prepareTrainFrame,
  productionSelection,
} from "./policy-harness.ts";

function variant(episode: PolicyEpisode, frameIndex: number): string {
  const frame = episode.frames[frameIndex]!;
  const resource = frame.resources[0];
  const kind = resource?.kind ?? "none";
  const poison = resource?.properties.includes("POISONED") ? ":poison" : "";
  return `${episode.stage} | ${frame.gold.action}${frame.gold.refToken === undefined ? "" : "+ref"} | ${kind}${poison}`;
}

function coverage(episodes: readonly PolicyEpisode[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const episode of episodes) {
    for (let i = 0; i < episode.frames.length; i++) {
      const key = variant(episode, i);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  return counts;
}

test("S7-S8 slice: capability + consequences generalize across foods and refs", async () => {
  const h = await getTrainingHarness();
  const catalog = buildFixtureActionCatalog();
  const split = buildCurriculum({
    stages: ["S7", "S8"],
    replayStages: ["S1", "S2", "S3", "S4", "S5", "S6"],
    trainSeeds: [0, 256],
    evalSeeds: [320, 384],
  });
  console.log(split.log.join("\n"));

  const trainCoverage = coverage(split.train);
  const evalCoverage = coverage(split.eval);
  console.log(`coverage S7-S8 train (${split.train.length} episodes):`);
  for (const [key, count] of [...trainCoverage].sort()) console.log(`  ${key}: ${count}`);
  console.log(`coverage S7-S8 eval (${split.eval.length} episodes):`);
  for (const [key, count] of [...evalCoverage].sort()) console.log(`  ${key}: ${count}`);

  // Every edible schema is learned and evaluated as a capability candidate,
  // not just the original Apple identity.
  for (const counts of [trainCoverage, evalCoverage]) {
    for (const food of ["apple", "berry", "bread"]) {
      expect(counts.get(`S7 | EAT+ref | ${food}`) ?? 0).toBeGreaterThan(0);
      expect(counts.get(`S8 | EAT+ref | ${food}`) ?? 0).toBeGreaterThan(0);
      expect(counts.get(`S8 | CRY | ${food}:poison`) ?? 0).toBeGreaterThan(0);
      expect(counts.get(`S8 | LAUGH | ${food}`) ?? 0).toBeGreaterThan(0);
    }
  }

  const trainFrames = split.train.flatMap((episode) =>
    episode.frames.map((frame) => ({ episode, frame })),
  );
  const runner = new KrystalForward(createBrainForwardWeights(POLICY_CONFIG, 42), POLICY_CONFIG);
  const trainer = new KrystalBackward(runner);
  const epochs = 3;
  let firstLoss: number | undefined;
  let lastLoss: number | undefined;
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let i = 0; i < trainFrames.length; i++) {
      const { episode, frame } = trainFrames[i]!;
      const step = epoch * trainFrames.length + i;
      const result = await trainer.trainStep({
        ...prepareTrainFrame(
          packBrainFrame(lowerPolicyFrame(frame, episode)).frame,
          frame.gold,
          catalog,
        ),
        learningRate: 0.01,
        telemetry: step === 0 || step === epochs * trainFrames.length - 1,
      });
      if (step === 0) firstLoss = result.loss;
      if (step === epochs * trainFrames.length - 1) lastLoss = result.loss;
    }
  }
  expect(firstLoss).toBeDefined();
  expect(lastLoss).toBeDefined();
  expect(firstLoss!).toBeGreaterThan(lastLoss!);

  const confusion = new Map<string, Map<string, number>>();
  const variantJoint = new Map<string, { correct: number; total: number }>();
  let intentCorrect = 0;
  let pointerFrames = 0;
  let pointerCorrect = 0;
  let pointerGivenIntent = 0;
  let pointerGivenIntentCorrect = 0;
  let jointCorrect = 0;
  let invalidPointer = 0;
  const evalFrames = split.eval.flatMap((episode) =>
    episode.frames.map((frame, frameIndex) => ({ episode, frame, frameIndex })),
  );
  for (const { episode, frame, frameIndex } of evalFrames) {
    const sel = await productionSelection(
      h,
      runner,
      packBrainFrame(lowerPolicyFrame(frame, episode)).frame,
      catalog,
    );
    const pred = sel ? emitPrediction(sel, catalog) : null;
    const needsPointer = frame.gold.refToken !== undefined;
    const intentOK = pred?.action === frame.gold.action;
    const pointerOK = !needsPointer || pred?.refToken === frame.gold.refToken;
    const jointOK = intentOK && pointerOK;
    if (intentOK) intentCorrect++;
    if (needsPointer) {
      pointerFrames++;
      if (pointerOK) pointerCorrect++;
      if (intentOK) {
        pointerGivenIntent++;
        if (pointerOK) pointerGivenIntentCorrect++;
      }
      if (pred !== null && !pointerOK) invalidPointer++;
    }
    if (jointOK) jointCorrect++;

    const key = variant(episode, frameIndex);
    const score = variantJoint.get(key) ?? { correct: 0, total: 0 };
    score.total++;
    if (jointOK) score.correct++;
    variantJoint.set(key, score);
    const row = confusion.get(frame.gold.action) ?? new Map<string, number>();
    const predAction = pred?.action ?? "(none)";
    row.set(predAction, (row.get(predAction) ?? 0) + 1);
    confusion.set(frame.gold.action, row);
  }

  const n = evalFrames.length;
  const intentAcc = intentCorrect / n;
  const pointerAcc = pointerCorrect / pointerFrames;
  const pointerGivenIntentAcc = pointerGivenIntentCorrect / pointerGivenIntent;
  const jointAcc = jointCorrect / n;
  const invalidRate = invalidPointer / n;
  console.log(`S7-S8 eval: ${n} held-out frames (seeds 320..383)`);
  console.log(`  intent accuracy            ${intentAcc}`);
  console.log(`  pointer accuracy           ${pointerAcc} (${pointerCorrect}/${pointerFrames})`);
  console.log(`  pointer | correct intent   ${pointerGivenIntentAcc} (${pointerGivenIntentCorrect}/${pointerGivenIntent})`);
  console.log(`  joint exact-match          ${jointAcc}`);
  console.log(`  invalid-pointer rate       ${invalidRate} (${invalidPointer}/${n})`);
  console.log("  variant joint:");
  for (const [key, score] of [...variantJoint].sort()) {
    console.log(`    ${key}: ${score.correct}/${score.total}`);
  }
  console.log("  confusion (gold -> pred):");
  for (const [gold, row] of confusion) {
    console.log(`    ${gold.padEnd(12)} -> ${[...row].map(([a, c]) => `${a}:${c}`).join(", ")}`);
  }

  expect(intentAcc).toBe(1);
  expect(pointerAcc).toBe(1);
  expect(pointerGivenIntentAcc).toBe(1);
  expect(jointAcc).toBe(1);
  expect(invalidRate).toBe(0);
  for (const score of variantJoint.values()) expect(score.correct).toBe(score.total);
  runner.destroy();
}, 240_000);
