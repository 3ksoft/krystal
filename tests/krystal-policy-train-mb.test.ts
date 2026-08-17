// M-B (S5-S6) end-to-end training proof: spatial reachability and active
// perception, with S1-S4 replay and held-out seeds/resource ids/layouts.
import { expect, test } from "bun:test";
import { buildFixtureActionCatalog } from "../packages/krystal/src/fixtures/action-intents.ts";
import { buildCurriculum } from "../packages/krystal/src/bridge/curriculum.ts";
import { lowerPolicyFrame, type PolicyEpisode } from "../packages/krystal/src/bridge/policy.ts";
import { KrystalBackward } from "../packages/webgpu/src/krystal-backward.ts";
import { KrystalForward } from "../packages/webgpu/src/krystal-forward.ts";
import {
  POLICY_CONFIG,
  createBrainForwardWeights,
  emitPrediction,
  getTrainingHarness,
  packBrainFrame,
  prepareTrainFrame,
  productionSelection,
} from "./policy-harness.ts";

function coverage(episodes: readonly PolicyEpisode[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const episode of episodes) {
    for (const frame of episode.frames) {
      const suffix = frame.gold.refToken === undefined ? "" : "+ref";
      const key = `${episode.stage} | ${frame.gold.action}${suffix}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  return counts;
}

function printCoverage(title: string, episodes: readonly PolicyEpisode[]): Map<string, number> {
  const counts = coverage(episodes);
  console.log(`coverage ${title} (${episodes.length} episodes):`);
  for (const [key, count] of [...counts].sort()) console.log(`  ${key}: ${count}`);
  return counts;
}

test("M-B (S5-S6): reachability + LOOK transitions generalize on unseen episodes", async () => {
  const h = await getTrainingHarness();
  const catalog = buildFixtureActionCatalog();
  const split = buildCurriculum({
    stages: ["S5", "S6"],
    replayStages: ["S1", "S2", "S3", "S4"],
    trainSeeds: [0, 256],
    evalSeeds: [256, 320],
  });
  console.log(split.log.join("\n"));

  const trainCoverage = printCoverage("M-B train", split.train);
  const evalCoverage = printCoverage("M-B eval", split.eval);
  for (const counts of [trainCoverage, evalCoverage]) {
    expect(counts.get("S5 | MOVE_TOWARDS+ref") ?? 0).toBeGreaterThan(0);
    expect(counts.get("S5 | EAT+ref") ?? 0).toBeGreaterThan(0);
    expect(counts.get("S6 | LOOK+ref") ?? 0).toBeGreaterThan(0);
    expect(counts.get("S6 | EAT+ref") ?? 0).toBeGreaterThan(0);
    expect(counts.get("S6 | CRY") ?? 0).toBeGreaterThan(0);
  }
  // Pin the held-out split: 32 episodes per stage and both S6 outcomes.
  expect(evalCoverage.get("S5 | MOVE_TOWARDS+ref")).toBe(32);
  expect(evalCoverage.get("S5 | EAT+ref")).toBe(32);
  expect(evalCoverage.get("S6 | LOOK+ref")).toBe(32);
  expect(evalCoverage.get("S6 | EAT+ref")).toBe(16);
  expect(evalCoverage.get("S6 | CRY")).toBe(16);

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
  const stageJoint = new Map<string, { correct: number; total: number }>();
  let intentCorrect = 0;
  let pointerFrames = 0;
  let pointerCorrect = 0;
  let pointerGivenIntent = 0;
  let pointerGivenIntentCorrect = 0;
  let jointCorrect = 0;
  let invalidPointer = 0;
  const evalFrames = split.eval.flatMap((episode) =>
    episode.frames.map((frame) => ({ episode, frame })),
  );
  for (const { episode, frame } of evalFrames) {
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

    const stage = stageJoint.get(episode.stage) ?? { correct: 0, total: 0 };
    stage.total++;
    if (jointOK) stage.correct++;
    stageJoint.set(episode.stage, stage);
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
  console.log(`M-B eval: ${n} held-out frames (seeds 256..319)`);
  console.log(`  intent accuracy            ${intentAcc}`);
  console.log(`  pointer accuracy           ${pointerAcc} (${pointerCorrect}/${pointerFrames})`);
  console.log(`  pointer | correct intent   ${pointerGivenIntentAcc} (${pointerGivenIntentCorrect}/${pointerGivenIntent})`);
  console.log(`  joint exact-match          ${jointAcc}`);
  console.log(`  invalid-pointer rate       ${invalidRate} (${invalidPointer}/${n})`);
  for (const [stage, score] of stageJoint) {
    console.log(`  ${stage} joint               ${score.correct / score.total} (${score.correct}/${score.total})`);
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
  for (const score of stageJoint.values()) expect(score.correct).toBe(score.total);
  runner.destroy();
}, 240_000);
