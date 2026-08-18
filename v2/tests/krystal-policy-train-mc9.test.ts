// Isolated M-C S9 slice: preserve the pending grounded action when the exact
// target leaves Vision and remains only in the Memory band.
import { expect, test } from "bun:test";
import { buildFixtureActionCatalog } from "../../packages/krystal/src/fixtures/action-intents.ts";
import { buildCurriculum } from "../../packages/krystal/src/bridge/curriculum.ts";
import { mix32 } from "../../packages/krystal/src/bridge/comfort.ts";
import {
  generatePolicyEpisode,
  lowerPolicyFrame,
  type PolicyEpisode,
  type PolicyStage,
} from "../../packages/krystal/src/bridge/policy.ts";
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
  const phase = frame.resources[0]?.band ?? "none";
  return `${episode.stage} | ${phase} | ${frame.gold.action}${frame.gold.refToken === undefined ? "" : "+ref"}`;
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

test("S9 slice: exact pending ref and action survive Vision -> Memory", async () => {
  const h = await getTrainingHarness();
  const catalog = buildFixtureActionCatalog();
  const split = buildCurriculum({
    stages: ["S9"],
    replayStages: ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
    trainSeeds: [0, 256],
    evalSeeds: [384, 448],
  });
  console.log(split.log.join("\n"));

  // With eight replay stages, seed-based stage selection picks S1 only on
  // even seeds; S1's legacy parity variant would therefore contain CRY only.
  // Balance that replay slice explicitly without changing the generator used
  // by the already-closed M-A/M-B milestones.
  const trainEpisodes = split.train.map((episode): PolicyEpisode => {
    if (episode.stage !== "S1") return episode;
    const bad = (mix32((episode.seed >>> 0) ^ 0x51) & 1) === 0;
    const frame = episode.frames[0]!;
    return {
      ...episode,
      frames: [{
        ...frame,
        comfort: bad ? -1 : 1,
        gold: { action: bad ? "CRY" : "LAUGH" },
      }],
    };
  });

  const trainCoverage = coverage(trainEpisodes);
  const evalCoverage = coverage(split.eval);
  console.log(`coverage S9 train (${split.train.length} episodes):`);
  for (const [key, count] of [...trainCoverage].sort()) console.log(`  ${key}: ${count}`);
  console.log(`coverage S9 eval (${split.eval.length} episodes):`);
  for (const [key, count] of [...evalCoverage].sort()) console.log(`  ${key}: ${count}`);
  for (const counts of [trainCoverage, evalCoverage]) {
    for (const phase of ["vision", "memory"]) {
      expect(counts.get(`S9 | ${phase} | EAT+ref`) ?? 0).toBeGreaterThan(0);
      expect(counts.get(`S9 | ${phase} | MOVE_TOWARDS+ref`) ?? 0).toBeGreaterThan(0);
    }
  }
  expect(evalCoverage.get("S9 | vision | EAT+ref")).toBe(32);
  expect(evalCoverage.get("S9 | memory | EAT+ref")).toBe(32);
  expect(evalCoverage.get("S9 | vision | MOVE_TOWARDS+ref")).toBe(32);
  expect(evalCoverage.get("S9 | memory | MOVE_TOWARDS+ref")).toBe(32);

  const trainFrames = trainEpisodes.flatMap((episode) =>
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
    const intentOK = pred?.action === frame.gold.action;
    const pointerOK = pred?.refToken === frame.gold.refToken;
    const jointOK = intentOK && pointerOK;
    if (intentOK) {
      intentCorrect++;
      pointerGivenIntent++;
      if (pointerOK) pointerGivenIntentCorrect++;
    }
    if (pointerOK) pointerCorrect++;
    if (jointOK) jointCorrect++;
    if (pred !== null && !pointerOK) invalidPointer++;

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
  const pointerAcc = pointerCorrect / n;
  const pointerGivenIntentAcc = pointerGivenIntentCorrect / pointerGivenIntent;
  const jointAcc = jointCorrect / n;
  const invalidRate = invalidPointer / n;
  console.log(`S9 eval: ${n} held-out frames (seeds 384..447)`);
  console.log(`  intent accuracy            ${intentAcc}`);
  console.log(`  pointer accuracy           ${pointerAcc} (${pointerCorrect}/${n})`);
  console.log(`  pointer | correct intent   ${pointerGivenIntentAcc} (${pointerGivenIntentCorrect}/${pointerGivenIntent})`);
  console.log(`  joint exact-match          ${jointAcc}`);
  console.log(`  invalid-pointer rate       ${invalidRate} (${invalidPointer}/${n})`);
  console.log("  phase/action joint:");
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

  // One-checkpoint retention measurement: audit every prior stage on a fresh
  // eval ref band after the S9 update. The comparison is intentionally kept
  // on this same runner/checkpoint boundary; the requested theoretical target
  // for S1-S8 after S9 is 0.646 (62/96 = 0.645833...).
  const priorStages: readonly PolicyStage[] = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"];
  const priorEpisodes = priorStages.flatMap((stage, stageIndex) =>
    Array.from({ length: 8 }, (_, i) => generatePolicyEpisode(stage, 448 + stageIndex * 8 + i, "eval")),
  );
  const priorScores = new Map<PolicyStage, { correct: number; total: number }>();
  const priorFailures: string[] = [];
  let priorJoint = 0;
  let priorTotal = 0;
  for (const episode of priorEpisodes) {
    for (const frame of episode.frames) {
      const sel = await productionSelection(
        h,
        runner,
        packBrainFrame(lowerPolicyFrame(frame, episode)).frame,
        catalog,
      );
      const pred = sel ? emitPrediction(sel, catalog) : null;
      const correct = pred?.action === frame.gold.action &&
        (frame.gold.refToken === undefined || pred.refToken === frame.gold.refToken);
      const score = priorScores.get(episode.stage) ?? { correct: 0, total: 0 };
      score.total++;
      if (correct) {
        score.correct++;
        priorJoint++;
      } else {
        priorFailures.push(
          `${episode.stage} seed=${episode.seed} gold=${frame.gold.action}` +
          `${frame.gold.refToken === undefined ? "" : `#${frame.gold.refToken.toString(16)}`}` +
          ` pred=${pred?.action ?? "(none)"}` +
          `${pred?.refToken === undefined ? "" : `#${pred.refToken.toString(16)}`}`,
        );
      }
      priorTotal++;
      priorScores.set(episode.stage, score);
    }
  }
  console.log(`S9 replay retention: ${priorJoint}/${priorTotal} held-out S1-S8 frames`);
  for (const [stage, score] of priorScores) {
    console.log(`  ${stage}: ${score.correct}/${score.total}`);
  }
  for (const failure of priorFailures) console.log(`  FAIL ${failure}`);
  // This is an isolated S9 slice, not the final S7-S10 milestone gate. Keep
  // this one-checkpoint measurement explicit; the combined M-C proof has a
  // different, stronger retention contract.
  expect(priorJoint / priorTotal).toBeCloseTo(0.646, 3);
  runner.destroy();
}, 240_000);
